from argparse import ArgumentParser
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import os
import pytorch_lightning as pl
import torch
import torchvision
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader

from centernet import CenterNet
from datasets.coco import CocoDetection
from decode.ctdet import ctdet_decode
from transforms import CategoryIdToClass, ImageAugmentation, ComposeSample
from transforms.ctdet import CenterDetectionSample
from transforms.sample import PoseFlip
from utils.decode import sigmoid_clamped
from utils.losses import RegL1Loss, FocalLoss
from utils.nms import soft_nms


class CenterNetDetection(CenterNet):
    def __init__(self, arch, heads, head_conv, learning_rate=1e-4, hm_weight=1, wh_weight=1, off_weight=0.1,
                 test_augmentation="none", num_classes=80, test_coco=None):
        super().__init__(arch, heads, head_conv)
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.test_coco = test_coco
        self.test_coco_ids = list(sorted(self.test_coco.imgs.keys()))
        self.test_max_per_image = 100
        self.test_scales = [0.5, 0.75, 1, 1.25, 1.5] if test_augmentation == "multi-scale" else [1]
        self.test_flip = test_augmentation == "flip"

        self.criterion = FocalLoss()
        self.criterion_regression = RegL1Loss()
        self.criterion_width_height = RegL1Loss()

    def forward(self, x):
        return self.model.forward(x)

    def loss(self, outputs, target):
        hm_loss, wh_loss, off_loss = 0, 0, 0
        num_stacks = len(outputs)

        for s in range(num_stacks):
            output = outputs[s]
            output['hm'] = sigmoid_clamped(output['hm'])

            hm_loss += self.criterion(output['hm'], target['hm'])
            wh_loss += self.criterion_width_height(output['wh'], target['reg_mask'], target['ind'], target['wh'])
            off_loss += self.criterion_regression(output['reg'], target['reg_mask'], target['ind'], target['reg'])

        loss = (self.hparams.hm_weight * hm_loss +
                self.hparams.wh_weight * wh_loss +
                self.hparams.off_weight * off_loss) / num_stacks
        loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'wh_loss': wh_loss, 'off_loss': off_loss}
        return loss, loss_stats

    def test_step(self, batch, batch_idx):
        img, target = batch
        image_id = self.test_coco_ids[batch_idx]

        # Test augmentation
        augmented_images = []
        meta = []
        for scale in self.test_scales:
            _, _, height, width = img.shape
            new_height = (int(height * scale) | self.padding) + 1
            new_width = (int(width * scale) | self.padding) + 1

            scaled_img = torchvision.transforms.Resize((new_height, new_width))(img)

            augmented_images.append(scaled_img)
            meta.append({
                "original_size": (height, width),
                "input_size": (new_height, new_height),
                "scale": (height/new_height, width/new_width),
            })

        flip = torchvision.transforms.RandomHorizontalFlip(1)
        if self.test_flip:
            flipped_images = [flip(img) for img in augmented_images]
            augmented_images.extend(flipped_images)

        # Forward
        outputs = []
        for image in augmented_images:
            outputs.append(self(image)[0])

        if self.test_flip:
            size = len(outputs) // 2
            for i in range(size):
                outputs[i]["hm"] = (outputs[i]["hm"] + flip(outputs[i + size]["hm"])) / 2
                outputs[i]["wh"] = (outputs[i]["wh"] + flip(outputs[i + size]["wh"])) / 2
            outputs = outputs[:size]

        return image_id, outputs, meta

    def test_step_end(self, outputs):
        image_id, outputs, metas = outputs

        detections = []
        for i in range(len(outputs)):
            output = outputs[i]
            meta = metas[i]

            dets = ctdet_decode(output['hm'].sigmoid_(), output['wh'], reg=output['reg']).cpu().detach().squeeze()

            # Scale coordinates to original image
            height_scale, width_scale = meta["scale"]
            height_scale, width_scale = height_scale * self.down_ratio, width_scale * self.down_ratio
            scales = torch.FloatTensor([width_scale, height_scale, width_scale, height_scale])
            dets[:, :4] = dets[:, :4] * scales

            # Group detections by class
            class_predictions = {}
            classes = dets[:, -1]
            for j in range(self.num_classes):
                indices = (classes == j)
                class_predictions[j + 1] = dets[indices, :5].numpy().reshape(-1, 5)

            detections.append(class_predictions)

        # Merge detections
        results = {}
        for j in range(1, self.num_classes + 1):
            boxes = np.concatenate([detection[j] for detection in detections], axis=0)
            results[j] = boxes
            if boxes.shape[0] > 0:
                keep_indices = soft_nms(boxes, Nt=0.5, method=2)
                results[j] = boxes[keep_indices]

        # Keep only k-best detections
        scores = np.hstack([results[j][:, 4] for j in range(1, self.num_classes + 1)])
        if len(scores) > self.test_max_per_image:
            kth = len(scores) - self.test_max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_indices = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_indices]

        return image_id, results

    def test_epoch_end(self, detections):
        # Convert to COCO eval format
        # Format: imageID, x1, y1, w, h, score, class
        data = []
        for image_id, detection in detections:
            for class_index, box in detection.items():
                if box.shape[0] == 0:
                    continue

                category_id = CocoDetection.valid_ids[class_index - 1]
                category_ids = np.repeat(category_id, box.shape[0]).reshape((-1, 1))
                image_ids = np.repeat(image_id, box.shape[0]).reshape((-1, 1))

                box[:, 2] -= box[:, 0]
                box[:, 3] -= box[:, 1]

                data.append(np.hstack((image_ids, box, category_ids)))

        data = np.concatenate(data, axis=0)

        coco_detections = self.test_coco.loadRes(data)

        coco_eval = COCOeval(self.test_coco, coco_detections, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        self.log("test/ap", coco_eval.stats[0])
        self.log("test/ap_50", coco_eval.stats[1])
        self.log("test/ap_75", coco_eval.stats[2])
        self.log("test/ap_S", coco_eval.stats[3])
        self.log("test/ap_M", coco_eval.stats[4])
        self.log("test/ap_L", coco_eval.stats[5])

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = CenterNet.add_model_specific_args(parent_parser)

        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--test_augmentation",
                            default="none",
                            choices=["none", "flip", "multi-scale"])
        return parser


def cli_main():
    pl.seed_everything(5318008)
    ia.seed(107734)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('image_root')
    parser.add_argument('annotation_root')

    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = CenterNetDetection.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    train_transform = ComposeSample([
        ImageAugmentation(
            iaa.Sequential([
                iaa.Fliplr(0.5),
                iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
                iaa.LinearContrast((0.75, 1.5)),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                iaa.Multiply((0.8, 1.2), per_channel=0.1),
                iaa.Affine(
                    scale={"x": (0.6, 1.4), "y": (0.6, 1.4)},
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    rotate=(-5, 5),
                    shear=(-3, 3)
                ),
                iaa.PadToFixedSize(width=512, height=512),
                iaa.CropToFixedSize(width=512, height=512)
            ]),
            torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(CocoDetection.mean, CocoDetection.std, inplace=True)
            ])
        ),
        CategoryIdToClass(CocoDetection.valid_ids),
        PoseFlip(0.5),
        CenterDetectionSample()
    ])

    valid_transform = ComposeSample([
        ImageAugmentation(
            iaa.Sequential([
                iaa.PadToFixedSize(width=512, height=512),
                iaa.CropToFixedSize(width=512, height=512)
            ]),
            torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(CocoDetection.mean, CocoDetection.std, inplace=True)
            ])
        ),
        CategoryIdToClass(CocoDetection.valid_ids),
        CenterDetectionSample()
    ])

    test_transform = ComposeSample([
        ImageAugmentation(
            iaa.Identity(),
            torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(CocoDetection.mean, CocoDetection.std, inplace=True)
            ])
        )
    ])

    coco_train = CocoDetection(os.path.join(args.image_root, 'train2017'),
                               os.path.join(args.annotation_root, 'instances_train2017.json'),
                               transforms=train_transform)

    coco_val = CocoDetection(os.path.join(args.image_root, 'val2017'),
                             os.path.join(args.annotation_root, 'instances_val2017.json'),
                             transforms=valid_transform)

    coco_test = CocoDetection(os.path.join(args.image_root, 'val2017'),
                              os.path.join(args.annotation_root, 'instances_val2017.json'),
                              transforms=test_transform)

    train_loader = DataLoader(coco_train, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(coco_val, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(coco_test, batch_size=1, num_workers=0)

    # ------------
    # model
    # ------------
    heads = {'hm': CocoDetection.num_classes, 'wh': 2, 'reg': 2}
    if 'dla' in args.arch:
        args.head_conv = 256
    model = CenterNetDetection(args.arch, heads, args.head_conv, args.learning_rate,
                               test_augmentation=args.test_augmentation, num_classes=CocoDetection.num_classes,
                               test_coco=coco_test.coco)
    model.load_model_weights("../model_weights/ctdet_coco_dla_1x.pth")

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    trainer.test(model, test_dataloaders=test_loader, ckpt_path=None)


if __name__ == '__main__':
    cli_main()
