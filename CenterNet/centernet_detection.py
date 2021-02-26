import os
from argparse import ArgumentParser
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as VF
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pycocotools.cocoeval import COCOeval
from torchvision.datasets import CocoDetection

from centernet import CenterNet
from models.heads import CenterHead
from transforms import CategoryIdToClass, ImageAugmentation
from transforms.sample import ComposeSample
from transforms.ctdet import CenterDetectionSample
from utils.losses import RegL1Loss, FocalLoss
from utils.decode import sigmoid_clamped
from decode.ctdet import ctdet_decode
from utils.nms import soft_nms


class CenterNetDetection(CenterNet):
    mean = [0.408, 0.447, 0.470]
    std = [0.289, 0.274, 0.278]
    max_objs = 128
    valid_ids = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
        14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
        58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
        72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
        82, 84, 85, 86, 87, 88, 89, 90
    ]

    def __init__(
        self,
        arch,
        learning_rate=1e-4,
        hm_weight=1,
        wh_weight=1,
        off_weight=0.1,
        num_classes=80,
        test_coco=None,
        test_scales=None,
        test_flip=False,
    ):
        super().__init__(arch)

        self.num_classes = num_classes
        heads = {
            "heatmap": self.num_classes,
            "width_height": 2,
            "regression": 2
        }
        self.heads = torch.nn.ModuleList([
            CenterHead(heads, self.backbone.out_channels, self.head_conv)
            for _ in range(self.num_stacks)
        ])

        # Test
        self.test_coco = test_coco
        self.test_coco_ids = list(sorted(self.test_coco.imgs.keys()))
        self.test_max_per_image = 100
        self.test_scales = [1] if test_scales is None else test_scales
        self.test_flip = test_flip

        self.criterion = FocalLoss()
        self.criterion_regression = RegL1Loss()
        self.criterion_width_height = RegL1Loss()

        self.save_hyperparameters()

    def forward(self, x):
        outputs = self.backbone(x)

        rets = []
        for head, output in zip(self.heads, outputs):
            rets.append(head(output))

        return rets

    def loss(self, outputs, target):
        hm_loss, wh_loss, off_loss = 0, 0, 0
        num_stacks = len(outputs)

        for s in range(num_stacks):
            output = outputs[s]
            output["heatmap"] = sigmoid_clamped(output["heatmap"])

            hm_loss += self.criterion(output["heatmap"], target["heatmap"])
            wh_loss += self.criterion_width_height(
                output["width_height"], target["regression_mask"], target["indices"], target["width_height"]
            )
            off_loss += self.criterion_regression(
                output["regression"], target["regression_mask"], target["indices"], target["regression"]
            )

        loss = (
            self.hparams.hm_weight * hm_loss
            + self.hparams.wh_weight * wh_loss
            + self.hparams.off_weight * off_loss
        ) / num_stacks
        loss_stats = {
            "loss": loss,
            "hm_loss": hm_loss,
            "wh_loss": wh_loss,
            "off_loss": off_loss,
        }
        return loss, loss_stats

    def test_step(self, batch, batch_idx):
        img, target = batch
        image_id = self.test_coco_ids[batch_idx]

        # Test augmentation
        images = []
        meta = []
        for scale in self.test_scales:
            _, _, height, width = img.shape
            new_height = int(height * scale)
            new_width = int(width * scale)
            pad_top_bottom = ((new_height | self.padding) + 1 - new_height) // 2
            pad_left_right = ((new_width | self.padding) + 1 - new_width) // 2

            img_scaled = VF.resize(img, [new_height, new_width])
            img_scaled = F.pad(
                img_scaled, (pad_left_right, pad_left_right, pad_top_bottom, pad_top_bottom)
            )
            img_scaled = VF.normalize(img_scaled, self.mean, self.std)

            if self.test_flip:
                img_scaled = torch.cat([img_scaled, VF.hflip(img_scaled)])

            images.append(img_scaled)
            meta.append({
                "scale": [new_width / width, new_height / height],
                "padding": [pad_left_right, pad_top_bottom]
            })

        # Forward
        outputs = []
        for image in images:
            outputs.append(self(image)[-1])

        if self.test_flip:
            for output in outputs:
                output["heatmap"] = (output["heatmap"][0:1] + VF.hflip(output["heatmap"][1:2])) / 2
                output["width_height"] = (output["width_height"][0:1] + VF.hflip(output["width_height"][1:2])) / 2
                output["regression"] = output["regression"][0:1]

        return image_id, outputs, meta

    def test_step_end(self, outputs):
        image_id, outputs, metas = outputs

        detections = []
        for i in range(len(outputs)):
            output = outputs[i]
            meta = metas[i]

            detection = ctdet_decode(
                output["heatmap"].sigmoid_(), output["width_height"], reg=output["regression"]
            )
            detection = detection.cpu().detach().squeeze()

            # Transform detection to original image
            padding = torch.FloatTensor(meta["padding"] + meta["padding"])
            scale = torch.FloatTensor(meta["scale"] + meta["scale"])
            detection[:, :4] *= self.down_ratio  # Scale to input
            detection[:, :4] -= padding  # Remove pad
            detection[:, :4] /= scale  # Compensate scale

            # Group detections by class
            class_predictions = {}
            classes = detection[:, -1]
            for j in range(self.num_classes):
                indices = classes == j
                class_predictions[j + 1] = detection[indices, :5].numpy().reshape(-1, 5)

            detections.append(class_predictions)

        # Merge detections
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate([detection[j] for detection in detections], axis=0)
            if len(self.test_scales) > 1:
                keep_indices = soft_nms(results[j], Nt=0.5, method=2)
                results[j] = results[j][keep_indices]

        # Keep only best detections
        scores = np.hstack([results[j][:, 4] for j in range(1, self.num_classes + 1)])
        if len(scores) > self.test_max_per_image:
            kth = len(scores) - self.test_max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_indices = results[j][:, 4] >= thresh
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

                category_id = self.valid_ids[class_index - 1]
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

        prefix = ""
        if len(self.test_scales) > 1:
            prefix += "multi-scale_"
        if self.test_flip:
            prefix += "flip_"

        self.log(f"test/{prefix}ap", coco_eval.stats[0])
        self.log(f"test/{prefix}ap_50", coco_eval.stats[1])
        self.log(f"test/{prefix}ap_75", coco_eval.stats[2])
        self.log(f"test/{prefix}ap_S", coco_eval.stats[3])
        self.log(f"test/{prefix}ap_M", coco_eval.stats[4])
        self.log(f"test/{prefix}ap_L", coco_eval.stats[5])


def cli_main():
    pl.seed_everything(5318008)
    ia.seed(107734)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument("image_root")
    parser.add_argument("annotation_root")

    parser.add_argument("--pretrained_weights_path")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = CenterNetDetection.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    train_transform = ComposeSample(
        [
            ImageAugmentation(
                iaa.Sequential([
                    iaa.Sequential(
                        [
                            iaa.Fliplr(0.5),
                            iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
                            iaa.LinearContrast((0.75, 1.5)),
                            iaa.AdditiveGaussianNoise(
                                loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                            ),
                            iaa.Multiply((0.8, 1.2), per_channel=0.1),
                            iaa.Affine(
                                scale={"x": (0.6, 1.4), "y": (0.6, 1.4)},
                                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                                rotate=(-5, 5),
                                shear=(-3, 3),
                            ),
                        ],
                        random_order=True
                    ),
                    iaa.PadToFixedSize(width=512, height=512),
                    iaa.CropToFixedSize(width=512, height=512)
                ]),
                torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(
                            CenterNetDetection.mean, CenterNetDetection.std
                        ),
                    ]
                ),
            ),
            CategoryIdToClass(CenterNetDetection.valid_ids),
            CenterDetectionSample(),
        ]
    )

    valid_transform = ComposeSample(
        [
            ImageAugmentation(
                iaa.Sequential([
                    iaa.PadToFixedSize(width=512, height=512),
                    iaa.CropToFixedSize(width=512, height=512)
                ]),
                torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(
                            CenterNetDetection.mean, CenterNetDetection.std
                        ),
                    ]
                ),
            ),
            CategoryIdToClass(CenterNetDetection.valid_ids),
            CenterDetectionSample(),
        ]
    )

    test_transform = ComposeSample(
        [ImageAugmentation(img_transforms=torchvision.transforms.ToTensor())]
    )

    coco_train = CocoDetection(
        os.path.join(args.image_root, "train2017"),
        os.path.join(args.annotation_root, "instances_train2017.json"),
        transforms=train_transform,
    )

    coco_val = CocoDetection(
        os.path.join(args.image_root, "val2017"),
        os.path.join(args.annotation_root, "instances_val2017.json"),
        transforms=valid_transform,
    )

    coco_test = CocoDetection(
        os.path.join(args.image_root, "val2017"),
        os.path.join(args.annotation_root, "instances_val2017.json"),
        transforms=test_transform,
    )

    train_loader = DataLoader(
        coco_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        coco_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(coco_test, batch_size=1, num_workers=0, pin_memory=True)

    # ------------
    # model
    # ------------
    model = CenterNetDetection(args.arch, args.learning_rate, test_coco=coco_test.coco)
    if args.pretrained_weights_path:
        model.load_pretrained_weights(args.pretrained_weights_path)

    # ------------
    # training
    # ------------
    args.callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            dirpath="model_weights",
            filename=args.arch + "detection-{epoch:02d}-{val_loss:.2f}",
        )
    ]

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    trainer.test(model, test_dataloaders=test_loader)


if __name__ == "__main__":
    cli_main()
