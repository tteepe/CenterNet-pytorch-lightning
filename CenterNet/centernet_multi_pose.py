import os
from argparse import ArgumentParser

import numpy as np
import imgaug.augmenters as iaa
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as VF
import pytorch_lightning as pl
from pycocotools.cocoeval import COCOeval
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection

from .centernet import CenterNet
from .models.heads import CenterHead
from .transforms import ImageAugmentation
from .transforms.ctdet import CenterDetectionSample
from .transforms.multi_pose import MultiPoseSample
from .transforms.sample import MultiSampleTransform, PoseFlip, ComposeSample
from .decode.multi_pose import multi_pose_decode
from .utils.decode import sigmoid_clamped
from .utils.losses import RegL1Loss, FocalLoss, RegWeightedL1Loss
from .utils.nms import soft_nms_39


class CenterNetMultiPose(CenterNet):
    mean = [0.408, 0.447, 0.470]
    std = [0.289, 0.274, 0.278]
    flip_idx = [
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15,
    ]

    def __init__(
        self,
        arch,
        learning_rate=1e-4,
        learning_rate_milestones=None,
        hm_weight=1,
        wh_weight=0.1,
        off_weight=1,
        hp_weight=1,
        hm_hp_weight=1,
        test_coco=None,
        test_coco_ids=None,
        test_scales=None,
        test_flip=True,
    ):
        super().__init__(arch)

        heads = {
            "heatmap": 1,
            "width_height": 2,
            "regression": 2,
            "keypoints": 34,
            "heatmap_keypoints": 17,
            "heatmap_keypoints_offset": 2,
        }
        self.heads = torch.nn.ModuleList(
            [
                CenterHead(heads, self.backbone.out_channels, self.head_conv)
                for _ in range(self.num_stacks)
            ]
        )

        self.learning_rate_milestones = (
            learning_rate_milestones if learning_rate_milestones is not None else []
        )

        # Test
        self.test_coco = test_coco
        self.test_coco_ids = test_coco_ids
        self.test_max_per_image = 20
        self.test_scales = [1] if test_scales is None else test_scales
        self.test_flip = test_flip

        # Loss
        self.criterion = FocalLoss()
        self.criterion_heatmap_keypoints = FocalLoss()
        self.criterion_keypoints = RegWeightedL1Loss()
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
        kp_loss, off_loss, hm_kp_loss, hm_offset_loss = 0, 0, 0, 0
        num_stacks = len(outputs)

        for s in range(num_stacks):
            output = outputs[s]
            output["heatmap"] = sigmoid_clamped(output["heatmap"])
            output["heatmap_keypoints"] = sigmoid_clamped(output["heatmap_keypoints"])

            hm_loss += self.criterion(output["heatmap"], target["heatmap"])
            wh_loss += self.criterion_width_height(
                output["width_height"],
                target["regression_mask"],
                target["indices"],
                target["width_height"],
            )
            off_loss += self.criterion_regression(
                output["regression"],
                target["regression_mask"],
                target["indices"],
                target["regression"],
            )

            kp_loss += self.criterion_keypoints(
                output["keypoints"],
                target["keypoints_mask"],
                target["indices"],
                target["keypoints"],
            )
            hm_kp_loss += self.criterion_heatmap_keypoints(
                output["heatmap_keypoints"], target["heatmap_keypoints"]
            )
            hm_offset_loss += self.criterion_regression(
                output["heatmap_keypoints_offset"],
                target["heatmap_keypoints_mask"],
                target["heatmap_keypoints_indices"],
                target["heatmap_keypoints_offset"],
            )

        loss = (
            self.hparams.hm_weight * hm_loss
            + self.hparams.wh_weight * wh_loss
            + self.hparams.off_weight * off_loss
            + self.hparams.hp_weight * kp_loss
            + self.hparams.hm_hp_weight * hm_kp_loss
            + self.hparams.off_weight * hm_offset_loss
        ) / num_stacks

        loss_stats = {
            "loss": loss,
            "hm_loss": hm_loss,
            "kp_loss": kp_loss,
            "hm_kp_loss": hm_kp_loss,
            "hm_offset_loss": hm_offset_loss,
            "wh_loss": wh_loss,
            "off_loss": off_loss,
        }
        return loss, loss_stats

    def test_step(self, batch, batch_idx):
        img, target = batch
        image_id = self.test_coco_ids[batch_idx] if self.test_coco_ids else batch_idx

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
                img_scaled,
                (pad_left_right, pad_left_right, pad_top_bottom, pad_top_bottom),
            )
            img_scaled = VF.normalize(img_scaled, self.mean, self.std)

            if self.test_flip:
                img_scaled = torch.cat([img_scaled, VF.hflip(img_scaled)])

            images.append(img_scaled)
            meta.append(
                {
                    "scale": [new_width / width, new_height / height],
                    "padding": [pad_left_right, pad_top_bottom],
                }
            )

        # Forward
        outputs = []
        for image in images:
            outputs.append(self(image)[-1])

        if self.test_flip:
            for output in outputs:
                output["heatmap"] = (output["heatmap"][0:1] + VF.hflip(output["heatmap"][1:2])) / 2
                output["width_height"] = (output["width_height"][0:1] + VF.hflip(output["width_height"][1:2])) / 2
                output["regression"] = output["regression"][0:1]

                # Flip pose aware
                num, points, height, width = output["keypoints"][1:2].shape
                flipped_keypoints = VF.hflip(output["keypoints"][1:2]).view(1, points // 2, 2, height, width)
                flipped_keypoints[:, :, 0, :, :] *= -1
                flipped_keypoints = flipped_keypoints[0:1, self.flip_idx].view(1, points, height, width)
                output["keypoints"] = (output["keypoints"][0:1] + flipped_keypoints) / 2

                flipped_heatmap = VF.hflip(output["heatmap_keypoints"][1:2])[0:1, self.flip_idx]
                output["heatmap_keypoints"] = (output["heatmap_keypoints"][0:1] + flipped_heatmap) / 2
                output["heatmap_keypoints_offset"] = output["heatmap_keypoints_offset"][0:1]

        return image_id, outputs, meta

    def test_step_end(self, outputs):
        image_id, outputs, metas = outputs

        detections = []
        for i in range(len(outputs)):
            output = outputs[i]
            meta = metas[i]

            detection = multi_pose_decode(
                output["heatmap"].sigmoid_(),
                output["width_height"],
                output["keypoints"],
                reg=output["regression"],
                hm_hp=output["heatmap_keypoints"].sigmoid_(),
                hp_offset=output["heatmap_keypoints_offset"],
            )
            detection = detection.cpu().detach().squeeze()

            # Transform detection to original image
            padding = torch.FloatTensor(meta["padding"])
            scale = torch.FloatTensor(meta["scale"])

            # Bounding Box
            detection[:, :4] *= self.down_ratio  # Scale to input
            detection[:, :4] -= torch.cat([padding, padding])  # Remove pad
            detection[:, :4] /= torch.cat([scale, scale])  # Compensate scale

            # Keypoints
            points = detection[:, 5:39].view(-1, 17, 2)
            points *= self.down_ratio
            points -= padding
            points /= scale
            detection[:, 5:39] = points.view(-1, 34)

            detections.append(detection.numpy())

        results = np.concatenate(detections, axis=0)
        if len(self.test_scales) > 1:
            keep_indices = soft_nms_39(results, Nt=0.5, method=2)
            results = results[keep_indices]

        # Keep only best detections
        scores = results[:, 4]
        if len(scores) > self.test_max_per_image:
            kth = len(scores) - self.test_max_per_image
            thresh = np.partition(scores, kth)[kth]
            keep_indices = results[:, 4] >= thresh
            results = results[keep_indices]

        return image_id, results.tolist()

    def test_epoch_end(self, results):
        if not self.test_coco:
            return

        category_id = 1

        # Convert to COCO annotation format
        data = []
        for image_id, detections in results:
            for detection in detections:
                bbox = detection[:4]
                bbox[2] -= bbox[0]
                bbox[3] -= bbox[1]
                score = detection[4]

                keypoints = (
                    np.concatenate(
                        [
                            np.array(detection[5:39], dtype=np.float32).reshape(-1, 2),
                            np.ones((17, 1), dtype=np.float32),
                        ],
                        axis=1,
                    )
                    .reshape(51)
                    .tolist()
                )

                data.append(
                    {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox,
                        "score": score,
                        "keypoints": keypoints,
                    }
                )

        coco_detections = self.test_coco.loadRes(data)

        coco_eval_kp = COCOeval(self.test_coco, coco_detections, "keypoints")
        coco_eval_kp.evaluate()
        coco_eval_kp.accumulate()
        coco_eval_kp.summarize()

        coco_eval = COCOeval(self.test_coco, coco_detections, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        prefix = ""
        if len(self.test_scales) > 1:
            prefix += "multi-scale_"
        if self.test_flip:
            prefix += "flip_"

        stats = ["ap", "ap_50", "ap_75", "ap_S", "ap_M", "ap_L"]
        for num, name in enumerate(stats):
            self.log(f"test/kp_{prefix}{name}", coco_eval_kp.stats[num], sync_dist=True)

        for num, name in enumerate(stats):
            self.log(f"test/bbox_{prefix}{name}", coco_eval.stats[num], sync_dist=True)


def cli_main():
    pl.seed_everything(5318008)

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
    parser = CenterNetMultiPose.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    train_transform = ComposeSample(
        [
            ImageAugmentation(
                iaa.Sequential(
                    [
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
                                    translate_percent={
                                        "x": (-0.2, 0.2),
                                        "y": (-0.2, 0.2),
                                    },
                                    rotate=(-5, 5),
                                    shear=(-3, 3),
                                ),
                            ],
                            random_order=True,
                        ),
                        iaa.PadToFixedSize(width=512, height=512),
                        iaa.CropToFixedSize(width=512, height=512),
                    ]
                ),
                torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(
                            CenterNetMultiPose.mean, CenterNetMultiPose.std
                        ),
                    ]
                ),
            ),
            PoseFlip(0.5),
            MultiSampleTransform([CenterDetectionSample(), MultiPoseSample()]),
        ]
    )

    valid_transform = ComposeSample(
        [
            ImageAugmentation(
                iaa.Sequential(
                    [
                        iaa.Resize(
                            {"shorter-side": "keep-aspect-ratio", "longer-side": 500}
                        ),
                        iaa.PadToFixedSize(width=512, height=512),
                    ]
                ),
                torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(
                            CenterNetMultiPose.mean,
                            CenterNetMultiPose.std,
                            inplace=True,
                        ),
                    ]
                ),
            ),
            MultiSampleTransform([CenterDetectionSample(), MultiPoseSample()]),
        ]
    )

    test_transform = ImageAugmentation(img_transforms=torchvision.transforms.ToTensor())

    coco_train = CocoDetection(
        os.path.join(args.image_root, "train2017"),
        os.path.join(args.annotation_root, "person_keypoints_train2017.json"),
        transforms=train_transform,
    )

    coco_val = CocoDetection(
        os.path.join(args.image_root, "val2017"),
        os.path.join(args.annotation_root, "person_keypoints_val2017.json"),
        transforms=valid_transform,
    )

    coco_test = CocoDetection(
        os.path.join(args.image_root, "val2017"),
        os.path.join(args.annotation_root, "person_keypoints_val2017.json"),
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
    args.learning_rate_milestones = list(
        map(int, args.learning_rate_milestones.split(","))
    )
    model = CenterNetMultiPose(
        args.arch,
        args.learning_rate,
        args.learning_rate_milestones,
        test_coco=coco_test.coco,
        test_coco_ids=list(sorted(coco_test.coco.imgs.keys())),
    )
    if args.pretrained_weights_path:
        model.load_pretrained_weights(args.pretrained_weights_path)

    # ------------
    # training
    # ------------
    args.callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=-1,
            save_last=True,
            period=10,
            dirpath="model_weights",
            filename=args.arch + "-detection-{epoch:02d}-{val_loss:.2f}",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    trainer.test(model, test_dataloaders=test_loader)


if __name__ == "__main__":
    cli_main()
