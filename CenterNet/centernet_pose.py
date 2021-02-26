import os
from argparse import ArgumentParser

import imgaug.augmenters as iaa

import torchvision
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection

from centernet import CenterNet
from utils.decode import sigmoid_clamped

from transforms import CategoryIdToClass, ImageAugmentation
from models import create_model
from utils.losses import RegL1Loss, FocalLoss, RegWeightedL1Loss
from transforms.ctdet import CenterDetectionSample
from transforms.multi_pose import MultiPoseSample
from transforms.sample import MultiSampleTransform, PoseFlip, ComposeSample


class CenterNetPose(CenterNet):
    def __init__(
        self,
        arch,
        learning_rate=1e-4,
        hm_weight=1,
        wh_weight=1,
        off_weight=0.1,
        hp_weight=1,
        hm_hp_weight=1,
    ):
        heads = {"heatmap": 1, "width_height": 2, "regression": 2, "heatmap_keypoints": 17, "heatpoint_offset": 2, "keypoints": 34}
        super().__init__(arch, heads)
        self.save_hyperparameters()

        self.criterion = FocalLoss()
        self.criterion_heatmap_heatpoints = FocalLoss()
        self.criterion_keypoint = RegWeightedL1Loss()
        self.criterion_regression = RegL1Loss()
        self.criterion_width_height = RegL1Loss()

    def forward(self, x):
        return self.model.forward(x)

    def loss(self, outputs, target):
        hm_loss, wh_loss, off_loss = 0, 0, 0
        hp_loss, off_loss, hm_hp_loss, hp_offset_loss = 0, 0, 0, 0
        num_stacks = len(outputs)

        for s in range(num_stacks):
            output = outputs[s]
            output["heatmap"] = sigmoid_clamped(output["heatmap"])
            output["heatmap_keypoints"] = sigmoid_clamped(output["heatmap_keypoints"])

            hm_loss += self.criterion(output["heatmap"], target["heatmap"])
            hp_loss += self.criterion_keypoint(
                output["keypoints"], target["keypoints_mask"], target["indices"], target["keypoints"]
            )
            wh_loss += self.criterion_width_height(
                output["width_height"], target["regression_mask"], target["indices"], target["width_height"]
            )

            off_loss += self.criterion_regularizer(
                output["regression"], target["regression_mask"], target["indices"], target["regression"]
            )
            hp_offset_loss += self.criterion_regression(
                output["heatpoint_offset"],
                target["heatpoint_mask"],
                target["heatpoint_indices"],
                target["heatpoint_offset"],
            )
            hm_hp_loss += self.criterion_heatmap_heatpoints(
                output["heatmap_keypoints"], target["heatmap_keypoints"]
            )

        loss = (
            self.hparams.hm_weight * hm_loss
            + self.hparams.wh_weight * wh_loss
            + self.hparams.off_weight * off_loss
            + self.hparams.hp_weight * hp_loss
            + self.hparams.hm_hp_weight * hm_hp_loss
            + self.hparams.off_weight * hp_offset_loss
        ) / num_stacks

        loss_stats = {
            "loss": loss,
            "hm_loss": hm_loss,
            "hp_loss": hp_loss,
            "hm_hp_loss": hm_hp_loss,
            "hp_offset_loss": hp_offset_loss,
            "wh_loss": wh_loss,
            "off_loss": off_loss,
        }
        return loss, loss_stats

    def test_step(self, batch, batch_idx):
        pass
        # x, y = batch
        # y_hat = self(x)
        # loss = F.cross_entropy(y_hat, y)
        # self.log('test_loss', loss)


def cli_main():
    pl.seed_everything(5318008)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument("image_root")
    parser.add_argument("annotation_root")

    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = CenterNetPose.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    train_transform = ComposeSample([
        ImageAugmentation(
            iaa.Sequential([
                # DO NOT use flip augmentation here, use PoseFlip instead
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
        MultiSampleTransform([
            CenterDetectionSample(),
            MultiPoseSample()
        ])
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
        MultiSampleTransform([
            CenterDetectionSample(),
            MultiPoseSample()
        ])
    ])

    coco_train = CocoDetection(os.path.join(args.image_root, 'train2017'),
                               os.path.join(args.annotation_root, 'person_keypoints_train2017.json'),
                               transforms=train_transform)

    coco_val = CocoDetection(os.path.join(args.image_root, 'val2017'),
                             os.path.join(args.annotation_root, 'person_keypoints_val2017.json'),
                             transforms=valid_transform)

    train_loader = DataLoader(coco_train, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(coco_val, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(coco_val, batch_size=1, num_workers=1)

    # ------------
    # model
    # ------------
    model = CenterNetPose(args.arch, args.learning_rate)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    trainer.test(test_dataloaders=test_loader)


if __name__ == "__main__":
    cli_main()
