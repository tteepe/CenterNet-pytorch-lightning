from argparse import ArgumentParser

import imgaug as ia
import imgaug.augmenters as iaa
import os
import pytorch_lightning as pl
import torch
import torchvision
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader

from centernet import CenterNet
from datasets.coco import CocoDetection
from decode.ctdet import ctdet_decode
from decode.utils import sigmoid_clamped
from utils.losses import RegL1Loss, FocalLoss
from transforms import CategoryIdToClass, ImageAugmentation, ComposeSample
from transforms.ctdet import CenterDetectionSample
from transforms.sample import PoseFlip


class CenterNetDetection(CenterNet):
    def __init__(self, arch, heads, head_conv, learning_rate=1e-4, hm_weight=1, wh_weight=1, off_weight=0.1):
        super().__init__(arch, heads, head_conv)
        self.save_hyperparameters()

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

    def test_step_end(self, parts):
        # TODO
        detections = []
        print(parts)
        for output in parts:
            output = output['output'][-1]
            dets = ctdet_decode(output['hm'].sigmoid_(), output['wh'], reg=output['reg'])
            dets = dets.detach().cpu().numpy()
            dets = dets.reshape(1, -1, dets.shape[2]).squeeze()

            detections.append(dets)

        coco_detections = self.convert_eval_format(detections)
        coco_eval = COCOeval(self.coco, coco_detections, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()


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

    coco_train = CocoDetection(os.path.join(args.image_root, 'train2017'),
                               os.path.join(args.annotation_root, 'instances_train2017.json'),
                               transforms=train_transform)

    coco_val = CocoDetection(os.path.join(args.image_root, 'val2017'),
                             os.path.join(args.annotation_root, 'instances_val2017.json'),
                             transforms=valid_transform)

    train_loader = DataLoader(coco_train, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(coco_val, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(coco_val, batch_size=1, num_workers=1)

    # ------------
    # model
    # ------------
    heads = {'hm': CocoDetection.num_classes, 'wh': 2, 'reg': 2}
    if 'dla' in args.arch:
        args.head_conv = 256
    model = CenterNetDetection(args.arch, heads, args.head_conv, args.learning_rate)
    model.load_model_weights("../model_weights/ctdet_coco_dla_1x.pth")

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    # trainer.test(test_dataloaders=test_loader)


if __name__ == '__main__':
    cli_main()
