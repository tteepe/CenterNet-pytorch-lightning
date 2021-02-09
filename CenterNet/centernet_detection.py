import os
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import ColorJitter, Normalize, ToTensor

from datasets.dataset.coco import CocoDetection
from torchvision import transforms

from datasets.transforms.transforms import CategoryIdToClass
from datasets.transforms.ctdet import CtDetTransform
from models.losses import RegL1Loss, FocalLoss
from models.model import create_model
from models.utils import _sigmoid


class CenterNetDetection(pl.LightningModule):
    def __init__(self, arch, heads, head_conv, learning_rate):
        super().__init__()
        self.save_hyperparameters()

        self.model = create_model(arch, heads, head_conv)

        self.criterion = FocalLoss()
        self.criterion_regularizer = RegL1Loss()
        self.criterion_width_height = RegL1Loss()

    def forward(self, x):
        return self.model.forward(x)

    def loss(self, outputs, batch):
        hm_loss, wh_loss, off_loss = 0, 0, 0
        hm_weight, wh_weight, off_weight = 1, 1, 0.1
        num_stacks = len(outputs)

        for s in range(num_stacks):
            output = outputs[s]
            output['hm'] = _sigmoid(output['hm'])

            hm_loss += self.criterion(output['hm'], batch['hm']) / num_stacks

            if wh_weight > 0:
                wh_loss += self.criterion_width_height(
                    output['wh'], batch['reg_mask'],
                    batch['ind'], batch['wh']) / num_stacks

            if off_weight > 0:
                off_loss += self.criterion_regularizer(output['reg'], batch['reg_mask'], batch['ind'], batch['reg']) / num_stacks

        loss = hm_weight * hm_loss + wh_weight * wh_loss + off_weight * off_loss
        loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'wh_loss': wh_loss, 'off_loss': off_loss}
        return loss, loss_stats

    def training_step(self, batch, batch_idx):
        outputs = self(batch['input'])
        loss, loss_stats = self.loss(outputs, batch)

        self.log('train_loss', loss)
        for key, value in loss_stats.items():
            self.log(f'train_loss_{key}', value)

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch['input'])
        loss, loss_stats = self.loss(outputs, batch)

        self.log('valid_loss', loss)
        for key, value in loss_stats.items():
            self.log(f'valid_loss_{key}', value)

        return loss

    def test_step(self, batch, batch_idx):
        pass
        # x, y = batch
        # y_hat = self(x)
        # loss = F.cross_entropy(y_hat, y)
        # self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--arch', default='dla_34',
                            help='model architecture. Currently tested '
                                 'res_18 | res_101 | resdcn_18 | resdcn_101 | dlav0_34 | dla_34 | hourglass')
        parser.add_argument('--head_conv', type=int, default=-1,
                            help='conv layer channels for output head'
                                 '0 for no conv layer, -1 for default setting, 64 for resnets and 256 for dla.')
        parser.add_argument('--down_ratio', type=int, default=4,
                            help='output stride. Currently only supports 4.')

        parser.add_argument('--learning_rate', type=float, default=2.5e-4)

        return parser


def cli_main():
    pl.seed_everything(80085)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('coco_image_root')
    parser.add_argument('coco_annotation')

    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = CenterNetDetection.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    coco_train = CocoDetection(os.path.join(args.coco_image_root, 'train2017'),
                               os.path.join(args.coco_annotation, 'instances_train2017.json'),
                               transform=transforms.Compose([
                                   ToTensor(),
                                   ColorJitter(0.4, 0.4, 0.4),
                                   Normalize(CocoDetection.mean, CocoDetection.std, inplace=True),
                                   ]),
                               target_transform=CategoryIdToClass(CocoDetection.valid_ids),
                               sample_transforms=CtDetTransform(augmented=True))

    coco_val = CocoDetection(os.path.join(args.coco_image_root, 'val2017'),
                             os.path.join(args.coco_annotation, 'instances_val2017.json'),
                             transform=transforms.Compose([
                                   ToTensor(),
                                   Normalize(CocoDetection.mean, CocoDetection.std, inplace=True),
                                   ]),
                             target_transform=CategoryIdToClass(CocoDetection.valid_ids),
                             sample_transforms=CtDetTransform(augmented=False))

    train_loader = DataLoader(coco_train, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = DataLoader(coco_val, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = DataLoader(coco_val, batch_size=args.batch_size, num_workers=args.num_workers)

    # ------------
    # model
    # ------------
    heads = {'hm': CocoDetection.num_classes, 'wh': 2, 'reg': 2}
    model = CenterNetDetection(args.arch, heads, args.head_conv, args.learning_rate)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    trainer.test(test_dataloaders=test_loader)


if __name__ == '__main__':
    cli_main()
