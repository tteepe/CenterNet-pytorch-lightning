import abc
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl

from models import create_model


class CenterNet(pl.LightningModule):
    def __init__(self, arch, heads, head_conv):
        super().__init__()

        self.model = create_model(arch, heads, head_conv)

    def forward(self, x):
        return self.model.forward(x)

    @abc.abstractmethod
    def loss(self, outputs, target):
        return 0, {}

    def training_step(self, batch, batch_idx):
        img, target = batch
        outputs = self(img)
        loss, loss_stats = self.loss(outputs, target)

        for key, value in loss_stats.items():
            self.log(f"train/{key}", value)

        return loss

    def validation_step(self, batch, batch_idx):
        img, target = batch
        outputs = self(img)
        loss, loss_stats = self.loss(outputs, target)

        for key, value in loss_stats.items():
            self.log(f"valid/{key}", value)

        return loss

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--arch",
            default="dla_34",
            help="model architecture. Currently tested "
            "res_18 | res_101 | resdcn_18 | resdcn_101 | dlav0_34 | dla_34 | hourglass",
        )
        parser.add_argument(
            "--head_conv",
            type=int,
            default=-1,
            help="conv layer channels for output head"
            "0 for no conv layer, -1 for default setting, 64 for resnets and 256 for dla.",
        )
        parser.add_argument(
            "--down_ratio",
            type=int,
            default=4,
            help="output stride. Currently only supports 4.",
        )

        parser.add_argument("--learning_rate", type=float, default=2.5e-4)

        return parser
