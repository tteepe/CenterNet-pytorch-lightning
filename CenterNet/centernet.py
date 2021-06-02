from argparse import ArgumentParser

import torch
import pytorch_lightning as pl

from CenterNet.models import create_model


class CenterNet(pl.LightningModule):
    def __init__(self, arch):
        super().__init__()
        self.arch = arch

        # Backbone specific args
        self.head_conv = 256 if "dla" in arch or "hourglass" in arch else 64
        self.num_stacks = 2 if "hourglass" in arch else 1
        self.padding = 127 if "hourglass" in arch else 31

        self.backbone = create_model(arch)

        self.down_ratio = 4

    def load_pretrained_weights(self, model_weight_path, strict=True):
        mapping = {
            "hm": "heatmap",
            "wh": "width_height",
            "reg": "regression",
            "hm_hp": "heatmap_keypoints",
            "hp_offset": "heatmap_keypoints_offset",
            "hps": "keypoints",
        }

        print(f"Loading weights from: {model_weight_path}")
        checkpoint = torch.load(model_weight_path)
        backbone = {
            k.replace("module.", ""): v
            for k, v in checkpoint["state_dict"].items()
            if k.split(".")[1] not in mapping
        }
        self.backbone.load_state_dict(backbone, strict=strict)

        # These next lines are some special magic.
        # Try not to touch them and enjoy their beauty.
        # (The new decoupled heads require these amazing mapping functions
        #  to load the old pretrained weights)
        heads = {
            ("0." if self.num_stacks == 1 else "")
            + ".".join(
                [mapping[k.replace("module.", "").split(".")[0]], "fc"]
                + k.split(".")[2:]
            ).replace("conv.", ""): v
            for k, v in checkpoint["state_dict"].items()
            if k.split(".")[1] in mapping
        }
        if self.arch == "hourglass":
            heads = {
                ".".join(
                    k.split(".")[2:3] + k.split(".")[:2] + k.split(".")[3:]
                ).replace("fc.1", "fc.2"): v
                for k, v in heads.items()
            }
        self.heads.load_state_dict(heads, strict=strict)

    def forward(self, x):
        return self.backbone.forward(x)

    def loss(self, outputs, target):
        return 0, {}

    def training_step(self, batch, batch_idx):
        img, target = batch
        outputs = self(img)
        loss, loss_stats = self.loss(outputs, target)

        self.log(f"train_loss", loss, on_epoch=True)

        for key, value in loss_stats.items():
            self.log(f"train/{key}", value)

        return loss

    def validation_step(self, batch, batch_idx):
        img, target = batch
        outputs = self(img)
        loss, loss_stats = self.loss(outputs, target)

        self.log(f"val_loss", loss, on_epoch=True, sync_dist=True)

        for name, value in loss_stats.items():
            self.log(f"val/{name}", value, on_epoch=True, sync_dist=True)

        return {"loss": loss, "loss_stats": loss_stats}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=self.learning_rate_milestones
            ),
            "name": "learning_rate",
            "interval": "epoch",
            "frequency": 1
        }

        return [optimizer], [lr_scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--arch",
            default="dla_34",
            help="backbone architecture. Currently tested "
            "res_18 | res_101 | resdcn_18 | resdcn_101 | dla_34 | hourglass",
        )

        parser.add_argument("--learning_rate", type=float, default=25e-5)
        parser.add_argument("--learning_rate_milestones", default="90, 120")
        return parser
