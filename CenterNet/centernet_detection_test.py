import os
from argparse import ArgumentParser
import pytorch_lightning as pl
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection

from centernet_detection import CenterNetDetection
from transforms import ImageAugmentation
from transforms.sample import ComposeSample


def cli_test():
    pl.seed_everything(5318008)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument("image_root")
    parser.add_argument("annotation_root")

    parser.add_argument("--pretrained_weights_path")
    parser.add_argument("--ckpt_path")

    parser.add_argument("--flip", action='store_true')
    parser.add_argument("--multi_scale", action='store_true')

    parser = pl.Trainer.add_argparse_args(parser)
    parser = CenterNetDetection.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    test_transform = ComposeSample(
        [ImageAugmentation(img_transforms=torchvision.transforms.ToTensor())]
    )

    coco_test = CocoDetection(
        os.path.join(args.image_root, "val2017"),
        os.path.join(args.annotation_root, "instances_val2017.json"),
        transforms=test_transform,
    )
    test_loader = DataLoader(coco_test, batch_size=1, num_workers=0, pin_memory=True)

    # ------------
    # model
    # ------------
    model = CenterNetDetection(
        args.arch,
        args.learning_rate,
        test_coco=coco_test.coco,
        test_coco_ids=list(sorted(coco_test.coco.imgs.keys()))
    )
    if args.pretrained_weights_path:
        model.load_pretrained_weights(args.pretrained_weights_path)

    if args.ckpt_path:
        ckpt = pl.utilities.cloud_io.load(args.ckpt_path)
        model.load_state_dict(ckpt['state_dict'])

    # ------------
    # testing
    # ------------
    args.test_flip = args.flip
    args.test_scales = [.5, .75, 1, 1.25, 1.5] if args.multi_scale else None
    trainer = pl.Trainer.from_argparse_args(args)

    trainer.test(model, test_dataloaders=test_loader)


if __name__ == "__main__":
    cli_test()
