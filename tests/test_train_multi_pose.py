from pytorch_lightning import Trainer, seed_everything
import torch
from torch.utils.data import DataLoader


from .utilities import CocoFakeDataset

from CenterNet.centernet_multi_pose import CenterNetMultiPose
from CenterNet.transforms import MultiSampleTransform
from CenterNet.sample.ctdet import CenterDetectionSample
from CenterNet.sample.multi_pose import MultiPoseSample


def test_multi_pose():
    """
    Simple smoke test for CenterNetMultiPose
    """
    seed_everything(1234)
    dataset = CocoFakeDataset(
        transforms=MultiSampleTransform([CenterDetectionSample(), MultiPoseSample()]),
    )

    test_val_loader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1,
        pin_memory=True,
    )
    test_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
    )

    model = CenterNetMultiPose(
        "dla_34",
        test_flip=True,
        test_scales=[.5, 1, 1.5]
    )

    trainer = Trainer(
        limit_train_batches=2,
        limit_val_batches=1,
        limit_test_batches=1,
        max_epochs=1,
        gpus=1 if torch.cuda.is_available() else 0
    )
    trainer.fit(model, test_val_loader, test_val_loader)

    trainer.test(model, test_dataloaders=test_loader)


if __name__ == "__main__":
    test_multi_pose()
