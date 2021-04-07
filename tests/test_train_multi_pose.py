from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import DataLoader

from tests.utilities import CocoFakeDataset

from centernet_multi_pose import CenterNetMultiPose
from transforms.ctdet import CenterDetectionSample


def test_multi_pose():
    """
    Simple smoke test for CenterNetMultiPose
    """
    seed_everything(1234)
    dataset = CocoFakeDataset(
        transforms=CenterDetectionSample()
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
        max_epochs=1
    )
    trainer.fit(model, test_val_loader, test_val_loader)

    trainer.test(model, test_dataloaders=test_loader)


if __name__ == "__main__":
    test_multi_pose()
