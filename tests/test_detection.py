from pycocotools.coco import COCO
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import DataLoader

from tests.utilities import CocoFakeDataset

from centernet_detection import CenterNetDetection
from transforms import ComposeSample, CategoryIdToClass
from transforms.ctdet import CenterDetectionSample


def test_detection():
    """
    Simple smoke test for CenterNetDetection
    """
    seed_everything(1234)
    dataset = CocoFakeDataset(
        transforms=ComposeSample([
            CategoryIdToClass(range(0, 100)),
            CenterDetectionSample(),
        ])
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

    model = CenterNetDetection(
        "dla_34",
    )

    trainer = Trainer(
        limit_train_batches=2,
        limit_val_batches=1,
        limit_test_batches=1,
        max_epochs=1
    )
    trainer.fit(model, test_val_loader, test_val_loader)

    results = trainer.test(model, test_dataloaders=test_loader)

    assert results


if __name__ == "__main__":
    test_detection()
