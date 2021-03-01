import json

import torch
from torch.utils.data import Dataset


class CocoFakeDataset(Dataset):
    def __init__(self,
                 transforms=None,
                 annotation_path="tests/data/coco_annotation.json",
                 length=1000
                 ):
        self.transforms = transforms
        with open(annotation_path) as json_file:
            self.coco_annotation = json.load(json_file)
        self.length = length

    def __getitem__(self, index):
        img = torch.rand((3, 512, 512))
        annotation = self.coco_annotation.copy()

        if self.transforms:
            img, annotation = self.transforms(img, annotation)

        return img, annotation

    def __len__(self):
        return self.length
