class CategoryIdToClass:
    def __init__(self, valid_ids):
        self.valid_ids = valid_ids
        self.category_ids = {v: i for i, v in enumerate(self.valid_ids)}

    def __call__(self, img, target):
        for ann in target:
            ann["class_id"] = int(self.category_ids[int(ann["category_id"])])

        return img, target
