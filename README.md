# CenterNet w/ PyTorchLightning 

![CI testing](https://github.com/tteepe/CenterNet-pytorch-lightning/workflows/CI%20testing/badge.svg?branch=main&event=push)
[![DOI](https://zenodo.org/badge/334429075.svg)](https://zenodo.org/badge/latestdoi/334429075)

 
## Description
My attempt at a cleaner implementation of the glorious [CenterNet](https://github.com/xingyizhou/CenterNet).

### Features
- Decoupled backbones and heads for easier backbone integration
- Split sample creation into image augmentation (with [imgaug](https://github.com/aleju/imgaug)) and actual sample creation
- Comes shipped with Lightning modules but can also be used with good ol' plain PyTorch
- Stripped all code not used to reproduce the results in the paper
- Smaller code base with more meaningful variable names
- Requires significantly less memory
- Same or slightly better results than the original implementation


### ToDos
Some features of the original repository are not implemented yet but pull requests are welcome!
- [ ] 3D bounding box detection
- [ ] ExtremeNet detection
- [ ] Pascal VOC dataset

## How to run   
First, install dependencies   
```bash
# clone CenterNet
git clone https://github.com/tteepe/CenterNet-pytorch-lightning

# install CenterNet
cd CenterNet-pytorch-lightning
pip install -e .   
pip install -r requirements.txt

# Make DCNv2 - a backbone dependency
cd CenterNet/models/backbones/DCNv2 && sh make.sh
 ```   
 Next, navigate to any file and run it.   
 ```bash
# module folder
cd CenterNet

# run module
python centernet_detection.py    
python centernet_multi_pose.py    
```

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:

```python
from CenterNet.datasets.coco import CocoDetection
from CenterNet.centernet_detection import CenterNetDetection
from pytorch_lightning import Trainer

# model
model = CenterNetDetection("dla_34")

# data
train = CocoDetection("train2017", "instances_train2017.json")
val = CocoDetection("val2017", "instances_val2017.json")

# train
trainer = Trainer()
trainer.fit(model, train, val)

# test using the best backbone!
test = CocoDetection("test2017", "image_info_test2017.json")
trainer.test(test_dataloaders=test)
```

## BibTeX
If you want to cite the implementation feel free to use this or [zendo](https://zenodo.org/record/4569502):

```bibtex
@article{teepe2021centernet,
  title={CenterNet PyTorch Lightning},
  author={Teepe, Torben and Gilg, Johannes},
  journal={GitHub. Note: https://github.com/tteepe/CenterNet-pytorch-lightning},
  volume={1},
  year={2021}
}
```
