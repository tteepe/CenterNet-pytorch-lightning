# CenterNet w/ PyTorchLightning 

![CI testing](https://github.com/tteepe/CenterNet-pytorch-lightning/workflows/CI%20testing/badge.svg?branch=main&event=push)

 
## Description
My attempt at a cleaner implementation of the glorious [CenterNet](https://github.com/xingyizhou/CenterNet).

### Features
- Decoupled backbones and heads for easier backbone integration
- Split sample creation into image augmentation (with [imgaug](https://github.com/aleju/imgaug)) and actual sample creation
- Comes shipped with PyTorch Lightning modules but can also be used with good ol' plain PyTorch
- Stripped all code not used to reproduce the results in the paper

### ToDos
Pull requests are welcome!

- [ ] Add 3D bounding box detection
- [ ] Add ExtremeNet detection


## How to run   
First, install dependencies   
```bash
# clone CenterNet
git clone https://github.com/tteepe/CenterNet-pytorch-lightning

# install CenterNet
cd CenterNet-pytorch-lightning
pip install -e .   
pip install -r requirements.txt
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
model = CenterNetDetection()

# data
train = CocoDetection("train2017", "instances_train2017.json")
val = CocoDetection("val2017", "instances_val2017.json")
test = CocoDetection("test2017", "image_info_test2017.json")

# train
trainer = Trainer()
trainer.fit(model, train, val)

# test using the best model!
trainer.test(test_dataloaders=test)
```

### Citation
If you like this work want to cite this package feel free to use this:
```
@article{teepe2021centernet,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```   
