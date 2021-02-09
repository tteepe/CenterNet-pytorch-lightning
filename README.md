# CenterNet w| PyTorchLightning 

![CI testing](https://github.com/tteepe/CenterNet-pytorch-lightning/workflows/CI%20testing/badge.svg?branch=main&event=push)

 
## Description   
Re-implementation of the glorious Network CenterNet as an easy to use PyTorchLightning package.

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

# run module (example: mnist as your main contribution)   
python lit_classifier_main.py    
```

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:

```python
from centernet.datasets.mnist import mnist
from centernet.lit_classifier_main import LitClassifier
from pytorch_lightning import Trainer

# model
model = LitClassifier()

# data
train, val, test = mnist()

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
