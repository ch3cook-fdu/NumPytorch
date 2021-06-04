# NumPytorch
A numpy based python package with pytorch-like interface for training a Convolution Neural Network.

To train and evaluate a model, please refer to `trainer.py` and `evalute.py`. We also show that we can also perform fine-tuning in `trainer.py`.

We implemented some image augmentation based on numpy only, please refer to `NumpyTorch/transforms.py`. The interface of data augmentation is like `torchvision.transforms`.

We provide some configs and pre-trained models in `config/` and `checkpoints/`. In our experiment, we show that the model trained in this package outperforms pytorch's implementation.

If you have any problem about this implementation, please let me know by opening an issue.
