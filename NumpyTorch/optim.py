"""
Author: Sijin Chen, Fudan University
Finished Date: 2021/06/04
"""

from .nn import NetworkWrapper
from collections import OrderedDict
from copy import deepcopy
from typing import Callable
import numpy as np


class Optimizer:
    """ Meta class for optimizers """
    def __init__(self, *args, **kwargs): 
        self.state_dict = OrderedDict()
    def load_state_dict(self, state_dict: OrderedDict):
        self.state_dict = deepcopy(state_dict)
    def step(self,): raise NotImplementedError("Overwrite this!")


class lr_scheduler:
    """ Meta class for adjusting optimizer learning rate """
    def __init__(self, optimizer: Optimizer, *args, **kwargs): 
        self.optimizer = optimizer
        self.state_dict = OrderedDict()
    def load_state_dict(self, state_dict: OrderedDict):
        self.state_dict = deepcopy(state_dict)


class SGD(Optimizer):
    """ The optimizer class to update the parameters from the network """
    def __init__(self, 
                 model: NetworkWrapper, 
                 lr: float, 
                 momentum: float=None, 
                 weight_decay: float=None
                 ):

        super(SGD, self).__init__()
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay if weight_decay is not None else 0.

        if self.momentum is not None:
            for ModuleName, Layer in self.model.ModuleDict.items():
                if "weight" in Layer.state_dict:
                    self.state_dict["{}-weight".format(ModuleName)] = 0.
                if "bias" in Layer.state_dict:
                    self.state_dict["{}-bias".format(ModuleName)] = 0.

        self.decay_rate = None
        self.batch_size = None

    def load_state_dict(self, state_dict: OrderedDict):
        self.state_dict = deepcopy(state_dict)

    def _step_without_momentum(self,):
        """ Update the layers without momentum
        Note:
            Since the update without momentum is a speial version of momentum update
        (momentum=0), we separate these two updating method to accelerate training.
        """
        for ModuleName, Layer in self.model.ModuleDict.items():
            self.batch_size = Layer["batch_size"]
            self.decay_rate = 1 - (self.weight_decay/self.batch_size)

            if "weight" in Layer.state_dict:
                Layer.state_dict["weight"] = Layer["weight"]*self.decay_rate - self.lr*Layer.grad["weight"]
            if "bias" in Layer.state_dict:
                Layer.state_dict["bias"] = Layer["bias"]*self.decay_rate - self.lr*Layer.grad["bias"]

    def _step_with_momentum(self,):
        """ Update the layers with momentum update:
                W(t+1) = W(t) - lr*dW + momentum*(W(t) - W(t-1))
        """
        for ModuleName, Layer in self.model.ModuleDict.items():
            self.batch_size = Layer["batch_size"]
            self.decay_rate = 1-(self.weight_decay/self.batch_size)

            if "weight" in Layer.state_dict:
                cache = deepcopy(Layer["weight"])
                momentum = cache - self.state_dict["{}-weight".format(ModuleName)]
                Layer.state_dict["weight"] = cache*self.decay_rate - self.lr*Layer.grad["weight"] + self.momentum*momentum
                self.state_dict["{}-weight".format(ModuleName)] = cache

            if "bias" in Layer.state_dict:
                cache = deepcopy(Layer["bias"])
                momentum = cache - self.state_dict["{}-bias".format(ModuleName)]
                Layer.state_dict["bias"] = cache*self.decay_rate - self.lr*Layer.grad["bias"] + self.momentum*momentum
                self.state_dict["{}-bias".format(ModuleName)] = cache

    def step(self,):
        """ We implemented two different ways of updating parameters """
        if self.momentum is not None:
            self._step_with_momentum()
        else: 
            self._step_without_momentum()


class LambdaLR(lr_scheduler):
    """ Using lambda function to adjust learning rate """
    # always update learning rate after running the entire epoch!
    def __init__(self, optimizer: Optimizer, lr_lambda: Callable, verbose: bool=False):
        super(lr_scheduler, self).__init__()
        self.optimizer = optimizer
        self.lr_lambda = lr_lambda
        self.verbose = verbose
        self.epoch = -1

    def step(self,):
        # adjusting learning rate using the lr-lambda function
        lr = self.optimizer.lr
        # Update learning rate
        self.optimizer.lr = self.lr_lambda(lr)
        if self.verbose is True:
            print("Adjusting learning rate with lr_lambda from {:.4f} to {:.4f}".format(
                    lr, self.optimizer.lr))
        self.epoch += 1


class CosineAnnealingLR(lr_scheduler):
    """ Using lambda function to adjust learning rate """
    # always update learning rate after running the entire epoch!
    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float=0., 
                 verbose: bool=False):
        super(lr_scheduler, self).__init__()
        self.optimizer = optimizer
        self.T_max = T_max
        self.T_cur = -1
        self.eta_min = eta_min
        self.eta_max = None
        self.verbose = verbose

    def step(self,):
        """ Update the learning rate using
            \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
                     \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)
        """
        lr = self.optimizer.lr
        if self.T_cur == -1: self.eta_max = lr
        # Cosine annealing function
        new_lr = self.eta_min + 1/2*(self.eta_max-self.eta_min) * \
                (1 + np.cos(self.T_cur/self.T_max*np.pi))
        # Update learning rate
        self.optimizer.lr = new_lr
        if self.verbose is True:
            print("Adjusting learning rate with cosine annealing from {:.4f} to {:.4f}".format(
                    lr, self.optimizer.lr))
        self.T_cur += 1
