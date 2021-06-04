"""
Author: Sijin Chen, Fudan University
Finished Date: 2021/06/04
"""

import numpy as np
from copy import deepcopy
from collections import OrderedDict, deque
import warnings
from typing import Union

dtype = np.float32

# %% Base Class (Meta Class)
class NetworkWrapper:
    ''' Base class of a network built on numpy.
    Notice that, only sequential model is supported!
    '''
    def __init__(self,):
        self.name = None
        self.state_dict = OrderedDict()
        self.ModuleDict = OrderedDict()
        self.istrain    = True
        self.cache      = OrderedDict()
        self.memory     = deque([])
        self.grad       = OrderedDict()

    def train(self,): 
        self.istrain = True
        for ModuleName, layer in self.ModuleDict.items():
            layer.istrain = True

    def eval(self,) : 
        self.istrain = False
        for ModuleName, layer in self.ModuleDict.items():
            layer.istrain = False

    def forward(self, *args) : raise NotImplementedError("Overwrite this!")
    def backward(self, *args): warnings.warn("Overwrite to train your network!")

    def _init_weights(self, ):
        # Initialize weights and bias using normal distribution
        for k, v in self.state_dict.items():
            self.state_dict[k] += 0.01*np.random.randn(*v.shape)

    def load_state_dict(self, state_dict: OrderedDict):
        if state_dict == OrderedDict(): pass
        flag = False
        if "weight" in state_dict:
            self.state_dict["weight"] = state_dict["weight"]
            flag = True
        if "bias" in state_dict:
            self.state_dict["bias"] = state_dict["bias"]
            flag = True
        if "batch_size" in state_dict:
            self.state_dict["batch_size"] = state_dict["batch_size"]
            flag = True
        if not flag:
            # load modules' state_dicts recursively
            for ModuleName, layer_dict in state_dict.items():
                self.ModuleDict[ModuleName].load_state_dict(layer_dict)

    def get_state_dict(self,):
        # State dict of all the parameters in this meta class
        state_dict = OrderedDict()
        for ModuleName, Module in self.ModuleDict.items():
            state_dict[ModuleName] = Module.state_dict
        self.state_dict = state_dict
        return state_dict

    def __call__(self, *args): return self.forward(*args)
    def __getitem__(self, arg: str): return self.state_dict[arg]
    def __str__(self,):
        [(k, v) for k, v in self.ModuleDict.items()]
        return """(Model)(\n{}\n)""".format(
                "\n".join(map(lambda kw: "\t({}){}".format(kw[0], kw[1].__str__()), 
                              self.ModuleDict.items()))
                )


class Criterion:
    ''' Meta class for loss functions. Warning: To avoid unexpected errors, 
    always call a new criterion class on different tasks.
    '''
    def __init__(self, reduction: str="mean"):
        assert reduction in {"mean", "sum"}, "reduction type error"
        self.reduction = reduction
        self.state_dict = OrderedDict()
        self.cache = OrderedDict()
        self.grad = OrderedDict()

    def forward(self, *args) : raise NotImplementedError("Overwrite this!")
    def backward(self, *args): raise NotImplementedError("Overwrite this!")
    def __call__(self, *args): 
        out = self.forward(*args)
        self.backward()
        return out


# %% Layers
class Sequential(NetworkWrapper):
    """ Fully feed forward network, a linked-list liked structure """
    def __init__(self, *args):
        super().__init__()
        for idx, layer in enumerate(args):
            self.ModuleDict["{}-{}".format(layer.name, idx)] = layer

    def forward(self, x: np.ndarray):
        for ModuleName, layer in self.ModuleDict.items():
            x = layer(x)
        return x

    def backward(self, criterion: Criterion):
        # Check the status of network first
        assert self.istrain is True, "the network is in evaluation status."

        ModuleList = [(k, v) for k, v in self.ModuleDict.items()]
        cache = criterion.grad
        for LayerName, layer in ModuleList[::-1]:
            assert isinstance(cache, OrderedDict), "gradient type error!"
            layer.backward(cache["propagate"])
            cache = layer.grad

    def __getitem__(self, arg): return self.ModuleDict[arg]
    def __str__(self,):
        return """(Sequential)(\n{}\n)""".format(
                "\n".join(map(lambda kw: "\t({}){}".format(kw[0], kw[1].__str__()), 
                              self.ModuleDict.items()))
                )


class Linear(NetworkWrapper):
    """ A Linear layer implemented with numpy. It should be called under Sequential class """
    def __init__(self, in_feat: int, out_feat: int, bias: bool=True):
        super(Linear, self).__init__()
        self.name = "linear"
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.bias = bias

        self.state_dict["weight"] = np.zeros( (in_feat, out_feat), dtype=dtype )
        if self.bias is True:
            self.state_dict["bias"] = np.zeros( (out_feat, ), dtype=dtype )
        self._init_weights()
        self.ModuleDict["self"] = self

    def forward(self, x: np.ndarray):
        # Checking input
        assert x.shape[-1] == self.in_feat, """
            Dimension unmatched Error, input is {}, weight's first dimension {}
        """.format(x.shape, self.state_dict["weight"].shape)

        if self.istrain is True: self.cache["running"] = deepcopy(x)
        self.state_dict["batch_size"] = x.shape[0]

        x = x @ self.state_dict["weight"]
        if self.bias is True: x += self.state_dict["bias"]
        return x

    def backward(self, dZ: Union[np.ndarray, Criterion]):
        # Check the status of network first
        assert self.istrain is True, "the network is in evaluation status."
        if isinstance(dZ, Criterion): dZ = dZ.grad["propagate"]

        BatchSize = self.cache["running"].shape[0]
        self.grad["weight"] = self.cache["running"].T @ dZ / BatchSize
        if self.bias is True: self.grad["bias"] = np.sum(dZ, axis=0) / BatchSize

        self.grad["propagate"] = dZ @ self.state_dict["weight"].T

    def __str__(self,) -> str:
        return "(Linear: in_feat={}, out_feat={}, bias={})".format(
                self.in_feat, self.out_feat, self.bias
                )


class Conv2d(NetworkWrapper):
    """ Convolution layer on 2d images, data shape: (B, C, H, W) """
    def __init__(self, in_channel: int, out_channel: int, 
                 kernel_size: Union[tuple, int], bias: bool=True, stride: int=1, 
                 padding: int=0):
        super(Conv2d, self).__init__()
        self.name = "conv2d"
        # store the super parameters
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        # make checks
        assert self.kernel_size[0] % 2 == 1 and self.kernel_size[1] % 2 == 1, "kernel size should be odd!"
        self.bias = bias
        self.stride = stride
        self.padding = padding

        self.state_dict["weight"] = np.zeros( (in_channel*self.kernel_size[0]*self.kernel_size[1], out_channel), dtype=dtype )
        if self.bias is True:
            self.state_dict["bias"] = np.zeros( (out_channel, ), dtype=dtype )
        self._init_weights()

        self.ModuleDict["self"] = self

    def forward(self, x: np.ndarray):
        p = self.padding
        self.cache["input_shape"] = x.shape
        x = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), 'constant', constant_values = 0)
        self.cache["padded_shape"] = x.shape
        # make checks
        BatchSize, Cin, H, W = x.shape
        assert Cin == self.in_channel, "input tensor's channel doesn't match in_channel"

        self.state_dict["batch_size"] = BatchSize

        weight = self.state_dict["weight"]
        bias = 0. if self.bias is False else self.state_dict["bias"]
        return self._conv2d(x, weight, bias)

    def _conv2d(self, input_: np.ndarray, kernel: np.ndarray, bias: Union[float, np.ndarray]):
        """ We treat convolution as a matrix multiplication, 
        size of input:
            input_: (BatchSize, in_channel, H_padded, W_padded)
            kernel: (KernelSize, out_channel)
            bias  : (out_channel, )
        size of output:
            output: (BatchSize, out_channel, H_padded, W_padded)
        """
        BatchSize, Cin, H, W = input_.shape
        (kh, kw), out_channel = self.kernel_size, kernel.shape[-1]
        # flatten the (B, Cin, H, W) image into (B, HW, Cin) tensor
        mat, match = [], OrderedDict()
        for ii, i in enumerate(range(0, H-kh+1, self.stride), 1):
            for jj, j in enumerate(range(0, W-kw+1, self.stride), 1):
                match[(ii, jj)] = (i, i+kh, j, j+kw)
                mat.append(input_[:, :, i:i+kh, j:j+kw].reshape(BatchSize, -1))
        mat = np.array(mat).transpose(1, 0, 2)
        # make a copy of the input flattened matrix
        self.cache["match"] = match
        self.cache["running"] = deepcopy(mat)
        # (B, HW, Cin) @ (Cin, Cout) = (B, HW, Cout)
        output = mat @ kernel + bias
        # reshape (B, HW, Cout) -> (B, Cout, H, W)
        output = output.transpose(0, 2, 1)
        output = output.reshape(BatchSize, out_channel, ii, jj)
        self.cache["output_shape"] = output.shape
        return output

    def backward(self, dZ: Union[np.ndarray, Criterion]):
        # Check the status of network first
        assert self.istrain is True, "the network is in evaluation status."
        if isinstance(dZ, Criterion): dZ = dZ.grad["propagate"]
        BatchSize, Cout, _, _ = dZ.shape
        # dZ(BatchSize, out_channel, H, W) -> dZ_flatten(BatchSize*H*W, out_channel)
        dZ_flatten = dZ.transpose(0, 2, 3, 1).reshape(-1, self.out_channel)
        self.cache["running"] = self.cache["running"].reshape(-1, self.state_dict["weight"].shape[0])

        # Compute gradient of kernel and bias
        self.grad["weight"] = self.cache["running"].T @ dZ_flatten / BatchSize
        if self.bias is True: self.grad["bias"] = np.sum(dZ_flatten, axis=0) / BatchSize
        # pad dZ (use interpolation when stride > 1)
        p = self.padding
        if self.stride != 1:
            dZ_interpolate = np.zeros( dZ.shape[:2] + self.cache["padded_shape"][-2:] )
            for (ii, jj), (xl, xu, yl, yu) in self.cache["match"].items():
                dZ_interpolate[..., (xl+xu)//2, (yl+yu)//2] = dZ[..., ii-1, jj-1]
            dZ = dZ_interpolate
        else:
            dZ = np.pad(dZ, ((0, 0), (0, 0), (p, p), (p, p)), 'constant', constant_values = 0)
        # transform kernel into (out_channel*kernel_size[0]*kernel_size[1], in_channel)
        kernel = self.state_dict["weight"]
        kernel = kernel.T.reshape( (self.out_channel, self.in_channel, -1) )
        # rotate the kernel 180 degrees
        kernel = np.vstack(kernel[:, :, ::-1].transpose(0, 2, 1))
        # Calculate convolution with matrix multiplication
        mat = []
        for (i, j), (xl, xu, yl, yu) in self.cache["match"].items():
            mat.append(dZ[:, :, xl:xu, yl:yu].reshape(BatchSize, -1))
        mat = np.array(mat).transpose(1, 0, 2)
        dZconv = (mat @ kernel).transpose(0, 2, 1)
        dZconv = dZconv.reshape( (BatchSize, self.in_channel)+self.cache["output_shape"][-2:] )
        # interpolation if the output size is smaller than input size
        if dZconv.shape == self.cache["input_shape"]:
            self.grad["propagate"] = dZconv
        else:
            padded = np.zeros(self.cache["padded_shape"])
            for (i, j), (xl, xu, yl, yu) in self.cache["match"].items():
                padded[:, :, xl:xu, yl:yu] = np.tile(dZconv[:, :, i, j], (1, 1, xu-xl, yu-yl))
            self.grad["propagate"] = padded[:, :, p:-p, p:-p]

    def __str__(self,) -> str:
        return "(Conv2d: in_channel={}, out_channel={}, kernel_size={}, bias={}, stride={}, padding={})".format(
                self.in_channel, self.out_channel, self.kernel_size, self.bias, self.stride, self.padding
                )


class MaxPool2d(NetworkWrapper):
    """ Max Pooling layer """
    def __init__(self, kernel_size: Union[int, tuple], stride: int=1):
        super(MaxPool2d, self).__init__()
        self.name = "maxpool2d"
        # store super parameters
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        assert kernel_size <= stride, "Only non-overlapping pooling is supported!"

    def forward(self, x: np.ndarray):
        BatchSize, Cin, H, W = x.shape
        kh, kw = self.kernel_size

        self.cache["input_shape"] = x.shape
        self.state_dict["batch_size"] = x.shape[0]
        # flatten the (B, Cin, H, W) image into (B, Cin, HW, Kernel_Size) tensor
        mat, self.cache["match"] = [], OrderedDict()
        for ii, i in enumerate(range(0, H-kh+1, self.stride), 1):
            for jj, j in enumerate(range(0, W-kw+1, self.stride), 1):
                self.cache["match"][(ii, jj)] = (i, i+kh, j, j+kw)
                mat.append(x[:, :, i:i+kh, j:j+kw].reshape(BatchSize, Cin, -1))
        mat = np.array(mat).transpose(1, 2, 0, 3)
        # Maxpooling
        output = np.max(mat, axis=3)
        output = output.reshape(BatchSize, Cin, ii, jj)
        Maximum_index = np.argmax(mat, axis=3)
        self.cache["Maximum_index"] = Maximum_index.reshape(output.shape)
        return output

    def backward(self, dZ: Union[np.ndarray, Criterion]):
        # make some checks
        assert self.istrain is True, "the network is in evaluation status."
        if isinstance(dZ, Criterion): dZ = dZ.grad["propagate"]
        BatchSize, Cout, H, W = dZ.shape

        mask = np.zeros(self.cache["input_shape"])
        # interpolate dZ's shape into input shape
        dZ_interpolate = np.zeros(self.cache["input_shape"])
        for (i, j), (xl, xu, yl, yu) in self.cache["match"].items():
            # Broadcast signal to full size
            dZ_interpolate[:, :, xl:xu, yl:yu] = np.tile(
                            dZ[:, :, i-1, j-1].reshape(BatchSize, Cout, 1, 1), 
                            (1, 1,) + self.kernel_size
                            )
            xx = self.cache["Maximum_index"][:, :, i-1, j-1] // self.kernel_size[0]
            yy = self.cache["Maximum_index"][:, :, i-1, j-1]  % self.kernel_size[0]
            # Use iterable index to indentify max element index on the mask
            for b in range(BatchSize):
                mask[b, range(Cout), xl+xx[b, :], yl+yy[b, :]] = 1
        self.grad["propagate"] = mask * dZ_interpolate

    def __str__(self,): return "(MaxPool2d: kernel_size=({}, {}), stride={})".format(
            self.kernel_size[0], self.kernel_size[1], self.stride
            )


class AvgPool2d(NetworkWrapper):
    """ Average Pooling layer """
    def __init__(self, kernel_size: Union[int, tuple], stride: int=1):
        super(AvgPool2d, self).__init__()
        self.name = "avgpool2d"
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        assert kernel_size <= stride, "Only non-overlapping pooling is supported!"

    def forward(self, x: np.ndarray):
        BatchSize, Cin, H, W = x.shape
        kh, kw = self.kernel_size

        self.cache["input_shape"] = x.shape
        self.state_dict["batch_size"] = x.shape[0]

        # flatten the (B, Cin, H, W) image into (B, Cin, HW, Kernel_Size) tensor
        mat, self.cache["mask"], self.cache["match"] = [], np.zeros_like(x), OrderedDict()
        for ii, i in enumerate(range(0, H-kh+1, self.stride), 1):
            for jj, j in enumerate(range(0, W-kw+1, self.stride), 1):
                # store links between input and output
                self.cache["mask"][:, :, i:i+kh, j:j+kw] += 1
                self.cache["match"][(ii, jj)] = (i, i+kh, j, j+kw)
                mat.append(x[:, :, i:i+kh, j:j+kw].reshape(BatchSize, Cin, -1))
        mat = np.array(mat).transpose(1, 2, 0, 3)

        # Average pooling
        output = np.mean(mat, axis=3)
        output = output.reshape(BatchSize, Cin, ii, jj)
        return output

    def backward(self, dZ: Union[np.ndarray, Criterion]):
        # make some checks
        assert self.istrain is True, "the network is in evaluation status."
        if isinstance(dZ, Criterion): dZ = dZ.grad["propagate"]
        BatchSize, Cout, H, W = dZ.shape

        # interpolate dZ's shape into input shape
        dZ_interpolate = np.zeros(self.cache["input_shape"])
        for (i, j), (xl, xu, yl, yu) in self.cache["match"].items():
            # Broadcast signal to full size
            dZ_interpolate[:, :, xl:xu, yl:yu] = np.tile(
                            dZ[:, :, i-1, j-1].reshape(BatchSize, Cout, 1, 1), 
                            (1, 1,) + self.kernel_size
                            )
        mask = self.cache["mask"] / self.kernel_size[0] / self.kernel_size[1]
        self.grad["propagate"] = mask * dZ_interpolate

    def __str__(self,): return "(AvgPool2d: kernel_size=({}, {}), stride={})".format(
            self.kernel_size[0], self.kernel_size[1], self.stride
            )


class Flatten(NetworkWrapper):
    """ Flatten layer, transform input from (B, ...) into size (B, F)"""
    def __init__(self,):
        super(Flatten, self).__init__()
        self.name = "flatten"

    def forward(self, x: np.ndarray):
        if self.istrain is True: self.cache["running"] = deepcopy(x.shape)
        self.state_dict["batch_size"] = x.shape[0]
        return x.reshape(x.shape[0], -1)

    def backward(self, dZ: Union[np.ndarray, Criterion]):
        assert self.istrain is True, "the network is in evaluation status."
        if isinstance(dZ, Criterion): dZ = dZ.grad["propagate"]

        self.grad["propagate"] = dZ.reshape(self.cache["running"])

    def __str__(self,): return "(Flatten())"


# %% Activation layers
class ReLU(NetworkWrapper):
    """ Activate function, relu(x)=max(0, x) """
    def __init__(self,):
        super(ReLU, self).__init__()
        self.name = "relu"

    def forward(self, x: np.ndarray):
        if self.istrain is True: self.cache["running"] = deepcopy(x)
        self.state_dict["batch_size"] = x.shape[0]
        return x.clip(0, )

    def backward(self, dZ: Union[np.ndarray, Criterion]):
        # Check the status of network first
        assert self.istrain is True, "the network is in evaluation status."
        if isinstance(dZ, Criterion): dZ = dZ.grad["propagate"]

        activate = (self.cache["running"] >= 0).astype(dtype)
        self.grad["propagate"] = dZ * activate

    def __str__(self,): return "(ReLU())"


class Sigmoid(NetworkWrapper):
    """ Activate function, sigmoid(x)=1/(1+exp(-x)) """
    def __init__(self,):
        super(Sigmoid, self).__init__()
        self.name = "sigmoid"

    def forward(self, x: np.ndarray):
        x = 1 / (1+np.exp(-x))
        if self.istrain is True: self.cache["running"] = deepcopy(x)
        self.state_dict["batch_size"] = x.shape[0]
        return x

    def backward(self, dZ: Union[np.ndarray, Criterion]):
        # Check the status of network first
        assert self.istrain is True, "the network is in evaluation status."
        if isinstance(dZ, Criterion): dZ = dZ.grad["propagate"]

        yout = self.cache["running"]
        activate = yout * (1 - yout)
        self.grad["propagate"] = dZ * activate

    def __str__(self,): return "(Sigmoid())"


class Dropout(NetworkWrapper):
    """ Random assign some outputs as 0 """
    def __init__(self, p: float=0.5):
        super(Dropout, self).__init__()
        self.name = "dropout"
        assert 0.0 < p < 1.0, "random dropout probability should between: 0 < p < 1"
        self.p = p

    def forward(self, x: np.ndarray):
        self.state_dict["batch_size"] = x.shape[0]
        # identity for evaluatio mode
        if self.istrain is False: return x
        # random dropout some outputs for training mode
        mask = np.random.uniform(size=x.shape) > self.p
        mask = mask.astype(dtype) / (1-self.p)
        self.cache["running"] = mask
        return mask * x

    def backward(self, dZ: np.ndarray):
        # Check the status of network first
        assert self.istrain is True, "the network is in evaluation status."
        if isinstance(dZ, Criterion): dZ = dZ.grad["propagate"]

        self.grad["propagate"] = dZ * self.cache["running"]

    def __str__(self,): return "(Dropout(p={}))".format(self.p)


# %% Loss function
class CrossEntropyLoss(Criterion):
    ''' Multinomial classification loss with function:(copy-paste from pytorch codes)
    
    loss(x, class) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                   = -x[class] + \log\left(\sum_j \exp(x[j])\right)
    
    '''
    def __init__(self, reduction: str="mean"):
        super(CrossEntropyLoss, self).__init__(reduction)

    def forward(self, input_: np.ndarray, target: np.ndarray):
        # make some checks
        assert input_.shape[0] == target.shape[0], "Batch Size doesn't match!"
        assert len(target.shape) == 1, "target size should be (batchsize, )"
        BatchSize, ClassNum = input_.shape
        assert ClassNum >= np.max(target), "Target class index out of classifier's boundary!"

        self.state_dict["batch_size"] = BatchSize
        # loss per item in batch data
        self.cache["exp_elm"] = np.exp(input_)
        self.cache["row_sum"] = np.sum(self.cache["exp_elm"], axis=-1)
        self.cache["target"]  = target
        batch_loss = -input_[range(BatchSize), target] + np.log(self.cache["row_sum"])

        # total loss in a batch, perform sum or average along the batch
        batch_loss = np.sum(batch_loss, axis=0)
        if self.reduction == "mean": batch_loss = batch_loss / BatchSize
        return batch_loss

    def backward(self,):
        """ First order derivative of softmax function is:
            d(loss) / d(x[j==class]) = \frac{exp(x[class])}{\sum_j \exp(x[j])} - 1
            d(loss) / d(x[j!=class]) = \frac{exp(x[class])}{\sum_j \exp(x[j])}
        """
        grad = self.cache["exp_elm"]
        BatchSize, ClassNum = grad.shape
        grad /= np.tile(self.cache["row_sum"].reshape(BatchSize, 1), (1, ClassNum))
        grad[range(BatchSize), self.cache["target"]] -= 1
        self.grad["propagate"] = grad


class MSELoss(Criterion):
    """ MSE regression loss:
                              ||                ||2
        loss(input, target) = || input - target ||
                              ||                ||2
    """
    def __init__(self, reduction: str="mean"):
        super(MSELoss, self).__init__(reduction)

    def forward(self, input_: np.ndarray, target: np.ndarray):
        assert input_.shape == target.shape, "Two inputs are in different shape!"
        target = target.astype(dtype)
        error = input_-target
        self.cache["running"] = error
        loss = np.sum(np.power(error, 2))
        # perform mean on each element
        if self.reduction == "mean": loss = loss / input_.size
        return loss

    def backward(self,):
        self.grad["propagate"] = 2*self.cache["running"] / self.cache["running"].size
