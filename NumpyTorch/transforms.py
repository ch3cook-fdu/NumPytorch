"""
Author: Sijin Chen, Fudan University
Finished Date: 2021/06/04
"""

import numpy as np
from typing import Union
from math import floor
from functools import reduce
from copy import deepcopy


class Transforms:
    def __init__(self, *args): pass
    def __call__(self, *args): raise NotImplementedError("Overwrite this!")


def BILINEARINTERPOLATION(pos: tuple, image: np.ndarray) -> float:
    m, n, _ = image.shape
    row, col = map(floor, pos)
    p0 = (row  , col  )     # North-East pixel
    p1 = (row  , col+1)     # North-West pixel
    p2 = (row+1, col+1)     # South-West pixel
    p3 = (row+1, col  )     # South-East pixel
    r, s = pos[0] - row, pos[1] - col
    if r == 0.: p3 = p0; p2 = p1
    if s == 0.: p1 = p0; p2 = p3
    weight = [(1.-r)*(1.-s), s*(1.-r), r*s, r*(1.-s)]
    return reduce(lambda x,y: x+y, [w * image[p] for w, p in zip(weight, [p0, p1, p2, p3])])


class Compose(Transforms):
    def __init__(self, *transforms):
        super(Compose, self).__init__()
        self.TransformList = []
        # Check the class of each input
        for trans in transforms:
            assert isinstance(trans, Transforms), "Only allow <class: Transforms> input"
            self.TransformList.append(trans)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        for trans in self.TransformList:
            image = trans(image)
        return image


class Apply(Transforms):
    """ Random apply transforms"""
    def __init__(self, func: Transforms, p: float=.5):
        super(Apply, self).__init__()
        self.func = func
        self.p = p

    def __call__(self, image: np.ndarray) -> np.ndarray:
        output = self.func(image) if np.random.uniform() < self.p else image
        return output


class Selection(Transforms):
    """ Random choose one of the transforms """
    def __init__(self, funcs: [Transforms], p: list=None):
        super(Selection, self).__init__()
        self.funcs = funcs
        if p is None: p = [1./len(funcs) for _ in range(len(funcs))]
        self.p = p

    def __call__(self, image: np.ndarray) -> np.ndarray:
        # random choose one of the transformation
        trans = np.random.choice(self.funcs, 1, p=self.p)[0]
        return trans(image)


class RandomGamma(Transforms):
    """ Gamma Transformation, s=cr^gamma """
    def __init__(self, gamma_range: tuple=(0.5, 1.0), force_gamma: float=None):
        super(RandomGamma, self).__init__()
        self.gamma_range = gamma_range
        self.force_gamma = force_gamma

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if self.force_gamma is not None: return np.power(image, self.force_gamma)
        # true gamma range from (gamma, 1/gamma)
        gamma = np.random.uniform(*self.gamma_range)
        gamma = pow(gamma, 1 if np.random.uniform() < 0.5 else -1)
        # check data type
        if image.dtype == np.uint8: image = image.astype(np.float32)/255.
        image = np.power(image, gamma)
        return image


class Normalize(Transforms):
    """ Normalize input to the desired distribution: y = (x-mu)/std """
    def __init__(self, mean: Union[float, list], std: Union[float, list]):
        super(Normalize, self).__init__()
        assert type(mean) == type(std), "mean and std should be the same datatype!"
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image = (image - self.mean) / self.std
        return image


class GaussianBlur(Transforms):
    """ Gaussian smoothing kernel [G_ij], G_ij = exp(-(s^2+t^2) / (2*var)) """
    def __init__(self, kernel_size: int=3, std: tuple=(0.1, 0.5), force_std: float=None):
        super(GaussianBlur, self).__init__()
        self.kernel_size = (kernel_size, kernel_size)
        pad = kernel_size // 2
        self.pad = pad
        self.std = std
        self.force_std = force_std

    def __call__(self, image: np.ndarray) -> np.ndarray:
        std = np.random.uniform(*self.std)
        if self.force_std is not None: std = self.force_std

        kernel = np.exp(np.array([
                [-((i-self.pad)**2+(j-self.pad)**2)/(2*std**2)] \
                for i in range(self.kernel_size[0]) for j in range(self.kernel_size[1])]))

        kernel = kernel / np.sum(kernel)
        # place holder, to store output
        output = np.zeros_like(image)
        inputshape = image.shape
        # Zero padding image
        p = self.pad
        if len(image.shape) == 2:
            image = np.pad(image, ((p,p), (p,p)), "constant", constant_values=0)
            image = image.reshape(image.shape+(1, ))
            output = output.reshape(output.shape+(1, ))
        elif len(image.shape) == 3:
            image = np.pad(image, ((p,p), (p,p), (0,0)), "constant", constant_values=0)
        else: raise TypeError("Image shape should be (H, W, 3) or (H, W)")
        # Convolution on each channel, and gather the output
        (H, W), (kh, kw) = image.shape[:-1], self.kernel_size
        for c in range(image.shape[-1]):
            mat = np.array([image[i:i+kh, j:j+kw, c].reshape(-1, ) \
                            for i in range(0, H-kh+1) for j in range(0, W-kw+1)])
            output[:,:,c] = (mat @ kernel).reshape(output.shape[:2])
        return output.reshape(inputshape)


class LaplacianShapen(Transforms):
    """ Laplace sharpening kernel 
                [-1, -1, -1]                [ 0, -1,  0]
    Full Kernel:[-1,  8, -1], Simple Kernel:[-1,  4, -1]
                [-1, -1, -1]                [ 0, -1,  0]
    """
    def __init__(self, strength: tuple=(0.1, 0.4), kernel_type: str="simple", pad: int=1):
        super(LaplacianShapen, self).__init__()
        self.pad = pad
        self.strength = strength
        assert kernel_type in {"simple", "full"}, "kernel_type should be simple or full!"
        if kernel_type == "simple":
            self.kernel = np.array([0., -1., 0., -1., 4., -1., 0., -1., 0.]).reshape(9, 1)
        else:
            self.kernel = np.array([-1., -1., -1., -1., 8., -1., -1., -1., -1.]).reshape(9, 1)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        kernel = self.kernel
        strength = np.random.uniform(*self.strength)
        # place holder, to store output
        output = np.zeros_like(image)
        cache = deepcopy(image)
        inputshape = image.shape
        # Zero padding image
        p = self.pad
        if len(image.shape) == 2:
            image = np.pad(image, ((p,p), (p,p)), "constant", constant_values=0)
            image = image.reshape(image.shape+(1, ))
            output = output.reshape(output.shape+(1, ))
        elif len(image.shape) == 3:
            image = np.pad(image, ((p,p), (p,p), (0,0)), "constant", constant_values=0)
        else: raise TypeError("Image shape should be (H, W, 3) or (H, W)")
        # Convolution on each channel, and gather the output
        (H, W), (kh, kw) = image.shape[:-1], (3, 3)
        for c in range(image.shape[-1]):
            mat = np.array([image[i:i+kh, j:j+kw, c].reshape(-1, ) \
                            for i in range(0, H-kh+1) for j in range(0, W-kw+1)])
            output[:,:,c] = (mat @ kernel).reshape(output.shape[:2])
        return np.clip(output.reshape(inputshape)*strength + cache, 0., 1.)


class Resize(Transforms):
    """ Resize the image into random size within range_ """
    def __init__(self, range_: tuple=(0.75, 1.0)):
        super(Resize, self).__init__()
        self.l, self.u = range_

    def __call__(self, image: np.ndarray) -> np.ndarray:
        input_shape = image.shape
        if len(image.shape) == 2: image = image.reshape(image.shape+(1, ))
        H, W, C = image.shape
        # Resize ratio
        H_ratio = np.random.uniform(self.l, self.u)
        W_ratio = np.random.uniform(self.l, self.u)
        outH, outW = int(H*H_ratio), int(W*W_ratio)
        output = np.zeros((outH, outW, C))
        # Interpolation to locate values of the image
        for h in range(int(H*H_ratio)):
            for w in range(int(W*W_ratio)):
                output[h, w, ...] = BILINEARINTERPOLATION((h/H_ratio, w/W_ratio), image)
        ph, pw = (H-outH)//2, (W-outW)//2
        output = np.pad(output, ((ph, H-outH-ph), (pw, W-outW-pw), (0, 0)), 
                        "constant", constant_values=0)
        return output.reshape(input_shape)


class Translation(Transforms):
    """ Translate the image along different axis """
    def __init__(self, distortion: float=0.1):
        super(Translation, self).__init__()
        self.distortion = distortion

    def __call__(self, image: np.ndarray) -> np.ndarray:
        input_shape = image.shape
        if len(image.shape) == 2: image = image.reshape(image.shape+(1, ))
        H, W, C = image.shape
        h, w = int(H*self.distortion), int(W*self.distortion)
        output = np.pad(image, ((h, h), (w, w), (0, 0)), "constant", constant_values=0)
        # Random translation on both axis
        HFLAG, WFLAG = np.random.uniform(size=2) > 0.5
        output = output[2*h*HFLAG: 2*h*HFLAG+H, 2*w*WFLAG: 2*w*WFLAG+W]
        return output.reshape(input_shape)


class Rotation(Transforms):
    """ Rotate the image for small angles """
    def __init__(self, range_: tuple=(-np.pi/12, np.pi/12)):
        super(Rotation, self).__init__()
        self.l, self.u = range_

    def __call__(self, image: np.ndarray) -> np.ndarray:
        input_shape = image.shape
        if len(image.shape) == 2: image = image.reshape(image.shape+(1, ))
        H, W, C = image.shape
        # determine rotation angle
        phi = np.random.uniform(self.l, self.u)
        rotate_matrix = np.array([[ np.cos(phi), np.sin(phi), 0.],
                                  [-np.sin(phi), np.cos(phi), 0.],
                                  [           0,           0, 1.]])
        rotate_matrix = np.linalg.inv(rotate_matrix)
        output = np.zeros_like(image)
        # Interpolation to locate values of the image
        center = np.array([[H/2], [W/2], [0]])
        location = np.array([[h, w, 1.] for h in range(H) for w in range(W)]).T
        target = rotate_matrix @ (location - center) + center
        location = location.astype(np.int)
        for h, w, tx, ty in zip(location[0], location[1], target[0], target[1]):
            if tx >= H-1 or tx < 0 or ty >= W-1 or ty < 0: continue
            output[h, w, :] = BILINEARINTERPOLATION((tx, ty), image)
        return output.reshape(input_shape)

