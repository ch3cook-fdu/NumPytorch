"""
Author: Sijin Chen, Fudan University
Finished Date: 2021/06/04
"""

import numpy as np
from math import ceil
from collections import OrderedDict
import pickle


class Dataset:
    def __init__(self, *args): pass
    def __getitem__(self, *args): raise NotImplementedError("Overwrite this!")
    def __len__(self,): raise NotImplementedError("Overwrite this!")


class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int=1, shuffle: bool=False, 
                 drop_last: bool=False):
        self.DATASET = dataset
        self.BATCHSIZE = batch_size

        self.cursor = -1
        datasize = len(self.DATASET)
        self.datasize = datasize
        self.NUM = datasize//batch_size if drop_last is True else ceil(datasize/batch_size)
        self.INDEX = np.arange(0, datasize)
        self.shuffle = shuffle

    def __len__(self, ): return self.NUM
    def __iter__(self,): return self

    def __next__(self):
        # shuffle data index at the begining of iteration
        if self.cursor == -1 and self.shuffle is True: 
            self.INDEX = np.random.choice(self.INDEX, self.datasize, replace=False)
        self.cursor += 1
        if self.cursor >= self.NUM: 
            self.cursor = -1; raise StopIteration()
        # Sampling from dataset
        start = self.BATCHSIZE*self.cursor
        end   = min(len(self.DATASET), self.BATCHSIZE*(self.cursor+1))
        # Stack together the output from the dataset
        tuple_flag = True
        datastack = OrderedDict()
        for index in self.INDEX[start : end]:
            sample = self.DATASET[index]
            if not isinstance(sample, tuple): 
                tuple_flag = False; sample = (sample, )
            for idx, spl in enumerate(sample):
                datastack[idx] = [spl] if idx not in datastack else datastack[idx]+[spl]
        if tuple_flag is True:
            return tuple(np.array(data) for data in datastack.values())
        else:
            return np.array(datastack[0])


def save(state_dict: OrderedDict, filename: str):
    f = open(filename, 'wb')
    pickle.dump(state_dict, f)
    f.close()


def load(filename: str) -> OrderedDict:
    f = open(filename, 'rb')
    state_dict = pickle.load(f)
    return state_dict
