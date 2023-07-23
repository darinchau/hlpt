### Provides a custom Dataset class for nice things to happen
from __future__ import annotations
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch import Tensor
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm as ProgressBar
import warnings
from torch.utils.data import get_worker_info
from typing import Iterable

def get_size(x):
    if isinstance(x, list):
        return int(x[0].size(0))
    elif isinstance(x, Tensor):
        return int(x.size(0))
    else:
        warnings.warn(f"No big deal, just wanna let you know x of type {type(x)} exists.")
        return 1

# TODO support stratification in the future
# TODO support multiworker data fetching
class DataIterator:
    """A wrapper for dataloader objects that provides lovely progress bars and all the low-level fiddling
    
    The dataset and dataloader is constructed insitu when __iter__ is called i.e. at the start of a for loop
    
    length: when not None, hints to the progress bar about the length of your data iterable
    otherwise we try to compute it by calling len() on your things"""
    def __init__(self, d: list[Iterable], batch_size: int = 1, shuffle = True, progress_bar = True, length: int | None = None):
        # Store the datas and some other variable
        try:
            self.datas = list(zip(*d, strict=True))
        except ValueError:
            raise RuntimeError("All datas must be of the same length")
        self.pbar = progress_bar
        self.shuffle = shuffle
        self.bs = batch_size

        # Attempt to precompute the size
        # Note: len(a) raises a type error if failed, but a.__len__() raises an attribute error if failed
        # Because I cant be fucked doing this, we catch both
        # if length is not None:
        #     self.len_t = length
        # else:
        #     try:
        #         lengths = list(map(lambda x: len(x), d))
        #         self.len_t = min(lengths)
        #     except (AttributeError, TypeError) as e:
        #         self.len_t = None

    def __iter__(self):
        pbar = ProgressBar(total = self.len_t, desc = "Loading data...")
        for d in self.datas:
            yield d
            s = get_size(d[0])
            pbar.update(s)

#     def __add__(self, other: DataLoader):
#         if not isinstance(other, DataLoader):
#             raise RuntimeError("Cannot add data iterators with other things")
#         tmp = self.pbar
#         self.pbar = False
#         if isinstance(other, (DataIterator, ConcatIterator)):
#             other.pbar = False
#         return ConcatIterator(self, other, self.bs, shuffle = self.shuffle, progress_bar = tmp, use_multiprocess = self.use_multiprocess)

# class ConcatIterator:
#     def __init__(self, d1: DataLoader, d2: DataLoader, batch_size: int = 1, shuffle = True, progress_bar = True, use_multiprocess = True):
#         self.d1 = d1
#         self.d2 = d2
#         super().__init__(self.d1.dataset + self.d2.dataset, batch_size=batch_size, shuffle = shuffle, num_workers=1 if use_multiprocess else 0)
#         self.pbar = progress_bar
#         self.shuffle = shuffle
#         self.use_multiprocess = use_multiprocess
    
#     def __iter__(self):        
#         p = self.get_pbar()
#         for x in self.d1:
#             yield x
#             p.update(get_size(x))
#         for x in self.d2:
#             yield x
#             p.update(get_size(x))
#         p.close()

#     def __add__(self, other: DataLoader):
#         # because I hate my life
#         return DataIterator.__add__(self, other)
    
#     def get_pbar(self):
#         try:
#             return ProgressBar(total = len(self.d1.dataset) + len(self.d2.dataset), disable = not self.pbar)
#         except AttributeError:
#             return ProgressBar(disable = not self.pbar)
