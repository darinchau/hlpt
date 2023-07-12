### Provides a custom Dataset class for nice things to happen
from __future__ import annotations
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset
from torch import Tensor
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm as ProgressBar
import warnings

def get_size(x):
    if isinstance(x, list):
        return int(x[0].size(0))
    elif isinstance(x, Tensor):
        return int(x.size(0))
    else:
        warnings.warn(f"No big deal, just wanna let you know x of type {type(x)} exists.")
        return 1

class DataIterator(DataLoader):
    """A wrapper for datasets and dataloaders that automatically perform train-test splits and stuff like a torch dataloader
    
    label: if not None, we will perform train-test split in a stratified manner"""
    def __init__(self, *t : Tensor, batch_size: int = 1, label: list | None = None, shuffle = True, progress_bar = True, use_multiprocess = True):
        assert len(t) > 0
        self.len_t = len(t[0])
        for tensor in t:
            assert len(tensor) == self.len_t
        
        super().__init__(TensorDataset(*t), batch_size=batch_size, shuffle = shuffle, num_workers=1 if use_multiprocess else 0)
        self.label = label
        self.pbar = progress_bar
        self.shuffle = shuffle
        self.use_multiprocess = use_multiprocess
        self.bs = batch_size
    
    def split(self, train_size = 0.85, *, seed = None) -> tuple[DataIterator, DataIterator]:
        """Performs train-test split"""
        trains = []
        tests = []

        if seed is None:
            seed = 42069

        for tensor in self.dataset[:]:
            train, test = train_test_split(tensor.detach().cpu().numpy(), test_size = 1 - train_size, random_state=seed, shuffle = False, stratify=self.label)
            train = torch.as_tensor(train, dtype = tensor.dtype, device = tensor.device)
            test = torch.as_tensor(test, dtype = tensor.dtype, device = tensor.device)
            trains.append(train)
            tests.append(test)
        
        # Create new TensorDatasets and DataLoaders for train and test sets
        train_loader = DataIterator(*trains, batch_size=self.bs, label=self.label, shuffle=self.shuffle, progress_bar=self.pbar, use_multiprocess=self.use_multiprocess)
        test_loader = DataIterator(*tests, batch_size=self.bs, label=self.label, shuffle=self.shuffle, progress_bar=self.pbar, use_multiprocess=self.use_multiprocess)
        return train_loader, test_loader

    def __iter__(self):
        p = ProgressBar(total = self.len_t, disable = not self.pbar)
        for x in super().__iter__():
            yield x
            p.update(get_size(x))
        p.close()

    def __add__(self, other: DataLoader):
        if not isinstance(other, DataLoader):
            raise RuntimeError("Cannot add data iterators with other things")
        self.pbar = False
        if isinstance(other, (DataIterator, ConcatIterator)):
            other.pbar = False
        return ConcatIterator(self, other, self.batch_size, shuffle = self.shuffle, progress_bar = self.pbar, use_multiprocess = self.use_multiprocess)

class ConcatIterator(DataLoader):
    def __init__(self, d1: DataLoader, d2: DataLoader, batch_size: int = 1, shuffle = True, progress_bar = True, use_multiprocess = True):
        self.d1 = d1
        self.d2 = d2
        super().__init__(self.d1.dataset + self.d2.dataset, batch_size=batch_size, shuffle = shuffle, num_workers=1 if use_multiprocess else 0)
        self.pbar = progress_bar
    
    def __iter__(self):        
        p = ProgressBar(total = len(self), disable = not self.pbar)
        for x in self.d1:
            yield x
            p.update(get_size(x))
        for x in self.d2:
            yield x
            p.update(get_size(x))
        p.close()

    def __add__(self, other: DataLoader):
        # because I hate my life
        return DataIterator.__add__(self, other)
