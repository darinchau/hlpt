### Provides a custom Dataset class for nice things to happen
from __future__ import annotations
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch import Tensor
from sklearn.model_selection import train_test_split
import numpy as np

class DataIterator:
    """A wrapper for datasets and dataloaders that automatically perform train-test splits and stuff like a torch dataloader
    
    label: if not None, we will perform train-test split in a stratified manner"""
    def __init__(self, *t : Tensor, batch_size: int = 1, label: list | None = None, shuffle = False):
        assert len(t) > 0
        self.len_t = len(t[0])
        for tensor in t:
            assert len(tensor) == self.len_t
        
        self.tensors = t
        self.bs = batch_size
        self.label = label
        self.should_shuffle = shuffle

    def split(self, train_size = 0.85, *, seed = None) -> tuple[DataIterator, DataIterator]:
        """Performs train-test split"""
        trains = []
        tests = []

        if seed is None:
            seed = 42069

        for tensor in self.tensors:
            train, test = train_test_split(tensor.detach().cpu().numpy(), test_size = 1 - train_size, random_state=seed, shuffle = False, stratify=self.label)
            train = torch.as_tensor(train, dtype = tensor.dtype, device = tensor.device)
            test = torch.as_tensor(test, dtype = tensor.dtype, device = tensor.device)
            trains.append(train)
            tests.append(test)
        
        # Create new TensorDatasets and DataLoaders for train and test sets
        train_loader = DataIterator(*trains, batch_size=self.bs, label=self.label)
        test_loader = DataIterator(*tests, batch_size=self.bs, label=self.label)
        return train_loader, test_loader
    
    def shuffle(self):
        """Shuffles the dataset lazily (on iteration)"""
        self.should_shuffle = True
    
    def __iter__(self):
        loader = DataLoader(TensorDataset(*self.tensors), batch_size = self.bs, num_workers = 1, shuffle = self.should_shuffle)
        for l in loader:
            yield l

    def __len__(self):
        return self.len_t
    
    def to(self, x):
        self.tensors = [t.to(x) for t in self.tensors]
        return self
    
    def double(self):
        self.tensors = [t.double() for t in self.tensors]
        return self
    
    def float(self):
        self.tensors = [t.float() for t in self.tensors]
        return self
    
    def cpu(self):
        self.tensors = [t.cpu() for t in self.tensors]
        return self
    
    def cuda(self):
        self.tensors = [t.cuda() for t in self.tensors]
        return self
