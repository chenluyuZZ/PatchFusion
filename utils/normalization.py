import torch
import torchvision

import copy
import torch
from torch import nn

class Normalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, x):
        # if bool(torch.max(x) <= 1 and torch.min(x) >= 0):
        #     return x
        return (x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None]

norm = Normalize()


class UnNormalize(object):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, tensors):

        assert len(tensors.shape)>=3
        if len(tensors.shape)==3:
            for i, m, s in zip(range(3), self.mean, self.std):
                tensors[i].mul(s).add(m)
            return tensors
        else:
            for i, m, s in zip(range(3), self.mean, self.std):
                tensors[:,i,:,:].mul(s).add(m)
            return tensors

unorm = UnNormalize()


