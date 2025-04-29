from torch import nn
from torch.nn import functional as F
import numpy as np
import torch

def get_topk_indices(x: torch.Tensor, topk: int):
    bsz, n_neurons = x.shape
    assert bsz == 1
    x = x.view(-1).abs().float()
    activation = F.softmax(x, dim=0)
    topk_indices = torch.argsort(activation, descending=True)[:topk]
    return topk_indices

class CollectorLinear(nn.Module):

    def __init__(self, in_features, out_features, topk_ratio):
        super(CollectorLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        self.activations = []
        self.topk_ratio = float(topk_ratio)
        self.topk = int(self.in_features * self.topk_ratio)

    @staticmethod
    def from_linear(linear, topk_ratio):
        collector_linear = CollectorLinear(linear.in_features, linear.out_features, topk_ratio)
        collector_linear.linear.weight = linear.weight
        collector_linear.linear.bias = linear.bias
        return collector_linear

    @property
    def weight(self):
        return self.linear.weight
    
    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x):
        bsz, seq, _ = x.shape
        
        x = x.view(bsz * seq, -1)
        activations = x.abs().detach()
        activations = F.softmax(activations, dim=-1)
        if bsz * seq == 1:
            self.activations.append(activations.cpu().numpy())
            topk_indices = get_topk_indices(x, self.topk)
            output = x[:, topk_indices] @ self.linear.weight[:, topk_indices].T
            if self.linear.bias is not None:
                output += self.linear.bias
            output = output.view(bsz, seq, -1)
            return output
        x = x.view(bsz, seq, -1)
        return self.linear(x)
    
    def get_activations(self):
        return np.concatenate(self.activations, axis=0)

