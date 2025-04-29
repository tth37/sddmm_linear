
from torch import nn
import torch
from torch.nn import functional as F


def get_topk_indices(x: torch.Tensor, topk: int):
    bsz, n_neurons = x.shape
    assert bsz == 1
    x = x.view(-1).abs().float()
    # activation = F.softmax(x, dim=0)
    topk_indices = torch.argsort(x, descending=True)[:topk]
    return topk_indices

def get_recall(indices: torch.Tensor, indices_ground_truth: torch.Tensor):
    indices = indices.cpu().numpy()
    indices_ground_truth = indices_ground_truth.cpu().numpy()
    indices_set = set(indices)
    indices_ground_truth_set = set(indices_ground_truth)
    intersection = len(indices_set.intersection(indices_ground_truth_set))
    recall = intersection / len(indices_ground_truth_set)
    return recall

class CachedSddmmLinear(nn.Module):

    def __init__(self, in_features, out_features, topk_ratio, recall_thres):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.topk_ratio = float(topk_ratio)
        self.topk = int(self.in_features * self.topk_ratio)
        self.recall_thres = float(recall_thres)
        self.topk_p70 = int(self.in_features * 0.7)
        self.topk_p50 = int(self.in_features * 0.5)
        self.verbose = False

        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.indices = []
        self.weight_slices = []

    @staticmethod
    def from_linear(linear, topk_ratio, recall_thres):
        cached_sddmm_linear = CachedSddmmLinear(linear.in_features, linear.out_features, topk_ratio, recall_thres)
        cached_sddmm_linear.weight = linear.weight
        cached_sddmm_linear.bias = linear.bias
        return cached_sddmm_linear.to(linear.weight.device)

    def cache_slice(self, indices):
        weight_slice = self.weight[:, indices]
        self.indices.append(indices)
        self.weight_slices.append(weight_slice)

    def find_closest(self, indices):
        best_idx, best_recall = 0, 0.0
        for i, cached_indices in enumerate(self.indices):
            recall = get_recall(cached_indices, indices)
            if recall > best_recall:
                best_recall = recall
                best_idx = i
        return best_idx, best_recall
    
    @torch.no_grad()
    def forward(self, x):
        bsz, seq, _ = x.shape
        if bsz * seq != 1:
            x = x.view(bsz * seq, -1)
            x_mean = x.mean(dim=0, keepdim=True)
            topk_indices = get_topk_indices(x_mean, self.topk)
            # self.cache_slice(topk_indices)
            output = x @ self.weight.T
            if self.bias is not None:
                output += self.bias
            output = output.view(bsz, seq, -1)
            return output
        
        x = x.view(bsz * seq, -1)
        topk_indices = get_topk_indices(x, self.topk)
        topk_p70_indices = get_topk_indices(x, self.topk_p70)
        topk_p50_indices = get_topk_indices(x, self.topk_p50)
        best_idx, best_recall = self.find_closest(topk_indices)
        if self.verbose:
            if len(self.indices) == 0:
                best_p70_recall, best_p50_recall = 0.0, 0.0
            else:
                best_p70_recall = get_recall(self.indices[best_idx], topk_p70_indices)
                best_p50_recall = get_recall(self.indices[best_idx], topk_p50_indices)
            print(f"Use existing: best_recall={best_recall:.2f}, best_p70_rc={best_p70_recall:.2f}, best_p50_rc={best_p50_recall:.2f}, cache_size={len(self.indices)}")
        if best_recall < self.recall_thres:
            self.cache_slice(topk_indices)
            best_idx = len(self.indices) - 1
            best_recall = 1.0
        indices = self.indices[best_idx]
        weight_slice = self.weight_slices[best_idx]
        output = x[:, indices] @ weight_slice.T
        if self.bias is not None:
            output += self.bias
        output = output.view(bsz, seq, -1)
        return output

