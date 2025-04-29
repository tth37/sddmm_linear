
from torch import nn
import torch
from torch.nn import functional as F


def get_topk_indices(x: torch.Tensor, topk: int):
    bsz, n_neurons = x.shape
    assert bsz == 1
    x = x.view(-1).abs().float()
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

def get_topk_mask(topk_indices, in_features):
    topk_mask = torch.zeros(in_features, dtype=torch.bool, device=topk_indices.device)
    topk_mask[topk_indices] = True
    return topk_mask

class FastCachedSddmmLinear(nn.Module):

    def __init__(self, in_features, out_features, topk_ratio, recall_thres):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.topk_ratio = float(topk_ratio)
        self.topk = int(self.in_features * self.topk_ratio)
        self.recall_thres = float(recall_thres)

        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.indices = []
        self.weight_slices = []
        self.register_buffer('topk_masks', torch.empty(0, self.in_features, dtype=torch.bool), persistent=False)

    @staticmethod
    def from_linear(linear, topk_ratio, recall_thres):
        cached_sddmm_linear = FastCachedSddmmLinear(linear.in_features, linear.out_features, topk_ratio, recall_thres)
        cached_sddmm_linear.weight = linear.weight
        cached_sddmm_linear.bias = linear.bias
        return cached_sddmm_linear.to(linear.weight.device)

    def cache_slice(self, indices, mask):
        weight_slice = self.weight[:, indices]
        self.indices.append(indices)
        self.weight_slices.append(weight_slice)
        self.topk_masks = torch.cat([self.topk_masks, mask.unsqueeze(0)], dim=0)

    def find_closest(self, indices, mask):
        # best_idx, best_recall = 0, 0.0
        # for i, cached_indices in enumerate(self.indices):
        #     recall = get_recall(cached_indices, indices)
        #     if recall > best_recall:
        #         best_recall = recall
        #         best_idx = i
        # return best_idx, best_recall
        if len(self.indices) == 0:
            return 0, 0.0
        intersection = (self.topk_masks & mask).sum(dim=1)
        recalls = intersection.float() / self.topk
        best_recall, best_idx = torch.max(recalls, dim=0)
        return best_idx.item(), best_recall.item()
    
    @torch.no_grad()
    def forward(self, x):
        bsz, seq, _ = x.shape
        if seq != 1:
            output = x @ self.weight.T
            if self.bias is not None:
                output += self.bias
            return output
        
        x = x.view(bsz * seq, -1)
        x_mean = x.abs().mean(dim=0, keepdim=True)
        topk_indices = get_topk_indices(x_mean, self.topk)
        topk_mask = get_topk_mask(topk_indices, self.in_features)

        best_idx, best_recall = self.find_closest(topk_indices, topk_mask)
        if best_recall < self.recall_thres:
            self.cache_slice(topk_indices, topk_mask)
            best_idx = len(self.indices) - 1
            best_recall = 1.0

        indices = self.indices[best_idx]
        weight_slice = self.weight_slices[best_idx]
        output = x[:, indices] @ weight_slice.T
        if self.bias is not None:
            output += self.bias
        output = output.view(bsz, seq, -1)
        return output

