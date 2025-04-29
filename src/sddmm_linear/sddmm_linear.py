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

def get_topk_recall(indices: torch.Tensor, indices_ground_truth: torch.Tensor):
    indices = indices.cpu().numpy()
    indices_ground_truth = indices_ground_truth.cpu().numpy()
    indices_set = set(indices)
    indices_ground_truth_set = set(indices_ground_truth)
    intersection = len(indices_set.intersection(indices_ground_truth_set))
    recall = intersection / len(indices_ground_truth_set)
    return recall

class SddmmLinear(nn.Module):

    def __init__(self, in_features, out_features, topk_ratio, cluster_result, weight, bias):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_clusters = int(cluster_result["n_clusters"])
        self.topk_ratio = float(topk_ratio)
        self.topk = int(self.in_features * self.topk_ratio)
        self.verbose = False

        cluster_centers_tensor = torch.from_numpy(cluster_result["cluster_centers"]).float()
        self.register_buffer("stacked_cluster_centers", cluster_centers_tensor)
        self.register_buffer('weight_full', weight.detach().clone())
        self.weight = self.weight_full
        if bias is not None:
            self.register_buffer('bias_full', bias.detach().clone())
        else:
            self.register_buffer('bias_full', None)
        self.bias = self.bias_full

        self.indices_buffers = []
        self.weight_slice_buffers = []

        for i in range(self.n_clusters):
            indices = torch.argsort(self.stacked_cluster_centers[i], descending=True)
            topk_indices = indices[:self.topk]
            buffer_name_indices = f'indices_{i}'
            self.register_buffer(buffer_name_indices, topk_indices)
            self.indices_buffers.append(getattr(self, buffer_name_indices))

            weight_slice = self.weight_full[:, topk_indices]
            buffer_name_weight = f'weight_slice_{i}'
            self.register_buffer(buffer_name_weight, weight_slice)
            self.weight_slice_buffers.append(getattr(self, buffer_name_weight))


    @torch.no_grad()
    def forward(self, x):
        bsz, seq, _ = x.shape
        if bsz * seq != 1:
            x_flat = x.view(bsz * seq, -1)
            intermediate_output = torch.matmul(x_flat, self.weight_full.T) # shape: (bsz * seq, out_features)
            output = intermediate_output.view(bsz, seq, -1)
            if self.bias_full is not None:
                output += self.bias_full
            return output
        # For batch size 1, we need to use the cluster centers to select the topk features
        x_flat = x.view(bsz * seq, -1)
        if self.verbose:
            topk_indices_ground_truth = get_topk_indices(x_flat, self.topk)
            # topk_indices = self.indices_buffers[0]
            # recall = get_topk_recall(topk_indices, topk_indices_ground_truth)
            # print(f"Recall (with cluster_center[0]): {recall:.4f}")
            for i in range(self.n_clusters):
                topk_indices = self.indices_buffers[i]
                recall = get_topk_recall(topk_indices, topk_indices_ground_truth)
                print(f"Recall (with cluster_center[{i}]): {recall:.4f}")
            print("" + "-" * 50)

        x_avg = x_flat.mean(dim=0).abs().float() # shape: (in_features, )
        activation = F.softmax(x_avg, dim=0) # shape: (in_features, )
        topk_indices_ground_truth = torch.argsort(activation, descending=True)[:self.topk]
        i = torch.argmin(torch.cdist(activation.unsqueeze(0), self.stacked_cluster_centers, p=2).squeeze(0))
        topk_indices = self.indices_buffers[i]
        weight_slice = self.weight_slice_buffers[i]
        x_flat = x_flat[:, topk_indices]
        intermediate_output = torch.matmul(x_flat, weight_slice.T) # shape: (bsz * seq, out_features)
        output = intermediate_output.view(bsz, seq, -1)
        if self.bias_full is not None:
            output += self.bias_full
        

        return output

    # @torch.no_grad()
    # def forward(self, x):
    #     bsz, seq, _ = x.shape
    #     if bsz * seq != 1:
    #         x_flat = x.view(bsz * seq, -1)
    #         intermediate_output = torch.matmul(x_flat, self.weight_full.T) # shape: (bsz * seq, out_features)
    #         output = intermediate_output.view(bsz, seq, -1)
    #         if self.bias_full is not None:
    #             output += self.bias_full
    #         return output
    #     # For batch size 1, we need to use the cluster centers to select the topk features
    #     x_flat = x.view(bsz * seq, -1)
    #     x_avg = x_flat.mean(dim=0).abs() # shape: (in_features, )
    #     activation = F.softmax(x_avg.float(), dim=0) # shape: (in_features, )
    #     indices = torch.argsort(activation, descending=True)
    #     topk_indices = indices[:self.topk]
    #     weight_slice = self.weight_full[:, topk_indices]
    #     x_flat = x_flat[:, topk_indices]
    #     intermediate_output = torch.matmul(x_flat, weight_slice.T)
    #     output = intermediate_output.view(bsz, seq, -1)
    #     if self.bias_full is not None:
    #         output += self.bias_full
    #     return output

    @staticmethod
    def from_linear(linear, topk_ratio, cluster_result):
        in_features = linear.in_features
        out_features = linear.out_features
        weight = linear.weight
        bias = linear.bias
        return SddmmLinear(in_features, out_features, topk_ratio, cluster_result, weight, bias)