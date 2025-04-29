import types
from ..collector_linear import CollectorLinear
from ..sddmm_linear import SddmmLinear
from ..cached_sddmm_linear import CachedSddmmLinear
from ..griffin_linear import GriffinLinear


def patch_qwen2_collective_linear(model, topk_ratio):
    for layer in model.model.layers:
        layer.mlp.gate_proj = CollectorLinear.from_linear(layer.mlp.gate_proj, topk_ratio)
        layer.mlp.up_proj = CollectorLinear.from_linear(layer.mlp.up_proj, topk_ratio)
        layer.mlp.down_proj = CollectorLinear.from_linear(layer.mlp.down_proj, topk_ratio)
        # layer.self_attn.q_proj = CollectorLinear.from_linear(layer.self_attn.q_proj)
        # layer.self_attn.k_proj = CollectorLinear.from_linear(layer.self_attn.k_proj)
        # layer.self_attn.v_proj = CollectorLinear.from_linear(layer.self_attn.v_proj)
        # layer.self_attn.o_proj = CollectorLinear.from_linear(layer.self_attn.o_proj)
    
    def get_activations(self):
        inputs = []
        for layer in self.model.layers:
            inputs.append(
                {
                    "gate_proj": layer.mlp.gate_proj.get_activations(),
                    "up_proj": layer.mlp.up_proj.get_activations(),
                    "down_proj": layer.mlp.down_proj.get_activations(),
                    # "q_proj": layer.self_attn.q_proj.get_activations(),
                    # "k_proj": layer.self_attn.k_proj.get_activations(),
                    # "v_proj": layer.self_attn.v_proj.get_activations(),
                    # "o_proj": layer.self_attn.o_proj.get_activations()
                }
            )
        return inputs
    model.get_activations = types.MethodType(get_activations, model)
    return model

def patch_qwen2_sddmm_linear(model, topk_ratio, cluster_results):
    for i, layer in enumerate(model.model.layers):
        layer.mlp.gate_proj = SddmmLinear.from_linear(layer.mlp.gate_proj, topk_ratio, cluster_results[i]["gate_proj"]).to(layer.mlp.gate_proj.weight.device)
        layer.mlp.up_proj = SddmmLinear.from_linear(layer.mlp.up_proj, topk_ratio, cluster_results[i]["up_proj"]).to(layer.mlp.up_proj.weight.device)
        layer.mlp.down_proj = SddmmLinear.from_linear(layer.mlp.down_proj, topk_ratio, cluster_results[i]["down_proj"]).to(layer.mlp.down_proj.weight.device)
        # layer.self_attn.q_proj = SddmmLinear.from_linear(layer.self_attn.q_proj, topk_ratio, cluster_results[i]["q_proj"]).to(layer.self_attn.q_proj.weight.device)
        # layer.self_attn.k_proj = SddmmLinear.from_linear(layer.self_attn.k_proj, topk_ratio, cluster_results[i]["k_proj"]).to(layer.self_attn.k_proj.weight.device)
        # layer.self_attn.v_proj = SddmmLinear.from_linear(layer.self_attn.v_proj, topk_ratio, cluster_results[i]["v_proj"]).to(layer.self_attn.v_proj.weight.device)
        # layer.self_attn.o_proj = SddmmLinear.from_linear(layer.self_attn.o_proj, topk_ratio, cluster_results[i]["o_proj"]).to(layer.self_attn.o_proj.weight.device)

    return model

def patch_qwen2_cached_sddmm_linear(model, topk_ratio, recall_thres):
    for layer in model.model.layers:
        layer.mlp.gate_proj = CachedSddmmLinear.from_linear(layer.mlp.gate_proj, topk_ratio, recall_thres)
        layer.mlp.up_proj = CachedSddmmLinear.from_linear(layer.mlp.up_proj, topk_ratio, recall_thres)
        layer.mlp.down_proj = CachedSddmmLinear.from_linear(layer.mlp.down_proj, topk_ratio, recall_thres)
        
    return model

def patch_qwen2_griffin_linear(model, topk_ratio):
    for layer in model.model.layers:
        layer.mlp.gate_proj = GriffinLinear.from_linear(layer.mlp.gate_proj, topk_ratio)
        layer.mlp.up_proj = GriffinLinear.from_linear(layer.mlp.up_proj, topk_ratio)
        layer.mlp.down_proj = GriffinLinear.from_linear(layer.mlp.down_proj, topk_ratio)
        
    return model
