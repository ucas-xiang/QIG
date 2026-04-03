import torch
import torch.nn as nn

from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.bloom.modeling_bloom import BloomBlock
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import (
    MistralDecoderLayer,
    MistralRMSNorm,
)
from transformers.models.mixtral.modeling_mixtral import (
    MixtralDecoderLayer,
    MixtralRMSNorm,
)
from transformers.models.falcon.modeling_falcon import FalconDecoderLayer

from transformers.models.clip.modeling_clip import CLIPEncoderLayer

@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, nn.LayerNorm)
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )

    ln.weight.div_(scales)
    ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))


@torch.no_grad()
def smooth_ln_fcs_llama_like(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    # assert isinstance(ln, (LlamaRMSNorm, MistralRMSNorm, MixtralRMSNorm))
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()
    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)
    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )

    ln.weight.div_(scales)
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))


@torch.no_grad()
def smooth_lm(model, scales, alpha=0.5):
    for name, module in model.named_modules():
        if isinstance(module, OPTDecoderLayer):
            attn_ln = module.self_attn_layer_norm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]
            qkv_input_scales = scales[name + ".self_attn.q_proj"]
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.final_layer_norm
            fc1 = module.fc1
            fc1_input_scales = scales[name + ".fc1"]
            smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)
        elif isinstance(module, BloomBlock):
            attn_ln = module.input_layernorm
            qkv = module.self_attention.query_key_value
            qkv_input_scales = scales[name + ".self_attention.query_key_value"]
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.post_attention_layernorm
            fc1 = module.mlp.dense_h_to_4h
            fc1_input_scales = scales[name + ".mlp.dense_h_to_4h"]
            smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)
        elif isinstance(module, FalconDecoderLayer):
            qkv = module.self_attention.query_key_value
            qkv_input_scales = scales[name + ".self_attention.query_key_value"]
            fc1_input_scales = scales[name + ".mlp.dense_h_to_4h"]
            fc1 = module.mlp.dense_h_to_4h

            if (
                not module.config.new_decoder_architecture
                and module.config.parallel_attn
            ):
                attn_ln = module.input_layernorm
                smooth_ln_fcs(attn_ln, [qkv, fc1], qkv_input_scales, alpha)
            else:
                attn_ln = (
                    module.ln_attn
                    if module.config.new_decoder_architecture
                    else module.input_layernorm
                )
                ffn_ln = (
                    module.ln_mlp
                    if module.config.new_decoder_architecture
                    else module.post_attention_layernorm
                )
                smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)
                smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)
        elif isinstance(module, (LlamaDecoderLayer, MistralDecoderLayer)):
            attn_ln = module.input_layernorm  # attention forward norm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]

            qkv_input_scales = scales[name + ".self_attn.q_proj"]
            smooth_ln_fcs_llama_like(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.post_attention_layernorm  # feed forward norm
            fcs = [module.mlp.gate_proj, module.mlp.up_proj]
            fcs_input_scales = scales[name + ".mlp.gate_proj"]

            smooth_ln_fcs_llama_like(ffn_ln, fcs, fcs_input_scales, alpha)
        elif isinstance(module, MixtralDecoderLayer):
            attn_ln = module.input_layernorm  # attention forward norm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]

            qkv_input_scales = scales[name + ".self_attn.q_proj"]
            smooth_ln_fcs_llama_like(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.post_attention_layernorm  # feed forward norm
            fcs = [module.block_sparse_moe.gate]
            for expert in module.block_sparse_moe.experts:
                fcs.append(expert.w1)
                fcs.append(expert.w3)
            fcs_input_scales = scales[name + ".block_sparse_moe.gate"]

            smooth_ln_fcs_llama_like(ffn_ln, fcs, fcs_input_scales, alpha)
        elif module.__class__.__name__ == "Qwen2DecoderLayer":
            attn_ln = module.input_layernorm  # attention forward norm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]

            qkv_input_scales = scales[name + ".self_attn.q_proj"]
            smooth_ln_fcs_llama_like(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.post_attention_layernorm  # feed forward norm
            fcs = [module.mlp.gate_proj, module.mlp.up_proj]
            fcs_input_scales = scales[name + ".mlp.gate_proj"]

            smooth_ln_fcs_llama_like(ffn_ln, fcs, fcs_input_scales, alpha)
        elif module.__class__.__name__ == "InternLM2DecoderLayer":
            attn_ln = module.attention_norm  # attention forward norm
            qkv = module.attention.wqkv
            qkv_input_scales = scales[name + ".attention.wqkv"]
            smooth_ln_fcs_llama_like(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.ffn_norm  # feed forward norm
            fcs = [module.feed_forward.w1, module.feed_forward.w3]
            fcs_input_scales = scales[name + ".feed_forward.w1"]

            smooth_ln_fcs_llama_like(ffn_ln, fcs, fcs_input_scales, alpha)
        elif module.__class__.__name__ == "Qwen2VLDecoderLayer":
            attn_ln = module.input_layernorm  # attention forward norm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]

            qkv_input_scales = scales[name + ".self_attn.q_proj"]
            smooth_ln_fcs_llama_like(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.post_attention_layernorm  # feed forward norm
            fcs = [module.mlp.gate_proj, module.mlp.up_proj]
            fcs_input_scales = scales[name + ".mlp.gate_proj"]

            smooth_ln_fcs_llama_like(ffn_ln, fcs, fcs_input_scales, alpha)
        
        elif module.__class__.__name__ == "Qwen2_5_VLDecoderLayer":
            attn_ln = module.input_layernorm  # attention forward norm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]

            qkv_input_scales = scales[name + ".self_attn.q_proj"]
            smooth_ln_fcs_llama_like(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.post_attention_layernorm  # feed forward norm
            fcs = [module.mlp.gate_proj, module.mlp.up_proj]
            fcs_input_scales = scales[name + ".mlp.gate_proj"]

            smooth_ln_fcs_llama_like(ffn_ln, fcs, fcs_input_scales, alpha)


@torch.no_grad()
def smooth_vit(model, scales, alpha=0.5):
    for name, module in model.named_modules():
        # if isinstance(module, InternVisionEncoderLayer_26B):
        #     # Only adapt to InternViT-6B
        #     attn_ln = module.norm1
        #     qkv = module.attn.qkv
        #     qkv_input_scales = scales[name + ".attn.qkv"]
        #     smooth_ln_fcs_llama_like(attn_ln, qkv, qkv_input_scales, alpha)
        #     print(f"[SmoothQuant] Success to handle {name}.attn")

        #     ffn_ln = module.norm2
        #     fcs = [module.mlp.fc1]
        #     fcs_input_scales = scales[name + ".mlp.fc1"]
        #     smooth_ln_fcs_llama_like(ffn_ln, fcs, fcs_input_scales, alpha)
        #     print(f"[SmoothQuant] Success to handle {name}.mlp")

        # elif isinstance(module, InternVisionEncoderLayer_8B):
        #     # Only adapt to InternViT-300M
        #     attn_ln = module.norm1
        #     qkv = module.attn.qkv
        #     qkv_input_scales = scales[name + ".attn.qkv"]
        #     smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)
        #     print(f"[SmoothQuant] Success to handle {name}.attn")

        #     ffn_ln = module.norm2
        #     fcs = [module.mlp.fc1]
        #     fcs_input_scales = scales[name + ".mlp.fc1"]
        #     smooth_ln_fcs(ffn_ln, fcs, fcs_input_scales, alpha)
        #     print(f"[SmoothQuant] Success to handle {name}.mlp")

        # elif isinstance(module, type(model.vision_tower.vision_model.encoder.layers[0])):
        if module.__class__.__name__ == "SigLipEncoderLayer":
            # Only adapt to llava-like model
            attn_ln = module.layer_norm1
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]
            qkv_input_scales = scales[name + ".self_attn.q_proj"]
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)
            print(f"[SmoothQuant] Success to handle {name}.self_attn")

            ffn_ln = module.layer_norm2
            fc1 = module.mlp.fc1
            fc1_input_scales = scales[name + ".mlp.fc1"]
            smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)
            print(f"[SmoothQuant] Success to handle {name}.mlp")
        elif isinstance(module, CLIPEncoderLayer):
            # Only adapt to llava-like model
            attn_ln = module.layer_norm1
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]
            qkv_input_scales = scales[name + ".self_attn.q_proj"]
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)
            print(f"[SmoothQuant] Success to handle {name}.self_attn")

            ffn_ln = module.layer_norm2
            fc1 = module.mlp.fc1
            fc1_input_scales = scales[name + ".mlp.fc1"]
            smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)
            print(f"[SmoothQuant] Success to handle {name}.mlp")

@torch.no_grad()
def smooth_vit_block(model, scales, block_idx, alpha=0.5):
    # Only adapt to InternViT now
    for name, module in model.named_modules():
        if module.__class__.__name__ == "InternVisionEncoderLayer_26B" and (int(name.split(".")[2]) in block_idx):
            attn_ln = module.norm1
            qkv = module.attn.qkv
            qkv_input_scales = scales[name + ".attn.qkv"]
            smooth_ln_fcs_llama_like(attn_ln, qkv, qkv_input_scales, alpha)
            print(f"[SmoothQuant] Success to handle {name}.attn")

            ffn_ln = module.norm2
            fcs = [module.mlp.fc1]
            fcs_input_scales = scales[name + ".mlp.fc1"]
            smooth_ln_fcs_llama_like(ffn_ln, fcs, fcs_input_scales, alpha)
            print(f"[SmoothQuant] Success to handle {name}.mlp")