from qmllm.quantization.qlinear import WALinear
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM

def quantize_opt(
    model, weight_quant="per_tensor", act_quant="per_tensor", w_bit=4, a_bit=8, quantize_bmm_input=True
):
    from transformers.models.opt.modeling_opt import (
        OPTAttention,
        OPTDecoderLayer,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, OPTDecoderLayer):
            m.fc1 = WALinear.from_float(
                m.fc1, weight_quant=weight_quant, act_quant=act_quant, w_bit=w_bit, a_bit=a_bit,
            )
            m.fc2 = WALinear.from_float(
                m.fc2, weight_quant=weight_quant, act_quant=act_quant, w_bit=w_bit, a_bit=a_bit,
            )
        elif isinstance(m, OPTAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = WALinear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                w_bit=w_bit, a_bit=a_bit,
                quantize_output=quantize_bmm_input,
            )
            m.k_proj = WALinear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                w_bit=w_bit, a_bit=a_bit,
                quantize_output=quantize_bmm_input,
            )
            m.v_proj = WALinear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                w_bit=w_bit, a_bit=a_bit,
                quantize_output=quantize_bmm_input,
            )
            m.out_proj = WALinear.from_float(
                m.out_proj, weight_quant=weight_quant, act_quant=act_quant, w_bit=w_bit, a_bit=a_bit,
            )
    return model


def quantize_llama_like(
    model, weight_quant="per_channel", act_quant="per_token", w_bit=4, a_bit=8, quantize_bmm_input=False
):
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        LlamaMLP,
    )

    from transformers.models.mistral.modeling_mistral import (
        MistralAttention,
        MistralMLP,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, (LlamaMLP, MistralMLP)):
            m.gate_proj = WALinear.from_float(
                m.gate_proj, weight_quant=weight_quant, act_quant=act_quant, w_bit=w_bit, a_bit=a_bit,
            )
            m.up_proj = WALinear.from_float(
                m.up_proj, weight_quant=weight_quant, act_quant=act_quant, w_bit=w_bit, a_bit=a_bit,
            )
            m.down_proj = WALinear.from_float(
                m.down_proj, weight_quant=weight_quant, act_quant=act_quant, w_bit=w_bit, a_bit=a_bit,
            )
        elif isinstance(m, (LlamaAttention, MistralAttention)):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = WALinear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                w_bit=w_bit, a_bit=a_bit,
            )
            m.k_proj = WALinear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                w_bit=w_bit, a_bit=a_bit,
            )
            m.v_proj = WALinear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                w_bit=w_bit, a_bit=a_bit,
                quantize_output=quantize_bmm_input,
            )
            m.o_proj = WALinear.from_float(
                m.o_proj, weight_quant=weight_quant, act_quant=act_quant, w_bit=w_bit, a_bit=a_bit,
            )
    return model


def quantize_mixtral(
    model, weight_quant="per_channel", act_quant="per_token", w_bit=4, a_bit=8, quantize_bmm_input=False
):
    from transformers.models.mixtral.modeling_mixtral import (
        MixtralAttention,
        MixtralSparseMoeBlock,
        MixtralBLockSparseTop2MLP,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, MixtralBLockSparseTop2MLP):
            m.w1 = WALinear.from_float(
                m.w1, weight_quant=weight_quant, act_quant=act_quant, w_bit=w_bit, a_bit=a_bit,
            )
            m.w2 = WALinear.from_float(
                m.w2, weight_quant=weight_quant, act_quant=act_quant, w_bit=w_bit, a_bit=a_bit,
            )
            m.w3 = WALinear.from_float(
                m.w3, weight_quant=weight_quant, act_quant=act_quant, w_bit=w_bit, a_bit=a_bit,
            )
        elif isinstance(m, MixtralAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = WALinear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                w_bit=w_bit, a_bit=a_bit,
                quantize_output=quantize_bmm_input,
            )
            m.k_proj = WALinear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                w_bit=w_bit, a_bit=a_bit,
                quantize_output=quantize_bmm_input,
            )
            m.v_proj = WALinear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                w_bit=w_bit, a_bit=a_bit,
                quantize_output=quantize_bmm_input,
            )
            m.o_proj = WALinear.from_float(
                m.o_proj, weight_quant=weight_quant, act_quant=act_quant, w_bit=w_bit, a_bit=a_bit,
            )
        elif isinstance(m, MixtralSparseMoeBlock):
            m.gate = WALinear.from_float(
                m.gate, weight_quant=weight_quant, act_quant=act_quant, w_bit=w_bit, a_bit=a_bit,
            )
    return model


def quantize_falcon(
    model, weight_quant="per_channel", act_quant="per_token", w_bit=4, a_bit=8, quantize_bmm_input=True
):
    from transformers.models.falcon.modeling_falcon import (
        FalconAttention,
        FalconMLP,
    )

    for name, m in model.named_modules():
        if isinstance(m, FalconMLP):
            m.dense_h_to_4h = WALinear.from_float(
                m.dense_h_to_4h, weight_quant=weight_quant, act_quant=act_quant, w_bit=w_bit, a_bit=a_bit,
            )
            m.dense_4h_to_h = WALinear.from_float(
                m.dense_4h_to_h, weight_quant=weight_quant, act_quant=act_quant, w_bit=w_bit, a_bit=a_bit,
            )
        elif isinstance(m, FalconAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.query_key_value = WALinear.from_float(
                m.query_key_value,
                weight_quant=weight_quant,
                act_quant=act_quant,
                w_bit=w_bit, a_bit=a_bit,
                quantize_output=quantize_bmm_input,
            )
            m.dense = WALinear.from_float(
                m.dense, weight_quant=weight_quant, act_quant=act_quant, w_bit=w_bit, a_bit=a_bit,
            )
    return model


def quantize_model(
    model, weight_quant="per_channel", act_quant="per_token", w_bit=4, a_bit=8, quantize_bmm_input=False
):
    from transformers.models.opt.modeling_opt import OPTPreTrainedModel
    from transformers.models.llama.modeling_llama import LlamaPreTrainedModel
    from transformers.models.mistral.modeling_mistral import MistralPreTrainedModel
    from transformers.models.mixtral.modeling_mixtral import MixtralPreTrainedModel
    from transformers.models.falcon.modeling_falcon import FalconPreTrainedModel

    if isinstance(model, OPTPreTrainedModel):
        return quantize_opt(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            w_bit=w_bit, a_bit=a_bit,
            quantize_bmm_input=quantize_bmm_input,
        )
    elif isinstance(model, (LlamaPreTrainedModel, MistralPreTrainedModel)):
        return quantize_llama_like(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            w_bit=w_bit, a_bit=a_bit,
            quantize_bmm_input=quantize_bmm_input,
        )
    elif isinstance(model, MixtralPreTrainedModel):
        return quantize_mixtral(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            w_bit=w_bit, a_bit=a_bit,
            quantize_bmm_input=quantize_bmm_input,
        )
    elif isinstance(model, FalconPreTrainedModel):
        return quantize_falcon(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            w_bit=w_bit, a_bit=a_bit,
            quantize_bmm_input=quantize_bmm_input,
        )
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}


def get_blocks(model):
    if model.__class__.__name__ == "LlamaForCausalLM":
        layers = model.model.layers
    elif model.__class__.__name__ == "LlavaLlamaForCausalLM":
        # layers = [model.model.layers, model.model.vision_tower.vision_tower.vision_model.encoder.layers]
        layers = model.model.layers
    elif model.__class__.__name__ == "LlavaQwenForCausalLM":
        layers = model.model.layers
    elif model.__class__.__name__ == "InternLM2ForCausalLM":
        layers = model.model.layers
    elif model.__class__.__name__ == "InternVLChatModel":
        layers = model.language_model.model.layers
    elif model.__class__.__name__ == "Qwen2VLForConditionalGeneration":
        layers = model.model.layers
    elif model.__class__.__name__ == "Qwen2_5_VLForConditionalGeneration":
        layers = model.model.layers
    elif model.__class__.__name__ == "LlavaLlamaModel":
        layers = model.llm.model.layers
    elif isinstance(model, OPTForCausalLM):
        layers = model.model.decoder.layers
    elif isinstance(model, BloomForCausalLM):
        layers = model.transformer.h
    elif "mpt" in str(model.__class__).lower():
        layers = model.transformer.blocks
    elif "falcon" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "bigcode" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "neox" in str(model.__class__).lower():
        layers = model.gpt_neox.layers
    else:
        raise NotImplementedError(type(model))
    return layers


def get_module_by_name_suffix(model, module_name: str):
    for name, module in model.named_modules():
        if name.endswith(module_name):
            return module
        

@torch.no_grad()
def pseudo_quantize_model_weight_act(
    model,
    w_bit,
    a_bit,
):
    
    layers = get_blocks(model)
    for i in tqdm(range(len(layers)), desc="pseudo weight activation quantization..."):
        named_linears = get_named_linears(layers[i])
        for n, m in named_linears.items():
            new_linear = WALinear.from_float(m, weight_quant="per_channel", act_quant="per_token", w_bit=w_bit, a_bit=a_bit)
            father_module = get_module_by_name_suffix(layers[i], '.'.join(n.split(".")[:-1]))
            setattr(father_module, n.split('.')[-1], new_linear)
            del new_linear, m
            torch.cuda.empty_cache()