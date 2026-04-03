import torch
import torch.nn as nn
from tqdm import tqdm
from qmllm.quantization.quant_funcs import pseudo_quantize_tensor
from qmllm.quantization.qlinear import WALinear
from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM

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


@torch.no_grad()
def pseudo_quantize_model_weight(
    model,
    w_bit,
    q_config,
):

    layers = get_blocks(model)
    for i in tqdm(range(len(layers)), desc="pseudo weight quantization..."):
        named_linears = get_named_linears(layers[i])
        for n, m in named_linears.items():
            # m.cuda()
            m.weight.data = pseudo_quantize_tensor(
                m.weight.data, n_bits=w_bit, **q_config
            )
            # m.cpu()


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