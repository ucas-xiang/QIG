import torch
import torch.nn as nn
import tqdm
import copy
import gc
import functools
from collections import defaultdict
from typing import List

import numpy as np
from torch.nn import CrossEntropyLoss
from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from qmllm.utils.search import append_str_prefix, get_op_name

from qmllm.methods.qig.quantize.auto_scale_wa_distort import auto_scale_block_wa_distort
from qmllm.methods.qig.quantize.auto_scale import auto_scale_block, apply_scale
from qmllm.quantization.qlinear import WALinear
from qmllm.quantization.quant_funcs import pseudo_quantize_tensor
from .quantizer import get_module_by_name_suffix


__all__ = ["run_qig"]


class GradCacheHook:
    def __init__(self, vis_masks, cap_masks):
        if vis_masks is None or cap_masks is None:
            raise ValueError
        self.hooks = []
        self.vis_masks = vis_masks.cpu()
        self.cap_masks = cap_masks.cpu()
        self.steps = {}
        self.grad_dict = {}


    def cache_grad_hook(self, module, inp, out, name):
        # initialize step counter, we use step counter to find the right mask for the grad
        if name not in self.steps:
            self.steps[name] = 0

        if name not in self.grad_dict:
            self.grad_dict[name] = {"vis_grad": [], "cap_grad": []}

        output_grad = out[0].float()
        step = self.steps[name]

        B, N, C = output_grad.shape

        for batch_idx in range(B):
            vis_mask = self.vis_masks[step]
            cap_mask = self.cap_masks[step]

            vis_grad = output_grad[batch_idx][vis_mask]
            cap_grad = output_grad[batch_idx][cap_mask]

            vis_grad_avg = vis_grad.abs().mean()
            cap_grad_avg = cap_grad.abs().mean()

            self.grad_dict[name]["vis_grad"].append(vis_grad_avg.detach().cpu())
            self.grad_dict[name]["cap_grad"].append(cap_grad_avg.detach().cpu())

            step = step + 1

        self.steps[name] = step


    def register_hooks(self, layers):
        for n, m in layers.named_modules():
            if isinstance(m, nn.Linear) and any([_ in n for _ in ["wo", "w2", "down_proj", "o_proj", "v_proj", "gate_proj", "up_proj", "w1", "w3"]]):
                # print(f"Registering hook for layer.{n}")
                self.hooks.append(
                    m.register_full_backward_hook(
                        functools.partial(self.cache_grad_hook, name=f"layers.{n}")
                    )
                )


    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


    def get_grad_dict(self):
        return self.grad_dict
    

    def get_avg_grad_dict(self):
        avg_grad_dict = {}

        for name, grad_values in self.grad_dict.items():
            mean_vis = torch.mean(torch.stack(grad_values["vis_grad"]))
            mean_cap = torch.mean(torch.stack(grad_values["cap_grad"]))

            avg_grad_dict[name] = {
                "vis_avg_grad": mean_vis.item(),
                "cap_avg_grad": mean_cap.item()
            }

        return avg_grad_dict
    

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
        try:
            layers = model.model.layers
        except:
            layers = model.model.language_model.layers
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


def move_embed(model, device):
    if isinstance(model, LlamaForCausalLM):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    elif isinstance(model, OPTForCausalLM):
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(
            device
        )
    elif isinstance(model, BloomForCausalLM):
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
        model.transformer.word_embeddings_layernorm = (
            model.transformer.word_embeddings_layernorm.to(device)
        )
    elif "mpt" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.emb_drop = model.transformer.emb_drop.to(device)
    elif "falcon" in str(model.__class__).lower():
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
    elif "bigcode" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.wpe = model.transformer.wpe.to(device)
        model.transformer.drop = model.transformer.drop.to(device)
    elif "neox" in str(model.__class__).lower():
        model.gpt_neox.embed_in = model.gpt_neox.embed_in.to(device)
        model.gpt_neox.emb_dropout = model.gpt_neox.emb_dropout.to(device)
        model.embed_out = model.embed_out.to(device)
    elif model.__class__.__name__ == "LlavaLlamaForCausalLM":
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.vision_tower.vision_tower.vision_model.embeddings.to(device)
    elif model.__class__.__name__ == "LlavaQwenForCausalLM":
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.rotary_emb = model.model.rotary_emb.to(device)
        model.model.norm = model.model.norm.to(device)
    elif model.__class__.__name__ == "InternLM2ForCausalLM":
        model.model.tok_embeddings = model.model.tok_embeddings.to(device)
    elif model.__class__.__name__ == "InternVLChatModel":
        model.language_model.model.tok_embeddings = model.language_model.model.tok_embeddings.to(device)  
    elif model.__class__.__name__ == "Qwen2VLForConditionalGeneration":
        try:
            model.model.embed_tokens = model.model.embed_tokens.to(device)
        except:
            model.model.language_model.embed_tokens = model.model.language_model.embed_tokens.to(device)
    elif model.__class__.__name__ == "LlavaLlamaModel":
        model.llm.model.embed_tokens = model.llm.model.embed_tokens.to(device)
    else:
        raise NotImplementedError(type(model))


def process_input(prompt_inputs, prompt_kwargs):
    inputs = {**prompt_inputs, **prompt_kwargs}
    inputs["use_cache"] = False
    vision_mask = inputs.pop("vision_mask", None)
    caption_mask = inputs.pop("caption_mask", None)
    
    return inputs, vision_mask, caption_mask


@torch.no_grad()
def run_qig(
    model,
    prompt_inputs,
    prompt_kwargs,
    w_bit,
    a_bit,
    q_config,
    auto_scale=True,
    loss_mode="mae",
    wa_quant=False,
    reweight=False,
    distort=False
):
    if "bigcode" in str(model.model.__class__).lower():
        # otherwise attention_mask will always be on cpu.
        model.transformer.bias = model.transformer.bias.to("cuda")

    layers = get_blocks(model.model)

    inps = []
    layer_kwargs = {}

    layers[0] = layers[0].cuda()
    move_embed(model.model, 'cpu')

    # get input and kwargs to layer 0
    # with_kwargs is only supported in PyTorch 2.0
    # use this Catcher hack for now
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            if hasattr(module, "attention_type"):
                self.attention_type = module.attention_type
        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference

    # patch layer 0 to catch input and kwargs
    layers[0] = Catcher(layers[0])

    inputs, vision_mask, caption_mask = process_input(prompt_inputs, prompt_kwargs)

    # model.to_cuda()
    try:
        if torch.cuda.device_count() > 1:
            model.to_cpu()
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to('cpu')
                # if list
                elif isinstance(v, list):
                    inputs[k] = [item.to('cpu') if torch.is_tensor(item) else item for item in v]
            # inputs = {k: v.to('cpu') if torch.is_tensor(v) else v for k, v in inputs.items()}           
        model(**inputs)
    except ValueError: # work with early exit
        pass

    model.to_cpu()
    layers[0] = layers[0].module  # restore
    inps = inps[0]
    layer_kwargs["use_cache"] = False

    layers[0] = layers[0].cpu()
    move_embed(model.model, "cpu")

    gc.collect()
    torch.cuda.empty_cache()

    qig_results = {
        "scale": [],
    }

    model.to_cpu()
    if reweight:
        model.to_cuda()
        # save gradient
        grad_cache = GradCacheHook(vis_masks=vision_mask, cap_masks=caption_mask)        
        grad_cache.register_hooks(layers=layers)
               
        with torch.enable_grad():
            mini_batch = 1
            total_samples = next(iter(prompt_inputs.values())).shape[0]
            accum_steps = int(total_samples/mini_batch)
            
            for i in range(0, total_samples, mini_batch):
                mini_inputs = {}
                for k in inputs:
                    if isinstance(inputs[k], torch.Tensor):
                        mini_inputs[k] = inputs[k][i:i+mini_batch]
                
                outputs = model(**mini_inputs)

                loss = outputs[0]

                loss = loss / accum_steps
                loss.backward()

        model.to_cpu()
        grad_avg_dict = grad_cache.get_avg_grad_dict()
        grad_cache.remove_hooks()
        del grad_cache

        attn_list = []
        mlp_list = []

        for key_name in grad_avg_dict:
            if "down_" in key_name or "w2" in key_name:
                mlp_list.append(grad_avg_dict[key_name]["vis_avg_grad"] / grad_avg_dict[key_name]["cap_avg_grad"])
            if "o_proj" in key_name or "wo" in key_name:
                attn_list.append(grad_avg_dict[key_name]["vis_avg_grad"] / grad_avg_dict[key_name]["cap_avg_grad"])

        attn_median = np.median(attn_list)
        mlp_median = np.median(mlp_list)


    if distort:
        assert wa_quant, "We only support distort input in weight-activation quantization!!!"
        print("Use distort input...")
        inps_distort = copy.deepcopy(inps)

    gc.collect()
    torch.cuda.empty_cache()

    model.to_cpu()
    # solve layer by layer
    for i in tqdm.tqdm(range(len(layers)), desc="Running QIG..."):
        layer = layers[i]
        layer = layer.cuda()
        named_linears = get_named_linears(layer)

        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        inps = inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input

        for k in layer_kwargs:
            if isinstance(layer_kwargs[k], torch.Tensor):
                layer_kwargs[k] = layer_kwargs[k].to(next(layer.parameters()).device)
            if isinstance(layer_kwargs[k], tuple) or isinstance(layer_kwargs[k], list):
                layer_kwargs[k] = [item.to(next(layer.parameters()).device) if torch.is_tensor(item) else item for item in layer_kwargs[k]]

        inps = layer(inps, **layer_kwargs)[0]
        for h in handles:
            h.remove()
        # now solve for scaling
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        # Clear GPU memory
        torch.cuda.empty_cache()

        if reweight:
            scale_reweight_ratio_dict = {}
            for key, value in grad_avg_dict.items():
                item_list = key.split(".")
                if str(i) in item_list:
                    if "wo" in item_list or "o_proj" in item_list:
                        scale_reweight_ratio_dict["attn"] = max((value["vis_avg_grad"] / value["cap_avg_grad"]), attn_median)
                    elif "w2" in item_list or "down_proj" in item_list:
                        scale_reweight_ratio_dict["mlp"] = max((value["vis_avg_grad"] / value["cap_avg_grad"]), mlp_median)
        else:
            scale_reweight_ratio_dict = {
                "attn": None,
                "mlp": None
            }

        if (
            auto_scale
        ):  # if it applies, we should also modify the input_feat with scales
            if not reweight:
                ans_mask = None
                vis_mask = None
            else:
                ans_mask = caption_mask
                vis_mask = vision_mask
            
            if wa_quant:
                scales_list = auto_scale_block_wa_distort(
                    layer,
                    layer_kwargs,
                    w_bit=w_bit,
                    a_bit=a_bit,
                    q_config=q_config,
                    input_feat=input_feat,
                    ans_mask=ans_mask,
                    vis_mask=vis_mask,
                    reweight_ratio_dict=scale_reweight_ratio_dict,
                    q_input=inps_distort,
                    loss_mode=loss_mode
                )
            else:
                scales_list = auto_scale_block(
                    layer,
                    layer_kwargs,
                    w_bit=w_bit,
                    q_config=q_config,
                    input_feat=input_feat,
                    ans_mask=ans_mask,
                    vis_mask=vis_mask,
                    reweight_ratio_dict=scale_reweight_ratio_dict,
                    loss_mode=loss_mode
                )

            # apply_scale(layer, scales_list, input_feat_dict=input_feat)
            apply_scale(layers[i], scales_list, input_feat_dict=input_feat)

            if distort:
                # get distort output as next layer's input
                if wa_quant:
                    layer_q = copy.deepcopy(layer)
                    layer_q = layer_q.cuda()
                    named_linears_q = get_named_linears(layer_q)
                    for n, m in named_linears_q.items():
                        new_linear = WALinear.from_float(m, weight_quant="per_channel", act_quant="per_token", w_bit=w_bit, a_bit=a_bit)
                        father_module = get_module_by_name_suffix(layer_q, '.'.join(n.split(".")[:-1]))
                        setattr(father_module, n.split('.')[-1], new_linear)
                        del new_linear, m
                        torch.cuda.empty_cache()
                    
                    inps_distort = inps_distort.to(next(layer_q.parameters()).device)  # in case multi-gpu
                    inps_distort = layer_q(inps_distort, **layer_kwargs)[0]
                    del layer_q

            # append prefix to make names global
            qig_results["scale"] += append_str_prefix(
                scales_list, get_op_name(model.model, layer) + "."
            )

        # Clear GPU memory
        torch.cuda.empty_cache()

        layer = layer.cpu()
        # Haotian: check activation replacement
        del input_feat
        gc.collect()
        torch.cuda.empty_cache()

    return qig_results


def apply_qig(model, qig_results):
    apply_scale(model, qig_results["scale"])