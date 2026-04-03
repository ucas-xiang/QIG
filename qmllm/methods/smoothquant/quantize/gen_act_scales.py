import copy
import torch
import torch.nn as nn

import functools
from tqdm import tqdm


def process_input(prompt_inputs, prompt_kwargs):
    inputs = {**prompt_inputs, **prompt_kwargs}
    inputs["use_cache"] = False
    vision_mask = inputs.pop("vision_mask", None)
    caption_mask = inputs.pop("caption_mask", None)
    
    return inputs

@torch.no_grad()
def get_act_scales(model, prompt_inputs, prompt_kwargs):
    model.model.eval()
    model.to_cuda()
    device = next(model.model.parameters()).device
    act_scales = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.model.named_modules():
        if isinstance(m, nn.Linear) and "vis" not in name and "mm_projector" not in name and "lm_head" not in name and "output" not in name and "mlp1" not in name:
            print(f"register hook for {name}")
            hooks.append(
                m.register_forward_hook(functools.partial(stat_input_hook, name=name))
            )
    
    num_samples = len(next(iter(prompt_inputs.values())))

    inputs = process_input(prompt_inputs, prompt_kwargs)

    for i in tqdm(range(num_samples)):
        mini_inputs = {}
        for k in inputs:
            if isinstance(inputs[k], torch.Tensor):
                mini_inputs[k] = copy.deepcopy(inputs[k][i:i+1]).to(device)
        
        # remove pad
        if "inputs_embeds" in mini_inputs:
            B, N, C = mini_inputs["inputs_embeds"].shape
            input_emb = mini_inputs["inputs_embeds"].reshape(B * N, C)
            attn_mask = mini_inputs["attention_mask"].reshape(B * N)
            label = mini_inputs["labels"].reshape(B * N)

            input_emb = input_emb[attn_mask.bool()]
            label = label[attn_mask.bool()]
            attn_mask = attn_mask[attn_mask.bool()]

            mini_inputs["inputs_embeds"] = input_emb.reshape(B, -1, C)
            mini_inputs["attention_mask"] = attn_mask.reshape(B, -1)
            mini_inputs["labels"] = label.reshape(B, -1)
            
        outputs = model(**mini_inputs)
        del mini_inputs
        torch.cuda.empty_cache()

    for h in hooks:
        h.remove()

    return act_scales