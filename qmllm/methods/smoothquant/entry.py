import os
import torch

from qmllm.methods.smoothquant.quantize.smooth import smooth_lm, smooth_vit
from qmllm.methods.smoothquant.quantize.quantizer import quantize_model, pseudo_quantize_model_weight_act
from qmllm.methods.smoothquant.quantize.gen_act_scales import get_act_scales

def smoothquant_entry(model, prompt_inputs, prompt_kwargs, run_sq_process: bool, pseudo_quant: bool, scale_path: str=None, w_bit: int=4, a_bit: int=8, alpha: float=0.5):
    '''
    model: here the model is the LLM, you have to extract the LLM first! 
    prompt_tokens: the prompt tokens
    prompt_mask: the prompt mask, mask the answer language tokens
    '''

    assert scale_path is not None

    if run_sq_process:
        act_scales = get_act_scales(model, prompt_inputs, prompt_kwargs)
        dirpath = os.path.dirname(scale_path)
        os.makedirs(dirpath, exist_ok=True)
        
        torch.save(act_scales, scale_path)
        print("SmoothQuant results saved at", scale_path)
    else:
        act_scales = torch.load(scale_path)
    
    if pseudo_quant:
        smooth_lm(model.model, act_scales, alpha)
        pseudo_quantize_model_weight_act(model.model, w_bit=w_bit, a_bit=a_bit)
    
    return model
