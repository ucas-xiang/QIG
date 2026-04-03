import os
import torch
from qmllm.methods.rtn.quantizer import pseudo_quantize_model_weight, pseudo_quantize_model_weight_act

def rtn_entry(model, pseudo_quant: bool, wa_quant: bool, zero_point: str=True, q_group_size: int=128, w_bit: int=4, a_bit: int=16):
    q_config = {
        "zero_point": zero_point,  # by default True
        "q_group_size": q_group_size,  # whether to use group quantization
    }

    if pseudo_quant:
        if not wa_quant:
            # weight quantization
            pseudo_quantize_model_weight(model.model, w_bit=w_bit, q_config=q_config)
        else:
            # weight activation quantization
            pseudo_quantize_model_weight_act(model.model, w_bit=w_bit, a_bit=a_bit)

    model.to_cuda()

    return model