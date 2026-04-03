import os

from qmllm.methods.awq.entry import awq_entry
from qmllm.methods.smoothquant.entry import smoothquant_entry
from qmllm.methods.mbq.entry import mbq_entry
from qmllm.methods.qig.entry import qig_entry
from qmllm.methods.rtn.entry import rtn_entry
from qmllm.methods.gptq.entry import gptq_entry

def qwrapper(model, prompt_inputs, prompt_kwargs, args):
    if args.method == "awq":
        model = awq_entry(model, prompt_inputs, prompt_kwargs, run_awq_process=args.run_process, pseudo_quant=args.pseudo_quant, scale_path=args.scale_path, q_group_size=args.w_group, w_bit=args.w_bit)
    elif args.method == "smoothquant":
        model = smoothquant_entry(model, prompt_inputs, prompt_kwargs, run_sq_process=args.run_process, pseudo_quant=args.pseudo_quant, scale_path=args.scale_path, w_bit=args.w_bit, a_bit=args.a_bit, alpha=args.alpha)
    elif args.method == "mbq":
        wa_quant = args.w_bit < 16 and args.a_bit < 16
        model = mbq_entry(model, prompt_inputs, prompt_kwargs, 
                                run_mbq_process=args.run_process, 
                                pseudo_quant=args.pseudo_quant, 
                                scale_path=args.scale_path, 
                                q_group_size=args.w_group, 
                                w_bit=args.w_bit, 
                                a_bit=args.a_bit, 
                                wa_quant=wa_quant, 
                                reweight=args.reweight,
                                distort=args.distort,
                                loss_mode=args.loss_mode)
    elif args.method == "qig":
        wa_quant = args.w_bit < 16 and args.a_bit < 16
        model = qig_entry(model, prompt_inputs, prompt_kwargs, 
                                run_qig_process=args.run_process, 
                                pseudo_quant=args.pseudo_quant, 
                                scale_path=args.scale_path, 
                                q_group_size=args.w_group, 
                                w_bit=args.w_bit, 
                                a_bit=args.a_bit, 
                                wa_quant=wa_quant, 
                                reweight=args.reweight,
                                distort=args.distort,
                                loss_mode=args.loss_mode)        
    elif args.method == "rtn":
        wa_quant = args.w_bit < 16 and args.a_bit < 16
        model = rtn_entry(model, pseudo_quant=args.pseudo_quant, wa_quant=wa_quant, q_group_size=args.w_group, w_bit=args.w_bit, a_bit=args.a_bit)
    elif args.method == "gptq":
        import torch
        if torch.cuda.device_count() > 1:
            model = gptq_entry(
                model,
                prompt_inputs,
                prompt_kwargs,
                pseudo_quant=args.pseudo_quant,
                w_bit=args.w_bit,
                q_group_size=args.w_group,
                percdamp=args.percdamp,
                model_args=args.model_args,
                model_type=args.model,
            )
        else:
            model = gptq_entry(
                model,
                prompt_inputs,
                prompt_kwargs,
                pseudo_quant=args.pseudo_quant,
                w_bit=args.w_bit,
                q_group_size=args.w_group,
                percdamp=args.percdamp,
            )
    else:
        raise NotImplementedError

    return model