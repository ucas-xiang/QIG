import os
import torch
from accelerate import Accelerator
import torch, gc
from qmllm.methods.gptq.quantize.pre_quant import run_gptq
from lmms_eval.models import get_model
from qmllm.models import get_process_model

accelerator = Accelerator()


def gptq_entry(model, 
               prompt_inputs, 
               prompt_kwargs, 
               pseudo_quant: bool,
               w_bit: int=4,
               q_group_size: int=128,
               zero_point: str=True,
               w_clip: bool=False,
               percdamp: float=0.01,
               act_order: bool=False,
               w_static_groups: bool=False,
               model_args: str=None,
               model_type: str=None
):
    if pseudo_quant:
        if torch.cuda.device_count() > 1:
            model.to_cpu()
            # accelerator = Accelerator()
            # model = accelerator.unwrap_model(model.model)
            # model.to("cpu")
            # del model
            # torch.cuda.empty_cache(); gc.collect()
            # ModelClass = get_model(model_type)
            # lm = ModelClass.create_from_arg_string(
            #     model_args,
            #     {
            #         "device": "cpu",
            #     },
            # )
            # Process_ModelClass = get_process_model(model_type)
            # model = Process_ModelClass(lm._model, 
            #                             lm._tokenizer, 
            #                             lm.processor if hasattr(lm, 'processor') else None)
        run_gptq(model=model, 
                prompt_inputs=prompt_inputs,
                prompt_kwargs=prompt_kwargs,
                w_bit=w_bit,
                w_asym=not zero_point,
                w_clip=w_clip,
                w_groupsize=q_group_size,
                percdamp=percdamp,
                act_order=act_order,
                w_static_groups=w_static_groups,)
    
    model.to_cuda()
    
    
    # 查看量化前和量化后的模型差异
    

    return model

