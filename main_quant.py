import argparse
import datetime
import importlib
import json
import os
import sys
import traceback
import warnings
from functools import partial

import numpy as np
import yaml

warnings.simplefilter("ignore", category=DeprecationWarning)

from typing import Union

from lmms_eval.models import get_model

from qmllm.quantization.quant_wrapper import qwrapper
from qmllm.models import get_process_model
from qmllm.calibration.pileval import get_calib_dataset
from qmllm.calibration.coco_vl import get_multimodal_calib_dataset
from qmllm.calibration.ocr_vl import get_ocr_parquet_calib_dataset
from qmllm.calibration.mix_vl import get_mix_vl_calib_dataset


def parse_quant_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default="", help="Path to a yaml file specifying all eval arguments, will ignore cli arguments if specified")
    parser.add_argument("--model", default="hf", help="Name of model e.g. `hf`")
    parser.add_argument(
        "--model_args",
        default="",
        help="String arguments for model, e.g. `pretrained=EleutherAI/pythia-160m,dtype=float32`",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=str,
        default=1,
        metavar="auto|auto:N|N",
        help="Acceptable values are 'auto', 'auto:N' or N, where N is an integer. Default 1.",
    )
    parser.add_argument(
        "--micro_batch_size",
        "--mbs",
        type=int,
        default=128,
        help="Micro batch size for models that support it.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g. cuda, cuda:0, cpu)",
    )
    # calibration parameters
    parser.add_argument("--calib_data", default="pileval", choices=["pileval", "coco", "ocr_parquet", "mix_vl", None])
    parser.add_argument("--n_samples", default=128, type=int)
    parser.add_argument("--data_path", default="", type=str)
    parser.add_argument("--image_folder", default="", type=str)
    parser.add_argument("--interleave_format", action="store_true")
    parser.add_argument("--few_shot_format", action="store_true")
    parser.add_argument("--text_data_path", default="", type=str)
    
    # mix_vl params
    parser.add_argument("--coco_data_path", default="", type=str)
    parser.add_argument("--coco_image_folder", default="", type=str)
    parser.add_argument("--ocr_data_path", default="", type=str)
    parser.add_argument("--n_coco", default=0, type=int)
    parser.add_argument("--n_ocr", default=0, type=int)
    parser.add_argument("--shuffle_merged", action="store_true")
    parser.add_argument("--seed_merged", default=42, type=int)


    # TODO: quantization parameters
    parser.add_argument("--method", default="awq", choices=["awq", "smoothquant", "mbq", "qig","rtn", "gptq", "omniquant", None])
    parser.add_argument("--w_bit", default=8, type=int)
    parser.add_argument("--a_bit", default=16, type=int)
    parser.add_argument("--w_group", default=128, type=int)
    parser.add_argument("--alpha", default=0.5, type=int)
    parser.add_argument("--reweight", action="store_true")
    parser.add_argument("--distort", action="store_true")
    parser.add_argument("--loss_mode", default="mae", choices=["mae", "mse"])
    parser.add_argument("--scale_path", default=None, type=str)
    parser.add_argument("--run_process", action="store_true")
    parser.add_argument("--pseudo_quant", action="store_true")
    
    args = parser.parse_args()
    return args


def cli_quant(args: Union[argparse.Namespace, None] = None) -> None:
    if not args:
        args = parse_quant_args()

    args_list = []
    if args.config:
        if not os.path.exists(args.config):
            raise ValueError(f"Config file does not exist: {args.config}")

        with open(args.config, "r") as file:
            config_args = yaml.safe_load(file)
        config_args = [config_args] if type(config_args) != list else config_args
        # multiple configs, create args list first
        for config in config_args:
            args_copy = argparse.Namespace(**vars(args))
            for key, value in config.items():
                setattr(args_copy, key, value)
            args_list.append(args_copy)
    else:
        args_list.append(args)

    for args in args_list:
        cli_quant_single(args)


def cli_quant_single(args: Union[argparse.Namespace, None] = None) -> None:
    # here we load MLLMs outside of the evaluator.
    if args.model_args is None:
        args.model_args = ""
    
    ModelClass = get_model(args.model)
    lm = ModelClass.create_from_arg_string(
        args.model_args,
        {
            "batch_size": args.batch_size,
            "device": args.device,
        },
    )

    # Preprocess the MLLM here, use "lm._model" to get the fp16 mllm.
    Process_ModelClass = get_process_model(args.model)
    process_model = Process_ModelClass(lm._model, 
                                       lm._tokenizer,
                                       lm.processor if hasattr(lm, 'processor') else None)

    # Generate the calibration tokens.
    prompt_inputs = None
    prompt_kwargs = None

    if args.calib_data == "pileval":
        prompt_inputs, prompt_kwargs = get_calib_dataset(data_path=args.data_path, tokenizer=lm._tokenizer, n_samples=args.n_samples)
    elif args.calib_data == "coco":
        prompt_inputs, prompt_kwargs = get_multimodal_calib_dataset(data_path=args.data_path,
                                                                    image_folder=args.image_folder,
                                                                    model=process_model,
                                                                    n_samples=args.n_samples,
                                                                    few_shot_format=args.few_shot_format,
                                                                    interleave_format=args.interleave_format,
                                                                    text_data_path=args.text_data_path,
                                                                    micro_bs=args.micro_batch_size)
    elif args.calib_data == "ocr_parquet":
        prompt_inputs, prompt_kwargs = get_ocr_parquet_calib_dataset(
            data_path=args.data_path,
            image_folder=args.image_folder,
            model=process_model,
            n_samples=args.n_samples,
            few_shot_format=args.few_shot_format,
            interleave_format=args.interleave_format,
            text_data_path=args.text_data_path,
            micro_bs=2,
            answers_take_first=False,
        )
        
    elif args.calib_data == "mix_vl":
        prompt_inputs, prompt_kwargs = get_mix_vl_calib_dataset(
            coco_data_path=args.coco_data_path,
            coco_image_folder=args.coco_image_folder,
            n_coco=args.n_coco,

            ocr_data_path=args.ocr_data_path,
            n_ocr=args.n_ocr,

            model=process_model,
            few_shot_format=args.few_shot_format,
            interleave_format=args.interleave_format,
            text_data_path=args.text_data_path,

            shuffle_each=True,
            shuffle_merged=args.shuffle_merged,
            seed_merged=args.seed_merged,

            coco_micro_bs=args.micro_batch_size,
            ocr_micro_bs=2,
        )

    # Wrapper the quantized model.
    qwrapper(process_model, prompt_inputs, prompt_kwargs, args)

    
if __name__ == "__main__":
    cli_quant()