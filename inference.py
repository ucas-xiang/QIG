import argparse
import datetime
import importlib
import json
import os
import re
import sys
import traceback
import warnings
from functools import partial
from PIL import Image
import torch
import numpy as np
import yaml

warnings.simplefilter("ignore", category=DeprecationWarning)

from typing import Union, List, Dict, Any

from lmms_eval.models import get_model

from qmllm.quantization.quant_wrapper import qwrapper
from qmllm.models import get_process_model
from qmllm.calibration.pileval import get_calib_dataset
from qmllm.calibration.coco_vl import get_multimodal_calib_dataset

try:
    from llava.constants import DEFAULT_IMAGE_TOKEN
except Exception:
    DEFAULT_IMAGE_TOKEN = "<image>"

def parse_quant_infer_args() -> argparse.Namespace:
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
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g. cuda, cuda:0, cpu)",
    )
    # calibration parameters
    parser.add_argument("--calib_data", default="pileval", choices=["pileval", "coco", None])
    parser.add_argument("--n_samples", default=128, type=int)
    parser.add_argument("--data_path", default="", type=str)
    parser.add_argument("--image_folder", default="", type=str)
    parser.add_argument("--interleave_format", action="store_true")
    parser.add_argument("--few_shot_format", action="store_true")
    parser.add_argument("--text_data_path", default="", type=str)

    # TODO: quantization parameters
    parser.add_argument("--method", default="awq", choices=["awq", "smoothquant", "mbq", "qig", "rtn", None])
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
    
    ## inference parameters
    parser.add_argument("--infer_pairs", default=None, type=str,
                        help="Path to a JSON or JSONL file. Each item: "
                             '{"images": ["a.jpg", "b.png"] or "a.jpg", "question": "...", "id": "optional"}')
    parser.add_argument("--save_path", default=None, type=str,
                        help="Where to save inference outputs as a JSON list.")
    parser.add_argument("--max_new_tokens", default=256, type=int)
    parser.add_argument("--temperature", default=0.2, type=float)
    parser.add_argument("--do_sample", action="store_true")

    
    args = parser.parse_args()
    return args


def cli_quant(args: Union[argparse.Namespace, None] = None) -> None:
    if not args:
        args = parse_quant_infer_args()

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
            # "use_flash_attention_2": False,
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
                                                                    text_data_path=args.text_data_path)

    # Wrapper the quantized model.
    qwrapper(process_model, prompt_inputs, prompt_kwargs, args)
    
    
    if args.infer_pairs and args.save_path:
        quant_meta = {
            "method": args.method,
            "w_bit": args.w_bit,
            "a_bit": args.a_bit,
            "w_group": args.w_group,
            "alpha": args.alpha,
            "reweight": bool(args.reweight),
            "distort": bool(args.distort),
            "loss_mode": args.loss_mode,
            "scale_path": args.scale_path,
            "pseudo_quant": bool(args.pseudo_quant),
        }

        run_inference(
            arch=args.model,
            process_model=process_model, 
            lm_model=lm._model,
            tokenizer=lm._tokenizer,
            infer_pairs=args.infer_pairs,
            save_path=args.save_path,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample,
            model_args_str=args.model_args,
            quant_meta=quant_meta,
            device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        )

#################### inferece ######################
def load_pairs(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"--infer_pairs not found: {path}")

    def normalize_item(it, idx):
        if "images" not in it and "image" in it:
            it["images"] = it.pop("image")
        imgs = it.get("images", [])
        if isinstance(imgs, (str, os.PathLike)):
            imgs = [str(imgs)]
        elif imgs is None:
            imgs = []
        elif isinstance(imgs, tuple):
            imgs = list(imgs)
        it["images"] = imgs

        q = it.get("question", "")
        if not isinstance(q, str):
            raise ValueError(f"[pairs[{idx}]] 'question' must be str, got {type(q)}")
        q = re.sub(r"\s*<image>\s*", " ", q, flags=re.IGNORECASE).strip()
        it["question"] = q

        it.setdefault("id", idx)
        return it

    items = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    items.append(normalize_item(json.loads(line), i))
        return items

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_items = data if isinstance(data, list) else data.get("data", [])
    if not isinstance(raw_items, list):
        raise TypeError(f"Unsupported JSON root type: {type(data)}")

    return [normalize_item(obj, i) for i, obj in enumerate(raw_items)]


def ensure_list_images(x):
    if x is None: return []
    if isinstance(x, (list, tuple)): return list(x)
    return [x]

def open_pils(paths: List[str]):
    imgs = []
    for p in paths:
        with Image.open(p) as im:
            imgs.append(im.convert("RGB"))
    return imgs

def pick_language_model(top_model):
    m = top_model
    base = getattr(m, "model", None) or getattr(m, "_model", None) or m
    for name in ["language_model", "llm", "lm", "transformer"]:
        llm = getattr(base, name, None)
        if llm is not None and hasattr(llm, "generate"):
            return llm
    if hasattr(m, "generate"):
        return m
    raise RuntimeError("Cannot locate underlying language model (ForCausalLM) for generate().")


@torch.inference_mode()
def safe_generate_with_embeds(top_model, tokenizer, inputs_embeds, attention_mask, **gen_kwargs):
    try:
        return top_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **gen_kwargs
        )
    except Exception:
        pass

    llm = pick_language_model(top_model)
    gconf = getattr(top_model, "generation_config", None)
    if gconf is not None:
        gen_kwargs.setdefault("eos_token_id", gconf.eos_token_id)
        gen_kwargs.setdefault("pad_token_id", getattr(gconf, "pad_token_id", None) or tokenizer.eos_token_id)
        if getattr(gconf, "bos_token_id", None) is not None:
            gen_kwargs.setdefault("bos_token_id", gconf.bos_token_id)
        if getattr(gconf, "do_sample", None) is not None:
            gen_kwargs.setdefault("do_sample", gconf.do_sample)
        if getattr(gconf, "temperature", None) is not None:
            gen_kwargs.setdefault("temperature", gconf.temperature)

    if hasattr(llm, "to"):
        llm.to(inputs_embeds.device)
    llm.eval()

    return llm.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        **gen_kwargs
    )


# =========================
# llava
# =========================
@torch.inference_mode()
def infer_llava_onevision(
    proc_model,
    raw_model,
    tokenizer,
    device: str,
    pairs: List[Dict[str, Any]],
    max_new_tokens=256,
    temperature=0.2,
    do_sample=False,
) -> List[Dict[str, Any]]:
    raw_model.eval()
    if hasattr(raw_model, "to"):
        raw_model.to(device)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    results = []

    for i, item in enumerate(pairs):
        q = item.get("question", "")
        img_paths = ensure_list_images(item.get("images"))
        pils = open_pils(img_paths) if img_paths else None
        n_img = len(pils) if pils else 0

        if n_img > 0:
            img_tokens = (DEFAULT_IMAGE_TOKEN + "\n") * n_img
            human_value = f"{img_tokens}{q}".strip()
        else:
            human_value = q

        data_item = {
            "id": item.get("id", i),
            "conversations": [
                {"from": "human", "value": human_value},
                {"from": "gpt",   "value": ""}
            ],
            "image": img_paths if pils else None
        }

        data_dict = proc_model.preprocess_data(pils, data_item)
        batch = proc_model.data_collator([data_dict])

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        param_dtype = next(raw_model.parameters()).dtype
        images = [im.to(device=device, dtype=param_dtype) for im in batch["images"]] if "images" in batch else None
        image_sizes = batch.get("image_sizes", None)
        modalities = batch.get("modalities", None)


        gen_ids = raw_model.generate(
            inputs=input_ids,
            images=images,
            image_sizes=image_sizes,
            modalities=modalities,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            temperature=temperature,
            do_sample=do_sample,
            use_cache=True,
            eos_token_id=getattr(raw_model.generation_config, "eos_token_id", tokenizer.eos_token_id),
            pad_token_id=pad_id,
        )

        full_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

        if q and full_text.startswith(q):
            answer = full_text[len(q):].strip()
        else:
            prompt_len = int(input_ids[0].ne(pad_id).sum().item())
            tail_ids = gen_ids[0][prompt_len:] if gen_ids.shape[1] > prompt_len else gen_ids[0]
            answer = tokenizer.decode(tail_ids, skip_special_tokens=True).strip()

        results.append({
            "id": data_item["id"],
            "question": q,
            "images": img_paths,
            "answer": answer
        })

    return results

# =========================
# InternVL2
# =========================
@torch.inference_mode()
def infer_internvl2(
    proc_model,
    raw_model,
    tokenizer,
    device: str,
    pairs: List[Dict[str, Any]],
    max_new_tokens=256,
    temperature=0.2,
    do_sample=False,
) -> List[Dict[str, Any]]:
    raw_model.eval()
    if hasattr(raw_model, "to"):
        raw_model.to(device)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    results = []

    for i, item in enumerate(pairs):
        q = item.get("question", "")
        img_paths = ensure_list_images(item.get("images"))
        pils = open_pils(img_paths) if img_paths else None

        data_item = {
            "id": item.get("id", i),
            "conversations": [
                {"from": "human", "value": q},
                {"from": "gpt",   "value": ""}
            ],
            "image": img_paths if pils else None
        }

        data_dict = proc_model.preprocess_data(pils, data_item)
        batch = proc_model.data_collator([data_dict])
        prompt_inps, kw = proc_model.generate_input(batch)

        inputs_embeds = prompt_inps["inputs_embeds"].to(device)
        attention_mask = kw["attention_mask"].to(device)

        gen_ids = safe_generate_with_embeds(
            top_model=raw_model,
            tokenizer=tokenizer,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            temperature=temperature,
            do_sample=do_sample,
            use_cache=True,
            eos_token_id=getattr(raw_model.generation_config, "eos_token_id", tokenizer.eos_token_id),
            pad_token_id=pad_id,
        )

        prompt_len = int(attention_mask[0].sum().item())
        tail_ids   = gen_ids[0][prompt_len:] if gen_ids.shape[1] > prompt_len else gen_ids[0]
        ans        = tokenizer.decode(tail_ids, skip_special_tokens=True).strip()

        results.append({
            "id": data_item["id"],
            "question": q,
            "images": img_paths,
            "answer": ans
        })

    return results

# =========================
# Qwen2-VL
# =========================
@torch.inference_mode()
def infer_qwen2_vl(
    proc_model, raw_model, tokenizer, device: str, pairs, 
    max_new_tokens=256, temperature=0.2, do_sample=False
):
    raw_model.eval()
    if hasattr(raw_model, "to"):
        raw_model.to(device)

    processor = proc_model.processor
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    results = []

    for i, item in enumerate(pairs):
        q = item.get("question", "")
        img_paths = ensure_list_images(item.get("images"))
        pils = open_pils(img_paths) if img_paths else None

        user_content = []
        if pils:
            for _ in pils:
                user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": q})
        chat_item = [{"role": "user", "content": user_content}]

        prompt_text = processor.apply_chat_template(
            chat_item, tokenize=False, add_generation_prompt=True
        )

        data_dict = processor(
            text=prompt_text, images=pils, videos=None,
            padding=True, return_tensors="pt",
        )

        for k, v in data_dict.items():
            if isinstance(v, torch.Tensor) and v.dim() > 1 and v.size(0) == 1:
                data_dict[k] = v.squeeze(0)

        samples = {
            "input_ids":        data_dict["input_ids"].unsqueeze(0),
            "attention_mask":   data_dict["attention_mask"].unsqueeze(0),
            "labels":           torch.full_like(data_dict["input_ids"].unsqueeze(0), -100),
            "pixel_values":     data_dict["pixel_values"],            # [B_total, C, H, W]
            "image_grid_thw":   data_dict["image_grid_thw"].unsqueeze(0),
        }

        prompt_inps, prompt_kwargs = proc_model.generate_input(samples)
        inputs_embeds  = prompt_inps["inputs_embeds"].to(device)
        attention_mask = prompt_kwargs["attention_mask"].to(device)

        gen_ids = safe_generate_with_embeds(
            top_model=raw_model,
            tokenizer=tokenizer,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            temperature=temperature,
            do_sample=do_sample,
            use_cache=True,
            eos_token_id=getattr(raw_model.generation_config, "eos_token_id", tokenizer.eos_token_id),
            pad_token_id=pad_id,
        )

        prompt_len = int(attention_mask[0].sum().item())
        ans_ids = gen_ids[0][prompt_len:] if gen_ids.shape[1] > prompt_len else gen_ids[0]
        ans = tokenizer.decode(ans_ids, skip_special_tokens=True).strip()

        if ans.lower().startswith("assistant"):
            ans = ans[len("assistant"):].lstrip(":： \n\t")

        results.append({
            "id": item.get("id", i),
            "question": q,
            "images": img_paths,
            "answer": ans
        })
    return results


def run_inference(
    arch: str,               # "llava_onevision" | "internvl2" | "qwen2_vl"
    process_model,
    lm_model,
    tokenizer,
    infer_pairs: str,        # input data（.json/.jsonl）
    save_path: str,          # save path
    max_new_tokens=256,
    temperature=0.2,
    do_sample=False,
    model_args_str: str = "",
    quant_meta: Dict[str, Any] = None,
    device: str = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pairs = load_pairs(infer_pairs)

    if arch.lower() == "llava_onevision":
        outputs = infer_llava_onevision(
            proc_model=process_model,
            raw_model=lm_model,
            tokenizer=tokenizer,
            device=device,
            pairs=pairs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample
        )
    elif arch.lower() == "internvl2":
        outputs = infer_internvl2(
            proc_model=process_model,
            raw_model=lm_model,
            tokenizer=tokenizer,
            device=device,
            pairs=pairs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample
        )
    elif arch.lower() == "qwen2_vl":
        outputs = infer_qwen2_vl(
            proc_model=process_model,
            raw_model=lm_model,
            tokenizer=tokenizer,
            device=device,
            pairs=pairs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample
        )
    else:
        raise ValueError(f"Unknown arch: {arch}")

    export = {
        "meta": {
            "timestamp": datetime.datetime.now().isoformat(),
            "arch": arch,
            "model_args": model_args_str,
            "quant": quant_meta or {},
        },
        "results": outputs
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved {len(outputs)} results to: {save_path}")



if __name__ == "__main__":
    model = cli_quant()
    