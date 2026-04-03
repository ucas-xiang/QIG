import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
from datasets import load_dataset


def _ensure_pil_rgb(img):
    """
    Convert various image formats to PIL.Image in RGB.
    Supports:
      - PIL.Image.Image
      - dict: {"bytes": ..., "path": ...}  (datasets Image feature)
      - bytes / bytearray
      - str path
    """
    if img is None:
        return None

    if isinstance(img, Image.Image):
        return img.convert("RGB")

    if isinstance(img, dict):
        b = img.get("bytes", None)
        p = img.get("path", None)
        if b is not None:
            return Image.open(BytesIO(b)).convert("RGB")
        if p is not None and isinstance(p, str) and len(p) > 0:
            return Image.open(p).convert("RGB")
        return None

    if isinstance(img, (bytes, bytearray)):
        return Image.open(BytesIO(img)).convert("RGB")

    if isinstance(img, str) and len(img) > 0:
        return Image.open(img).convert("RGB")

    return None


def _answers_to_str(answers, joiner=" ", take_first=False):
    """
    answers: list[str] or str or other
    - take_first=True: use first non-empty
    - else: join all non-empty with joiner
    """
    if answers is None:
        return ""

    if isinstance(answers, str):
        return answers.strip()

    if isinstance(answers, (list, tuple)):
        parts = [str(a).strip() for a in answers if a is not None and str(a).strip() != ""]
        if not parts:
            return ""
        return parts[0] if take_first else joiner.join(parts)

    return str(answers).strip()


def _build_llava_style_item(question, answer_str, idx, image_ref):
    """
    Convert OCR-QA sample into LLaVA-style json item.
    IMPORTANT for qwen2_vl:
      - Must include key "image" (even if not used to load image)
      - image_ref can be PIL.Image or a dummy string path
    """
    return {
        "id": f"ocr_parquet_{idx:09d}",
        "image": image_ref,  # qwen2_vl's convert_data_item will read.
        "conversations": [
            {"from": "human", "value": f"{question}\n<image>"},
            {"from": "gpt", "value": answer_str},
        ],
    }


def get_ocr_parquet_calib_dataset(
    data_path,
    image_folder,  # kept for API compatibility; unused for parquet-with-PIL case
    model,
    n_samples=128,
    few_shot_format=False,
    interleave_format=False,
    text_data_path=None,
    shuffle=True,
    micro_bs=8,
    answers_take_first=False,
    answers_joiner=" | ",
):
    # ------------- load parquet dataset -------------
    if not data_path.endswith(".parquet"):
        raise ValueError(f"OCR parquet loader only supports .parquet, got: {data_path}")

    dataset = load_dataset("parquet", data_files=data_path, split="train")
    if shuffle:
        dataset = dataset.shuffle(seed=444)

    # ------------- sanity check -------------
    if few_shot_format and interleave_format:
        raise ValueError("You cannot specify both few_shot_format and interleave_format at the same time!")

    # ------------- build pure_text once (if needed) -------------
    pure_text = None
    if interleave_format:
        if not text_data_path:
            text_ds = load_dataset("mit-han-lab/pile-val-backup", split="validation")
        else:
            text_ds = load_dataset(text_data_path, split="validation")

        if shuffle:
            text_ds = text_ds.shuffle(seed=444)

        samples = []
        n_run = 0
        for item in text_ds:
            line = item["text"].strip()
            ids = model.tokenizer.encode(line)
            if len(ids) > 512:
                samples.append(torch.tensor(ids[:512], dtype=torch.long))
                n_run += 1
            if n_run == 128:
                break
        pure_text = samples

    # ------------- helpers: pad to target length -------------
    def pad_3d_to_len(x: torch.Tensor, target_len: int, pad_value: float = 0.0):
        cur = x.size(1)
        if cur == target_len:
            return x
        if cur > target_len:
            return x[:, :target_len, :]
        pad = target_len - cur
        return F.pad(x, (0, 0, 0, pad), value=pad_value)

    def pad_2d_to_len(x: torch.Tensor, target_len: int, pad_value):
        cur = x.size(1)
        if cur == target_len:
            return x
        if cur > target_len:
            return x[:, :target_len]
        pad = target_len - cur
        if x.dtype == torch.bool:
            x_u8 = x.to(torch.uint8)
            x_u8 = F.pad(x_u8, (0, pad), value=0)
            return x_u8.to(torch.bool)
        return F.pad(x, (0, pad), value=pad_value)

    # ------------- accumulate (pad-aware) -------------
    prompt_inputs_all = None
    prompt_kwargs_all = None
    global_max_len = None

    for st in range(0, n_samples, micro_bs):
        micro_data_list = []
        ed = min(st + micro_bs, n_samples)

        for i in range(st, ed):
            idx = i % len(dataset)
            raw = dataset[idx]

            question = raw.get("question", "")
            answers = raw.get("answers", None)
            answer_str = _answers_to_str(
                answers,
                joiner=answers_joiner,
                take_first=answers_take_first
            )

            # image is stored in parquet as actual data
            pil = _ensure_pil_rgb(raw.get("image", None))
            images = [pil] if pil is not None else None

            # For qwen2_vl: data_item, "image" is required
            image_ref = pil if pil is not None else "dummy.jpg"

            data_item = _build_llava_style_item(question, answer_str, idx, image_ref)

            data_dict = model.preprocess_data(images, data_item)
            micro_data_list.append(data_dict)

        examples = model.data_collator(micro_data_list)

        if few_shot_format:
            examples = model.few_shot_data_samples(examples)

        if interleave_format:
            examples = model.interleave_data_samples(examples, pure_text=pure_text)

        prompt_inputs, prompt_kwargs = model.generate_input(examples)

        cur_embeds = prompt_inputs["inputs_embeds"].detach()
        cur_labels = prompt_kwargs["labels"].detach()
        cur_attn   = prompt_kwargs["attention_mask"].detach()
        cur_vmask  = prompt_kwargs["vision_mask"].detach()
        cur_cmask  = prompt_kwargs["caption_mask"].detach()

        cur_len = cur_embeds.size(1)

        if prompt_inputs_all is None:
            global_max_len = cur_len
            prompt_inputs_all = {"inputs_embeds": cur_embeds}
            prompt_kwargs_all = {
                "labels": cur_labels,
                "attention_mask": cur_attn,
                "vision_mask": cur_vmask,
                "caption_mask": cur_cmask,
            }
        else:
            target_len = max(global_max_len, cur_len)

            if global_max_len != target_len:
                prompt_inputs_all["inputs_embeds"] = pad_3d_to_len(prompt_inputs_all["inputs_embeds"], target_len, pad_value=0.0)
                prompt_kwargs_all["labels"]        = pad_2d_to_len(prompt_kwargs_all["labels"], target_len, pad_value=-100)
                prompt_kwargs_all["attention_mask"]= pad_2d_to_len(prompt_kwargs_all["attention_mask"], target_len, pad_value=False)
                prompt_kwargs_all["vision_mask"]   = pad_2d_to_len(prompt_kwargs_all["vision_mask"], target_len, pad_value=False)
                prompt_kwargs_all["caption_mask"]  = pad_2d_to_len(prompt_kwargs_all["caption_mask"], target_len, pad_value=False)
                global_max_len = target_len

            if cur_len != global_max_len:
                cur_embeds = pad_3d_to_len(cur_embeds, global_max_len, pad_value=0.0)
                cur_labels = pad_2d_to_len(cur_labels, global_max_len, pad_value=-100)
                cur_attn   = pad_2d_to_len(cur_attn,   global_max_len, pad_value=False)
                cur_vmask  = pad_2d_to_len(cur_vmask,  global_max_len, pad_value=False)
                cur_cmask  = pad_2d_to_len(cur_cmask,  global_max_len, pad_value=False)

            prompt_inputs_all["inputs_embeds"] = torch.cat([prompt_inputs_all["inputs_embeds"], cur_embeds], dim=0)
            prompt_kwargs_all["labels"]        = torch.cat([prompt_kwargs_all["labels"], cur_labels], dim=0)
            prompt_kwargs_all["attention_mask"]= torch.cat([prompt_kwargs_all["attention_mask"], cur_attn], dim=0)
            prompt_kwargs_all["vision_mask"]   = torch.cat([prompt_kwargs_all["vision_mask"], cur_vmask], dim=0)
            prompt_kwargs_all["caption_mask"]  = torch.cat([prompt_kwargs_all["caption_mask"], cur_cmask], dim=0)

        del micro_data_list, examples, prompt_inputs, prompt_kwargs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return prompt_inputs_all, prompt_kwargs_all
