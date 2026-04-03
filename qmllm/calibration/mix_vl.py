import torch
import torch.nn.functional as F

from qmllm.calibration.coco_vl import get_multimodal_calib_dataset
from qmllm.calibration.ocr_vl import get_ocr_parquet_calib_dataset


def _pad_3d_to_len(x: torch.Tensor, target_len: int, pad_value: float = 0.0):
    """
    x: [B, N, C]
    """
    cur = x.size(1)
    if cur == target_len:
        return x
    if cur > target_len:
        return x[:, :target_len, :]
    pad = target_len - cur
    return F.pad(x, (0, 0, 0, pad), value=pad_value)


def _pad_2d_to_len(x: torch.Tensor, target_len: int, pad_value):
    """
    x: [B, N]
    """
    cur = x.size(1)
    if cur == target_len:
        return x
    if cur > target_len:
        return x[:, :target_len]
    pad = target_len - cur

    # bool mask needs special handling
    if x.dtype == torch.bool:
        x_u8 = x.to(torch.uint8)
        x_u8 = F.pad(x_u8, (0, pad), value=0)
        return x_u8.to(torch.bool)

    return F.pad(x, (0, pad), value=pad_value)


def _merge_prompt_tensors(
    prompt_inputs_a,
    prompt_kwargs_a,
    prompt_inputs_b,
    prompt_kwargs_b,
    shuffle: bool = True,
    seed: int = 42,
):
    """
    Merge two (prompt_inputs, prompt_kwargs) with pad-aware alignment.
    Expect both to contain:
      prompt_inputs: {"inputs_embeds": [B, N, C]}
      prompt_kwargs: {
        "labels": [B, N],
        "attention_mask": [B, N] (bool),
        "vision_mask": [B, N] (bool),
        "caption_mask": [B, N] (bool),
      }
    """
    if prompt_inputs_a is None:
        return prompt_inputs_b, prompt_kwargs_b
    if prompt_inputs_b is None:
        return prompt_inputs_a, prompt_kwargs_a

    a_emb = prompt_inputs_a["inputs_embeds"]
    b_emb = prompt_inputs_b["inputs_embeds"]

    # Determine max seq len
    max_len = max(a_emb.size(1), b_emb.size(1))

    # Pad embeds
    if a_emb.size(1) != max_len:
        prompt_inputs_a["inputs_embeds"] = _pad_3d_to_len(a_emb, max_len, pad_value=0.0)
    if b_emb.size(1) != max_len:
        prompt_inputs_b["inputs_embeds"] = _pad_3d_to_len(b_emb, max_len, pad_value=0.0)

    # Pad kwargs
    def pad_kwargs_inplace(pk):
        pk["labels"] = _pad_2d_to_len(pk["labels"], max_len, pad_value=-100)
        pk["attention_mask"] = _pad_2d_to_len(pk["attention_mask"], max_len, pad_value=False)
        pk["vision_mask"] = _pad_2d_to_len(pk["vision_mask"], max_len, pad_value=False)
        pk["caption_mask"] = _pad_2d_to_len(pk["caption_mask"], max_len, pad_value=False)

    pad_kwargs_inplace(prompt_kwargs_a)
    pad_kwargs_inplace(prompt_kwargs_b)

    # Cat on batch dim
    out_inputs = {
        "inputs_embeds": torch.cat(
            [prompt_inputs_a["inputs_embeds"], prompt_inputs_b["inputs_embeds"]],
            dim=0,
        )
    }
    out_kwargs = {
        "labels": torch.cat([prompt_kwargs_a["labels"], prompt_kwargs_b["labels"]], dim=0),
        "attention_mask": torch.cat([prompt_kwargs_a["attention_mask"], prompt_kwargs_b["attention_mask"]], dim=0),
        "vision_mask": torch.cat([prompt_kwargs_a["vision_mask"], prompt_kwargs_b["vision_mask"]], dim=0),
        "caption_mask": torch.cat([prompt_kwargs_a["caption_mask"], prompt_kwargs_b["caption_mask"]], dim=0),
    }

    # Optional shuffle on batch dim
    if shuffle:
        g = torch.Generator(device=out_inputs["inputs_embeds"].device)
        g.manual_seed(seed)
        perm = torch.randperm(out_inputs["inputs_embeds"].size(0), generator=g, device=out_inputs["inputs_embeds"].device)

        out_inputs["inputs_embeds"] = out_inputs["inputs_embeds"][perm]
        out_kwargs["labels"] = out_kwargs["labels"][perm]
        out_kwargs["attention_mask"] = out_kwargs["attention_mask"][perm]
        out_kwargs["vision_mask"] = out_kwargs["vision_mask"][perm]
        out_kwargs["caption_mask"] = out_kwargs["caption_mask"][perm]

    return out_inputs, out_kwargs


def get_mix_vl_calib_dataset(
    *,
    # coco
    coco_data_path: str,
    coco_image_folder: str,
    n_coco: int,
    coco_micro_bs: int = 16,

    # ocr parquet
    ocr_data_path: str,
    ocr_image_folder: str = "",  # kept for API compatibility
    n_ocr: int = 0,
    ocr_micro_bs: int = 2,
    answers_take_first: bool = False,
    answers_joiner: str = " | ",

    # shared
    model=None,
    few_shot_format: bool = False,
    interleave_format: bool = False,
    text_data_path: str = None,
    shuffle_each: bool = True,     # shuffle inside each loader
    shuffle_merged: bool = True,   # shuffle after merging
    seed_merged: int = 42,
    empty_cache: bool = True,
):
    """
    Build a mixed calibration dataset by sampling n_coco from coco(json/jsonl)
    and n_ocr from ocr(parquet), then merging their prompt tensors.

    Return:
      prompt_inputs_all, prompt_kwargs_all
    """
    if model is None:
        raise ValueError("model must be provided (process_model).")

    # 1) COCO part
    coco_inputs, coco_kwargs = (None, None)
    if n_coco and n_coco > 0:
        coco_inputs, coco_kwargs = get_multimodal_calib_dataset(
            data_path=coco_data_path,
            image_folder=coco_image_folder,
            model=model,
            n_samples=n_coco,
            few_shot_format=few_shot_format,
            interleave_format=interleave_format,
            text_data_path=text_data_path,
            shuffle=shuffle_each,
            micro_bs=coco_micro_bs,
        )
        if empty_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 2) OCR part
    ocr_inputs, ocr_kwargs = (None, None)
    if n_ocr and n_ocr > 0:
        ocr_inputs, ocr_kwargs = get_ocr_parquet_calib_dataset(
            data_path=ocr_data_path,
            image_folder=ocr_image_folder,
            model=model,
            n_samples=n_ocr,
            few_shot_format=few_shot_format,
            interleave_format=interleave_format,
            text_data_path=text_data_path,
            shuffle=shuffle_each,
            micro_bs=ocr_micro_bs,
            answers_take_first=answers_take_first,
            answers_joiner=answers_joiner,
        )
        if empty_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 3) Merge (pad-aware)
    mixed_inputs, mixed_kwargs = _merge_prompt_tensors(
        coco_inputs, coco_kwargs,
        ocr_inputs, ocr_kwargs,
        shuffle=shuffle_merged,
        seed=seed_merged,
    )
    return mixed_inputs, mixed_kwargs
