import os, json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from datasets import load_dataset

def load_image(image_path):
    # Load the image using use PIL, we don't support tcs_loader
    return Image.open(image_path).convert('RGB')


def get_multimodal_calib_dataset(
    data_path,
    image_folder,
    model,
    n_samples=128,
    few_shot_format=False,
    interleave_format=False,
    text_data_path=None,
    shuffle=True,
    micro_bs=16,
):
    # ------------- load dataset -------------
    if data_path.endswith(".jsonl"):
        dataset = []
        with open(data_path, "r") as json_file:
            for line in json_file:
                dataset.append(json.loads(line.strip()))
    elif data_path.endswith(".json"):
        with open(data_path, "r") as json_file:
            dataset = json.load(json_file)
    else:
        raise ValueError(f"Unsupported file type: {data_path}")

    if shuffle:
        rng = np.random.default_rng(seed=42)
        rng.shuffle(dataset)

    # ------------- sanity check -------------
    if few_shot_format and interleave_format:
        raise ValueError("You cannot specify both few_shot_format and interleave_format at the same time!")

    # ------------- build pure_text once (if needed) -------------
    pure_text = None
    if interleave_format:
        from datasets import load_dataset
        if not text_data_path:
            text_ds = load_dataset("mit-han-lab/pile-val-backup", split="validation")
        else:
            text_ds = load_dataset(text_data_path, split="validation")

        if shuffle:
            text_ds = text_ds.shuffle(seed=42)

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
        # x: [B, N, C]
        cur = x.size(1)
        if cur == target_len:
            return x
        if cur > target_len:
            return x[:, :target_len, :]
        pad = target_len - cur
        return F.pad(x, (0, 0, 0, pad), value=pad_value)

    def pad_2d_to_len(x: torch.Tensor, target_len: int, pad_value):
        # x: [B, N]
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
            data_item = dataset[idx]

            # load images
            if "image" in data_item and data_item["image"] and len(data_item["image"]) != 0:
                images = []
                if isinstance(data_item["image"], list):
                    for image_path in data_item["image"]:
                        full_image_path = os.path.join(image_folder, image_path)
                        images.append(load_image(full_image_path))
                else:
                    full_image_path = os.path.join(image_folder, data_item["image"])
                    images.append(load_image(full_image_path))
            else:
                images = None

            data_dict = model.preprocess_data(images, data_item)
            micro_data_list.append(data_dict)

        examples = model.data_collator(micro_data_list)

        if few_shot_format:
            examples = model.few_shot_data_samples(examples)

        if interleave_format:
            examples = model.interleave_data_samples(examples, pure_text=pure_text)

        # generate inputs on GPU (per your model.generate_input), then move to CPU for accumulation
        prompt_inputs, prompt_kwargs = model.generate_input(examples)

        cur_embeds = prompt_inputs["inputs_embeds"].detach()             # [B, N, C]
        cur_labels = prompt_kwargs["labels"].detach()                    # [B, N]
        cur_attn   = prompt_kwargs["attention_mask"].detach()            # [B, N]
        cur_vmask  = prompt_kwargs["vision_mask"].detach()               # [B, N]
        cur_cmask  = prompt_kwargs["caption_mask"].detach()              # [B, N]

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

            # pad existing accumulated tensors if needed
            if global_max_len != target_len:
                prompt_inputs_all["inputs_embeds"] = pad_3d_to_len(prompt_inputs_all["inputs_embeds"], target_len, pad_value=0.0)
                prompt_kwargs_all["labels"]        = pad_2d_to_len(prompt_kwargs_all["labels"], target_len, pad_value=-100)
                prompt_kwargs_all["attention_mask"]= pad_2d_to_len(prompt_kwargs_all["attention_mask"], target_len, pad_value=False)
                prompt_kwargs_all["vision_mask"]   = pad_2d_to_len(prompt_kwargs_all["vision_mask"], target_len, pad_value=False)
                prompt_kwargs_all["caption_mask"]  = pad_2d_to_len(prompt_kwargs_all["caption_mask"], target_len, pad_value=False)
                global_max_len = target_len

            # pad current tensors if needed
            if cur_len != global_max_len:
                cur_embeds = pad_3d_to_len(cur_embeds, global_max_len, pad_value=0.0)
                cur_labels = pad_2d_to_len(cur_labels, global_max_len, pad_value=-100)
                cur_attn   = pad_2d_to_len(cur_attn,   global_max_len, pad_value=False)
                cur_vmask  = pad_2d_to_len(cur_vmask,  global_max_len, pad_value=False)
                cur_cmask  = pad_2d_to_len(cur_cmask,  global_max_len, pad_value=False)

            # now cat along batch dim
            prompt_inputs_all["inputs_embeds"] = torch.cat([prompt_inputs_all["inputs_embeds"], cur_embeds], dim=0)
            prompt_kwargs_all["labels"]        = torch.cat([prompt_kwargs_all["labels"], cur_labels], dim=0)
            prompt_kwargs_all["attention_mask"]= torch.cat([prompt_kwargs_all["attention_mask"], cur_attn], dim=0)
            prompt_kwargs_all["vision_mask"]   = torch.cat([prompt_kwargs_all["vision_mask"], cur_vmask], dim=0)
            prompt_kwargs_all["caption_mask"]  = torch.cat([prompt_kwargs_all["caption_mask"], cur_cmask], dim=0)

        # cleanup
        del micro_data_list, examples, prompt_inputs, prompt_kwargs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return prompt_inputs_all, prompt_kwargs_all

