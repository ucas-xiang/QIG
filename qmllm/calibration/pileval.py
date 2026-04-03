import torch
from datasets import load_dataset

def get_calib_dataset(data_path="", tokenizer=None, n_samples=512, block_size=512):
    if not data_path:
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    else:
        dataset = load_dataset(data_path, split="validation")
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")

    samples = [cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)]
    samples = torch.cat(samples, dim=0)

    prompt_inputs = {
        "input_ids": samples
    }

    prompt_kawrgs = {
        "labels": samples,
        "attention_mask": None,
        "vision_mask": None,
        "caption_mask": None,
    }

    return prompt_inputs, prompt_kawrgs