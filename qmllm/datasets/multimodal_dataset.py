import os
import json
import copy
import torch
import transformers
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from typing import  Dict, Optional, Sequence, List
from torch.utils.data import Dataset, DataLoader

# Import LLaVA modules
try:
    from llava.train.train import LazySupervisedDataset as LLaVA_LazySupervisedDataset
    from llava.train.train import DataCollatorForSupervisedDataset as LLaVA_DataCollatorForSupervisedDataset
except ImportError as e:
    print(f"LLaVA is not installed. Please install LLaVA to use this model.\nError: {e}")

# Import InternVL modules
try:
    from internvl.patch import concat_pad_data_collator
    from internvl.train.internvl_chat_finetune import build_datasets
except ImportError as e:
    print(f"internvl-chat is not installed. Please install internvl-chat to use this model.\nError: {e}")

def build_llava_dataloader(n_samples, tokenizer, shuffle, data_args):
    train_dataset = LLaVA_LazySupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)
    data_collator = LLaVA_DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    dataloader = DataLoader(dataset=train_dataset, batch_size=n_samples, shuffle=shuffle, collate_fn=data_collator)

    return dataloader

def build_internvl_dataloader(n_samples, model, tokenizer, shuffle, data_args):
    train_dataset = build_datasets(
        data_args=data_args, 
        tokenizer=tokenizer,
        tcs_loader=None,
        model=model,
        group_by_length=True,
        dynamic_image_size=data_args.dynamic_image_size,
        use_thumbnail=data_args.use_thumbnail,
        min_dynamic_patch=data_args.min_dynamic_patch,
        max_dynamic_patch=data_args.max_dynamic_patch,
        normalize_type=data_args.normalize_type
    )
    data_collator = concat_pad_data_collator
    dataloader = DataLoader(dataset=train_dataset, batch_size=n_samples, shuffle=shuffle, collate_fn=data_collator)

    return dataloader

