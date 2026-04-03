import torch

import os
import sys
import math
import json
import random
import traceback
import transformers
import numpy as np
from copy import deepcopy
from torch.nn import CrossEntropyLoss
from accelerate import dispatch_model
from accelerate.hooks import remove_hook_from_submodules
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Dict, Optional
from .conversation import get_conv_template
from .constants import (CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD,
                        IMG_CONTEXT_TOKEN, IMG_END_TOKEN, IMG_START_TOKEN,
                        SIGLIP_MEAN, SIGLIP_STD)
from .dataset import (build_transform, dynamic_preprocess, preprocess, preprocess_internlm, preprocess_mpt, preprocess_phi3)

from PIL import Image, ImageFile, PngImagePlugin, UnidentifiedImageError
from transformers.trainer_pt_utils import LabelSmoother
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

from qmllm.models.base import BaseModel
from qmllm.utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register("internvl2")
class InternVL2(BaseModel):
    def __init__(self, model, tokenizer, processor=None):
        self.model = model
        self.tokenizer = tokenizer

        self.num_params = sum(p.numel() for p in self.model.parameters())
        self.template_name = "internlm2-chat"
        self.num_image_token = self.model.num_image_token
        self.image_size = 448
        self.pad2square = False
        self.dynamic_image_size = True
        self.use_thumbnail = True
        self.min_dynamic_patch = 1
        self.max_dynamic_patch = 1
        self.normalize_type = "imagenet"
        self.group_by_length = True

    def fetch_vit(self):
        return self.model.vision_model

    def fetch_llm(self):
        return self.model.language_model

    def fetch_proj(self):
        return self.model.mlp1

    def vision_preprocess(self, image):
        transform = self.get_transform()
        if len(image) == 1:
            # single image
            num_tiles = []
            img = image[0]
            if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
                images = dynamic_preprocess(img, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                                            image_size=self.image_size, use_thumbnail=self.use_thumbnail)
                num_tiles.append(len(images))
            else:  # Otherwise, use the original image as a single patch
                images = [img]
                num_tiles.append(1)

            # Apply the transformation to each image and stack the results into a tensor
            pixel_values = [transform(i) for i in images]
            pixel_values = torch.stack(pixel_values)
            
            # Ensure that there is only one patch if dynamic image size is not enabled
            num_patches = pixel_values.size(0)
            if not self.dynamic_image_size:
                assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        else:
            # multi images
            images, num_tiles = [], []
            num_image = len(image)
            for i in range(num_image):
                img = image[i]
                if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
                    img = dynamic_preprocess(img, min_num=self.min_dynamic_patch,
                                            max_num=self.max_dynamic_patch // num_image,
                                            image_size=self.image_size, use_thumbnail=self.use_thumbnail)
                    images += img
                    num_tiles.append(len(img))
                else:  # Otherwise, use the original image as a single patch
                    images.append(img)
                    num_tiles.append(1)
            pixel_values = [transform(i) for i in images]
            pixel_values = torch.stack(pixel_values)
            num_patches = pixel_values.size(0)

        return pixel_values, num_patches, num_tiles
    
         
    def language_preprocess(self, text):
        return self.tokenizer(text)

    
    def forward(
            self, 
            input_ids: torch.LongTensor = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        
        if input_ids is not None:
            lm = self.fetch_llm()
            outputs = lm(
                input_ids=input_ids.to(next(lm.parameters()).device),
                attention_mask=attention_mask.to(next(lm.parameters()).device) if attention_mask is not None else None,
                use_cache=use_cache
            )
        if inputs_embeds is not None:
            lm = self.fetch_llm()
            outputs = lm(
                inputs_embeds=inputs_embeds.to(next(lm.parameters()).device),
                attention_mask=attention_mask.to(next(lm.parameters()).device) if attention_mask is not None else None,
                use_cache=use_cache
            )

        logits = outputs.logits

        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, lm.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    
    def split_model(self, num_layers, vit_alpha=0.5):
        device_map = {}
        world_size = torch.cuda.device_count()
        # Since the first GPU will be used for ViT, treat it as half a GPU.
        num_layers_per_gpu = math.ceil(num_layers / (world_size - vit_alpha))
        num_layers_per_gpu = [num_layers_per_gpu] * world_size
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * (1 - vit_alpha))
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f'language_model.model.layers.{layer_cnt}'] = i
                layer_cnt += 1
        device_map['vision_model'] = 0
        device_map['mlp1'] = 0
        device_map['language_model.model.tok_embeddings'] = 0
        device_map['language_model.model.embed_tokens'] = 0
        device_map['language_model.output'] = 0
        device_map['language_model.model.norm'] = 0
        device_map['language_model.lm_head'] = 0
        device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

        return device_map
         

    def to_cuda(self):
        # if self.num_params > 20 * 10 ** 9: # 20B model
        if torch.cuda.device_count() > 1:
            # TODO: ugly comment the previous code for InternVL2-26B eval, remove it!!!
            device_map = self.split_model(self.model.language_model.config.num_hidden_layers)
            self.model = dispatch_model(self.model, device_map=device_map)
            # self.model = self.model.cuda()
        else:
            self.model = self.model.cuda()

    def to_cpu(self):
        if self.num_params > 20 * 10 ** 9: # 20B model
            remove_hook_from_submodules(self.model)
        self.model = self.model.cpu()


    def get_preprocess_function(self):
        # Select the appropriate preprocessing function based on the template name
        if self.template_name == 'Hermes-2':
            preprocess_function = preprocess_mpt
        elif self.template_name == 'internlm2-chat':
            preprocess_function = preprocess_internlm
            # preprocess_function = preprocess_internlm_pure_pair
        elif self.template_name == 'phi3-chat':
            preprocess_function = preprocess_phi3
        else:
            preprocess_function = preprocess
        return preprocess_function


    def get_transform(self):
        # Build transformation function
        transform = build_transform(is_train=False, input_size=self.image_size,
                                    pad2square=self.pad2square, normalize_type=self.normalize_type)
        return transform


    def preprocess_data(self, images, data_item):
        if images is not None:                
            pixel_values, num_patches, num_tiles = self.vision_preprocess(images)

            preprocess_function = self.get_preprocess_function()

            if len(images) == 1:
                # single image

                # Ensure the first conversation contains an image placeholder
                if '<image>' not in data_item['conversations'][0]['value']:
                    data_item['conversations'][0]['value'] = '<image>\n' + data_item['conversations'][0]['value']

                # Preprocess the conversations and generate the return dictionary
                ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                          self.tokenizer, [self.num_image_token * num_patches],
                                          group_by_length=self.group_by_length, ds_name="sharegpt4v")
            else:
                # multi images
                num_image = len(data_item['image'])
                # Preprocess the conversations and generate the return dictionary
                num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]
                ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                          self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
                                          ds_name="sharegpt4v", num_image=num_image)
            
            # Create the final return dictionary
            data_dict = dict(
                input_ids=ret['input_ids'][0],
                labels=ret['labels'][0],
                attention_mask=ret['attention_mask'][0],
                pixel_values=pixel_values,
                image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
            )

        else:
            # pure_text

            # Create a blank white image
            image = Image.new('RGB', (224, 224), (255, 255, 255))

            images = [image]

            pixel_values, num_patches, num_tiles = self.vision_preprocess(images)
            
            # Select the appropriate preprocessing function based on the template name
            preprocess_function = self.get_preprocess_function()

            # Preprocess the conversations and generate the return dictionary
            ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                      self.tokenizer, [self.num_image_token * num_patches], text_only=True,
                                      group_by_length=self.group_by_length, ds_name="sharegpt4v")
            
            # Create the final return dictionary
            data_dict = dict(
                input_ids=ret['input_ids'][0],
                labels=ret['labels'][0],
                attention_mask=ret['attention_mask'][0],
                pixel_values=pixel_values,
                image_flags=torch.tensor([0] * num_patches, dtype=torch.long)
            )

        if "id" in data_item:
            data_dict["sample_id"] = data_item["id"]

        return data_dict
    
    @torch.no_grad() 
    def few_shot_data_samples(self, data_samples, pad_side="right", interleave_freq=2):
        input_ids = data_samples["input_ids"]
        labels = data_samples["labels"]
        attention_mask = data_samples["attention_mask"]
        pixel_values = data_samples["pixel_values"]
        image_flags = data_samples["image_flags"]
        sample_id = data_samples["sample_id"]

        # process input_ids, labels, attention_mask
        new_input_ids = []
        new_labels = []

        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        for i in range(0, len(input_ids) - interleave_freq + 1, interleave_freq):
            cur_input_ids = input_ids[i: i+interleave_freq]
            concat_input_ids = torch.cat(cur_input_ids, dim=0)
            new_input_ids.append(concat_input_ids)

        for i in range(0, len(labels) - interleave_freq + 1, interleave_freq):
            cur_labels = labels[i: i+interleave_freq]
            concat_labels = torch.cat(cur_labels, dim=0)
            new_labels.append(concat_labels)

        # maybe we should add some codes to ensure the sequences we make are not longer than tokenizer max length
        # but now don't care it
         
        max_len = max(x.shape[0] for x in new_input_ids)
        batch_size = len(new_input_ids)
        new_input_ids_padded = torch.zeros((batch_size, max_len), dtype=new_input_ids[0].dtype, device=new_input_ids[0].device)
        new_labels_padded = torch.full((batch_size, max_len), -100, dtype=new_labels[0].dtype, device=new_labels[0].device)
        new_attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)

        for i, (cur_new_input_ids, cur_new_labels) in enumerate(zip(new_input_ids, new_labels)):
            cur_len = cur_new_input_ids.shape[0]
            if pad_side == "left":
                if cur_len > 0:
                    new_input_ids_padded[i, -cur_len:] = cur_new_input_ids
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    new_attention_mask[i, -cur_len:] = True
            else:
                if cur_len > 0:
                    new_input_ids_padded[i, :cur_len] = cur_new_input_ids
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    new_attention_mask[i, :cur_len] = True
                    
        # process pixel_values, image_flags, sample_id
        # according to the org code of internvl, we find we don't need to modify pixel_values and image_flags
        new_sample_id = []
        
        for i in range(0, len(sample_id) - interleave_freq + 1, interleave_freq):
            cur_sample_id = sample_id[i: i+interleave_freq]
            new_sample_id.append([item for item in cur_sample_id])

        new_data_samples = {}
        new_data_samples["input_ids"] = new_input_ids_padded
        new_data_samples["labels"] = new_labels_padded
        new_data_samples["attention_mask"] = new_attention_mask
        new_data_samples["pixel_values"] = pixel_values
        new_data_samples["image_flags"] = image_flags
        new_data_samples["sample_id"] = new_sample_id

        return new_data_samples


    @torch.no_grad() 
    def interleave_data_samples(self, data_samples, pure_text=None, pad_side="right", interleave_freq=2):

        input_ids = data_samples["input_ids"]
        labels = data_samples["labels"]
        attention_mask = data_samples["attention_mask"]
        pixel_values = data_samples["pixel_values"]
        image_flags = data_samples["image_flags"]
        sample_id = data_samples["sample_id"]

        # process input_ids, labels, attention_mask
        new_input_ids = []
        new_labels = []

        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        for i in range(0, len(input_ids) - interleave_freq + 1, interleave_freq):
            cur_input_ids = [input_ids[i]]

            for j in range(interleave_freq-1):
                cur_input_ids.append(pure_text[i+j])
                cur_input_ids.append(input_ids[i+1+j])

            concat_input_ids = torch.cat(cur_input_ids, dim=0)
            new_input_ids.append(concat_input_ids)

        for i in range(0, len(labels) - interleave_freq + 1, interleave_freq):
            cur_labels = [labels[i]]

            for j in range(interleave_freq-1):
                cur_labels.append(pure_text[i+j])
                cur_labels.append(labels[i+1+j])

            concat_labels = torch.cat(cur_labels, dim=0)
            new_labels.append(concat_labels)

        # maybe we should add some codes to ensure the sequences we make are not longer than tokenizer max length
        # but now don't care it
         
        max_len = max(x.shape[0] for x in new_input_ids)
        batch_size = len(new_input_ids)
        new_input_ids_padded = torch.zeros((batch_size, max_len), dtype=new_input_ids[0].dtype, device=new_input_ids[0].device)
        new_labels_padded = torch.full((batch_size, max_len), -100, dtype=new_labels[0].dtype, device=new_labels[0].device)
        new_attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)

        for i, (cur_new_input_ids, cur_new_labels) in enumerate(zip(new_input_ids, new_labels)):
            cur_len = cur_new_input_ids.shape[0]
            if pad_side == "left":
                if cur_len > 0:
                    new_input_ids_padded[i, -cur_len:] = cur_new_input_ids
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    new_attention_mask[i, -cur_len:] = True
            else:
                if cur_len > 0:
                    new_input_ids_padded[i, :cur_len] = cur_new_input_ids
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    new_attention_mask[i, :cur_len] = True
                    
        # process pixel_values, image_flags, sample_id
        # according to the org code of internvl, we find we don't need to modify pixel_values and image_flags
        new_sample_id = []
        
        for i in range(0, len(sample_id) - interleave_freq + 1, interleave_freq):
            cur_sample_id = sample_id[i: i+interleave_freq]
            new_sample_id.append([item for item in cur_sample_id])

        new_data_samples = {}
        new_data_samples["input_ids"] = new_input_ids_padded
        new_data_samples["labels"] = new_labels_padded
        new_data_samples["attention_mask"] = new_attention_mask
        new_data_samples["pixel_values"] = pixel_values
        new_data_samples["image_flags"] = image_flags
        new_data_samples["sample_id"] = new_sample_id

        return new_data_samples

    @torch.no_grad()   
    def generate_input(self, data_samples):
        input_ids = data_samples['input_ids'].cuda()
        attention_mask = data_samples['attention_mask'].cuda()
        labels = data_samples['labels'].cuda() 
        pixel_values = data_samples['pixel_values'].to(self.model.dtype).cuda()
        image_flags = data_samples['image_flags'].cuda() 
        
        # generate input embeddings
        # copied from the InternVLChatModel.forward
        
        img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.model.language_model.get_input_embeddings()(input_ids)

        vit_embeds = self.model.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.model.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        vision_mask = selected.reshape(B, N)
        answer_mask = (labels != -100) # ignore token id
        


        prompt_inputs = {
            "inputs_embeds": input_embeds
        }

        prompt_kwargs = {
            "labels": labels,
            "attention_mask": attention_mask,
            "vision_mask": vision_mask,
            "caption_mask": answer_mask
        }

        return prompt_inputs, prompt_kwargs
        

    def data_collator(self, instances):
        pad_id = 0
        IGNORE_INDEX = -100
        first = instances[0]
        batch = {}

        batch_lens = [feat['input_ids'].shape for feat in instances]
        max_item_length = max(batch_lens)[0]
        for idx in range(len(instances)):
            feat = instances[idx]
            temp_input_ids = torch.LongTensor([pad_id] * max_item_length)
            temp_input_ids[:feat['input_ids'].shape[0]] = feat['input_ids']
            feat['input_ids'] = temp_input_ids
            temp_labels = torch.LongTensor([IGNORE_INDEX] * max_item_length)
            temp_labels[:feat['labels'].shape[0]] = feat['labels']
            feat['labels'] = temp_labels
            feat['attention_mask'] = feat['input_ids'].ne(pad_id)

        # Special handling for labels.
        # Ensure that tensor is created with the correct type
        # (it should be automatically the case, but let's make sure of it.)
        if 'label' in first and first['label'] is not None:
            label = first['label'].item() if isinstance(first['label'], torch.Tensor) else first['label']
            dtype = torch.long if isinstance(label, int) else torch.float
            batch['labels'] = torch.tensor([f['label'] for f in instances], dtype=dtype)
        elif 'label_ids' in first and first['label_ids'] is not None:
            if isinstance(first['label_ids'], torch.Tensor):
                batch['labels'] = torch.stack([f['label_ids'] for f in instances])
            else:
                dtype = torch.long if isinstance(first['label_ids'][0], int) else torch.float
                batch['labels'] = torch.tensor([f['label_ids'] for f in instances], dtype=dtype)

        # Handling of all other possible keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        for k, v in first.items():
            if k not in ('label', 'label_ids', 'pixel_values', 'image_flags') and \
                    v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in instances])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in instances]))
                else:
                    batch[k] = torch.tensor([f[k] for f in instances])
            if k in ('pixel_values', 'image_flags'):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.concat([f[k] for f in instances])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.concat(np.stack([f[k] for f in instances]))
                else:
                    batch[k] = torch.concat([f[k] for f in instances])
            if k in ('sample_id'):
                batch[k] = [f[k] for f in instances]
        return batch

