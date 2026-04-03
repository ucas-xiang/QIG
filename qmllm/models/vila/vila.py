import re
import math
import copy
import torch
import transformers

from PIL import Image, ImageFile
from accelerate import dispatch_model
from accelerate.hooks import remove_hook_from_submodules
from typing import Dict, Optional, Sequence, List
from torch.utils.data import ConcatDataset, Dataset, default_collate

from qmllm.models.vila import conversation as conversation_lib
from .tokenizer import preprocess_conversation, tokenizer_image_token
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX

from qmllm.models.base import BaseModel
from qmllm.utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register("vila")
class VILA(BaseModel):
    def __init__(self, model, tokenizer, processor=None):
        self.model = model
        self.tokenizer = tokenizer

        self.num_params = sum(p.numel() for p in self.model.parameters())
        self.device_map = self.init_device_map()
        self.vision_tower = model.get_vision_tower()
        self.image_aspect_ratio = 'resize'
        self.image_processor = self.vision_tower.image_processor
        self.is_multimodal = True
        self.mm_use_im_start_end = model.config.mm_use_im_start_end
        self.image_grid_pinpoints = getattr(model.config, 'image_grid_pinpoints', None)
        self.image_crop_resolution = getattr(model.config, 'image_crop_resolution', None)
        self.image_split_resolution = getattr(model.config, 'image_split_resolution', None)

        # conversation_lib.default_conversation = conversation_lib.conv_templates[self.conv_templates]

    def fetch_vit(self):
        return self.model.model.vision_tower

    def fetch_llm(self):
        return self.model.model

    def fetch_proj(self):
        return self.model.model.mm_projector

    def vision_preprocess(self, image: torch.FloatTensor):
        image = image.convert("RGB")
        image_aspect_ratio = self.image_aspect_ratio
        if image_aspect_ratio == "resize":
            if hasattr(self.image_processor, "crop_size"):
                # CLIP vision tower
                crop_size = self.image_processor.crop_size
            else:
                # SIGLIP vision tower
                assert hasattr(self.image_processor, "size")
                crop_size = self.image_processor.size
            image = image.resize((crop_size["height"], crop_size["width"]))     
        if image_aspect_ratio == "pad":

            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(image, tuple(int(x * 255) for x in self.image_processor.image_mean))
            image = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        else:
            # Using default behavior of the vision encoder
            # For CLIP, default is central crop
            # For Radio, default is central crop
            # For Siglip, default is resize
            # For InternVIT, default is resize
            image = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        return image

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
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        
        outputs = self.model.llm(
            input_ids=input_ids.to(next(self.model.parameters()).device) if input_ids is not None else None,
            inputs_embeds=inputs_embeds.to(next(self.model.parameters()).device) if inputs_embeds is not None else None,
            attention_mask=attention_mask.to(next(self.model.parameters()).device) if attention_mask is not None else None,
            labels=labels.to(next(self.model.parameters()).device),
            use_cache=use_cache,
            return_dict=return_dict,   
        )

        return outputs


    def init_device_map(self):
        device_dict = {}
        device_dict["vision_tower"] = 0
        device_dict["mm_projector"] = 0
        device_dict["llm.model.embed_tokens"] = 0
        device_dict["llm.model.norm"] = 0
        device_dict["llm.model.rotary_emb"] = 0
        device_dict["llm.lm_head"] = 0
        for name, param in self.model.named_parameters():
            if "llm.model.layers" in name:
                layer_name = ".".join(name.split(".")[0:4])
                layer_device = int(str(param.device).split(":")[-1])
                device_dict[layer_name] = layer_device

        return device_dict


    def to_cuda(self):
        if self.num_params > 20 * 10 ** 9: # 20B model
            self.model = dispatch_model(self.model, device_map=self.device_map)
        else:
            self.model = self.model.cuda()

    def to_cpu(self):
        if self.num_params > 20 * 10 ** 9: # 20B model
            remove_hook_from_submodules(self.model)
        self.model = self.model.cpu()


    def preprocess_multimodal(self, sources: Sequence[str]) -> Dict:
        is_multimodal = self.is_multimodal
        if not is_multimodal:
            return sources

        for source in sources:
            concat_values = "".join([sentence["value"] for sentence in source])
            for sid, sentence in enumerate(source):
                # In multimodal conversations, we automatically prepend '<image>' at the start of the first sentence if it doesn't already contain one.
                if sid == 0 and DEFAULT_IMAGE_TOKEN not in concat_values:
                    sentence["value"] = f"{DEFAULT_IMAGE_TOKEN}\n" + sentence["value"]
                if DEFAULT_IMAGE_TOKEN in sentence["value"]:
                    sentence_chunks = [chunk.strip() for chunk in sentence["value"].split(DEFAULT_IMAGE_TOKEN)]
                    sentence_chunks = [
                        chunk + " " if not (chunk.endswith("\n")) else chunk for chunk in sentence_chunks[:-1]
                    ] + [sentence_chunks[-1]]
                    sentence["value"] = f"{DEFAULT_IMAGE_TOKEN}\n".join(sentence_chunks).strip()

                    replace_token = DEFAULT_IMAGE_TOKEN
                    if "mmtag" in conversation_lib.default_conversation.version:
                        replace_token = "<Image>" + replace_token + "</Image>"
                    if self.mm_use_im_start_end:
                        replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                    sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
                # ensure every DEFAULT_IMAGE_TOKEN is followed by a newline character.
                # If it has one already, we don't add another one.
                if DEFAULT_IMAGE_TOKEN in sentence["value"]:
                    sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, f"{DEFAULT_IMAGE_TOKEN}\n")
                    sentence["value"] = sentence["value"].replace(f"{DEFAULT_IMAGE_TOKEN}\n\n", f"{DEFAULT_IMAGE_TOKEN}\n")

        return sources

    def preprocess_plain(
        self,
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
    ) -> Dict:
        # add end signal and concatenate together
        conversations = []
        for source in sources:
            assert len(source) == 2
            assert DEFAULT_IMAGE_TOKEN in source[0]["value"]
            source[0]["value"] = DEFAULT_IMAGE_TOKEN
            conversation = source[0]["value"] + source[1]["value"] + conversation_lib.default_conversation.sep
            conversations.append(conversation)
        # tokenize conversations
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
        targets = copy.deepcopy(input_ids)
        for target, source in zip(targets, sources):
            tokenized_len = len(tokenizer_image_token(source[0]["value"], tokenizer))
            target[:tokenized_len] = IGNORE_INDEX

        return dict(input_ids=input_ids, labels=targets)

    def preprocess(
        self,
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False,
        no_system_prompt: bool = False,
    ) -> Dict:
        if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
            return self.preprocess_plain(sources, tokenizer)
        return default_collate(
            [
                preprocess_conversation(conversation, tokenizer, no_system_prompt=no_system_prompt)
                for conversation in sources
            ]
        )


    def preprocess_data(self, images, data_item):
        sources = [data_item]
        if images is not None:
            if len(images) == 1:
                image = self.vision_preprocess(images[0])
            else:
                image = [self.vision_preprocess(f) for f in images]
                image_tensor = torch.stack(image)

            sources = self.preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]))

        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        has_image = "image" in data_item
        data_dict = self.preprocess(sources, self.tokenizer, has_image=has_image)
        # data_dict = preprocess_pure_pair(sources, self.tokenizer, has_image=has_image)

        data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        if images is not None:
            # image exist in the data
            if len(images) == 1:
                data_dict["image"] = image.unsqueeze(0).half()
            else:
                data_dict["image"] = image_tensor.half()
        else:
            # image does not exist in the data, but the model is multimodal
            data_dict["image"] = None
        
        if "id" in data_item:
            data_dict["id"] = data_item["id"]

        return data_dict

    @torch.no_grad() 
    def few_shot_data_samples(self, data_samples, pad_side="right", interleave_freq=2):
        input_ids = data_samples["input_ids"]
        labels = data_samples["labels"]
        attention_mask = data_samples["attention_mask"]
        images = data_samples["images"]
        # sample_id = data_samples["sample_id"]

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
                    
        # process image_sizes, modalities, images, sample_id
        # according to the org code of llava, we find we don't need to modify pixel_values and image_flags
        # new_sample_id = []
        
        # for i in range(0, len(sample_id) - interleave_freq + 1, interleave_freq):
        #     cur_sample_id = sample_id[i: i+interleave_freq]
        #     new_sample_id.append([item for item in cur_sample_id])

        new_data_samples = {}
        new_data_samples["input_ids"] = new_input_ids_padded
        new_data_samples["labels"] = new_labels_padded
        new_data_samples["attention_mask"] = new_attention_mask
        new_data_samples["images"] = images
        # new_data_samples["sample_id"] = new_sample_id

        return new_data_samples


    @torch.no_grad() 
    def interleave_data_samples(self, data_samples, pure_text=None, pad_side="right", interleave_freq=2):
    
        input_ids = data_samples["input_ids"]
        labels = data_samples["labels"]
        attention_mask = data_samples["attention_mask"]
        images = data_samples["images"]
        # sample_id = data_samples["sample_id"]

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
                    
        # process image_sizes, modalities, images, sample_id
        # according to the org code of llava, we find we don't need to modify pixel_values and image_flags
        # new_sample_id = []
        
        # for i in range(0, len(sample_id) - interleave_freq + 1, interleave_freq):
        #     cur_sample_id = sample_id[i: i+interleave_freq]
        #     new_sample_id.append([item for item in cur_sample_id])

        new_data_samples = {}
        new_data_samples["input_ids"] = new_input_ids_padded
        new_data_samples["labels"] = new_labels_padded
        new_data_samples["attention_mask"] = new_attention_mask
        new_data_samples["images"] = images
        # new_data_samples["sample_id"] = new_sample_id

        return new_data_samples
    
    
    @torch.no_grad()
    def generate_input(self, data_samples):
        data_samples['input_ids'] = data_samples['input_ids'].cuda()
        data_samples['attention_mask'] = data_samples['attention_mask'].cuda()
        data_samples['labels'] = data_samples['labels'].cuda() 
        for i, _ in enumerate(data_samples['images']):
            data_samples['images'][i] = data_samples['images'][i].to(self.model.dtype)
    
        (   input_ids, 
            position_ids, 
            attention_mask, 
            past_key_values, 
            input_embeds, 
            labels  
        ) = self.model.prepare_inputs_labels_for_multimodal(
                    data_samples['input_ids'],
                    None,
                    data_samples['attention_mask'],
                    None,
                    data_samples['labels'],
                    data_samples['images'],
            )
        
        # image_embeds = []
        # image_labels = []
        # image_attn_mask = []
        # caption_embeds = []
        # caption_labels = []
        # caption_attn_mask = []

        vision_sel = []

        for batch_idx, pre_input_ids in enumerate(data_samples['input_ids']):
            num_images = (pre_input_ids == IMAGE_TOKEN_INDEX).sum()
        
            # remove the padding using attention mask
            pre_labels = data_samples['labels'][batch_idx]
            pre_attn_mask = data_samples['attention_mask'][batch_idx]        
            pre_labels_rm_pad = pre_labels[pre_attn_mask]
            pre_len = pre_labels_rm_pad.shape[0]

            post_labels = labels[batch_idx]
            post_attn_mask = attention_mask[batch_idx]
            post_labels_rm_pad = post_labels[post_attn_mask]
            post_len = post_labels_rm_pad.shape[0]

            image_emb_len = int((post_len - pre_len + num_images) / num_images)
            image_emb_start = torch.where(pre_input_ids == IMAGE_TOKEN_INDEX)[0]
            for idx, _ in enumerate(image_emb_start):
                image_emb_start[idx] = image_emb_start[idx] + (image_emb_len - 1) * idx

            image_emb_end = image_emb_start + image_emb_len

            cur_vision_sel = torch.zeros(post_attn_mask.shape[0], dtype=torch.bool)
            
            for jdx in range(num_images):
                cur_im_emb_start = image_emb_start[jdx]
                cur_im_emb_end = image_emb_end[jdx]

                cur_vision_sel[cur_im_emb_start: cur_im_emb_end] = True

            # cur_image_embeds = input_embeds[batch_idx][cur_vision_sel]
            # cur_image_labels = labels[batch_idx][cur_vision_sel]
            # cur_image_attn_mask = attention_mask[batch_idx][cur_vision_sel]

            # cur_caption_embeds = input_embeds[batch_idx][~cur_vision_sel]
            # cur_caption_labels = labels[batch_idx][~cur_vision_sel]
            # cur_caption_attn_mask = attention_mask[batch_idx][~cur_vision_sel]

            # image_embeds.append(cur_image_embeds)
            # image_labels.append(cur_image_labels)
            # image_attn_mask.append(cur_image_attn_mask)
            # caption_embeds.append(cur_caption_embeds)
            # caption_labels.append(cur_caption_labels)
            # caption_attn_mask.append(cur_caption_attn_mask)
            vision_sel.append(cur_vision_sel)

        # image_embeds = torch.stack(image_embeds)
        # image_labels = torch.stack(image_labels)
        # image_attn_mask = torch.stack(image_attn_mask)
        # caption_embeds = torch.stack(caption_embeds)
        # caption_labels = torch.stack(caption_labels)
        # caption_attn_mask = torch.stack(caption_attn_mask)

        vision_sel = torch.stack(vision_sel)
            
        vision_mask = vision_sel
        answer_mask = (labels != -100) # ignore token id
        
        prompt_inputs = {
            "inputs_embeds": input_embeds
        }

        prompt_kwargs = {
            "labels": labels,
            "attention_mask": attention_mask,
            "vision_mask": vision_mask,
            "caption_mask": answer_mask,
        }

        return prompt_inputs, prompt_kwargs
    

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids


    def data_collator(self, instances):
        input_ids, labels, images = [], [], []
        for instance in instances:
            if not isinstance(instance["input_ids"], list):
                input_ids.append(instance["input_ids"])
            else:
                input_ids += instance["input_ids"]
            if not isinstance(instance["labels"], list):
                labels.append(instance["labels"])
            else:
                labels += instance["labels"]
            # Note (kentang-mit@: we do not directly push tensors to
            # images, but list of tensors.
            if instance["image"] is not None:
                cur_image = instance["image"]
                assert len(cur_image.shape) == 4
                # n_images, 3, size, size
                if not isinstance(instance["input_ids"], list):
                    # datasets other than coyo, not packing >1 samples together
                    images.append(cur_image)
                else:
                    # coyo-like datasets
                    images.extend(cur_image.chunk(cur_image.size(0), 0))
            else:
                images.append([])
        # kentang-mit@: we need to make sure these two lists have
        # the same length. We will use input_ids to filter out images corresponding
        # to truncated <image> tokens later.
        for _images, _input_ids in zip(images, input_ids):
            assert (
                len(_images) == (_input_ids == IMAGE_TOKEN_INDEX).sum().item()
            ), f"Number mismatch between images and placeholder image tokens in 'len(_images) == (_input_ids == IMAGE_TOKEN_INDEX).sum().item()'.\
                Expect to have {len(_images)} images but only found {(_input_ids == IMAGE_TOKEN_INDEX).sum().item()} images in tokens. \
                Error input_ids: {_input_ids} {self.tokenizer.decode([x if x != -200 else 200 for x in _input_ids])}"

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        new_images = []
        # kentang-mit@: it is possible that some <image> tokens get removed
        # after truncation. It is important to also remove corresponding images.
        # otherwise, text and image will mismatch in the model.
        for ix in range(len(input_ids)):
            num_images = (input_ids[ix] == IMAGE_TOKEN_INDEX).sum().item()
            cur_images = images[ix]
            cur_images = cur_images[:num_images]
            if len(cur_images) > 0:
                new_images.append(cur_images)
        if len(new_images) > 0:
            batch["images"] = torch.cat(new_images, dim=0)
        else:
            # the entire batch is text-only
            if hasattr(self.data_args.image_processor, "crop_size"):
                crop_size = self.data_args.image_processor.crop_size
            else:
                crop_size = self.data_args.image_processor.size
            # we still need 1 dummy image for the vision tower
            batch["images"] = torch.zeros(1, 3, crop_size["height"], crop_size["width"])

        return batch