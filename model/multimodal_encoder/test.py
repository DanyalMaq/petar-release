import sys
sys.path.append("/mym3d/")

import os
import logging
from typing import Optional, List, Dict
import numpy as np
import torch
from torch.utils.data import DataLoader
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM
from dataclasses import dataclass, field
from LaMed.src.dataset.multi_dataset import UniDatasets, CapDataset, TextDatasets, VQADataset, MyCapDataset
from LaMed.src.model.language_model import LamedLlamaForCausalLM, LamedPhi3ForCausalLM
from LaMed.src.train.lamed_trainer import LaMedTrainer

from .builder import build_vision_tower

@dataclass
class DataArguments:
    data_root: str = field(default="/mym3d/Data/data/", metadata={"help": "Root directory for all data."})

    # caption data
    cap_data_path: str = field(default="/mym3d/Data/output.json", metadata={"help": "Path to caption data."})
    seg_enable: bool = field(default=False)
    per_device_train_batch_size: int = field(default=2)
    proj_out_num: int = field(default=256) 
    max_length: int = field(default=256)

@dataclass
class ModelArguments:
    version: Optional[str] = field(default="v0")
    model_name_or_path: Optional[str] = field(default="microsoft/Phi-3-mini-4k-instruct", metadata={"help": "Path to the LLM or MLLM."})
    model_type: Optional[str] = field(default=None, metadata={"help": "llama2, phi3"})

    freeze_backbone: bool = field(default=False)
    pretrain_mllm: Optional[str] = field(default=None)

    tune_mm_mlp_adapter: bool = field(default=False, metadata={"help": "Used in pretrain: tune mm_projector and embed_tokens"})
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None, metadata={"help": "Path to pretrained mm_projector and embed_tokens."})

    # image
    image_channel: int = field(default=1)
    image_size: tuple = field(default=(32, 256, 256))
    patch_size: tuple = field(default=(4, 16, 16))

    # vision
    vision_tower: Optional[str] = field(default="vit3d") # None, "vit3d"
    vision_select_layer: Optional[int] = field(default=-1)
    vision_select_feature: Optional[str] = field(default="patch")
    pretrain_vision_model: str = field(default=None, metadata={"help": "Path to pretrained model for ViT."})
    freeze_vision_tower: bool = field(default=False)

@dataclass
class DataCollator:
    def __init__(self, seg_enable):
        self.seg_enable = seg_enable
    def __call__(self, batch: list) -> dict:
        if self.seg_enable:
            images, contours, input_ids, labels, attention_mask, segs = tuple(
                [b[key] for b in batch] for key in ('image', 'contour', 'input_id', 'label', 'attention_mask', 'seg'))

            images = torch.cat([_.unsqueeze(0) for _ in images], dim=0)
            contours = torch.cat([_.unsqueeze(0) for _ in contours], dim=0)
            input_ids = torch.cat([_.unsqueeze(0) for _ in input_ids], dim=0)
            labels = torch.cat([_.unsqueeze(0) for _ in labels], dim=0)
            attention_mask = torch.cat([_.unsqueeze(0) for _ in attention_mask], dim=0)

            for i, seg in enumerate(segs):
                if seg.sum() == 0:
                    segs[i] = torch.zeros((1, 1, 32, 256, 256))
                else:
                    segs[i] = seg.unsqueeze(0)
            segs = torch.cat(segs, dim=0)

            return_dict = dict(
                images=images,
                contours=contours,
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                segs=segs,
            )
        else:
            images, contours, input_ids, labels, attention_mask = tuple(
                [b[key] for b in batch] for key in ('image', 'contour', 'input_id', 'label', 'attention_mask'))

            images = torch.cat([_.unsqueeze(0) for _ in images], dim=0)
            contours = torch.cat([_.unsqueeze(0) for _ in contours], dim=0)
            input_ids = torch.cat([_.unsqueeze(0) for _ in input_ids], dim=0)
            labels = torch.cat([_.unsqueeze(0) for _ in labels], dim=0)
            attention_mask = torch.cat([_.unsqueeze(0) for _ in attention_mask], dim=0)

            return_dict = dict(
                images=images,
                contours=contours,
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )

        return return_dict

def main():
    # Setting up device and args
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    # Seting up tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(
        "GoodBaiBai88/M3D-LaMed-Phi-3-4B",
        model_max_length=512,
        padding_side="right",
        use_fast=False,
    )

    # Setting up dataset
    train_dataset = UniDatasets(data_args, tokenizer, mode='train')
    data_collator = DataCollator(data_args.seg_enable)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=data_args.per_device_train_batch_size, # Assuming this is available in data_args
        shuffle=True,
        collate_fn=data_collator,
    )

    # Setting up vision modules

    # Sets up vision encoder from m3d and merlin combined
    vision_encoder = build_vision_tower(model_args)
    vision_encoder.to(device)
    vision_encoder.eval() # Set the model to evaluation mode

    # Performs inference on a single batch with
    with torch.no_grad(): 
        for batch in train_dataloader:
            images = batch['images'].to(device)
            contours = batch['contours'].to(device)

            outputs = vision_encoder(images, contours)

            print(outputs.shape)
            break

if __name__ == "__main__":
    main()