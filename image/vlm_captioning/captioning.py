import argparse
import copy
from copy import deepcopy
import logging
import os
from pathlib import Path
from collections import OrderedDict
import json
import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

import sys
import os

from dataset import CustomTemporaryDataset


logger = get_logger(__name__)


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def main(args):
    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator_kwargs = {
        "mixed_precision": args.mixed_precision,
        "project_config": accelerator_project_config,
    }

    accelerator = Accelerator(**accelerator_kwargs)

    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # Setup data:
    train_dataset = CustomTemporaryDataset(args.data_dir)
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=False,  # disable shuffle for debugging
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # Setup model
    model_name_or_path = args.model_name_or_path
    if "Qwen2_5" in model_name_or_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
    elif "Qwen2" in model_name_or_path:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
    else:
        raise ValueError(f"Model {model_name_or_path} not found")

    model = model.to(device)
    model.requires_grad_(False)
    model.eval()

    processor = AutoProcessor.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    for raw_image, x, y, idx in tqdm(train_dataloader):
        dialogue_list = []
        for i in range(raw_image.shape[0]):
            dialogue_list.append(
                [{"role": "user", "content": [{"type": "image", "image": raw_image[i]}, {"type": "text", "text": "Describe this image."}]}]
            )
        text = [processor.apply_chat_template(dialogue, tokenize=False, add_generation_prompt=True) for dialogue in dialogue_list]
        inputs = processor(text=text, images=[raw_image[i] for i in range(raw_image.shape[0])], return_tensors="pt")
        inputs = inputs.to(device)
        generate_output = model.generate(**inputs, max_new_tokens=200)
        count = 0
        output_text_clean = processor.batch_decode(generate_output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        full_list = []
        for i in range(raw_image.shape[0]):
            full_list.append(
                [{"role": "user", "content": [{"type": "image", "image": raw_image[i]}, {"type": "text", "text": "Describe this image."}]},
                 {"role": "assistant", "content": [{"type": "text", "text": output_text_clean[i]}]}
                ]
            )
        full_text = [processor.apply_chat_template(dialogue, tokenize=False, add_generation_prompt=True) for dialogue in full_list]
        # image_inputs, video_inputs = process_vision_info(full_list)
        image_inputs = [raw_image[i] for i in range(raw_image.shape[0])]
        inputs_full = processor(text=full_text, images=image_inputs, return_tensors="pt", padding=True, truncation=True)
        inputs_full = inputs_full.to(device)
        with torch.no_grad():
            model_output = model(**inputs_full, output_hidden_states=True)

        text_embeds = model_output.hidden_states[-1]
        
        for i in idx:
            i = i.item()
            img_dir = train_dataset.image_fnames[i]
            img_ext = train_dataset._file_ext(img_dir)
            img_dir = os.path.join(train_dataset.images_dir, img_dir)
            text_caption_dir = img_dir.replace("images", "text_captions").replace(img_ext, ".txt")
            text_embeds_dir = img_dir.replace("images", "text_embeds_" + model_name_or_path + "_last").replace(img_ext, ".npy")
            
            os.makedirs(os.path.dirname(text_caption_dir), exist_ok=True)
            os.makedirs(os.path.dirname(text_embeds_dir), exist_ok=True)
            with open(text_caption_dir, 'w') as f:
                f.write(output_text_clean[count])
            np.save(text_embeds_dir, text_embeds[count].mean(dim=0).detach().cpu().float().numpy())
            count += 1



def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="ImageNet/captioning")  # "exps"
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="none")  # "wandb"
    parser.add_argument("--sampling-steps", type=int, default=10000)
    parser.add_argument("--resume-step", type=int, default=0)

    # dataset
    parser.add_argument("--data-dir", type=str, default="ImageNet")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=32)

    # precision
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="no", choices=["no", "fp16", "bf16"])

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # cpu
    parser.add_argument("--num-workers", type=int, default=4)

    # model
    parser.add_argument("--model-name-or-path", type=str, default="Qwen2-VL-2B-Instruct")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    main(args)
