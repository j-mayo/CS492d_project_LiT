#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import logging
import math
import os
import random
import shutil

from PIL import Image
from contextlib import nullcontext
from pathlib import Path

import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers

from safetensors.torch import load_file
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torchvision import transforms
from tqdm.auto import tqdm
from utils.utils import image_grid
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision import transforms
from termcolor import cprint

import diffusers
from diffusers import (DDPMScheduler,
                       DDIMScheduler,
                       PNDMScheduler,
                       AutoencoderKL,
                       UNet2DConditionModel,
                       DiffusionPipeline,
                       StableDiffusionPipeline)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

# from ..ldm.modules.encoders.modules import LightFieldEncoder  # added
from glob import glob
from utils.dataloader_vae import LitDataset
from utils.diagonal_gaussian import DiagonalGaussianDistribution
from models.DiffRelight import DiffRelight
from models.encoder import CompoundEncoder #LightingEncoder

opj = os.path.join

if is_wandb_available():
    import wandb

logger = get_logger(__name__, log_level="INFO")


def convert2img(tensor):
    return transforms.ToPILImage()(tensor)

def norm_img(img):
    #return (img + 1.)/2.
    return transforms.Normalize([0.5], [0.5])(img)

def unnorm_img(img):
    #return (2. * img) - 1.
    return transforms.Normalize([-1.0], [2.0])(img)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--eval_img_path",
        type=str,
        default=None,
        help=(
            "A folder containing the training images. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument("--pose_data_path", type=str, default=None)
    parser.add_argument("--eval_json_path", type=str, default=None)
    parser.add_argument(
        "--num_eval_generation",
        type=int,
        default=4,
        help="Number of images that should be generated during validation.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the generated images will be saved.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default=None,
        help="The logging directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument("--eval_batch_size", type=int, default=4, help="Batch size (per device) for the validation dataloader.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument("--lighting_layers", type=int, default=4)
    parser.add_argument("--latent_layers", type=int, default=2)

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args



def generate_img(pipeline, img, src_cond, tgt_cond, pose):
    images = pipeline(image=img,
                      src_lighting_condition=src_cond,
                      tgt_lighting_condition=tgt_cond,
                      pose_condition=pose,
                      negative_src_lighting_condition=src_cond,
                      negative_tgt_lighting_condition=src_cond,
                      num_inference_steps=100,
                      guidance_scale=1.25,
                      output_type="pt").images
    return images

def save_image(images, names, dirname):
    for img, name in zip(images, names):
        convert2img(img).save(opj(dirname, name+'.png'))

def main():
    args = parse_args()

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(project_config=accelerator_project_config)

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    save_path = Path(args.output_dir, "generated_images")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = PNDMScheduler.from_config(args.model_path, subfolder='scheduler')

    lighting_encoder = CompoundEncoder.from_config(args.model_path, subfolder='encoder')
    lighting_encoder.load_state_dict(
            load_file(opj(args.model_path, 'encoder', 'encoder.safetensors')),
            strict=False)

    vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="vae")  

    unet = UNet2DConditionModel.from_config(args.model_path, subfolder="unet")
    unet.load_state_dict(load_file(opj(args.model_path, 'unet', 'unet.safetensors')))

    lighting_encoder.eval()
    vae.eval()
    unet.eval()

    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    lighting_encoder.requires_grad_(False)

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device)
    vae.to(accelerator.device)
    lighting_encoder.to(accelerator.device) # trainable params are in float32

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    eval_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
    )
    # pose map 어디서?
    pose_map = {
        "NA3": 0, "NE7": 1, "CB5": 2, "CF8": 3, "NA7": 4, "CC7": 5, "CA2": 6, "NE1": 7, "NC3": 8, "CE2": 9
    }
    eval_dataset = LitDataset(img_path=args.eval_img_path,
                              vae_path=None,
                              json_path=args.eval_json_path,
                              pose_map=pose_map,
                              transform=eval_transforms,
                              use_base=False,
                              mode='eval')

    ##################
    # TODO - Dataloader 구조 알맞게 수정
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        shuffle=False,
        batch_size=args.eval_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Prepare everything with our `accelerator`.
    # add light encoder

    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.eval_batch_size}")

    progress_bar = tqdm(
        range(0, len(eval_dataloader)),
        dynamic_ncols=True,
        initial=0,
        desc="Iters",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    pipeline = DiffRelight(unet=unwrap_model(unet),
                           vae=unwrap_model(vae),
                           scheduler=noise_scheduler)
    pipeline.add_lighting_encoder(unwrap_model(lighting_encoder))
    pipeline, eval_dataloader = accelerator.prepare(pipeline, eval_dataloader)

    generator = torch.Generator(device=accelerator.device)
    if args.seed is not None:
        generator = generator.manual_seed(args.seed)

    concat_path = Path(args.output_dir, "concatenated_images")
    os.makedirs(concat_path, exist_ok=True)
    for step, batch in enumerate(eval_dataloader):
        gen_img = generate_img(pipeline, batch["src_img"], batch["src_condition"], batch["tgt_condition"], batch["pose"])
        save_image(gen_img, batch["name"], save_path)

        image_logs = [[s, t, g] for s, t, g in zip(batch["src_img"], batch["tgt_img"], gen_img)]

        image_list = []
        for image_log in image_logs:
            imgs = list(map(convert2img, image_log))
            image_concat = image_grid(imgs, 1, len(imgs))
            image_list.append(image_concat)

        image_full = image_grid(image_list, len(image_list), 1)
        image_full.save(opj(concat_path, f"batch-{step:02d}.png"))

    torch.cuda.empty_cache()
    accelerator.end_training()


if __name__ == "__main__":
    main()
