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
from contextlib import nullcontext
from pathlib import Path

import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers

from safetensors.torch import load_file, save_file
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
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel
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

def log_validation(
    pipeline,
    args,
    accelerator,
    epoch,
    is_final_validation=False,
    val_dataloader=None,
    batch=None,
    save_dir=None,
):
    logger.info(
        f"Running validation... \n Generating validation images:"
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=accelerator.device)
    if args.seed is not None:
        generator = generator.manual_seed(args.seed)
    #images = []

    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    image_logs = []
    val_error_list = []

    with autocast_ctx:
        def gen_img(img, src_cond, tgt_cond, pose):
            images = pipeline(
                    image=img,
                    src_lighting_condition=src_cond,
                    tgt_lighting_condition=tgt_cond,
                    pose_condition=pose,
                    #negative_light_conditioning=batch["src_condition"],
                    #negative_pose_conditioning=batch["pose"],
                    guidance_scale=1.0,
                    output_type="pt"
            ).images
            return images
        
        if val_dataloader is not None:
            for batch in val_dataloader:
                src_img, tgt_img = batch["src_img"], batch["tgt_img"]
                images = gen_img(src_img,
                                 batch["src_condition"],
                                 batch["tgt_condition"],
                                 batch["pose"])

                val_error = torch.mean((tgt_img - images)**2).item()
                val_error_list.append(val_error)
                image_logs.append({"images": [src_img[0],
                                              tgt_img[0],
                                              images[0]],
                                   "val_error": val_error})

        elif batch is not None:
            src_img, tgt_img = batch["src_img"], batch["tgt_img"]
            images = gen_img(src_img,
                             batch["src_condition"],
                             batch["tgt_condition"],
                             batch["pose"])

            val_error = torch.mean((tgt_img - images)**2, dim=(1, 2, 3))
            val_error_list = [v for v in val_error]
            image_logs = [{"images": [b, t, g], "val_error": v} for b, t, g, v
                          in zip(src_img,
                                 tgt_img,
                                 images,
                                 val_error_list)
                         ]
        else:
            raise ValueError

    # Save the concatenated validation output
    if save_dir is not None:
        image_list = []
        for image_log in image_logs:
            imgs = list(map(convert2img, image_log["images"]))
            image_concat = image_grid(imgs, 1, len(imgs))
            image_list.append(image_concat)
        
        image_val_full = image_grid(image_list, len(image_list), 1)
        image_val_full.save(
            opj(save_dir, f"epoch_{epoch:06d}.png")
        )
    print(f"Epoch {epoch} | Validation error: {torch.mean(torch.tensor(val_error_list))}")

    return images

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/workspace/Project/stable-diffusion-2-1-base",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
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
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_img_path",
        type=str,
        default=None,
        help=(
            "A folder containing the training images. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--train_lat_path",
        type=str,
        default=None,
        help=(
            "A folder containing the training latents. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--val_img_path",
        type=str,
        default=None,
        help=(
            "A folder containing the validation images. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--val_lat_path",
        type=str,
        default=None,
        help=(
            "A folder containing the validation latents. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )

    parser.add_argument("--pose_data_path", type=str, default=None)
    parser.add_argument("--train_json_path", type=str, default=None)
    parser.add_argument("--val_json_path", type=str, default=None)
    parser.add_argument("--train_encoder", action='store_true')
    parser.add_argument("--loss_type", type=str, default="mse") #action='store_true')
    parser.add_argument("--dream_detail_preservation", type=float, default=1.)
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run validation every X epochs. The validation process consists of running the prompt"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
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
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--val_batch_size", type=int, default=4, help="Batch size (per device) for the validation dataloader.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.train_lat_path is None and args.val_lat_path is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args


DATASET_NAME_MAPPING = {
    "lambdalabs/naruto-blip-captions": ("image", "text"),
}


def main():
    args = parse_args()
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

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
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(opj(args.output_dir, "validation"), exist_ok=True)

    # Load scheduler and models.
    noise_scheduler = DDPMScheduler.from_config(args.pretrained_model_name_or_path,
                                                subfolder="scheduler")


    loaded = load_file(opj(args.pretrained_model_name_or_path, "encoder", "encoder.safetensors"))
    if loaded is None:
        cprint("no any pretrained files for encoder", color='yellow')
        light_map = []
        for flight in glob(os.path.expanduser("~/diffusion/Project/dataset/generate_light_gt_sg/hdrnpys/*")):
            light_map.append(np.load(flight))
        light_map = torch.from_numpy(np.array(light_map))
        light_pos_maps = torch.stack(tuple(light_map), dim=0).permute(0, 3, 1, 2) # n_labels, 3, 256, 512
        lighting_encoder = CompoundEncoder.from_config(args.pretrained_model_name_or_path,
                                                       subfolder="encoder",
                                                       light_pos_maps=light_pos_maps) # load light encoder
    else:
        cprint("loading variables from pretrained encoder", color='green')
        lighting_encoder = CompoundEncoder.from_config(args.pretrained_model_name_or_path,
                                                       subfolder="encoder")
        lighting_encoder.load_state_dict(loaded, strict=False)
    if args.train_encoder: lighting_encoder.train()

    vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="vae")

    unet = UNet2DConditionModel.from_config(args.pretrained_model_name_or_path,
                                            subfolder="unet")
    unet.train()

    # freeze parameters of models to save more memory
    unet.requires_grad_(True)
    vae.requires_grad_(False)
    lighting_encoder.requires_grad_(args.train_encoder)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    lighting_encoder.to(accelerator.device, dtype=weight_dtype) # trainable params are in float32

    # Add adapter and make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(lighting_encoder, dtype=torch.float16)
        cast_training_params(vae, dtype=torch.float16)

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

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    unet_parameters = list(filter(lambda p: p.requires_grad, unet.parameters()))
    lighting_encoder_parameters = list(filter(lambda p: p.requires_grad, lighting_encoder.parameters()))

    cprint("==============================TRAINABLE-PARAMETERS==============================", color='green')
    cprint(f"UNET Parameters: {len(list(unet_parameters))}", color='green')
    cprint(f"LightNET Parameters: {len(list(lighting_encoder_parameters))}", color='green')
    cprint("================================================================================", color='green')

    params_to_optimize = [
        {"params": unet_parameters, "lr": args.learning_rate},
        {"params": lighting_encoder_parameters, "lr": args.learning_rate * 0.1},
    ]

    optimizer = optimizer_cls(
        params_to_optimize,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.

    # prepare LiT dataset
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
    )
    # pose map 어디서?
    pose_map = {
        "NA3": 0, "NE7": 1, "CB5": 2, "CF8": 3, "NA7": 4, "CC7": 5, "CA2": 6, "NE1": 7, "NC3": 8, "CE2": 9
    }
    train_dataset = LitDataset(args.train_img_path,
                               args.train_lat_path,
                               args.train_json_path,
                               pose_map,
                               train_transforms,
                               use_base=False)
    val_dataset = LitDataset(args.val_img_path,
                             args.val_lat_path,
                             args.val_json_path,
                             pose_map,
                             train_transforms,
                             use_base=False)

    ##################
    # TODO - Dataloader 구조 알맞게 수정
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=args.val_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    ###################
    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # Prepare everything with our `accelerator`.
    # add light encoder
    unet, lighting_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, lighting_encoder, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(opj(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        dynamic_ncols=True,
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        if args.train_encoder: lighting_encoder.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            if step >= 10: break
            src_enc, tgt_enc = batch["src_lat"], batch["tgt_lat"]
            acc_modules = [unet, lighting_encoder] if args.train_encoder else [unet]
            with accelerator.accumulate(*acc_modules):

                # Convert src images to latent space
                #src_latents = vae.encode(base_img.to(dtype=weight_dtype)).latent_dist.sample()
                #src_latents = src_latents * vae.config.scaling_factor
                src_latents = DiagonalGaussianDistribution(src_enc).sample() *  0.18215 # scaling factor of vae

                # Convert tgt images to latent space
                #tgt_latents = vae.encode(tgt_img.to(dtype=weight_dtype)).latent_dist.sample()
                #tgt_latents = tgt_latents * vae.config.scaling_factor
                tgt_latents = DiagonalGaussianDistribution(tgt_enc).sample() * 0.18215 # scaling factor of vae

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(src_latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (src_latents.shape[0], src_latents.shape[1], 1, 1), device=src_latents.device
                    )

                bsz = tgt_latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=tgt_latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_src_latents = noise_scheduler.add_noise(src_latents, noise, timesteps)
                noisy_tgt_latents = noise_scheduler.add_noise(tgt_latents, noise, timesteps)

                # Get the light embedding for conditioning
                # encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]

                src_cond, tgt_cond = batch["src_condition"], batch["tgt_condition"]
                #encoder_hidden_states = lighting_encoder(tgt_cond, batch["pose"])
                encoder_hidden_states = lighting_encoder(src_latents,
                                                         src_cond,
                                                         tgt_cond,
                                                         batch["pose"])

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    # tgt이 다른 lighting에 대한 latent이므로, v prediction이나 x0 prediction을 사용해야 하지 않을까? img 단에서의 loss를 사용하는 편이 좋아 보인다.
                    target = noise_scheduler.get_velocity(tgt_latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                if args.loss_type == "dream":
                    alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)[timesteps, None, None, None]
                    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

                    # The paper uses lambda = sqrt(1 - alpha) ** p, with p = 1 in their experiments.
                    dream_lambda = sqrt_one_minus_alphas_cumprod**args.dream_detail_preservation

                    with torch.no_grad():
                        stop_grad_pred = unet(noisy_tgt_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

                    _noisy_latents, _target = (None, None)
                    if noise_scheduler.config.prediction_type == "epsilon":
                        delta_noise = (noise - stop_grad_pred).detach()
                        delta_noise.mul_(dream_lambda)

                        noisy_tgt_latents = noisy_tgt_latents.add(sqrt_one_minus_alphas_cumprod * delta_noise)
                        target = target.add(delta_noise)

                        model_pred = unet(noisy_tgt_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        raise NotImplementedError("DREAM has not been implemented for v-prediction")
                    else:
                        raise ValueError

                    if epoch >= 15:
                        loss = torch.max(torch.abs(model_pred.float()-target.float()), dim=0).values.mean()
                    else:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction='mean')

                else:
                    model_pred = unet(noisy_tgt_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

                    if args.loss_type == 'centroid':
                        raise NotImplementedError
                    else:
                        if args.snr_gamma is None:
                            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                        else:
                            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                            # This is discussed in Section 4.2 of the same paper.
                            snr = compute_snr(noise_scheduler, timesteps)
                            mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
                            if noise_scheduler.config.prediction_type == "epsilon":
                                mse_loss_weights = mse_loss_weights / snr
                            elif noise_scheduler.config.prediction_type == "v_prediction":
                                mse_loss_weights = mse_loss_weights / (snr + 1)

                            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                            loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = unet_parameters + lighting_encoder_parameters
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = opj(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = opj(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)

                        accelerator.save_state(f"{save_path}")

                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if (
                epoch % args.validation_epochs == 0
            ):
                with torch.no_grad():
                    unet.eval()
                    #vae.eval()
                    lighting_encoder.eval()

                    pipeline = DiffRelight(
                        unet=unwrap_model(unet),
                        vae=unwrap_model(vae),
                        scheduler=noise_scheduler
                    )
                    pipeline.add_lighting_encoder(unwrap_model(lighting_encoder))

                    images = log_validation(
                        pipeline,
                        args,
                        accelerator,
                        epoch,
                        batch=batch,
                        save_dir=opj(args.output_dir, "validation")
                    )

                    del pipeline
                    unet.train()
                    #vae.train()
                    if args.train_encoder: lighting_encoder.train()
                torch.cuda.empty_cache()

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet, vae, lighting_encoder = unet.to(torch.float32), vae.to(torch.float32), lighting_encoder.to(torch.float32)

        save_path = opj(args.output_dir, f"checkpoint-{global_step}")
        accelerator.save_state(f"{save_path}")

        save_file(unet.state_dict(), opj(args.output_dir, 'unet.safetensors'))
        if args.train_encoder:
            save_file(lighting_encoder.state_dict(), opj(args.output_dir, 'encoder.safetensors'))

    accelerator.end_training()


if __name__ == "__main__":
    main()
