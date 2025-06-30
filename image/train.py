import argparse
import copy
from copy import deepcopy
import logging
import os
from pathlib import Path
from collections import OrderedDict
import json

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from models.sit import SiT_models
from loss import SILoss
from utils import load_encoders

from dataset import CustomDataset
from diffusers.models import AutoencoderKL
# import wandb_utils
import wandb
import math
from torchvision.utils import make_grid
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize
import datetime

logger = get_logger(__name__)

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)

TEXT_Z_DIM_DICT = {'text_embeds_qwenvl': 1536, 'text_embeds_open_clip': 1280, 'text_embeds_qwenvl_7b': 3584, 
                   'text_embeds_qwenvl_7b_layer_0': 3584, 'text_embeds_qwenvl_7b_layer_1': 3584, 'text_embeds_qwenvl_7b_layer_15': 3584, 
                   'text_embeds_qwenvl_2.5_3B': 2048, 'text_embeds_qwenvl_2.5_7B': 3584, 'text_embeds_qwenvl_2.5_7B_layer_15': 3584,
                   'text_embeds_qwenvl_2.5_7B_layer_1': 3584}


# cosine schedule
def cosine_schedule(step, total_steps, start, end):
    if step >= total_steps:
        return end
    return end + (start - end) / 2 * (1 + math.cos(step / total_steps * math.pi))


def preprocess_raw_image(x, enc_type):
    resolution = x.shape[-1]
    if 'clip' in enc_type:
        x = x / 255.
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif 'mocov3' in enc_type or 'mae' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'dinov2' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    elif 'dinov1' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'jepa' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')

    return x


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    device = moments.device
    
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z * latents_scale + latents_bias) 
    return z 


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


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


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):    
    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
        )
    
    accelerator_kwargs = {
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "mixed_precision": args.mixed_precision,
        "project_config": accelerator_project_config,
    }

    report_to = args.report_to if args.report_to and args.report_to.lower() != "none" else None

    if report_to is not None:
        accelerator_kwargs["log_with"] = report_to
    accelerator = Accelerator(**accelerator_kwargs)

    curr_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.resume_step > 0:
        exp_name_with_date = args.exp_name
    else:
        exp_name_with_date = f"{args.exp_name}_{curr_time}"

    if accelerator.is_main_process and report_to is not None:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        save_dir = os.path.join(args.output_dir, exp_name_with_date)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False    
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)
    
    # Create model:
    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution // 8

    if args.enc_type != None:
        encoders, encoder_types, architectures = load_encoders(
            args.enc_type, device, args.resolution
            )
        enc_names = encoder_types
    else:
        raise NotImplementedError()
    z_dims = [encoder.embed_dim for encoder in encoders] if args.enc_type != 'None' else [0]
    z_types = ['i' for _ in encoders] if args.enc_type != 'None' else []
    
    # text encoders
    if args.text_embeds_dir is not None:
        enc_names.append(args.text_embeds_dir)
        z_dims.append(TEXT_Z_DIM_DICT[args.text_embeds_dir])
        z_types.append('t')

    assert len(enc_names) >= 1, "At least one encoder must be provided for alignment."
    if accelerator.is_main_process and report_to is not None:
        logger.info(f"Encoders for Alignment {enc_names}")
        logger.info(f"Encoder Alignment Weights {args.repa_coeff}")

    assert len(args.repa_coeff) == len(
        enc_names), f"Number of alignment loss coefficients {len(args.repa_coeff)} must match the total number of encoders {len(enc_names)}."
    
    
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg = (args.cfg_prob > 0),
        z_dims = z_dims,
        z_types = z_types,
        encoder_depth=args.encoder_depth,
        encoder_depth_text=args.encoder_depth_text,
        **block_kwargs
    )

    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(device)
    requires_grad(ema, False)
    
    latents_scale = torch.tensor(
        [0.18215, 0.18215, 0.18215, 0.18215]
        ).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor(
        [0., 0., 0., 0.]
        ).view(1, 4, 1, 1).to(device)

    # create loss function
    loss_fn = SILoss(
        prediction=args.prediction,
        path_type=args.path_type, 
        encoders=encoders,
        enc_names=enc_names,
        accelerator=accelerator,
        latents_scale=latents_scale,
        latents_bias=latents_bias,
        weighting=args.weighting,
        loss_weights={enc_name: args.repa_coeff[i] for i, enc_name in enumerate(enc_names)},
        time_schedule=args.time_schedule,
        cutoffs=args.cutoffs,
    )
    if accelerator.is_main_process and report_to is not None:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )    
    
    # Setup data:
    train_dataset = CustomDataset(args.data_dir, text_embeds_dir=args.text_embeds_dir)
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process and report_to is not None:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")
    
    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    
    # resume:
    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) +'.pt'
        ckpt = torch.load(
            f'{os.path.join(args.output_dir, exp_name_with_date)}/checkpoints/{ckpt_name}',
            map_location='cpu',
            )
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['opt'])
        global_step = ckpt['steps']

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    if accelerator.is_main_process and report_to is not None:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            project_name="REPA", 
            config=tracker_config,
            init_kwargs={
                "wandb": {"name": f"{exp_name_with_date}", "dir": os.path.join(args.output_dir, exp_name_with_date)},
            },
        )
        
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Labels to condition the model with (feel free to change):
    sample_batch_size = 64 // accelerator.num_processes
    gt_raw_images, gt_xs, _, _ = next(iter(train_dataloader))
    assert gt_raw_images.shape[-1] == args.resolution
    gt_xs = gt_xs[:sample_batch_size]
    gt_xs = sample_posterior(
        gt_xs.to(device), latents_scale=latents_scale, latents_bias=latents_bias
        )
    ys = torch.randint(1000, size=(sample_batch_size,), device=device)
    ys = ys.to(device)
    if not args.cfg:
        ys = torch.zeros_like(ys)
    # Create sampling noise:
    n = ys.size(0)
    xT = torch.randn((n, 4, latent_size, latent_size), device=device)
        
    for epoch in range(args.epochs):
        model.train()
        for raw_image, x, y, textemb in train_dataloader:
            raw_image = raw_image.to(device)
            x = x.squeeze(dim=1).to(device)
            y = y.to(device)
            z = None
            if args.legacy:
                # In our early experiments, we accidentally apply label dropping twice: 
                # once in train.py and once in sit.py. 
                # We keep this option for exact reproducibility with previous runs.
                drop_ids = torch.rand(y.shape[0], device=y.device) < args.cfg_prob
                labels = torch.where(drop_ids, args.num_classes, y)
            else:
                labels = y
            if not args.cfg:
                labels = torch.zeros_like(labels)
            with torch.no_grad():
                x = sample_posterior(x, latents_scale=latents_scale, latents_bias=latents_bias)
                zs = []
                with accelerator.autocast():
                    for encoder, encoder_type, arch in zip(encoders, encoder_types, architectures):
                        raw_image_ = preprocess_raw_image(raw_image, encoder_type)
                        z = encoder.forward_features(raw_image_)
                        if 'mocov3' in encoder_type: z = z = z[:, 1:] 
                        if 'dinov2' in encoder_type: z = z['x_norm_patchtokens']
                        zs.append(z)
                    if args.text_embeds_dir is not None:
                        textemb = textemb.to(device)
                        zs.append(textemb)

            with accelerator.accumulate(model):
                if args.repa_weight_decay == "constant":
                    _repa_weight_decay = 1.0
                elif args.repa_weight_decay == "linear":
                    _repa_weight_decay = max(1.0 - global_step / args.repa_steps, 0.)
                elif args.repa_weight_decay == "cosine":
                    _repa_weight_decay = max((1.0 + np.cos(np.pi * global_step / args.repa_steps)) / 2, 0.)
                else:
                    raise NotImplementedError

                top_steps = args.diffusion_warm_up_steps + args.start_diffusion_steps
                if global_step < args.start_diffusion_steps:
                    _diffusion_loss_decay = 0.0
                elif args.start_diffusion_steps <= global_step < top_steps:
                    _diffusion_loss_decay = (global_step - args.start_diffusion_steps) / args.diffusion_warm_up_steps
                else:
                    if args.diffusion_decay == "constant":
                        _diffusion_loss_decay = 1.0
                    elif args.diffusion_decay == "linear":
                        _diffusion_loss_decay = 1.0 - (global_step - top_steps) / (args.max_train_steps - top_steps)
                    elif args.diffusion_decay == "cosine":
                        _diffusion_loss_decay = (1.0 + np.cos(np.pi * (global_step - top_steps) / args.max_train_steps - top_steps)) / 2
                    else:
                        raise NotImplementedError

                model_kwargs = dict(y=labels)
                loss_dict = loss_fn(model, x, model_kwargs, zs=zs, save_projloss=False)
                loss, proj_loss = loss_dict["denoising_loss"], loss_dict["proj_loss"]
                if args.text_embeds_dir != None:
                    img_proj_loss, text_proj_loss = loss_dict["img_proj_loss"], loss_dict["text_proj_loss"]
                    img_proj_loss_mean, text_proj_loss_mean = img_proj_loss.mean(), text_proj_loss.mean()
                else:
                    img_proj_loss_mean = proj_loss.mean()

                loss_mean = loss.mean()
                proj_loss_mean = proj_loss.mean()
                loss = loss_mean * _diffusion_loss_decay + proj_loss_mean * args.proj_coeff * _repa_weight_decay
                    
                ## optimization
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                if accelerator.num_processes==1:
                    params_to_clip = model.parameters()
                    grad_norm = torch.nn.utils.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients or accelerator.num_processes==1:
                    update_ema(ema, model) # change ema function
            
            ### enter
            if accelerator.sync_gradients or accelerator.num_processes==1:
                progress_bar.update(1)
                global_step += 1     
            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                if accelerator.is_main_process and report_to is not None:
                    checkpoint = {
                        "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            if (global_step == 1 or (global_step % args.sampling_steps == 0 and global_step > 0)):
                from samplers import euler_sampler
                with torch.no_grad():
                    samples = euler_sampler(
                        model, 
                        xT, 
                        ys,
                        num_steps=50, 
                        cfg_scale=4.0,
                        guidance_low=0.,
                        guidance_high=1.,
                        path_type=args.path_type,
                        heun=False,
                        prediction=args.prediction,
                    ).to(torch.float32)
                    samples = vae.decode((samples -  latents_bias) / latents_scale).sample
                    gt_samples = vae.decode((gt_xs - latents_bias) / latents_scale).sample
                    samples = (samples + 1) / 2.
                    gt_samples = (gt_samples + 1) / 2.
                out_samples = accelerator.gather(samples.to(torch.float32))
                gt_samples = accelerator.gather(gt_samples.to(torch.float32))
                accelerator.log({"samples": wandb.Image(array2grid(out_samples)),
                                 "gt_samples": wandb.Image(array2grid(gt_samples))})
                logging.info("Generating EMA samples done.")

            logs = {
                "proj_loss": accelerator.gather(proj_loss_mean).mean().detach().item(),
                "grad_norm": accelerator.gather(grad_norm).mean().detach().item(),
                "training_denoising_loss": accelerator.gather(loss_mean).mean().detach().item(), # denoising loss used for training
            }
            if args.text_embeds_dir is not None:
                logs.update({"img_proj_loss": accelerator.gather(img_proj_loss_mean).mean().detach().item(),
                             "text_proj_loss": accelerator.gather(text_proj_loss_mean).mean().detach().item()})
            else:
                logs.update({"img_proj_loss": accelerator.gather(img_proj_loss_mean).mean().detach().item()})

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process and report_to is not None:
        logger.info("Done!")
    accelerator.end_training()

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--sampling-steps", type=int, default=10000)
    parser.add_argument("--resume-step", type=int, default=0)

    # model
    parser.add_argument("--model", type=str)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--encoder-depth-text", type=int, default=None) # None means the same as encoder-depth
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm",  action=argparse.BooleanOptionalAction, default=False)

    # dataset
    parser.add_argument("--data-dir", type=str, default="../data/imagenet256")
    parser.add_argument("--resolution", type=int, choices=[256], default=256)
    parser.add_argument("--batch-size", type=int, default=256)

    # precision
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=400000)
    parser.add_argument("--checkpointing-steps", type=int, default=50000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # cpu
    parser.add_argument("--num-workers", type=int, default=4)

    # loss
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"]) # currently we only support v-prediction
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--enc-type", type=str, default='dinov2-vit-b')
    parser.add_argument("--proj-coeff", type=float, default=0.5)
    parser.add_argument("--weighting", default="uniform", type=str, help="Max gradient norm.")
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--time-schedule", type=str, default="constant", choices=["constant", "linear", "cosine", "loglinear", "cutoff"])
    parser.add_argument("--repa-coeff", type=float, nargs='+', default=[1.0])  # alignment loss coefficients for different image/text encoders
    parser.add_argument("--cutoffs", type=float, nargs='+', default=[0.0, 1.0])
    parser.add_argument("--cfg", action=argparse.BooleanOptionalAction, default=True)
    
    parser.add_argument("--text-embeds-dir", type=str, default=None)
    parser.add_argument("--repa-weight-decay", type=str, default="constant", help="scheduler for repa loss weight w.r.t. training epoch")
    parser.add_argument("--repa-steps", type=int, default=400000, help="total number of epochs with repa loss")
    parser.add_argument("--start-diffusion-steps", type=int, default=0, help="before this epoch only use repa loss as a pretrain stage")
    parser.add_argument("--diffusion-warm-up-steps", type=int, default=50000, help="warm up diffusion loss weight")
    parser.add_argument("--diffusion-decay", type=str, default="constant", help="diffusion loss decay")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
        
    return args

if __name__ == "__main__":
    args = parse_args()

    main(args)
