#!/usr/bin/env python3
"""
Train LoRA for style transfer on Wan2.2 using Ditto-1M dataset.

This script implements LoRA fine-tuning for Wan2.2-TI2V-5B model
to learn style transfer from the Ditto-1M dataset.
"""

import argparse
import contextlib
import json
import logging
import os
import sys
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as torch_ckpt
import yaml
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from functools import partial
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from peft import LoraConfig, get_peft_model
from wan.textimage2video import WanTI2V
from wan.configs import WAN_CONFIGS
from wan.utils.utils import save_video


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def resolve_lora_target_modules(base_model: nn.Module, requested_modules):
    """
    Resolve LoRA target module suffixes against actual linear module names.

    PEFT matches target modules by module-name suffix, so this validates that
    each requested suffix maps to at least one Linear layer.
    """
    linear_module_names = [
        name for name, module in base_model.named_modules()
        if isinstance(module, nn.Linear)
    ]
    requested_modules = list(requested_modules)
    resolved = []
    missing = []

    for suffix in requested_modules:
        if any(name == suffix or name.endswith(f".{suffix}") for name in linear_module_names):
            resolved.append(suffix)
        else:
            missing.append(suffix)

    if missing:
        sample_linear = linear_module_names[:30]
        raise ValueError(
            "LoRA target modules not found in WanModel: "
            f"{missing}. "
            "Use suffixes matching nn.Linear layers (e.g. q/k/v/o). "
            f"Sample linear modules: {sample_linear}"
        )

    return resolved


class DittoStyleDataset(Dataset):
    """Dataset for Ditto-1M style transfer videos."""
    
    def __init__(self, metadata_path, data_dir, num_frames=81, resolution=(1280, 704)):
        """
        Args:
            metadata_path: Path to metadata JSON file
            data_dir: Base directory containing videos
            num_frames: Number of frames to sample
            resolution: Target resolution (width, height)
        """
        self.data_dir = Path(data_dir)
        self.num_frames = num_frames
        self.resolution = resolution
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.samples = json.load(f)
        
        logger.info(f"Loaded {len(self.samples)} samples from {metadata_path}")

    def resolve_video_path(self, video_path):
        """
        Resolve Ditto video path across common layouts:
        - <data_dir>/<source/...>
        - <data_dir>/videos/<source/...>
        """
        rel = Path(video_path)
        candidates = [
            self.data_dir / rel,
            self.data_dir / "videos" / rel,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None
    
    def __len__(self):
        return len(self.samples)
    
    def load_video(self, video_path, num_frames):
        """Load and preprocess video."""
        full_path = self.resolve_video_path(video_path)
        
        if full_path is None:
            raise FileNotFoundError(
                f"Video not found for relative path: {video_path}. "
                f"Tried: {self.data_dir / video_path} and {self.data_dir / 'videos' / video_path}"
            )
        
        # Read video
        cap = cv2.VideoCapture(str(full_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames uniformly
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                # If frame read fails, duplicate last frame
                if frames:
                    frames.append(frames[-1])
                continue
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize to target resolution
            frame = cv2.resize(frame, self.resolution)
            
            frames.append(frame)
        
        cap.release()
        
        # Convert to tensor [T, H, W, C] -> [T, C, H, W]
        frames = np.stack(frames)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        
        return frames
    
    def __getitem__(self, idx):
        """Get a training sample."""
        sample = self.samples[idx]
        
        try:
            # Load source and edited videos
            source_video = self.load_video(sample['source_path'], self.num_frames)
            edited_video = self.load_video(sample['edited_path'], self.num_frames)
            
            # Get text instruction
            instruction = sample['instruction']
            source_caption = sample.get('source_caption', '')
            
            return {
                'source_video': source_video,
                'edited_video': edited_video,
                'instruction': instruction,
                'source_caption': source_caption,
                'source_path': sample['source_path'],
                'edited_path': sample['edited_path']
            }
        
        except Exception as e:
            logger.warning(f"Error loading sample {idx}: {e}")
            # Try a bounded number of fallback samples to avoid infinite recursion.
            for offset in range(1, min(32, len(self.samples))):
                next_idx = (idx + offset) % len(self.samples)
                next_sample = self.samples[next_idx]
                try:
                    source_video = self.load_video(next_sample['source_path'], self.num_frames)
                    edited_video = self.load_video(next_sample['edited_path'], self.num_frames)
                    instruction = next_sample['instruction']
                    source_caption = next_sample.get('source_caption', '')
                    return {
                        'source_video': source_video,
                        'edited_video': edited_video,
                        'instruction': instruction,
                        'source_caption': source_caption,
                        'source_path': next_sample['source_path'],
                        'edited_path': next_sample['edited_path']
                    }
                except Exception:
                    continue
            raise RuntimeError(
                "Unable to load a valid video pair after multiple retries. "
                "Dataset paths may be wrong or video files are missing."
            )


class LoRATrainer:
    """Trainer for LoRA fine-tuning on Wan2.2."""
    
    def __init__(self, config, rank=0, world_size=1, local_rank=0):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.is_main = (rank == 0)
        # Each rank owns its assigned GPU.
        self.device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(self.device)

        training_cfg = config.get('training', {})
        self.debug_loss = bool(training_cfg.get('debug_loss', False))
        self.debug_loss_steps = int(training_cfg.get('debug_loss_steps', 3))
        self.debug_breakpoint = bool(training_cfg.get('debug_breakpoint', False))
        self._debug_loss_seen = 0
        self.memory_debug = bool(training_cfg.get('memory_debug', False))
        # Whether to use FSDP (set when torchrun provides RANK env var and config enables it)
        self.use_fsdp = world_size > 1 and bool(training_cfg.get('fsdp', False))

        # Create output directories and TensorBoard writer on rank 0 only.
        self.output_dir = Path(config['training']['output_dir'])
        self.logging_dir = Path(config['training']['logging_dir'])
        if self.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.logging_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(self.logging_dir))
            with open(self.output_dir / "config.yaml", 'w') as f:
                yaml.dump(config, f)
            logger.info(f"Output directory: {self.output_dir}")
            logger.info(f"Logging directory: {self.logging_dir}")
        else:
            self.writer = None  # non-rank-0 processes never write to TB
        self.global_step = 0
        self._num_train_timesteps = None  # set lazily from model config

    def _debug(self, msg):
        if self.debug_loss:
            logger.info(f"[DEBUG_LOSS] {msg}")

    def _mem(self, tag):
        """Log GPU memory at a named checkpoint. Only active when memory_debug=true."""
        if not self.memory_debug:
            return
        torch.cuda.synchronize()
        alloc  = torch.cuda.memory_allocated()  / 1024**3
        reserv = torch.cuda.memory_reserved()   / 1024**3
        peak   = torch.cuda.max_memory_allocated() / 1024**3
        total  = torch.cuda.get_device_properties(self.local_rank).total_memory / 1024**3
        logger.info(
            f"[MEM][rank{self.rank}] {tag:<45s} "
            f"alloc={alloc:6.2f}GB  reserved={reserv:6.2f}GB  "
            f"peak={peak:6.2f}GB  total={total:.2f}GB"
        )

    def _mem_reset_peak(self):
        if self.memory_debug:
            torch.cuda.reset_peak_memory_stats()
    
    def setup_model(self):
        """Initialize Wan2.2 model with LoRA."""
        logger.info("Setting up Wan2.2 model...")
        self._mem("before model load")

        # Load base model config
        task = self.config['model']['task']
        wan_config = WAN_CONFIGS[task]

        # Initialize model — each rank loads to its own GPU.
        self.model = WanTI2V(
            config=wan_config,
            checkpoint_dir=self.config['model']['base_model_path'],
            device_id=self.local_rank,
            rank=self.rank,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=self.config['inference']['t5_cpu'],
            init_on_cpu=False,
            convert_model_dtype=self.config['inference']['convert_model_dtype']
        )

        self._mem("after WanTI2V init (DiT + T5 + VAE loaded)")

        # Expand input projection for channel-concat conditioning:
        # x and y are concatenated along channels in WanModel.forward when y is passed.
        # TI2V checkpoints may ship with patch_embedding.in_channels = Cz, while concat needs 2*Cz.
        self._expand_patch_embedding_for_concat_condition()
        
        # Configure LoRA
        target_modules = resolve_lora_target_modules(
            self.model.model,
            self.config['model']['target_modules'],
        )
        logger.info(f"Resolved LoRA target modules: {target_modules}")
        lora_config = LoraConfig(
            r=self.config['model']['lora_rank'],
            lora_alpha=self.config['model']['lora_alpha'],
            lora_dropout=self.config['model']['lora_dropout'],
            target_modules=target_modules,
            bias="none",
            modules_to_save=["patch_embedding"],
        )
        
        # Apply LoRA to the DiT model
        logger.info("Applying LoRA to DiT model...")
        self.model.model = get_peft_model(self.model.model, lora_config)

        # Keep input projection trainable so the newly-added conditioning channels can learn.
        base_model = self.model.model.get_base_model()
        for p in base_model.patch_embedding.parameters():
            p.requires_grad = True

        self._mem("after LoRA applied")

        # ── FSDP wrapping ────────────────────────────────────────────────────
        # Must happen BEFORE gradient checkpointing when using FSDP.
        # Applying grad-ckpt first via manual block.forward patching causes a
        # dtype mismatch: torch.utils.checkpoint recomputes outside FSDP's
        # forward hook, so FSDP's MixedPrecision cast (float32→bfloat16) is
        # not re-applied during recompute — weights are bfloat16 but activations
        # are still float32 → RuntimeError in the LoRA linear layers.
        if self.use_fsdp:
            sharding_str = self.config['training'].get('fsdp_sharding_strategy', 'FULL_SHARD')
            sharding_strategy = getattr(ShardingStrategy, sharding_str)
            base_model = self.model.model.get_base_model()
            fsdp_policy = partial(
                lambda_auto_wrap_policy,
                lambda_fn=lambda m: m in base_model.blocks,
            )
            self.model.model = FSDP(
                self.model.model,
                sharding_strategy=sharding_strategy,
                auto_wrap_policy=fsdp_policy,
                # No param_dtype: keep weights in float32 so FSDP doesn't fight
                # the model's own internal dtype contracts (WanAttentionBlock has
                # explicit `assert e.dtype == float32` and `.float()` casts that
                # break when param_dtype=bfloat16 + cast_forward_inputs=True).
                # Mixed-precision compute is handled via torch.autocast in
                # compute_loss instead, which respects the model's internal casts.
                # reduce_dtype=bfloat16 keeps gradient all-reduce efficient.
                mixed_precision=MixedPrecision(
                    param_dtype=None,
                    reduce_dtype=torch.bfloat16,
                    buffer_dtype=torch.float32,
                ),
                device_id=self.local_rank,
                sync_module_states=True,   # broadcast rank-0 weights to all ranks
                use_orig_params=True,      # required for LoRA / PEFT save_pretrained
            )
            logger.info(
                f"[rank{self.rank}] FSDP wrapping done "
                f"(strategy={sharding_str}, world_size={self.world_size})"
            )

        # ── Gradient checkpointing ───────────────────────────────────────────
        # The DiT has 30 transformer blocks. Without checkpointing, every
        # block's activations stay in VRAM for the backward pass (~2-3 GB each).
        # With checkpointing we trade one extra forward pass per block for
        # freeing those activations — typically cuts peak VRAM by ~50-60%.
        #
        # FSDP path: use apply_activation_checkpointing AFTER FSDP wrapping.
        #   This uses CheckpointWrapper which re-enters FSDP's forward context
        #   during recompute, ensuring parameters are re-gathered and re-cast to
        #   bfloat16 before the recomputed forward runs.
        #
        # Single-GPU path: manual block.forward patching (unchanged behaviour).
        if self.config['training'].get('gradient_checkpointing', False):
            if self.use_fsdp:
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                    apply_activation_checkpointing,
                    checkpoint_wrapper,
                    CheckpointImpl,
                )
                from wan.modules.model import WanAttentionBlock
                apply_activation_checkpointing(
                    self.model.model,
                    checkpoint_wrapper_fn=partial(
                        checkpoint_wrapper,
                        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                    ),
                    check_fn=lambda m: isinstance(m, WanAttentionBlock),
                )
                logger.info(
                    "Gradient checkpointing enabled via apply_activation_checkpointing "
                    "(FSDP-safe, NO_REENTRANT)"
                )
            else:
                dit = self.model.model.get_base_model()
                for block in dit.blocks:
                    block._original_forward = block.forward

                    def _make_ckpt_forward(mod):
                        def _ckpt_forward(*args, **kwargs):
                            def run(*a):
                                return mod._original_forward(*a, **kwargs)
                            return torch_ckpt.checkpoint(run, *args, use_reentrant=False)
                        return _ckpt_forward

                    block.forward = _make_ckpt_forward(block)
                logger.info(
                    f"Gradient checkpointing enabled for {len(dit.blocks)} DiT blocks"
                )
        else:
            logger.warning(
                "Gradient checkpointing is DISABLED — expect high VRAM usage. "
                "Set training.gradient_checkpointing=true if you hit OOM."
            )

        if self.is_main:
            self.model.model.print_trainable_parameters()
        trainable_params = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.model.parameters())
        logger.info(f"[rank{self.rank}] Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

        return self.model

    def _expand_patch_embedding_for_concat_condition(self):
        """
        Expand Conv3d input channels from C -> 2C so channel-concat conditioning
        (passing y to WanModel.forward) is valid.

        Initialization:
        - first C channels: copy pretrained weights
        - second C channels: zero init
        """
        patch = self.model.model.patch_embedding
        old_in = int(patch.in_channels)
        new_in = old_in * 2

        # If already expanded, do nothing.
        if old_in == new_in // 2 and patch.in_channels == new_in:
            return
        if old_in % 2 == 0 and patch.in_channels == old_in and old_in * 2 == new_in:
            # proceed
            pass

        new_patch = nn.Conv3d(
            in_channels=new_in,
            out_channels=patch.out_channels,
            kernel_size=patch.kernel_size,
            stride=patch.stride,
            padding=patch.padding,
            dilation=patch.dilation,
            groups=patch.groups,
            bias=patch.bias is not None,
            padding_mode=patch.padding_mode,
            device=patch.weight.device,
            dtype=patch.weight.dtype,
        )

        with torch.no_grad():
            new_patch.weight.zero_()
            new_patch.weight[:, :old_in].copy_(patch.weight)
            if patch.bias is not None:
                new_patch.bias.copy_(patch.bias)

        self.model.model.patch_embedding = new_patch
        logger.info(
            "Expanded patch_embedding in_channels for concat conditioning: "
            f"{old_in} -> {new_in} (new half zero-initialized)"
        )

    def _get_num_train_timesteps(self):
        if self._num_train_timesteps is None:
            self._num_train_timesteps = int(self.model.num_train_timesteps)
        return self._num_train_timesteps

    def _compute_seq_len_from_latent(self, latent):
        """
        latent: [Cz, Fz, Hz, Wz]
        """
        _, fz, hz, wz = latent.shape
        ph, pw = self.model.patch_size[1], self.model.patch_size[2]
        seq_len = math.ceil((hz * wz) / (ph * pw) * fz / self.model.sp_size) * self.model.sp_size
        return int(seq_len)

    def _supports_channel_concat_condition(self, latent_channels):
        in_ch = int(self.model.model.patch_embedding.in_channels)
        return in_ch == int(latent_channels) * 2

    def _encode_video_batch_to_latents(self, video_btchw):
        """
        video_btchw: [B, T, C, H, W] in [0,1]
        returns: list of [Cz, Fz, Hz, Wz]
        """
        latents = []
        if self.debug_breakpoint or os.getenv("WAN_LORA_DEBUG_BREAKPOINT", "0") == "1":
            breakpoint()
        with torch.no_grad():
            for i in range(video_btchw.size(0)):
                # VAE expects [C, F, H, W] and normalized to [-1, 1].
                v = video_btchw[i].permute(1, 0, 2, 3).to(self.device)  # [C, T, H, W]
                v = v.mul(2.0).sub(1.0)
                z = self.model.vae.encode([v])[0]
                latents.append(z.to(self.device, dtype=torch.float32))
        return latents

    def _encode_text_single(self, prompt: str):
        """
        Encode one prompt string → context tensor on CPU (float32).
        Called only by _build_text_cache; never called per-step.
        """
        with torch.no_grad():
            if not self.model.t5_cpu:
                self.model.text_encoder.model.to(self.device)
                ctx = self.model.text_encoder([prompt], self.device)
                ctx = [t.cpu().float() for t in ctx]
            else:
                ctx = self.model.text_encoder([prompt], torch.device('cpu'))
                ctx = [t.float() for t in ctx]
        return ctx[0]  # single tensor [seq_len, 4096]

    def _build_text_cache(self, datasets):
        """
        Pre-encode every unique instruction string that appears in the given
        datasets.  Results are kept on CPU and moved to GPU per-step, so the
        T5 model itself is never called during training.

        In the mini dataset all samples share one instruction, so this runs
        T5 exactly once.  In medium/full datasets the number of unique
        instructions is still small relative to the number of training steps.
        """
        unique_prompts = set()
        for ds in datasets:
            for sample in ds.samples:
                unique_prompts.add(sample['instruction'])

        logger.info(
            f"Pre-encoding {len(unique_prompts)} unique instruction(s) with T5 "
            f"(runs once — cached for all training steps)..."
        )
        self._text_cache = {}
        for i, prompt in enumerate(sorted(unique_prompts)):
            self._text_cache[prompt] = self._encode_text_single(prompt)
            logger.info(f"  [{i+1}/{len(unique_prompts)}] cached: '{prompt[:80]}'")

        logger.info("T5 cache built. T5 will not run again during training.")

    def _encode_text(self, prompts):
        """
        prompts: list[str]
        returns context list for WanModel forward (tensors on GPU, float32).

        Serves from _text_cache if available (built once before training),
        otherwise falls back to live T5 inference (e.g. during preview generation).
        """
        if hasattr(self, '_text_cache'):
            context = []
            for p in prompts:
                if p not in self._text_cache:
                    # Unseen prompt at inference time — encode and cache it
                    logger.warning(f"Cache miss for prompt '{p[:60]}' — encoding now")
                    self._text_cache[p] = self._encode_text_single(p)
                context.append(self._text_cache[p].to(self.device).float())
            return context

        # Fallback: live T5 inference (only hit before _build_text_cache is called)
        with torch.no_grad():
            if not self.model.t5_cpu:
                self.model.text_encoder.model.to(self.device)
                context = self.model.text_encoder(prompts, self.device)
            else:
                context = self.model.text_encoder(prompts, torch.device('cpu'))
                context = [t.to(self.device) for t in context]
        context = [t.float() for t in context]
        return context
    
    def setup_data(self):
        """Setup data loaders."""
        logger.info("Setting up data loaders...")
        
        data_config = self.config['data']
        
        # Create datasets
        train_dataset = DittoStyleDataset(
            metadata_path=data_config['train_metadata'],
            data_dir=data_config['data_dir'],
            num_frames=data_config['num_frames'],
            resolution=tuple(data_config['resolution'])
        )
        
        val_dataset = DittoStyleDataset(
            metadata_path=data_config['val_metadata'],
            data_dir=data_config['data_dir'],
            num_frames=data_config['num_frames'],
            resolution=tuple(data_config['resolution'])
        )
        self.val_dataset = val_dataset

        # Pre-encode all unique instructions once so T5 is never called per-step
        self._build_text_cache([train_dataset, val_dataset])

        # Create data loaders
        train_num_workers = int(self.config['training'].get('num_workers', 0))
        val_num_workers = int(self.config['training'].get('val_num_workers', train_num_workers))

        # DistributedSampler ensures each rank sees a disjoint subset of samples.
        # shuffle=False on the loader because the sampler handles shuffling.
        if self.use_fsdp:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=True,
            )
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                drop_last=False,
            )
            train_shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            train_shuffle = True

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=train_shuffle,
            sampler=train_sampler,
            num_workers=train_num_workers,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            sampler=val_sampler,
            num_workers=val_num_workers,
            pin_memory=True,
        )

        # Store sampler so train_epoch can call set_epoch for proper shuffling.
        self.train_sampler = train_sampler
        
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Val samples: {len(val_dataset)}")
        
        return train_loader, val_loader

    def _load_source_image_from_video(self, rel_video_path):
        """
        Read first frame from source video and return PIL RGB image.
        """
        full_path = self.val_dataset.resolve_video_path(rel_video_path)
        if full_path is None:
            raise FileNotFoundError(f"Cannot resolve source video path: {rel_video_path}")
        cap = cv2.VideoCapture(str(full_path))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Failed to read first frame from {full_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)

    def _log_video_to_tensorboard(self, tag, vid_tensor, global_step, fps=16):
        """
        Log a video tensor to TensorBoard.

        vid_tensor: [C, T, H, W] float in [0, 1]
        Tries add_video first; falls back to a frame grid image if moviepy is
        not compatible (e.g. moviepy >= 2.0 dropped moviepy.editor).
        """
        # [C, T, H, W] -> [1, T, C, H, W] for add_video
        tb_vid = vid_tensor.permute(1, 0, 2, 3).unsqueeze(0)
        try:
            self.writer.add_video(tag, tb_vid, global_step=global_step, fps=fps)
        except Exception:
            # Fallback: log evenly-spaced frames as a horizontal image strip
            frames = vid_tensor.permute(1, 2, 3, 0).mul(255).byte().numpy()  # [T, H, W, C]
            T = frames.shape[0]
            indices = np.linspace(0, T - 1, min(8, T), dtype=int)
            grid = np.concatenate([frames[j] for j in indices], axis=1)  # [H, W*N, C]
            grid_tensor = torch.from_numpy(grid).permute(2, 0, 1).float().div(255)
            self.writer.add_image(tag, grid_tensor, global_step=global_step)

    def generate_val_previews(self, epoch):
        """
        Generate qualitative previews from validation samples.
        """
        # Preview generation calls model.generate() which runs a full FSDP forward.
        # All ranks must participate in FSDP all-gathers, so we cannot run inference
        # on rank 0 while other ranks wait at a barrier — that deadlocks and triggers
        # the NCCL watchdog timeout after 600 s. Skip previews entirely when FSDP is
        # active; run inference manually after training using inference_with_lora.py.
        if self.use_fsdp:
            if self.is_main:
                logger.info(
                    "Skipping val previews (FSDP active — run inference_with_lora.py "
                    "after training to generate previews)."
                )
            return

        preview_every = int(self.config['training'].get('preview_every_epochs', 1))
        if preview_every <= 0 or (epoch + 1) % preview_every != 0:
            return
        preview_n = int(self.config['training'].get('preview_num_samples', 5))
        preview_n = max(0, min(preview_n, len(self.val_dataset.samples)))
        if preview_n == 0:
            logger.info("Skipping val previews: no validation samples.")
            return

        infer_cfg = self.config.get('inference', {})
        sampling_steps = int(infer_cfg.get('preview_sampling_steps', infer_cfg.get('num_inference_steps', 30)))
        guide_scale = float(infer_cfg.get('guidance_scale', 5.0))
        offload_model = bool(infer_cfg.get('offload_model', True))
        seed = int(infer_cfg.get('seed', 42))
        width, height = self.config['data']['resolution']
        frame_num = int(self.config['data'].get('num_frames', 49))

        preview_dir = self.output_dir / "val_previews" / f"epoch_{epoch + 1:03d}"
        preview_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating {preview_n} val previews for epoch {epoch + 1} ...")
        was_training = self.model.model.training
        self.model.model.eval()
        saved_files = []
        preview_tensors = []
        with torch.no_grad():
            for i in range(preview_n):
                sample = self.val_dataset.samples[i]
                prompt = sample.get("instruction", "")
                try:
                    src_img = self._load_source_image_from_video(sample["source_path"])
                    video = self.model.generate(
                        input_prompt=prompt,
                        img=src_img,
                        size=(width, height),
                        frame_num=frame_num,
                        sample_solver='unipc',
                        sampling_steps=sampling_steps,
                        guide_scale=guide_scale,
                        seed=seed + i,
                        offload_model=offload_model,
                    )
                    out_path = preview_dir / f"sample_{i:02d}.mp4"
                    save_video(video, str(out_path))
                    saved_files.append(str(out_path))
                    # video is [C, T, H, W] float tensor in [-1,1]; normalize to [0,1]
                    preview_tensors.append(video.cpu().float().add(1).div(2).clamp(0, 1))
                except Exception as e:
                    logger.warning(f"Preview generation failed for val sample {i}: {e}")
        if was_training:
            self.model.model.train()

        if saved_files:
            logger.info(f"Saved {len(saved_files)} val previews to {preview_dir}")
            for i, (path, vid_tensor) in enumerate(zip(saved_files, preview_tensors)):
                self._log_video_to_tensorboard(f"val/preview_{i:02d}", vid_tensor, epoch)
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler."""
        logger.info("Setting up optimizer...")
        
        opt_config = self.config['optimizer']
        train_config = self.config['training']
        
        # Only optimize LoRA parameters
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.model.parameters()),
            lr=train_config['learning_rate'],
            weight_decay=opt_config['weight_decay'],
            betas=opt_config['betas'],
            eps=opt_config['eps']
        )
        
        # Cosine annealing scheduler
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=train_config['num_epochs'],
            eta_min=train_config['min_lr']
        )
        
        return optimizer, scheduler
    
    def compute_loss(self, source_video, edited_video, instruction):
        """
        Diffusion epsilon-prediction loss in latent space.

        source_video, edited_video: [B, T, C, H, W] in [0,1]
        instruction: list[str] of length B
        """
        should_debug = self.debug_loss and self._debug_loss_seen < self.debug_loss_steps
        t0 = time.perf_counter()
        if should_debug:
            self._debug(
                f"compute_loss start | source={tuple(source_video.shape)} "
                f"edited={tuple(edited_video.shape)} batch={source_video.size(0)}"
            )
        T = self._get_num_train_timesteps()
        if should_debug:
            self._debug(f"num_train_timesteps={T}")

        # 1) Encode source and target videos to latent space.
        self._mem("compute_loss start")
        t1 = time.perf_counter()
        source_latents = self._encode_video_batch_to_latents(source_video)
        self._mem("after source VAE encode")
        target_latents = self._encode_video_batch_to_latents(edited_video)
        self._mem("after target VAE encode")
        # Free the raw video tensors — latents are all we need from here on.
        del source_video, edited_video
        torch.cuda.empty_cache()
        self._mem("after del videos + empty_cache")
        batch_size = len(target_latents)
        use_source_concat = self._supports_channel_concat_condition(
            target_latents[0].shape[0]
        )
        if not use_source_concat:
            raise RuntimeError(
                "Channel-concat conditioning requested but patch_embedding is not expanded "
                f"(in_channels={self.model.model.patch_embedding.in_channels}, "
                f"latent_channels={target_latents[0].shape[0]})."
            )
        if should_debug:
            self._debug(
                f"vae encode done in {(time.perf_counter() - t1):.3f}s | "
                f"latent[0]={tuple(target_latents[0].shape)} | "
                f"use_source_concat={use_source_concat} | "
                f"patch_in_ch={self.model.model.patch_embedding.in_channels}"
            )

        # 2) Text conditioning.
        t2 = time.perf_counter()
        prompts = list(instruction)
        context = self._encode_text(prompts)
        self._mem("after T5 text encode")
        if should_debug:
            self._debug(f"text encode done in {(time.perf_counter() - t2):.3f}s | prompts={len(prompts)}")

        # 3) Random timestep per sample (flow matching: t in [1, T] as integer index).
        t3 = time.perf_counter()
        t_idx = torch.randint(
            low=1,
            high=T + 1,
            size=(batch_size,),
            device=self.device,
            dtype=torch.long,
        )
        if should_debug:
            self._debug(f"sampled timesteps in {(time.perf_counter() - t3):.3f}s")

        # 4) Flow matching forward process: x_t = (1 - sigma_t) * x0 + sigma_t * eps
        #    sigma_t = t / T,  velocity target = eps - x0
        t4 = time.perf_counter()
        noisy_latents = []
        velocity_targets = []
        for i in range(batch_size):
            x0 = target_latents[i]
            eps = torch.randn_like(x0)
            sigma_t = t_idx[i].float() / T
            x_t = (1.0 - sigma_t) * x0 + sigma_t * eps
            noisy_latents.append(x_t)
            velocity_targets.append(eps - x0)
        if should_debug:
            self._debug(f"flow matching forward build in {(time.perf_counter() - t4):.3f}s")
        self._mem("after noise build (noisy_latents ready)")

        # WanModel supports t as [B]; it internally expands to [B, seq_len].
        seq_len = self._compute_seq_len_from_latent(target_latents[0])
        t_for_model = t_idx.float()

        # 5) Predict velocity, conditioned on source latents + text instruction.
        t5 = time.perf_counter()
        # Cast inputs to match model parameter dtype (float32 for both FSDP and
        # single-GPU paths — FSDP no longer uses param_dtype=bfloat16).
        model_dtype = next(self.model.model.parameters()).dtype
        noisy_latents_model = [x.to(dtype=model_dtype) for x in noisy_latents]
        source_latents_model = [x.to(dtype=model_dtype) for x in source_latents]
        context = [c.to(dtype=model_dtype) for c in context]
        model_kwargs = dict(
            t=t_for_model,
            context=context,
            seq_len=seq_len,
        )
        model_kwargs["y"] = source_latents_model
        self._mem("before DiT forward")
        # torch.autocast lets linear layers (including LoRA) compute in bfloat16
        # while the model's own internal float32 assertions and .float() casts
        # remain respected — autocast only promotes ops that are safe to run in
        # lower precision, and explicit .float() casts inside the model override it.
        autocast_ctx = (
            torch.autocast('cuda', dtype=torch.bfloat16)
            if self.use_fsdp
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            eps_pred_list = self.model.model(noisy_latents_model, **model_kwargs)
        self._mem("after DiT forward")
        if should_debug:
            self._debug(
                f"model forward in {(time.perf_counter() - t5):.3f}s | seq_len={seq_len} "
                f"preds={len(eps_pred_list)}"
            )

        # 6) MSE flow matching velocity loss.
        t6 = time.perf_counter()
        loss = 0.0
        for v_pred, v_tgt in zip(eps_pred_list, velocity_targets):
            loss = loss + F.mse_loss(v_pred.float(), v_tgt.float())
        loss = loss / batch_size
        self._mem("after loss compute")
        if should_debug:
            self._debug(
                f"loss reduce in {(time.perf_counter() - t6):.3f}s | "
                f"total compute_loss={(time.perf_counter() - t0):.3f}s | loss={loss.item():.6f}"
            )
            self.writer.add_scalar(
                "timing/compute_loss_total_s",
                time.perf_counter() - t0,
                self.global_step
            )
            self._debug_loss_seen += 1
        return loss
    
    def train_epoch(self, train_loader, optimizer, epoch):
        """Train for one epoch."""
        # DistributedSampler must know the epoch for correct per-epoch shuffling.
        if self.use_fsdp and self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

        self.model.model.train()

        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not self.is_main)

        for step, batch in enumerate(progress_bar):
            # Keep videos on CPU — _encode_video_batch_to_latents moves one
            # sample at a time, so there's no need to hold the full [B,T,C,H,W]
            # tensor on GPU throughout the forward pass.
            source_video = batch['source_video']
            edited_video = batch['edited_video']
            instruction = batch['instruction']

            # Reset peak stats each step so peak reflects THIS step only
            self._mem_reset_peak()

            # Forward pass
            loss = self.compute_loss(source_video, edited_video, instruction)

            # Backward pass
            self._mem("before backward")
            loss.backward()
            self._mem("after backward")

            # Gradient accumulation
            if (step + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                # With FSDP, clip_grad_norm_ must be called on the FSDP module
                # itself so it can access sharded parameters across all ranks.
                if self.use_fsdp:
                    self.model.model.clip_grad_norm_(
                        self.config['training']['max_grad_norm']
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.model.parameters(),
                        self.config['training']['max_grad_norm']
                    )
                optimizer.step()
                optimizer.zero_grad()
                self._mem("after optimizer step")

            # Logging — only rank 0 writes to TensorBoard.
            total_loss += loss.item()
            if self.is_main:
                progress_bar.set_postfix({'loss': loss.item()})
                self.writer.add_scalar("train/loss_step", loss.item(), self.global_step)
                self.writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], self.global_step)
            self.global_step += 1

            # Save checkpoint (rank 0 does the actual writing; all ranks must
            # participate in FSDP state-dict gathering regardless).
            if (step + 1) % self.config['training']['save_steps'] == 0:
                self.save_checkpoint(epoch, step)

        avg_loss = total_loss / len(train_loader)
        logger.info(f"[rank{self.rank}] Epoch {epoch} - Average loss: {avg_loss:.4f}")
        if self.is_main:
            self.writer.add_scalar("train/loss_epoch", avg_loss, epoch)

        return avg_loss
    
    def validate(self, val_loader):
        """Run validation."""
        self.model.model.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", disable=not self.is_main):
                source_video = batch['source_video']
                edited_video = batch['edited_video']
                instruction = batch['instruction']

                loss = self.compute_loss(source_video, edited_video, instruction)
                total_loss += loss.item()
                num_batches += 1

        # Average across ranks so the reported loss is the global average.
        if self.use_fsdp:
            loss_tensor = torch.tensor([total_loss, float(num_batches)], device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = (loss_tensor[0] / loss_tensor[1]).item()
        else:
            avg_loss = total_loss / max(num_batches, 1)

        if self.is_main:
            logger.info(f"Validation loss: {avg_loss:.4f}")
            self.writer.add_scalar("val/loss_epoch", avg_loss, self.current_epoch)

        return avg_loss
    
    def save_checkpoint(self, epoch, step):
        """Save model checkpoint.

        With FSDP all ranks must participate in state_dict gathering (it's a
        collective operation), but only rank 0 writes the files to disk.
        """
        checkpoint_path = self.output_dir / f"checkpoint-epoch{epoch}-step{step}"

        if self.use_fsdp:
            # FullStateDictConfig with rank0_only=True gathers all shards onto
            # rank-0 CPU; other ranks receive an empty dict.
            save_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(
                self.model.model,
                StateDictType.FULL_STATE_DICT,
                save_cfg,
            ):
                full_state = self.model.model.state_dict()

            if self.is_main:
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                # Persist only LoRA / modules_to_save parameters.
                # PEFT's loader expects modules_to_save keys WITHOUT the
                # '.modules_to_save.<adapter_name>' segment — it strips that
                # part on save and re-inserts it on load.
                # e.g. FSDP full key:
                #   base_model.model.patch_embedding.modules_to_save.default.weight
                # must become:
                #   base_model.model.patch_embedding.weight
                import re as _re
                lora_state = {}
                for k, v in full_state.items():
                    if 'lora_' in k:
                        lora_state[k] = v
                    elif 'modules_to_save' in k:
                        # strip '.modules_to_save.<adapter_name>' from key
                        norm_k = _re.sub(r'\.modules_to_save\.[^.]+', '', k)
                        lora_state[norm_k] = v
                torch.save(lora_state, checkpoint_path / "adapter_model.bin")
                # Save PEFT adapter config so the checkpoint is loadable with
                # PeftModel.from_pretrained. to_dict() can return Python sets
                # (e.g. target_modules) which are not JSON-serializable; convert
                # them to sorted lists before dumping.
                try:
                    import json as _json

                    def _make_serializable(obj):
                        if isinstance(obj, set):
                            return sorted(list(obj))
                        if isinstance(obj, dict):
                            return {k: _make_serializable(v) for k, v in obj.items()}
                        if isinstance(obj, (list, tuple)):
                            return [_make_serializable(v) for v in obj]
                        if hasattr(obj, 'value'):  # enum
                            return obj.value
                        return obj

                    for adapter_name, peft_cfg in self.model.model.peft_config.items():
                        cfg_dict = _make_serializable(peft_cfg.to_dict())
                        with open(checkpoint_path / "adapter_config.json", 'w') as f:
                            _json.dump(cfg_dict, f, indent=2)
                except Exception as cfg_err:
                    logger.warning(f"Could not save adapter_config.json: {cfg_err}")
                logger.info(f"[rank0] 💾 Saved FSDP checkpoint to {checkpoint_path}")
        else:
            if self.is_main:
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                self.model.model.save_pretrained(checkpoint_path)
                logger.info(f"💾 Saved checkpoint to {checkpoint_path}")
    
    def train(self):
        """Main training loop."""
        logger.info("=" * 80)
        logger.info("Starting LoRA Training for Wan2.2 Style Transfer")
        logger.info("=" * 80)
        
        # Setup
        self.setup_model()
        train_loader, val_loader = self.setup_data()
        optimizer, scheduler = self.setup_optimizer()
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(self.config['training']['num_epochs']):
            self.current_epoch = epoch
            logger.info(f"\n{'='*80}")
            logger.info(f"Epoch {epoch + 1}/{self.config['training']['num_epochs']}")
            logger.info(f"{'='*80}")
            
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, epoch)
            
            # Validate
            eval_every_epochs = max(
                1,
                self.config['training']['eval_steps'] // max(1, len(train_loader)),
            )
            if (epoch + 1) % eval_every_epochs == 0:
                val_loss = self.validate(val_loader)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(epoch, "best")
                    logger.info(f"🌟 New best model! Val loss: {val_loss:.4f}")
            
            # Step scheduler
            scheduler.step()

            # Qualitative preview generation
            self.generate_val_previews(epoch)
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, "latest")
        
        # Save final model — reuse save_checkpoint logic.
        self.save_checkpoint("final", "last")
        logger.info(f"\n✅ Training complete! Final model saved under {self.output_dir / 'checkpoint-epochfinal-steplast'}")
        if self.is_main:
            self.writer.flush()
            self.writer.close()


def load_config(config_path):
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def validate_data_paths(config):
    """
    Ensure all data paths point to /mnt/localssd and actually exist.
    Fails fast before any model loading so the error is immediately obvious.
    """
    DATA_ROOT = "/mnt/localssd"
    data_cfg = config.get('data', {})
    checks = {
        'data_dir':       data_cfg.get('data_dir', ''),
        'train_metadata': data_cfg.get('train_metadata', ''),
        'val_metadata':   data_cfg.get('val_metadata', ''),
    }

    errors = []
    for key, path in checks.items():
        if not path:
            errors.append(f"  - data.{key} is not set in config")
            continue
        if not path.startswith(DATA_ROOT):
            errors.append(
                f"  - data.{key} = '{path}'\n"
                f"    Expected path under {DATA_ROOT}/ (not the repo ./data/ folder)"
            )
        elif not Path(path).exists():
            errors.append(
                f"  - data.{key} = '{path}'\n"
                f"    Path does not exist — run the download/preprocess scripts first"
            )

    if errors:
        logger.error(
            "❌ Data path validation failed:\n" + "\n".join(errors) + "\n\n"
            "  Fix: update your config so all data.* paths point to "
            f"{DATA_ROOT}/datasets/ and re-run the download script."
        )
        raise SystemExit(1)


def main():
    # Reduce CUDA allocator fragmentation — helps when many tensors of varying
    # sizes are allocated/freed (e.g. per-layer activations during grad ckpt).
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

    parser = argparse.ArgumentParser(
        description="Train LoRA for Wan2.2 style transfer"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/ghibli_style_lora.yaml",
        help="Path to training configuration YAML"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )

    args = parser.parse_args()

    # ── Distributed init ─────────────────────────────────────────────────────
    # torchrun sets RANK / LOCAL_RANK / WORLD_SIZE automatically.
    # When launched with plain `python` these vars are absent → single-GPU mode.
    is_distributed = 'RANK' in os.environ and int(os.environ.get('WORLD_SIZE', 1)) > 1
    if is_distributed:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    torch.cuda.set_device(local_rank)

    # Load config — all ranks load independently (cheap, avoids a broadcast).
    config = load_config(args.config)

    # Data-path validation only on rank 0 to avoid duplicate error spam.
    if rank == 0:
        validate_data_paths(config)
    if is_distributed:
        dist.barrier()  # wait for rank-0 validation before proceeding

    # Check CUDA
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available! Training requires GPU.")
        if is_distributed:
            dist.destroy_process_group()
        return

    if rank == 0:
        logger.info(f"🎮 Using device: {torch.cuda.get_device_name(local_rank)}")
        logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(local_rank).total_memory / 1e9:.2f} GB")
        if is_distributed:
            logger.info(f"🌐 Distributed: {world_size} GPUs, FSDP={'enabled' if config.get('training', {}).get('fsdp') else 'disabled'}")

    # Create trainer and start training
    trainer = LoRATrainer(config, rank=rank, world_size=world_size, local_rank=local_rank)
    trainer.train()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
