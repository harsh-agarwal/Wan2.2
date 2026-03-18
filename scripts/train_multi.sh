#!/bin/bash
# Multi-GPU FSDP training script for Wan2.2 Style Transfer LoRA
# Replicates the "medium" config from quick_start.sh but launches with
# torchrun so every GPU gets an FSDP shard of the DiT.
#
# Usage:
#   ./scripts/train_multi.sh              # auto-detect all GPUs
#   ./scripts/train_multi.sh --gpus 4     # pin to 4 GPUs

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   Wan2.2 Style Transfer LoRA — Multi-GPU FSDP Training        ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# ── Sanity checks ────────────────────────────────────────────────────────────
if [ ! -f "generate.py" ]; then
    echo "❌ Error: run this script from the Wan2.2 root directory"
    exit 1
fi

if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "❌ Error: CUDA is required for training"
    exit 1
fi

if ! huggingface-cli whoami >/dev/null 2>&1; then
    echo "❌ Please login to HuggingFace first:"
    echo "   huggingface-cli login"
    exit 1
fi

# ── Parse arguments ──────────────────────────────────────────────────────────
NUM_GPUS=""
CONFIG_ONLY=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            NUM_GPUS="$2"; shift 2 ;;
        --config-only)
            CONFIG_ONLY=1; shift ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--gpus N] [--config-only]"
            exit 1 ;;
    esac
done

# Auto-detect GPU count if not pinned
if [ -z "$NUM_GPUS" ]; then
    NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
fi

if [ "$NUM_GPUS" -lt 2 ]; then
    echo "⚠️  Only ${NUM_GPUS} GPU detected."
    echo "   FSDP will still work on 1 GPU but offers no memory/speed benefit."
    echo "   Consider using scripts/quick_start.sh medium for single-GPU runs."
    read -p "   Continue anyway? (y/n) " -n 1 -r; echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then exit 0; fi
fi

echo "✓ Found ${NUM_GPUS} GPU(s)"
for i in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name($i))")
    GPU_MEM=$(python3 -c "import torch; print(f'{torch.cuda.get_device_properties($i).total_memory/1e9:.1f}')")
    echo "  GPU $i: $GPU_NAME (${GPU_MEM}GB)"
done
echo ""

# Effective batch size note
BATCH_PER_GPU=2
GRAD_ACCUM=8
EFFECTIVE_BATCH=$((BATCH_PER_GPU * GRAD_ACCUM * NUM_GPUS))
echo "📊 Batch config:"
echo "   Per-GPU batch size : $BATCH_PER_GPU"
echo "   Gradient accumulation: $GRAD_ACCUM"
echo "   Effective global batch: $EFFECTIVE_BATCH (${BATCH_PER_GPU} × ${GRAD_ACCUM} × ${NUM_GPUS} GPUs)"
echo ""

read -p "Continue with ${NUM_GPUS}-GPU FSDP medium training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then exit 0; fi

# ── Step 1: Download medium dataset ──────────────────────────────────────────
# The download script is fully idempotent:
#   - HuggingFace snapshot_download uses resume_download=True, so already-downloaded
#     shards are skipped automatically.
#   - Tar extraction is guarded by a .extracted marker file inside each category dir,
#     so already-extracted archives are not re-extracted.
#   - medium_selected.json (the curated pair list) is always regenerated from what
#     is actually on disk, which is required for preprocessing to find valid pairs.
# We therefore always run it — just like quick_start.sh does.
DATA_DIR="/mnt/localssd/datasets/ditto_medium"
PROCESSED_DIR="/mnt/localssd/datasets/processed_medium"
TRAIN_META="${PROCESSED_DIR}/train_metadata.json"
VAL_META="${PROCESSED_DIR}/val_metadata.json"

echo ""
echo "Step 1/4: Downloading ~10k style-transfer video pairs (resumes if interrupted)..."
python3 scripts/download_ditto_style.py \
    --medium \
    --medium_pairs 10000 \
    --output_dir "$DATA_DIR"

# ── Step 2: Preprocess ───────────────────────────────────────────────────────
# Skip only when both processed metadata files already exist AND extraction is
# confirmed complete (both .extracted markers are present).  An incomplete
# extraction would leave metadata pointing at files that don't exist yet.
SOURCE_EXTRACTED="${DATA_DIR}/videos/source/.extracted"
STYLE_EXTRACTED="${DATA_DIR}/videos/global_style2/.extracted"

echo ""
if [ -f "$TRAIN_META" ] && [ -f "$VAL_META" ] \
   && [ -f "$SOURCE_EXTRACTED" ] && [ -f "$STYLE_EXTRACTED" ]; then
    echo "Step 2/4: Videos extracted and processed metadata exists — skipping preprocessing."
else
    echo "Step 2/4: Preprocessing data..."
    python3 scripts/preprocess_ditto.py \
        --data_dir "$DATA_DIR" \
        --output_dir "$PROCESSED_DIR" \
        --val_ratio 0.05
fi

# ── Step 3: Write FSDP config ─────────────────────────────────────────────
echo ""
echo "Step 3/4: Writing multi-GPU FSDP training config..."

mkdir -p configs

cat > configs/ghibli_style_lora_medium_fsdp.yaml <<EOF
# Medium training configuration — multi-GPU FSDP variant
# Mirrors ghibli_style_lora_medium.yaml but enables FSDP sharding.
#
# Launch with:
#   torchrun --standalone --nproc_per_node=NUM_GPUS \\
#       scripts/train_style_lora.py \\
#       --config configs/ghibli_style_lora_medium_fsdp.yaml

model:
  base_model_path: "/mnt/localssd/Wan2.2-TI2V-5B"
  task: "ti2v-5B"
  lora_rank: 256
  lora_alpha: 256
  lora_dropout: 0.05
  target_modules: ["q", "k", "v", "o"]
  param_dtype: "bfloat16"

training:
  output_dir: "./checkpoints/ghibli_style_lora_medium_fsdp"
  logging_dir: "./logs/ghibli_style_lora_medium_fsdp"
  num_epochs: 15
  # Per-GPU batch size — effective global batch = batch_size × grad_accum × num_gpus
  batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-5
  max_grad_norm: 1.0
  lr_scheduler: "cosine"
  warmup_steps: 300
  min_lr: 1.0e-6
  save_steps: 500
  save_total_limit: 5
  eval_steps: 500
  eval_samples: 4
  logging_steps: 20
  report_to: "tensorboard"
  gradient_checkpointing: true
  mixed_precision: "bf16"
  num_workers: 8
  val_num_workers: 4
  preview_every_epochs: 2
  preview_num_samples: 5
  # ── FSDP settings ──────────────────────────────────────────────────────────
  fsdp: true
  # FULL_SHARD: maximum memory savings (all params sharded across GPUs)
  # SHARD_GRAD_OP: shard gradients+optimizer state only (faster, more VRAM)
  fsdp_sharding_strategy: "FULL_SHARD"

data:
  data_dir: "/mnt/localssd/datasets/ditto_medium"
  train_metadata: "/mnt/localssd/datasets/processed_medium/train_metadata.json"
  val_metadata: "/mnt/localssd/datasets/processed_medium/val_metadata.json"
  resolution: [1280, 704]
  num_frames: 49
  fps: 24

optimizer:
  type: "AdamW"
  weight_decay: 0.01
  betas: [0.9, 0.999]
  eps: 1.0e-8

inference:
  lora_scale: 0.8
  num_inference_steps: 50
  preview_sampling_steps: 20
  guidance_scale: 5.0
  seed: 42
  offload_model: false
  convert_model_dtype: false
  t5_cpu: false
EOF

echo "✓ Config written to configs/ghibli_style_lora_medium_fsdp.yaml"

if [ "$CONFIG_ONLY" -eq 1 ]; then
    echo ""
    echo "--config-only set. Skipping training launch."
    exit 0
fi

# ── Step 4: Launch with torchrun ──────────────────────────────────────────────
echo ""
echo "Step 4/4: Launching ${NUM_GPUS}-GPU FSDP training with torchrun..."
echo "   Monitor with: tensorboard --logdir logs/ghibli_style_lora_medium_fsdp"
echo ""

# PYTORCH_CUDA_ALLOC_CONF reduces allocator fragmentation across ranks.
# NCCL_DEBUG=WARN surfaces NCCL init/topology issues without flooding stdout.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN

torchrun \
    --standalone \
    --nproc_per_node="${NUM_GPUS}" \
    scripts/train_style_lora.py \
    --config configs/ghibli_style_lora_medium_fsdp.yaml

echo ""
echo "✅ Multi-GPU FSDP medium training complete!"
echo "   Checkpoint: checkpoints/ghibli_style_lora_medium_fsdp/final"
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    All Done!                                   ║"
echo "╚════════════════════════════════════════════════════════════════╝"
