#!/bin/bash
# Quick start script for Wan2.2 Style Transfer LoRA Training

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   Wan2.2 Style Transfer LoRA Training - Quick Start           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if we're in the right directory
if [ ! -f "generate.py" ]; then
    echo "❌ Error: Please run this script from the Wan2.2 root directory"
    exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python
if ! command_exists python3; then
    echo "❌ Error: Python 3 is required"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"

# Check CUDA
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "❌ Error: CUDA is required for training"
    exit 1
fi

GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
GPU_MEM=$(python3 -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}')")
echo "✓ GPU found: $GPU_NAME (${GPU_MEM}GB)"

# Parse arguments
MODE=${1:-"mini"}  # mini, medium, full, or train

case $MODE in
    mini)
        echo ""
        echo "📦 Mode: MINI (Quick test with small dataset)"
        echo "   - Downloads a trainable mini subset (matched source+style pairs)"
        echo "   - Processes 30 samples"
        echo "   - Trains for 5 epochs (~2 hours)"
        echo ""
        read -p "Continue? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 0
        fi
        
        # Download mini dataset
        echo ""
        echo "Step 1/4: Downloading mini test dataset..."
        python3 scripts/download_ditto_style.py --mini --mini_pairs 30 --output_dir /mnt/localssd/datasets/ditto_mini

        # Preprocess
        echo ""
        echo "Step 2/4: Preprocessing data..."
        python3 scripts/preprocess_ditto.py \
            --data_dir /mnt/localssd/datasets/ditto_mini \
            --output_dir /mnt/localssd/datasets/processed \
            --max_samples 30
        
        # Update config for mini training
        echo ""
        echo "Step 3/4: Configuring for mini training..."
        cat > configs/ghibli_style_lora_mini.yaml <<EOF
# Mini training configuration (for testing)
model:
  base_model_path: "/mnt/localssd/Wan2.2-TI2V-5B"
  task: "ti2v-5B"
  lora_rank: 32
  lora_alpha: 32
  lora_dropout: 0.1
  target_modules: ["q", "k", "v", "o"]
  param_dtype: "bfloat16"

training:
  output_dir: "./checkpoints/ghibli_style_lora_mini"
  logging_dir: "./logs/ghibli_style_lora_mini"
  num_epochs: 5
  batch_size: 1
  gradient_accumulation_steps: 4
  learning_rate: 5.0e-5
  max_grad_norm: 1.0
  lr_scheduler: "cosine"
  warmup_steps: 50
  min_lr: 1.0e-6
  save_steps: 50
  save_total_limit: 3
  eval_steps: 50
  eval_samples: 2
  logging_steps: 10
  report_to: "tensorboard"
  gradient_checkpointing: true
  mixed_precision: "bf16"

data:
  data_dir: "/mnt/localssd/datasets/ditto_mini"
  train_metadata: "/mnt/localssd/datasets/processed/train_metadata.json"
  val_metadata: "/mnt/localssd/datasets/processed/val_metadata.json"
  resolution: [640, 352]  # Smaller for faster training
  num_frames: 49          # Fewer frames
  fps: 24

optimizer:
  type: "AdamW"
  weight_decay: 0.01
  betas: [0.9, 0.999]
  eps: 1.0e-8

inference:
  lora_scale: 0.8
  num_inference_steps: 30
  guidance_scale: 5.0
  seed: 42
  offload_model: true
  convert_model_dtype: true
  t5_cpu: true
EOF
        
        # Train
        echo ""
        echo "Step 4/4: Starting mini training..."
        echo "   This will take ~2 hours"
        echo ""
        python3 scripts/train_style_lora.py --config configs/ghibli_style_lora_mini.yaml
        
        echo ""
        echo "✅ Mini training complete!"
        echo "   Checkpoint: checkpoints/ghibli_style_lora_mini/final"
        ;;
    
    medium)
        echo ""
        echo "📦 Mode: MEDIUM (~10k real style-transfer pairs, full-fledged training)"
        echo "   - Downloads metadata, selects 10k pairs, fetches only those videos (~20-40GB)"
        echo "   - Trains for 15 epochs at full 1280x704 resolution"
        echo "   - Estimated wall time: ~12-18 hours on A100"
        echo ""
        read -p "Continue? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 0
        fi

        # Check HuggingFace login
        if ! huggingface-cli whoami >/dev/null 2>&1; then
            echo "❌ Please login to HuggingFace first:"
            echo "   huggingface-cli login"
            exit 1
        fi

        # Download medium dataset
        echo ""
        echo "Step 1/4: Downloading ~10k style-transfer video pairs..."
        python3 scripts/download_ditto_style.py \
            --medium \
            --medium_pairs 10000 \
            --output_dir /mnt/localssd/datasets/ditto_medium

        # Preprocess
        echo ""
        echo "Step 2/4: Preprocessing data..."
        python3 scripts/preprocess_ditto.py \
            --data_dir /mnt/localssd/datasets/ditto_medium \
            --output_dir /mnt/localssd/datasets/processed_medium \
            --val_ratio 0.05

        # Write config
        echo ""
        echo "Step 3/4: Writing medium training config..."
        cat > configs/ghibli_style_lora_medium.yaml <<EOF
# Medium training configuration (~10k real style-transfer pairs)
# Full-fledged training — not a smoke test.
# Sits between mini (30 pairs, 5 epochs) and full (300k pairs, 30 epochs).

model:
  base_model_path: "/mnt/localssd/Wan2.2-TI2V-5B"
  task: "ti2v-5B"
  lora_rank: 16
  lora_alpha: 16
  lora_dropout: 0.05
  target_modules: ["q", "k", "v", "o"]
  param_dtype: "bfloat16"

training:
  output_dir: "./checkpoints/ghibli_style_lora_medium"
  logging_dir: "./logs/ghibli_style_lora_medium"
  num_epochs: 15
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

        # Train
        echo ""
        echo "Step 4/4: Starting medium training..."
        echo "   Monitor with: tensorboard --logdir logs/ghibli_style_lora_medium"
        echo ""
        python3 scripts/train_style_lora.py --config configs/ghibli_style_lora_medium.yaml

        echo ""
        echo "✅ Medium training complete!"
        echo "   Checkpoint: checkpoints/ghibli_style_lora_medium/final"
        ;;

    full)
        echo ""
        echo "📦 Mode: FULL (Complete training pipeline)"
        echo "   - Downloads ~350GB of style videos"
        echo "   - Processes ~300k samples"
        echo "   - Trains for 30 epochs (~60 hours on A100)"
        echo ""
        read -p "Continue? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 0
        fi
        
        # Check HuggingFace login
        if ! huggingface-cli whoami >/dev/null 2>&1; then
            echo "❌ Please login to HuggingFace first:"
            echo "   huggingface-cli login"
            exit 1
        fi
        
        # Download full dataset
        echo ""
        echo "Step 1/4: Downloading full style dataset (~350GB)..."
        echo "   This will take 4-8 hours depending on your connection"
        python3 scripts/download_ditto_style.py --output_dir /mnt/localssd/datasets/ditto

        # Preprocess
        echo ""
        echo "Step 2/4: Preprocessing data..."
        python3 scripts/preprocess_ditto.py \
            --data_dir /mnt/localssd/datasets/ditto \
            --output_dir /mnt/localssd/datasets/processed \
            --val_ratio 0.1
        
        # Train
        echo ""
        echo "Step 3/4: Starting full training..."
        echo "   This will take ~60 hours on A100"
        echo "   Monitor with: tensorboard --logdir logs/ghibli_style_lora"
        echo ""
        python3 scripts/train_style_lora.py --config configs/ghibli_style_lora.yaml
        
        echo ""
        echo "✅ Training complete!"
        echo "   Checkpoint: checkpoints/ghibli_style_lora/final"
        ;;
    
    train)
        echo ""
        echo "🚀 Mode: TRAIN (Resume or start training)"
        echo ""
        
        # Check if data is ready
        if [ ! -f "data/processed/train_metadata.json" ]; then
            echo "❌ Error: Preprocessed data not found"
            echo "   Please run preprocessing first:"
            echo "   ./scripts/quick_start.sh mini"
            exit 1
        fi
        
        # Check if config exists
        if [ ! -f "configs/ghibli_style_lora.yaml" ]; then
            echo "❌ Error: Config file not found"
            exit 1
        fi
        
        # Start training
        python3 scripts/train_style_lora.py --config configs/ghibli_style_lora.yaml
        ;;
    
    *)
        echo "Usage: ./scripts/quick_start.sh [mode]"
        echo ""
        echo "Modes:"
        echo "  mini   - Pipeline smoke test, 30 pairs, 5 epochs (~2 hours)"
        echo "  medium - 10k real pairs, 15 epochs, full resolution (~12-18 hours)"
        echo "  full   - Complete training, ~300k pairs, 30 epochs (~60 hours)"
        echo "  train  - Resume or start training (assumes data is ready)"
        echo ""
        echo "Examples:"
        echo "  ./scripts/quick_start.sh mini    # Quick smoke test"
        echo "  ./scripts/quick_start.sh medium  # Medium-scale training"
        echo "  ./scripts/quick_start.sh full    # Full training"
        echo "  ./scripts/quick_start.sh train   # Just train"
        exit 1
        ;;
esac

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    🎉 All Done!                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
