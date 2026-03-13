# Wan2.2 Style Transfer LoRA Training Guide

This guide walks you through training a LoRA adapter for style transfer on Wan2.2 using the Ditto-1M dataset.

## 📋 Overview

**Goal:** Train a LoRA that can transfer videos to specific artistic styles (anime, cyberpunk, watercolor, etc.)

**Dataset:** Ditto-1M (1M video editing triplets, we'll use ~350GB style subset)

**Model:** Wan2.2-TI2V-5B (5B parameter text-image-to-video model)

**Training Time:** ~48-72 hours on 1x A100 or 2x RTX 4090

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install huggingface_hub peft datasets transformers accelerate
pip install opencv-python pandas pyyaml tensorboard
```

### Step 2: Download Dataset

**Option A: Mini test set (1GB, for quick testing)**
```bash
python scripts/download_ditto_style.py --mini
```

**Option B: Full style subset (350GB)**
```bash
# Login to HuggingFace first
huggingface-cli login

# Accept dataset terms at:
# https://huggingface.co/datasets/QingyanBai/Ditto-1M

# Download
python scripts/download_ditto_style.py --output_dir ./data/ditto
```

This will download:
- `videos/source/` - Original videos (~180GB)
- `videos/global_style1/` - Style transfer examples (~230GB)
- `videos/global_style2/` - More style examples (~120GB)
- `training_metadata/` - Metadata JSONs
- `csvs_for_DiffSynth/` - Pre-made training CSVs

**Download time:** 4-8 hours depending on connection

### Step 3: Preprocess Data

```bash
# Process the downloaded data
python scripts/preprocess_ditto.py \
    --data_dir ./data/ditto \
    --output_dir ./data/processed \
    --val_ratio 0.1

# For quick testing with limited samples:
python scripts/preprocess_ditto.py \
    --data_dir ./data/ditto \
    --output_dir ./data/processed \
    --max_samples 100 \
    --no_verify
```

This creates:
- `data/processed/train_metadata.json` - Training samples
- `data/processed/val_metadata.json` - Validation samples
- `data/processed/train.csv` - Training CSV
- `data/processed/val.csv` - Validation CSV

### Step 4: Configure Training

Edit `configs/ghibli_style_lora.yaml` to customize:
- LoRA rank/alpha (default: 64/64)
- Learning rate (default: 5e-5)
- Batch size and gradient accumulation
- Number of epochs (default: 30)

### Step 5: Train LoRA

```bash
# Start training
python scripts/train_style_lora.py --config configs/ghibli_style_lora.yaml

# Monitor training
tensorboard --logdir logs/ghibli_style_lora
```

**Expected training time:**
- 1x A100 (80GB): ~60 hours
- 2x RTX 4090: ~72 hours
- 8x A100: ~12 hours (with FSDP)

### Step 6: Test Trained LoRA

```bash
# Test with default prompts
python scripts/test_style_lora.py \
    --lora_path checkpoints/ghibli_style_lora/final

# Test with custom prompts
python scripts/test_style_lora.py \
    --lora_path checkpoints/ghibli_style_lora/final \
    --prompts "A cat walking in a garden" "Ocean waves at sunset"
```

---

## 📊 Dataset Details

### Ditto-1M Structure

```
data/ditto/
├── videos/
│   ├── source/              # Original videos (180GB)
│   ├── global_style1/       # Style transfer 1 (230GB)
│   └── global_style2/       # Style transfer 2 (120GB)
├── training_metadata/       # JSON metadata files
├── source_video_captions/   # Video captions
└── csvs_for_DiffSynth/      # Pre-made CSVs
```

### Style Categories in Ditto-1M

The dataset includes these style transfers:
- **Anime/Cartoon** - Studio Ghibli, anime, cartoon styles
- **Pixel Art** - 8-bit, 16-bit retro game aesthetics
- **Cyberpunk** - Neon, futuristic, blade runner style
- **Artistic** - Watercolor, oil painting, sketch, impressionist
- **Vintage** - Film noir, sepia, retro, vintage film
- **Abstract** - Geometric, surreal, artistic interpretations

### Sample Counts (Estimated)

- Total style samples: ~300,000
- Anime/cartoon: ~80,000
- Pixel art: ~40,000
- Cyberpunk: ~30,000
- Artistic styles: ~100,000
- Other: ~50,000

---

## ⚙️ Training Configuration

### Key Hyperparameters

```yaml
# LoRA configuration
lora_rank: 64              # Higher = more capacity, slower training
lora_alpha: 64             # Scaling factor (usually same as rank)
lora_dropout: 0.1          # Regularization

# Training
learning_rate: 5e-5        # Conservative for stable training
batch_size: 1              # Limited by GPU memory
gradient_accumulation: 8   # Effective batch size = 8
num_epochs: 30             # Adjust based on dataset size

# Memory optimization
gradient_checkpointing: true
mixed_precision: "bf16"
offload_model: true
t5_cpu: true
```

### Memory Requirements

| Setup | GPU | VRAM | Training Time |
|-------|-----|------|---------------|
| Single GPU | A100 80GB | ~75GB | 60 hours |
| Single GPU | RTX 4090 24GB | ~22GB* | 90 hours |
| Multi GPU | 2x A100 | ~40GB each | 30 hours |
| Multi GPU | 8x A100 | ~20GB each | 12 hours |

*With aggressive offloading (`offload_model=true`, `t5_cpu=true`, `convert_model_dtype=true`)

---

## 🎯 Training Tips

### 1. Start Small
```bash
# Test with 100 samples first
python scripts/preprocess_ditto.py --max_samples 100
python scripts/train_style_lora.py --config configs/ghibli_style_lora.yaml
```

### 2. Monitor Training
```bash
# Watch tensorboard
tensorboard --logdir logs/ghibli_style_lora --port 6006

# Check GPU usage
watch -n 1 nvidia-smi
```

### 3. Checkpoint Management
- Checkpoints saved every 500 steps
- Best model saved based on validation loss
- Keep last 5 checkpoints (configurable)

### 4. Hyperparameter Tuning

**If training is unstable:**
- Reduce learning rate: `5e-5` → `1e-5`
- Increase warmup steps: `500` → `1000`
- Reduce LoRA rank: `64` → `32`

**If overfitting:**
- Increase LoRA dropout: `0.1` → `0.2`
- Add more data augmentation
- Reduce number of epochs

**If underfitting:**
- Increase LoRA rank: `64` → `128`
- Increase learning rate: `5e-5` → `1e-4`
- Train for more epochs

---

## 🧪 Evaluation

### Metrics

1. **Visual Quality**
   - FID (Fréchet Inception Distance)
   - CLIP score similarity to target style

2. **Temporal Consistency**
   - Frame-to-frame SSIM
   - Optical flow consistency

3. **Style Adherence**
   - CLIP text-image similarity
   - Human evaluation (1-5 scale)

### Test Prompts

Use diverse prompts to test generalization:
```python
test_prompts = [
    # Nature scenes
    "A person walking through a forest, Studio Ghibli style",
    "Ocean waves crashing on rocks, anime style",
    "Mountains at sunrise, hand-drawn animation",
    
    # Urban scenes
    "A busy city street, cyberpunk neon style",
    "A cafe interior, watercolor painting style",
    
    # Characters
    "A cat sitting on a windowsill, Ghibli animation",
    "A person riding a bicycle, cartoon style",
]
```

---

## 📦 Sharing Your LoRA

### 1. Prepare Model Card

```markdown
# Wan2.2 [Your Style] LoRA

## Description
A LoRA adapter for Wan2.2-TI2V-5B trained on Ditto-1M dataset for [style] transfer.

## Training Details
- Base model: Wan2.2-TI2V-5B
- Dataset: Ditto-1M (style subset)
- Training samples: [X]
- Training time: [Y] hours on [GPU]
- LoRA rank: 64

## Usage
\`\`\`python
from wan.textimage2video import WanTI2V

model = WanTI2V(config, checkpoint_dir="./Wan2.2-TI2V-5B")
# TODO: Load LoRA
video = model.generate("Your prompt, [style] style")
\`\`\`

## Examples
[Include comparison videos]

## License
CC-BY-NC-SA-4.0 (following Ditto-1M dataset license)
```

### 2. Upload to HuggingFace

```bash
# Install huggingface_hub
pip install huggingface_hub

# Login
huggingface-cli login

# Upload
huggingface-cli upload \
    your-username/wan2.2-ghibli-lora \
    checkpoints/ghibli_style_lora/final \
    --repo-type model
```

### 3. Share

- GitHub: Create release on your fork
- Reddit: r/StableDiffusion, r/MachineLearning
- Twitter: Tag @Wan_AI
- Discord: Wan community server

---

## 🐛 Troubleshooting

### Out of Memory (OOM)

```yaml
# In config, enable aggressive memory optimization:
training:
  gradient_checkpointing: true
  mixed_precision: "bf16"

inference:
  offload_model: true
  t5_cpu: true
  convert_model_dtype: true
```

### Slow Training

- Reduce `num_frames`: 81 → 49
- Reduce resolution: [1280, 704] → [640, 352]
- Use fewer samples for initial testing

### Poor Results

- Train longer (more epochs)
- Increase LoRA rank: 64 → 128
- Check if style samples are diverse enough
- Adjust loss weights

---

## 📚 References

- [Ditto Paper](https://arxiv.org/abs/2510.15742)
- [Ditto Dataset](https://huggingface.co/datasets/QingyanBai/Ditto-1M)
- [Wan2.2 Paper](https://arxiv.org/abs/2503.20314)
- [PEFT Documentation](https://huggingface.co/docs/peft)

---

## 💬 Support

Questions? Open an issue or reach out on:
- Wan Discord: https://discord.gg/AKNgpMK4Yj
- GitHub Issues: https://github.com/harsh-agarwal/Wan2.2/issues
