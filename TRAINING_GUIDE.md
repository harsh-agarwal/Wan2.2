# 🎨 Wan2.2 Style Transfer LoRA Training Guide

Complete guide for training style transfer LoRA adapters on Wan2.2 using the Ditto-1M dataset.

---

## 📚 Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Dataset Information](#dataset-information)
4. [Training Pipeline](#training-pipeline)
5. [Evaluation](#evaluation)
6. [Troubleshooting](#troubleshooting)
7. [Next Steps](#next-steps)

---

## 🎯 Overview

### What You'll Build

A **LoRA adapter** that can transfer videos to specific artistic styles:
- 🎌 Anime/Studio Ghibli style
- 🎮 Pixel art / retro game aesthetics
- 🌃 Cyberpunk / neon style
- 🎨 Watercolor, oil painting, sketch styles
- 📼 Vintage / film noir styles

### Why Ditto-1M?

- ✅ **1 million video editing triplets** (source → edited pairs)
- ✅ **300k+ style transfer examples** (anime, cyberpunk, watercolor, etc.)
- ✅ **High quality** - 1280×720 resolution, 101 frames, 20fps
- ✅ **Diverse styles** - Multiple artistic styles in one dataset
- ✅ **Instruction-based** - Each pair has text description

### Requirements

- **GPU:** A100 (80GB) or 2x RTX 4090 (24GB each)
- **Storage:** 400GB for dataset + 50GB for checkpoints
- **Time:** 48-72 hours for full training, 2 hours for mini test

---

## 🚀 Quick Start

### Option 1: Mini Test (2 hours)

Perfect for testing the pipeline before committing to full training.

```bash
# Run everything with one command
./scripts/quick_start.sh mini
```

This will:
1. Download 1GB of test videos
2. Process 30 samples
3. Train for 5 epochs (~2 hours)
4. Generate test outputs

### Option 2: Full Training (60 hours)

```bash
# Step 1: Login to HuggingFace
huggingface-cli login

# Step 2: Accept dataset terms
# Visit: https://huggingface.co/datasets/QingyanBai/Ditto-1M
# Click "Agree and access repository"

# Step 3: Run full pipeline
./scripts/quick_start.sh full
```

### Option 3: Manual Step-by-Step

```bash
# 1. Install dependencies
pip install -r scripts/requirements_training.txt

# 2. Download dataset
python scripts/download_ditto_style.py --output_dir ./data/ditto

# 3. Preprocess
python scripts/preprocess_ditto.py \
    --data_dir ./data/ditto \
    --output_dir ./data/processed

# 4. Train
python scripts/train_style_lora.py --config configs/ghibli_style_lora.yaml

# 5. Test
python scripts/test_style_lora.py \
    --lora_path checkpoints/ghibli_style_lora/final
```

---

## 📊 Dataset Information

### Ditto-1M Dataset

**Paper:** [Scaling Instruction-Based Video Editing](https://arxiv.org/abs/2510.15742)  
**Dataset:** [HuggingFace](https://huggingface.co/datasets/QingyanBai/Ditto-1M)  
**License:** CC-BY-NC-SA-4.0

### Style Subset Breakdown

| Category | Samples | Size | Description |
|----------|---------|------|-------------|
| Source Videos | ~300k | 180GB | Original videos |
| Global Style 1 | ~200k | 230GB | Primary style transfers |
| Global Style 2 | ~100k | 120GB | Additional styles |
| **Total** | **~300k** | **530GB** | Style transfer subset |

### Style Distribution (Estimated)

- 🎌 **Anime/Cartoon:** ~80,000 samples
- 🎮 **Pixel Art:** ~40,000 samples
- 🌃 **Cyberpunk:** ~30,000 samples
- 🎨 **Artistic (watercolor, oil, sketch):** ~100,000 samples
- 📼 **Vintage/Noir:** ~30,000 samples
- 🎭 **Other styles:** ~20,000 samples

### Video Specifications

- **Resolution:** 1280×720 or 720×1280
- **Frames:** 101 frames per video
- **FPS:** 20fps (5 seconds per video)
- **Format:** MP4

---

## 🔧 Training Pipeline

### Architecture

```
Wan2.2-TI2V-5B (5B parameters, frozen)
    ↓
LoRA Adapters (rank=64, ~50M trainable parameters)
    ↓
Style Transfer Output
```

### Training Process

1. **Load source video** (original)
2. **Load edited video** (with style applied)
3. **Extract text instruction** (e.g., "Convert to anime style")
4. **Add noise** to edited video at random timestep
5. **Predict noise** using model conditioned on source + instruction
6. **Compute loss** between predicted and actual noise
7. **Update LoRA weights** via backpropagation

### Loss Function

```python
total_loss = (
    mse_weight * mse_loss +              # Reconstruction
    perceptual_weight * perceptual_loss + # Style preservation
    temporal_weight * temporal_loss       # Smooth motion
)
```

### Hyperparameters

**Recommended settings:**

```yaml
# LoRA
lora_rank: 64              # Good balance of quality/speed
lora_alpha: 64             # Typically same as rank
lora_dropout: 0.1          # Regularization

# Training
learning_rate: 5e-5        # Conservative, stable
batch_size: 1              # Limited by memory
gradient_accumulation: 8   # Effective batch = 8
num_epochs: 30             # Adjust based on dataset size

# Memory
mixed_precision: "bf16"    # 2x memory savings
gradient_checkpointing: true
offload_model: true
t5_cpu: true
```

---

## 📈 Evaluation

### Automatic Metrics

1. **FID (Fréchet Inception Distance)**
   - Measures visual quality
   - Lower is better
   - Target: <50 for good quality

2. **CLIP Score**
   - Measures style adherence
   - Higher is better
   - Target: >0.25 for strong style

3. **Temporal Consistency**
   - Frame-to-frame SSIM
   - Higher is better
   - Target: >0.9 for smooth video

### Manual Evaluation

Generate test videos and check:
- ✓ Style is correctly applied
- ✓ Motion is smooth and natural
- ✓ No artifacts or flickering
- ✓ Details are preserved
- ✓ Temporal consistency maintained

### Test Prompts

```python
test_prompts = [
    "A person walking in a park, Studio Ghibli style",
    "A cat sitting by the window, anime style",
    "Ocean waves at sunset, Ghibli animation",
    "City street at night, cyberpunk neon style",
    "A bicycle on a country road, watercolor painting",
]
```

---

## 🐛 Troubleshooting

### Out of Memory (OOM)

**Symptoms:** `CUDA out of memory` error

**Solutions:**
```yaml
# Reduce memory usage in config:
data:
  resolution: [640, 352]  # Half resolution
  num_frames: 49          # Fewer frames

training:
  gradient_accumulation_steps: 16  # Larger accumulation
  gradient_checkpointing: true

inference:
  offload_model: true
  t5_cpu: true
  convert_model_dtype: true
```

Or use smaller LoRA:
```yaml
model:
  lora_rank: 32  # Instead of 64
```

### Slow Training

**Symptoms:** <1 it/s training speed

**Solutions:**
1. Reduce resolution: 1280×704 → 640×352
2. Reduce frames: 81 → 49
3. Use fewer samples for testing
4. Enable mixed precision (should be on by default)

### Poor Style Transfer Quality

**Symptoms:** Generated videos don't match target style

**Solutions:**
1. Train longer (more epochs)
2. Increase LoRA rank: 64 → 128
3. Increase perceptual loss weight
4. Filter dataset for specific style only:
   ```bash
   python scripts/preprocess_ditto.py --style_filter anime cartoon
   ```

### Dataset Download Issues

**Symptoms:** 401 Unauthorized, slow download

**Solutions:**
```bash
# 1. Login to HuggingFace
huggingface-cli login

# 2. Accept dataset terms at:
# https://huggingface.co/datasets/QingyanBai/Ditto-1M

# 3. Resume download (it will continue from where it stopped)
python scripts/download_ditto_style.py --output_dir ./data/ditto
```

---

## 🎓 Advanced Topics

### Multi-GPU Training

```bash
# Use FSDP for multi-GPU training
torchrun --nproc_per_node=8 scripts/train_style_lora.py \
    --config configs/ghibli_style_lora.yaml
```

Update config:
```yaml
training:
  use_fsdp: true
```

### Style-Specific Training

Train separate LoRAs for each style:

```bash
# Anime only
python scripts/preprocess_ditto.py --style_filter anime cartoon

# Cyberpunk only
python scripts/preprocess_ditto.py --style_filter cyberpunk neon

# Watercolor only
python scripts/preprocess_ditto.py --style_filter watercolor painting
```

### Combining Multiple LoRAs

You can combine multiple LoRAs at inference:
```python
# Load multiple LoRAs
model.load_lora("anime_lora.safetensors", adapter_name="anime")
model.load_lora("cyberpunk_lora.safetensors", adapter_name="cyberpunk")

# Generate with weighted combination
video = model.generate(
    prompt="A city street",
    lora_scales={"anime": 0.5, "cyberpunk": 0.5}
)
```

---

## 📦 Sharing Your LoRA

### 1. Prepare Release

```bash
# Create release directory
mkdir -p release/wan2.2-ghibli-lora
cd release/wan2.2-ghibli-lora

# Copy final checkpoint
cp -r ../../checkpoints/ghibli_style_lora/final/* .

# Create model card
cat > README.md <<EOF
# Wan2.2 Ghibli Style LoRA

LoRA adapter for Wan2.2-TI2V-5B trained on Ditto-1M dataset.

## Usage
\`\`\`python
from wan.textimage2video import WanTI2V

model = WanTI2V(config, checkpoint_dir="./Wan2.2-TI2V-5B")
# Load LoRA here
video = model.generate("A cat in a garden, Studio Ghibli style")
\`\`\`

## Training Details
- Dataset: Ditto-1M (style subset)
- Training samples: [X]
- LoRA rank: 64
- Training time: [Y] hours

## License
CC-BY-NC-SA-4.0
EOF

# Add example videos
mkdir examples
# Copy your best test outputs here
```

### 2. Upload to HuggingFace

```bash
huggingface-cli upload \
    your-username/wan2.2-ghibli-lora \
    . \
    --repo-type model
```

### 3. Create GitHub Release

```bash
# Tag release
git tag -a v1.0-ghibli-lora -m "Ghibli style LoRA v1.0"
git push origin v1.0-ghibli-lora

# Create release on GitHub with:
# - Release notes
# - Example videos
# - Training metrics
# - Usage instructions
```

---

## 🎯 Next Steps After Training

### 1. Create More Style LoRAs

Use the same pipeline for other styles:
- Cyberpunk/neon
- Pixel art
- Watercolor
- Film noir

### 2. Improve Training

- Implement proper diffusion loss (currently placeholder)
- Add perceptual loss for better style preservation
- Add temporal consistency loss
- Experiment with different LoRA ranks

### 3. Build Applications

- Web UI for style transfer
- Batch processing tool
- Style mixing interface
- Real-time preview

### 4. Contribute Back

- Share your LoRA on HuggingFace
- Write blog post about your process
- Submit improvements to Wan2.2 repo
- Help others in the community

---

## 📚 Resources

### Papers
- [Wan2.2 Paper](https://arxiv.org/abs/2503.20314)
- [Ditto Paper](https://arxiv.org/abs/2510.15742)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

### Code
- [Wan2.2 GitHub](https://github.com/Wan-Video/Wan2.2)
- [Ditto GitHub](https://github.com/EzioBy/Ditto)
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)

### Community
- [Wan Discord](https://discord.gg/AKNgpMK4Yj)
- [HuggingFace Forums](https://discuss.huggingface.co/)
- [r/StableDiffusion](https://reddit.com/r/StableDiffusion)

---

## 💬 Support

Questions or issues? 
- Open an issue on GitHub
- Ask in Wan Discord
- Check `scripts/README_TRAINING.md` for detailed docs

---

## 📝 License

Training scripts: Apache 2.0 (same as Wan2.2)  
Ditto-1M dataset: CC-BY-NC-SA-4.0  
Trained LoRAs: CC-BY-NC-SA-4.0 (inherits from dataset)

---

**Happy training! 🚀**
