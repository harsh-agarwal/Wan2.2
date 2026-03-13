# 🎨 Wan2.2 Style Transfer LoRA Training Scripts

Complete toolkit for training style transfer LoRA adapters on Wan2.2.

---

## 📦 What's Included

### Core Scripts

| Script | Purpose | Time |
|--------|---------|------|
| `download_ditto_style.py` | Download Ditto-1M dataset | 4-8 hours |
| `preprocess_ditto.py` | Process videos for training | 1-2 hours |
| `train_style_lora.py` | Train LoRA adapter | 48-72 hours |
| `test_style_lora.py` | Evaluate trained LoRA | 1 hour |
| `inference_with_lora.py` | Generate with LoRA | 5-10 min/video |
| `preview_dataset.py` | Preview dataset samples | 10 min |
| `quick_start.sh` | One-command setup | 2-60 hours |

### Configuration

| File | Purpose |
|------|---------|
| `ghibli_style_lora.yaml` | Training configuration |
| `requirements_training.txt` | Python dependencies |

### Documentation

| File | Purpose |
|------|---------|
| `README_TRAINING.md` | Detailed training guide |
| `../TRAINING_GUIDE.md` | Complete project guide |
| `../LORA_PROJECT_SUMMARY.md` | Project overview |
| `../QUICKSTART.md` | Quick reference |

---

## 🚀 Quick Start

### Fastest Way (Mini Test - 2 hours)

```bash
./scripts/quick_start.sh mini
```

### Full Training (60 hours)

```bash
# 1. Login to HuggingFace
huggingface-cli login

# 2. Accept dataset terms
# Visit: https://huggingface.co/datasets/QingyanBai/Ditto-1M

# 3. Run
./scripts/quick_start.sh full
```

---

## 📖 Detailed Usage

### 1. Download Dataset

**Mini test (1GB):**
```bash
python scripts/download_ditto_style.py --mini
```

**Full dataset (350GB):**
```bash
python scripts/download_ditto_style.py --output_dir ./data/ditto
```

**Options:**
- `--output_dir`: Where to save dataset (default: `./data/ditto`)
- `--mini`: Download only test videos (1GB)
- `--extract`: Auto-extract tar.gz archives

---

### 2. Preprocess Data

**Basic usage:**
```bash
python scripts/preprocess_ditto.py \
    --data_dir ./data/ditto \
    --output_dir ./data/processed
```

**Filter for specific styles:**
```bash
python scripts/preprocess_ditto.py \
    --data_dir ./data/ditto \
    --output_dir ./data/processed \
    --style_filter anime cartoon ghibli
```

**Quick test with limited samples:**
```bash
python scripts/preprocess_ditto.py \
    --data_dir ./data/ditto \
    --output_dir ./data/processed \
    --max_samples 100 \
    --no_verify
```

**Options:**
- `--data_dir`: Input dataset directory
- `--output_dir`: Output directory for processed data
- `--style_filter`: Filter for specific styles (space-separated)
- `--val_ratio`: Validation split ratio (default: 0.1)
- `--max_samples`: Limit number of samples (for testing)
- `--no_verify`: Skip file existence check (faster)

---

### 3. Preview Dataset

**Analyze dataset statistics:**
```bash
python scripts/preview_dataset.py --analyze
```

**Create preview videos:**
```bash
python scripts/preview_dataset.py --preview --num_previews 10
```

**Both:**
```bash
python scripts/preview_dataset.py
```

**Options:**
- `--metadata`: Path to metadata JSON
- `--data_dir`: Base directory with videos
- `--analyze`: Show dataset statistics
- `--preview`: Create side-by-side preview videos
- `--num_previews`: Number of previews to create
- `--output_dir`: Where to save previews

---

### 4. Train LoRA

**Basic training:**
```bash
python scripts/train_style_lora.py \
    --config configs/ghibli_style_lora.yaml
```

**Resume from checkpoint:**
```bash
python scripts/train_style_lora.py \
    --config configs/ghibli_style_lora.yaml \
    --resume checkpoints/ghibli_style_lora/checkpoint-epoch10-step5000
```

**Monitor training:**
```bash
# In another terminal
tensorboard --logdir logs/ghibli_style_lora --port 6006
```

---

### 5. Test Trained LoRA

**Test with default prompts:**
```bash
python scripts/test_style_lora.py \
    --lora_path checkpoints/ghibli_style_lora/final
```

**Test with custom prompts:**
```bash
python scripts/test_style_lora.py \
    --lora_path checkpoints/ghibli_style_lora/final \
    --prompts \
        "A cat walking in a garden" \
        "Ocean waves at sunset" \
        "City street at night"
```

**Test different LoRA scales:**
```bash
python scripts/test_style_lora.py \
    --lora_path checkpoints/ghibli_style_lora/final \
    --lora_scales 0.0 0.3 0.5 0.7 0.9 1.0
```

---

### 6. Generate with LoRA

**Text-to-Video:**
```bash
python scripts/inference_with_lora.py \
    --lora_path checkpoints/ghibli_style_lora/final \
    --prompt "A person walking through a forest, Studio Ghibli style" \
    --lora_scale 0.8
```

**Image-to-Video:**
```bash
python scripts/inference_with_lora.py \
    --lora_path checkpoints/ghibli_style_lora/final \
    --prompt "Gentle breeze, peaceful atmosphere, Ghibli animation" \
    --image examples/i2v_input.JPG \
    --lora_scale 0.8
```

**Custom resolution:**
```bash
python scripts/inference_with_lora.py \
    --lora_path checkpoints/ghibli_style_lora/final \
    --prompt "Your prompt here" \
    --size "640*352" \
    --frame_num 49
```

---

## ⚙️ Configuration

Edit `configs/ghibli_style_lora.yaml` to customize:

### LoRA Settings
```yaml
model:
  lora_rank: 64        # 32=fast, 64=balanced, 128=high quality
  lora_alpha: 64       # Usually same as rank
  lora_dropout: 0.1    # Regularization
```

### Training Settings
```yaml
training:
  num_epochs: 30            # More = better (but slower)
  learning_rate: 5.0e-5     # Lower = more stable
  batch_size: 1             # Limited by GPU memory
  gradient_accumulation_steps: 8  # Effective batch size
```

### Memory Optimization
```yaml
data:
  resolution: [1280, 704]   # Reduce if OOM: [640, 352]
  num_frames: 81            # Reduce if OOM: 49

inference:
  offload_model: true       # Enable for low VRAM
  t5_cpu: true              # Move T5 to CPU
  convert_model_dtype: true # Use BF16
```

---

## 📊 Training Profiles

### Profile 1: Quick Test (RTX 4090, 2 hours)
```yaml
data:
  resolution: [640, 352]
  num_frames: 49
training:
  num_epochs: 5
  max_samples: 30
model:
  lora_rank: 32
```

### Profile 2: Balanced (A100 80GB, 60 hours)
```yaml
data:
  resolution: [1280, 704]
  num_frames: 81
training:
  num_epochs: 30
model:
  lora_rank: 64
```

### Profile 3: High Quality (8x A100, 12 hours)
```yaml
data:
  resolution: [1280, 704]
  num_frames: 81
training:
  num_epochs: 50
  use_fsdp: true
model:
  lora_rank: 128
```

---

## 🎯 Training Checklist

- [ ] Install dependencies: `pip install -r scripts/requirements_training.txt`
- [ ] Login to HuggingFace: `huggingface-cli login`
- [ ] Accept dataset terms: https://huggingface.co/datasets/QingyanBai/Ditto-1M
- [ ] Download dataset: `python scripts/download_ditto_style.py`
- [ ] Preprocess data: `python scripts/preprocess_ditto.py`
- [ ] Preview samples: `python scripts/preview_dataset.py`
- [ ] Configure training: Edit `configs/ghibli_style_lora.yaml`
- [ ] Start training: `python scripts/train_style_lora.py`
- [ ] Monitor progress: `tensorboard --logdir logs/ghibli_style_lora`
- [ ] Test LoRA: `python scripts/test_style_lora.py`
- [ ] Generate videos: `python scripts/inference_with_lora.py`
- [ ] Share on HuggingFace: `huggingface-cli upload`

---

## 💾 Storage Requirements

| Component | Size | Notes |
|-----------|------|-------|
| Source videos | 180GB | Original videos |
| Style videos | 350GB | Styled versions |
| Processed data | 10GB | Metadata, CSVs |
| Checkpoints | 50GB | Model checkpoints |
| Outputs | 20GB | Generated test videos |
| **Total** | **610GB** | Full training setup |

**Mini test:** Only 5GB total

---

## ⏱️ Time Estimates

| Task | Time | GPU |
|------|------|-----|
| Download (mini) | 10 min | - |
| Download (full) | 4-8 hours | - |
| Preprocessing | 1-2 hours | - |
| Training (mini) | 2 hours | RTX 4090 |
| Training (full) | 60 hours | A100 80GB |
| Training (full) | 12 hours | 8x A100 |
| Testing | 1 hour | Any GPU |
| Inference | 5-10 min | Any GPU |

---

## 🎓 Next Steps

1. ✅ **Run mini test** - Validate pipeline
2. ⏳ **Train first LoRA** - Full training
3. ⏳ **Evaluate results** - Test and iterate
4. ⏳ **Share on HuggingFace** - Contribute to community
5. ⏳ **Train more styles** - Expand collection

---

## 📞 Support

**Quick help:** See `QUICKSTART.md`  
**Detailed docs:** See `TRAINING_GUIDE.md`  
**Issues:** Open GitHub issue  
**Community:** Wan Discord

---

**Start now:** `./scripts/quick_start.sh mini` 🚀
