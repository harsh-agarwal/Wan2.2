# ⚡ Quick Start: Style Transfer LoRA Training

## 🎯 One-Command Start

```bash
# Test the pipeline (2 hours)
./scripts/quick_start.sh mini

# Full training (60 hours)
./scripts/quick_start.sh full
```

---

## 📝 Manual Commands

### 1. Install Dependencies
```bash
pip install -r scripts/requirements_training.txt
```

### 2. Download Dataset
```bash
# Login first
huggingface-cli login

# Download (~350GB, 4-8 hours)
python scripts/download_ditto_style.py
```

### 3. Preprocess
```bash
python scripts/preprocess_ditto.py \
    --data_dir ./data/ditto \
    --output_dir ./data/processed
```

### 4. Preview Dataset (Optional)
```bash
python scripts/preview_dataset.py --analyze --preview
```

### 5. Train
```bash
python scripts/train_style_lora.py \
    --config configs/ghibli_style_lora.yaml
```

### 6. Test
```bash
python scripts/test_style_lora.py \
    --lora_path checkpoints/ghibli_style_lora/final
```

### 7. Generate
```bash
python scripts/inference_with_lora.py \
    --lora_path checkpoints/ghibli_style_lora/final \
    --prompt "A cat in a garden, Ghibli style"
```

---

## 🎨 Training Specific Styles

### Anime Only
```bash
python scripts/preprocess_ditto.py --style_filter anime cartoon
```

### Cyberpunk Only
```bash
python scripts/preprocess_ditto.py --style_filter cyberpunk neon
```

### Watercolor Only
```bash
python scripts/preprocess_ditto.py --style_filter watercolor painting
```

---

## 📊 Monitor Training

```bash
# TensorBoard
tensorboard --logdir logs/ghibli_style_lora

# GPU usage
watch -n 1 nvidia-smi

# Training logs
tail -f logs/ghibli_style_lora/train.log
```

---

## 🐛 Common Issues

### Out of Memory
```bash
# Edit configs/ghibli_style_lora.yaml:
# - Set resolution: [640, 352]
# - Set num_frames: 49
# - Set lora_rank: 32
```

### Dataset Download Fails
```bash
# Accept terms at:
# https://huggingface.co/datasets/QingyanBai/Ditto-1M
```

### Slow Training
```bash
# Use fewer samples for testing:
python scripts/preprocess_ditto.py --max_samples 100
```

---

## 📚 Documentation

- **Full Guide:** `TRAINING_GUIDE.md`
- **Detailed Docs:** `scripts/README_TRAINING.md`
- **Project Summary:** `LORA_PROJECT_SUMMARY.md`

---

## 💬 Help

- Discord: https://discord.gg/AKNgpMK4Yj
- Issues: GitHub Issues
- Docs: See above files

---

**Ready? Run:** `./scripts/quick_start.sh mini`
