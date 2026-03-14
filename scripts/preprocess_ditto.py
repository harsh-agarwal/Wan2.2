#!/usr/bin/env python3
"""
Preprocess Ditto-1M dataset for Wan2.2 LoRA training.

This script:
1. Reads the training metadata JSON files
2. Filters for style transfer examples
3. Creates train/val splits
4. Generates training CSV compatible with the training pipeline
"""

import json
import os
import random
from pathlib import Path
import argparse
from tqdm import tqdm
import pandas as pd


def resolve_dataset_video_path(base_dir, relative_path):
    """
    Resolve Ditto video path across common layouts:
    - <base>/<source/...>
    - <base>/videos/<source/...>
    """
    base_path = Path(base_dir)
    rel = Path(relative_path)
    candidates = [
        base_path / rel,
        base_path / "videos" / rel,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_metadata(metadata_dir):
    """Load all metadata JSON files."""
    metadata_path = Path(metadata_dir)
    all_samples = []
    
    print("📖 Loading metadata files...")
    
    # Load all JSON files in training_metadata/
    json_files = list(metadata_path.glob("*.json"))
    
    for json_file in tqdm(json_files, desc="Loading metadata"):
        with open(json_file, 'r') as f:
            data = json.load(f)
            all_samples.extend(data)
    
    print(f"✅ Loaded {len(all_samples)} total samples")
    return all_samples


def filter_style_samples(samples, style_keywords=None):
    """
    Filter samples for style transfer tasks.

    Primary filter: edited_path must be under a global_style* folder (or mini_test_videos
    for smoke-test mode). This is the reliable structural indicator that a sample is a
    style-transfer pair — independent of how the instruction is worded.

    Optional secondary filter: if style_keywords are explicitly provided (via --style_filter),
    the instruction must also contain one of those keywords. This lets callers narrow to a
    specific style (e.g. only 'anime'). When style_keywords is None (the default), all
    structurally-valid style pairs are kept regardless of instruction wording.

    Args:
        samples: List of sample dictionaries
        style_keywords: If provided, only keep samples whose instruction contains one of
                        these keywords. If None, keep all structural style-transfer samples.
    """
    print(f"\n🔍 Filtering for style transfer samples...")
    if style_keywords:
        print(f"   Style keywords (instruction filter): {style_keywords}")
    else:
        print("   No keyword filter — keeping all global_style* pairs")

    style_samples = []

    for sample in tqdm(samples, desc="Filtering"):
        edited_path = sample.get('edited_path', '')

        # Primary: structural check — is this a style-transfer output?
        is_style_pair = 'global_style' in edited_path or 'mini_test_videos/' in edited_path
        if not is_style_pair:
            continue

        # Secondary (optional): narrow by instruction keywords
        if style_keywords:
            instruction = sample.get('instruction', '').lower()
            if not any(kw in instruction for kw in style_keywords):
                continue

        style_samples.append(sample)

    print(f"✅ Found {len(style_samples)} style transfer samples")
    return style_samples


def verify_video_exists(sample, base_dir):
    """Check if both source and edited videos exist."""
    source_path = resolve_dataset_video_path(base_dir, sample['source_path'])
    edited_path = resolve_dataset_video_path(base_dir, sample['edited_path'])
    return source_path is not None and edited_path is not None


def create_train_val_split(samples, base_dir, val_ratio=0.1, verify_files=True):
    """
    Create train/val split.
    
    Args:
        samples: List of samples
        base_dir: Base directory containing videos
        val_ratio: Ratio of validation samples
        verify_files: If True, only include samples where files exist
    """
    print(f"\n📊 Creating train/val split (val_ratio={val_ratio})...")
    # Filter samples with existing files if requested
    if verify_files:
        print("   Verifying video files exist...")
        valid_samples = []
        for sample in tqdm(samples, desc="Verifying files"):
            if verify_video_exists(sample, base_dir):
                valid_samples.append(sample)
        
        print(f"   ✅ {len(valid_samples)}/{len(samples)} samples have valid files")
        samples = valid_samples
        if len(samples) == 0:
            raise RuntimeError(
                "No valid source/edited video pairs found after verification. "
                "Your dataset likely has metadata only, or videos are not downloaded/extracted. "
                "Ensure Ditto video folders exist (source/global_style*) under "
                "`data/ditto/videos/` (or `data/ditto/`)."
            )
    
    # Shuffle
    random.seed(42)
    random.shuffle(samples)
    
    # Split
    val_size = int(len(samples) * val_ratio)
    val_samples = samples[:val_size]
    train_samples = samples[val_size:]
    
    print(f"✅ Train: {len(train_samples)} samples")
    print(f"✅ Val: {len(val_samples)} samples")
    
    return train_samples, val_samples


def create_training_csv(samples, output_path, base_dir):
    """
    Create CSV file for training.
    
    CSV format:
    source_video_path, edited_video_path, instruction, source_caption
    """
    print(f"\n📝 Creating training CSV: {output_path}")
    
    rows = []
    for sample in tqdm(samples, desc="Creating CSV"):
        rows.append({
            'source_video': sample['source_path'],
            'edited_video': sample['edited_path'],
            'instruction': sample['instruction'],
            'source_caption': sample.get('source_caption', ''),
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    
    print(f"✅ Saved {len(rows)} samples to {output_path}")
    return df


def analyze_style_distribution(samples):
    """Analyze the distribution of styles in the dataset."""
    print("\n📊 Style Distribution Analysis:")
    print("-" * 60)
    
    style_counts = {}
    
    for sample in samples:
        instruction = sample['instruction'].lower()
        
        # Extract style keywords
        styles = []
        if 'anime' in instruction:
            styles.append('anime')
        if 'pixel' in instruction or '8-bit' in instruction:
            styles.append('pixel_art')
        if 'cyberpunk' in instruction or 'neon' in instruction:
            styles.append('cyberpunk')
        if 'watercolor' in instruction:
            styles.append('watercolor')
        if 'oil painting' in instruction:
            styles.append('oil_painting')
        if 'sketch' in instruction or 'pencil' in instruction:
            styles.append('sketch')
        if 'comic' in instruction or 'cartoon' in instruction:
            styles.append('comic')
        if 'vintage' in instruction or 'retro' in instruction:
            styles.append('vintage')
        if 'noir' in instruction or 'black and white' in instruction:
            styles.append('noir')
        
        for style in styles:
            style_counts[style] = style_counts.get(style, 0) + 1
    
    # Print distribution
    for style, count in sorted(style_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {style:20s}: {count:6d} samples")
    
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Ditto-1M dataset for Wan2.2 LoRA training"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/ditto",
        help="Directory containing downloaded Ditto-1M dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/processed",
        help="Directory to save processed training data"
    )
    parser.add_argument(
        "--style_filter",
        type=str,
        nargs="+",
        default=None,
        help="Filter for specific styles (e.g., 'anime' 'cyberpunk')"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1)"
    )
    parser.add_argument(
        "--no_verify",
        action="store_true",
        help="Skip file existence verification (faster but may include missing files)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of samples (for quick testing)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    metadata_dir = Path(args.data_dir) / "training_metadata"
    if not metadata_dir.exists():
        print(f"❌ Error: Metadata directory not found: {metadata_dir}")
        print("\nPlease run download script first:")
        print("  python scripts/download_ditto_style.py")
        return
    
    samples = load_metadata(metadata_dir)
    
    # Filter for style transfer
    style_samples = filter_style_samples(samples, args.style_filter)
    
    # Analyze distribution
    analyze_style_distribution(style_samples)
    
    # Limit samples if requested
    if args.max_samples:
        print(f"\n⚠️  Limiting to {args.max_samples} samples for testing")
        style_samples = style_samples[:args.max_samples]
    
    # Create train/val split
    train_samples, val_samples = create_train_val_split(
        style_samples,
        args.data_dir,
        args.val_ratio,
        verify_files=not args.no_verify
    )
    
    # Save metadata JSONs
    train_json_path = output_path / "train_metadata.json"
    val_json_path = output_path / "val_metadata.json"
    
    with open(train_json_path, 'w') as f:
        json.dump(train_samples, f, indent=2)
    
    with open(val_json_path, 'w') as f:
        json.dump(val_samples, f, indent=2)
    
    print(f"\n💾 Saved metadata:")
    print(f"   Train: {train_json_path}")
    print(f"   Val: {val_json_path}")
    
    # Create training CSVs
    train_csv = create_training_csv(
        train_samples,
        output_path / "train.csv",
        args.data_dir
    )
    
    val_csv = create_training_csv(
        val_samples,
        output_path / "val.csv",
        args.data_dir
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("✨ Preprocessing Complete!")
    print("=" * 80)
    print(f"\n📊 Dataset Summary:")
    print(f"   Total samples: {len(style_samples)}")
    print(f"   Training: {len(train_samples)}")
    print(f"   Validation: {len(val_samples)}")
    print(f"   Output directory: {output_path.absolute()}")
    
    print("\n📋 Sample instruction examples:")
    for i, sample in enumerate(train_samples[:5]):
        print(f"   {i+1}. {sample['instruction'][:80]}...")
    
    print("\n🚀 Next steps:")
    print("   1. Review the data: ls -lh data/processed/")
    print("   2. Configure training: edit configs/ghibli_style_lora.yaml")
    print("   3. Start training: python scripts/train_style_lora.py")


if __name__ == "__main__":
    main()
