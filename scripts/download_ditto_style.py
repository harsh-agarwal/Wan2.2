#!/usr/bin/env python3
"""
Download Ditto-1M style transfer subset for Wan2.2 LoRA training.

This script downloads only the style transfer videos (global_style1 and global_style2)
from the Ditto-1M dataset, which is ~350GB instead of the full 2TB dataset.
"""

import os
import argparse
import json
from huggingface_hub import snapshot_download
from pathlib import Path


def _download_mini_smoke_subset(output_path, mini_pairs=30):
    """
    Download a tiny but trainable subset:
    - metadata
    - exact source/edited video files for selected style pairs
    """
    print("\n📦 Downloading trainable MINI subset...")
    print(f"Target size: ~{mini_pairs} paired examples")

    # 1) Download mini test videos + metadata
    snapshot_download(
        repo_id="QingyanBai/Ditto-1M",
        repo_type="dataset",
        local_dir=str(output_path),
        allow_patterns=[
            "mini_test_videos/*",
            "training_metadata/*",
            "source_video_captions/*",
        ],
        resume_download=True,
        max_workers=4,
    )

    # Build synthetic paired metadata from mini test videos for end-to-end smoke test.
    mini_videos = sorted((output_path / "mini_test_videos").glob("*.mp4"))
    if not mini_videos:
        raise RuntimeError(
            "No mini_test_videos found after download. Cannot build mini smoke subset."
        )
    mini_videos = mini_videos[:max(2, mini_pairs)]

    selected = []
    for i, video_path in enumerate(mini_videos):
        rel = f"mini_test_videos/{video_path.name}"
        # For smoke tests we pair the same clip as source/edited with a style instruction.
        selected.append({
            "source_path": rel,
            "edited_path": rel,
            "instruction": "Make it a Japanese anime style, cel shading.",
            "source_caption": "",
            "is_mini_smoke": True,
            "mini_index": i,
        })

    metadata_dir = output_path / "training_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    # Keep only mini metadata to make preprocessing deterministic in mini mode.
    for old_json in metadata_dir.glob("*.json"):
        try:
            old_json.unlink()
        except OSError:
            pass
    mini_metadata_path = metadata_dir / "mini_selected.json"
    with open(mini_metadata_path, "w") as f:
        json.dump(selected, f, indent=2)

    print("\n✅ Mini trainable subset download complete!")
    print(f"   Metadata: {mini_metadata_path}")
    print(f"   Pairs: {len(selected)}")
    print(f"   Videos downloaded: {len(mini_videos)}")
    print("   Note: MINI mode is for pipeline validation, not quality training.")


def download_ditto_style_subset(output_dir="./data/ditto", use_mini=False, mini_pairs=30):
    """
    Download Ditto-1M style transfer subset.
    
    Args:
        output_dir: Directory to save the dataset
        use_mini: If True, only download mini_test_videos for quick testing
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Downloading Ditto-1M Style Transfer Dataset")
    print("=" * 80)
    
    if use_mini:
        print("\n📦 MINI mode: trainable subset for end-to-end validation")
        print("Includes matched source + styled videos and mini metadata.")
    else:
        print("\n📦 Downloading style transfer subset...")
        patterns = [
            "videos/source/*",           # Source videos (~180GB)
            "videos/global_style1/*",    # Style transfer 1 (~230GB)
            "videos/global_style2/*",    # Style transfer 2 (~120GB)
            "training_metadata/*",       # Metadata
            "source_video_captions/*",   # Captions
            "csvs_for_DiffSynth/*"       # Training CSVs
        ]
        print("Total size: ~530GB (source + style1 + style2)")
        print("\nThis will take several hours depending on your connection.")
        print("You can cancel and restart - downloads will resume.")
    
    print(f"\n📁 Saving to: {output_path.absolute()}")
    print("\n⏳ Starting download...\n")
    
    try:
        if use_mini:
            _download_mini_smoke_subset(output_path, mini_pairs=mini_pairs)
        else:
            snapshot_download(
                repo_id="QingyanBai/Ditto-1M",
                repo_type="dataset",
                local_dir=str(output_path),
                allow_patterns=patterns,
                resume_download=True,  # Resume if interrupted
                max_workers=4,         # Parallel downloads
            )
        
        print("\n✅ Download complete!")
        print(f"📁 Dataset saved to: {output_path.absolute()}")
        
        # Print directory structure
        print("\n📊 Dataset structure:")
        for root, dirs, files in os.walk(output_path):
            level = root.replace(str(output_path), '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:3]:  # Show first 3 files
                print(f"{subindent}{file}")
            if len(files) > 3:
                print(f"{subindent}... and {len(files) - 3} more files")
        
    except Exception as e:
        print(f"\n❌ Error during download: {e}")
        print("\nYou may need to:")
        print("1. Login to HuggingFace: huggingface-cli login")
        print("2. Accept the dataset terms on: https://huggingface.co/datasets/QingyanBai/Ditto-1M")
        raise


def extract_tar_files(data_dir):
    """
    Extract split tar.gz files if they exist.
    
    The dataset may contain split archives like:
    - global_style1.tar.gz.aa
    - global_style1.tar.gz.ab
    etc.
    """
    import subprocess
    
    data_path = Path(data_dir)
    
    # Find all .tar.gz.aa files (first part of split archives)
    tar_parts = list(data_path.glob("**/*.tar.gz.aa"))
    
    if not tar_parts:
        print("\n✓ No split archives found - videos are already extracted.")
        return
    
    print(f"\n📦 Found {len(tar_parts)} split archives to extract...")
    
    for tar_first_part in tar_parts:
        tar_prefix = str(tar_first_part)[:-3]  # Remove '.aa'
        output_dir = tar_first_part.parent
        
        print(f"\n⏳ Extracting {tar_first_part.name}...")
        
        # Concatenate all parts and extract
        cmd = f"cd {output_dir} && cat {tar_prefix}.* | tar -zxv"
        
        try:
            subprocess.run(cmd, shell=True, check=True)
            print(f"✅ Extracted successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error extracting: {e}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Ditto-1M style transfer dataset for Wan2.2 LoRA training"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/ditto",
        help="Directory to save the dataset (default: ./data/ditto)"
    )
    parser.add_argument(
        "--mini",
        action="store_true",
        help="Download a small trainable subset (source+style pairs) for quick testing"
    )
    parser.add_argument(
        "--mini_pairs",
        type=int,
        default=30,
        help="Number of source/style pairs for --mini mode (default: 30)"
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract tar.gz split archives after download"
    )
    
    args = parser.parse_args()
    
    # Download dataset
    download_ditto_style_subset(args.output_dir, args.mini, mini_pairs=args.mini_pairs)
    
    # Extract if requested
    if args.extract:
        extract_tar_files(args.output_dir)
    
    print("\n" + "=" * 80)
    print("✨ Ready to proceed with data preprocessing!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Run: python scripts/preprocess_ditto.py")
    print("2. Run: python scripts/train_style_lora.py")
