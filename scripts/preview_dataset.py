#!/usr/bin/env python3
"""
Preview Ditto-1M dataset samples.

This script helps you explore the dataset before training:
- View random samples
- Check video quality
- Analyze style distribution
- Verify file integrity
"""

import argparse
import json
import random
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


def load_metadata(metadata_path):
    """Load metadata JSON."""
    with open(metadata_path, 'r') as f:
        return json.load(f)


def display_video_info(video_path):
    """Display video information."""
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        return None
    
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    }
    
    cap.release()
    return info


def create_side_by_side_preview(source_path, edited_path, output_path):
    """Create side-by-side comparison of source and edited video."""
    source_cap = cv2.VideoCapture(str(source_path))
    edited_cap = cv2.VideoCapture(str(edited_path))
    
    if not source_cap.isOpened() or not edited_cap.isOpened():
        print(f"❌ Could not open videos")
        return False
    
    # Get video properties
    fps = source_cap.get(cv2.CAP_PROP_FPS)
    width = int(source_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(source_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width * 2, height))
    
    frame_count = 0
    while True:
        ret1, source_frame = source_cap.read()
        ret2, edited_frame = edited_cap.read()
        
        if not ret1 or not ret2:
            break
        
        # Concatenate horizontally
        combined = np.hstack([source_frame, edited_frame])
        
        # Add text labels
        cv2.putText(combined, "Source", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Styled", (width + 10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(combined)
        frame_count += 1
    
    source_cap.release()
    edited_cap.release()
    out.release()
    
    print(f"✅ Created preview with {frame_count} frames: {output_path}")
    return True


def analyze_dataset(metadata_path, data_dir, num_samples=100):
    """Analyze dataset statistics."""
    print("\n" + "=" * 80)
    print("📊 Dataset Analysis")
    print("=" * 80)
    
    samples = load_metadata(metadata_path)
    
    print(f"\nTotal samples: {len(samples)}")
    
    # Sample random subset for analysis
    analysis_samples = random.sample(samples, min(num_samples, len(samples)))
    
    # Analyze styles
    style_counts = {}
    instruction_lengths = []
    
    for sample in analysis_samples:
        instruction = sample['instruction'].lower()
        instruction_lengths.append(len(instruction))
        
        # Extract style keywords
        if 'anime' in instruction or 'cartoon' in instruction:
            style_counts['anime'] = style_counts.get('anime', 0) + 1
        if 'pixel' in instruction or '8-bit' in instruction:
            style_counts['pixel_art'] = style_counts.get('pixel_art', 0) + 1
        if 'cyberpunk' in instruction or 'neon' in instruction:
            style_counts['cyberpunk'] = style_counts.get('cyberpunk', 0) + 1
        if 'watercolor' in instruction:
            style_counts['watercolor'] = style_counts.get('watercolor', 0) + 1
        if 'oil painting' in instruction:
            style_counts['oil_painting'] = style_counts.get('oil_painting', 0) + 1
        if 'sketch' in instruction or 'pencil' in instruction:
            style_counts['sketch'] = style_counts.get('sketch', 0) + 1
        if 'noir' in instruction or 'vintage' in instruction:
            style_counts['vintage'] = style_counts.get('vintage', 0) + 1
    
    # Print style distribution
    print("\n📈 Style Distribution (from sample):")
    print("-" * 60)
    for style, count in sorted(style_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(analysis_samples)) * 100
        print(f"  {style:20s}: {count:4d} ({percentage:5.1f}%)")
    print("-" * 60)
    
    # Instruction statistics
    print(f"\n📝 Instruction Statistics:")
    print(f"  Average length: {np.mean(instruction_lengths):.1f} characters")
    print(f"  Min length: {min(instruction_lengths)}")
    print(f"  Max length: {max(instruction_lengths)}")
    
    # Check file existence
    print(f"\n🔍 Checking file existence (sample of {min(10, len(analysis_samples))})...")
    base_path = Path(data_dir)
    existing = 0
    
    for sample in analysis_samples[:10]:
        source_path = base_path / sample['source_path']
        edited_path = base_path / sample['edited_path']
        
        if source_path.exists() and edited_path.exists():
            existing += 1
            # Get video info
            info = display_video_info(source_path)
            if info:
                print(f"  ✓ {source_path.name}: {info['width']}×{info['height']}, {info['frames']} frames, {info['fps']:.1f}fps")
        else:
            print(f"  ✗ Missing: {source_path.name}")
    
    print(f"\n✅ {existing}/10 sample files exist")
    
    # Print example instructions
    print(f"\n📋 Example Instructions:")
    print("-" * 60)
    for i, sample in enumerate(random.sample(samples, 5)):
        print(f"{i+1}. {sample['instruction']}")
    print("-" * 60)


def preview_samples(metadata_path, data_dir, num_previews=5, output_dir="./previews"):
    """Create preview videos for random samples."""
    print("\n" + "=" * 80)
    print("🎬 Creating Preview Videos")
    print("=" * 80)
    
    samples = load_metadata(metadata_path)
    preview_samples = random.sample(samples, min(num_previews, len(samples)))
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    base_path = Path(data_dir)
    
    for idx, sample in enumerate(preview_samples):
        print(f"\n📹 Preview {idx + 1}/{len(preview_samples)}")
        print(f"   Instruction: {sample['instruction'][:70]}...")
        
        source_path = base_path / sample['source_path']
        edited_path = base_path / sample['edited_path']
        
        if not source_path.exists() or not edited_path.exists():
            print(f"   ⚠️  Files not found, skipping")
            continue
        
        # Create side-by-side preview
        preview_output = output_path / f"preview_{idx:02d}.mp4"
        create_side_by_side_preview(source_path, edited_path, preview_output)
    
    print(f"\n✅ Previews saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Preview Ditto-1M dataset samples"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="./data/processed/train_metadata.json",
        help="Path to metadata JSON file"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/ditto",
        help="Base directory containing videos"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze dataset statistics"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Create preview videos"
    )
    parser.add_argument(
        "--num_previews",
        type=int,
        default=5,
        help="Number of preview videos to create"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./previews",
        help="Directory to save preview videos"
    )
    
    args = parser.parse_args()
    
    # Check if metadata exists
    if not Path(args.metadata).exists():
        print(f"❌ Error: Metadata file not found: {args.metadata}")
        print("\nPlease run preprocessing first:")
        print("  python scripts/preprocess_ditto.py")
        return
    
    # Run analysis
    if args.analyze:
        analyze_dataset(args.metadata, args.data_dir)
    
    # Create previews
    if args.preview:
        preview_samples(
            args.metadata,
            args.data_dir,
            args.num_previews,
            args.output_dir
        )
    
    # If neither flag specified, run both
    if not args.analyze and not args.preview:
        analyze_dataset(args.metadata, args.data_dir)
        preview_samples(
            args.metadata,
            args.data_dir,
            args.num_previews,
            args.output_dir
        )


if __name__ == "__main__":
    main()
