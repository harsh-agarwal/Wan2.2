#!/usr/bin/env python3
"""
Simple inference script to generate videos with trained LoRA.

Usage:
    python scripts/inference_with_lora.py \
        --lora_path checkpoints/ghibli_style_lora/final \
        --prompt "A cat walking in a garden, Studio Ghibli style" \
        --lora_scale 0.8
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import torch
from peft import PeftModel
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from wan.textimage2video import WanTI2V
from wan.configs import WAN_CONFIGS
from wan.utils.utils import save_video


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_lora_scale(peft_model: torch.nn.Module, lora_scale: float):
    """
    Best-effort runtime LoRA scaling for PEFT LoRA layers.
    """
    for module in peft_model.modules():
        if hasattr(module, "scaling") and isinstance(module.scaling, dict):
            if not hasattr(module, "_base_lora_scaling"):
                module._base_lora_scaling = dict(module.scaling)
            for adapter_name, base_value in module._base_lora_scaling.items():
                module.scaling[adapter_name] = float(base_value) * float(lora_scale)


def generate_with_lora(
    model_path,
    lora_path,
    prompt,
    image_path=None,
    lora_scale=0.8,
    size=(1280, 704),
    frame_num=81,
    seed=42,
    output_dir="./outputs"
):
    """
    Generate video with LoRA.
    
    Args:
        model_path: Path to base Wan2.2 model
        lora_path: Path to trained LoRA checkpoint
        prompt: Text prompt
        image_path: Optional image for I2V
        lora_scale: LoRA strength (0.0-1.0)
        size: Video resolution (width, height)
        frame_num: Number of frames
        seed: Random seed
        output_dir: Directory to save output
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available! Generation requires GPU.")
        return None
    
    logger.info("=" * 80)
    logger.info("Wan2.2 Style Transfer LoRA Inference")
    logger.info("=" * 80)
    logger.info(f"\n🎮 GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    logger.info(f"\n📝 Prompt: {prompt}")
    logger.info(f"🎨 LoRA scale: {lora_scale}")
    logger.info(f"📐 Resolution: {size[0]}×{size[1]}")
    logger.info(f"🎞️  Frames: {frame_num}")
    
    # Load model
    logger.info(f"\n⏳ Loading Wan2.2 model from {model_path}...")
    config = WAN_CONFIGS["ti2v-5B"]
    
    model = WanTI2V(
        config=config,
        checkpoint_dir=model_path,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=True,
        init_on_cpu=False,
        convert_model_dtype=True
    )
    
    # Load LoRA
    if lora_path:
        logger.info(f"⏳ Loading LoRA from {lora_path}...")
        model.model = PeftModel.from_pretrained(model.model, lora_path)
        set_lora_scale(model.model, lora_scale)
        model.model.eval()
    
    # Load image if provided
    img = None
    if image_path:
        logger.info(f"🖼️  Loading image from {image_path}...")
        img = Image.open(image_path).convert('RGB')
    
    # Generate video
    logger.info(f"\n🎬 Generating video...")
    
    video = model.generate(
        input_prompt=prompt,
        img=img,
        size=size,
        frame_num=frame_num,
        shift=5.0,
        sample_solver='unipc',
        sampling_steps=50,
        guide_scale=5.0,
        n_prompt="",
        seed=seed,
        offload_model=True
    )
    
    # Save output
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"lora_output_{timestamp}.mp4"
    output_file = output_path / output_filename
    
    logger.info(f"\n💾 Saving video to {output_file}...")
    save_video(video, str(output_file))
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ Generation Complete!")
    logger.info("=" * 80)
    logger.info(f"\n📁 Output: {output_file}")
    
    return str(output_file)


def main():
    parser = argparse.ArgumentParser(
        description="Generate videos with trained LoRA"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./Wan2.2-TI2V-5B",
        help="Path to base Wan2.2 model"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to trained LoRA checkpoint"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for generation"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional image for I2V generation"
    )
    parser.add_argument(
        "--lora_scale",
        type=float,
        default=0.8,
        help="LoRA strength (0.0-1.0, default: 0.8)"
    )
    parser.add_argument(
        "--size",
        type=str,
        default="1280*704",
        help="Video resolution (width*height, default: 1280*704)"
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=81,
        help="Number of frames (default: 81)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory (default: ./outputs)"
    )
    
    args = parser.parse_args()
    
    # Parse size
    width, height = map(int, args.size.split('*'))
    
    # Generate
    output_file = generate_with_lora(
        model_path=args.model_path,
        lora_path=args.lora_path,
        prompt=args.prompt,
        image_path=args.image,
        lora_scale=args.lora_scale,
        size=(width, height),
        frame_num=args.frame_num,
        seed=args.seed,
        output_dir=args.output_dir
    )
    
    if output_file:
        print(f"\n✨ Video generated successfully: {output_file}")
    else:
        print("\n❌ Generation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
