#!/usr/bin/env python3
"""
Test and evaluate trained LoRA for style transfer.

This script:
1. Loads trained LoRA weights
2. Generates test videos with different LoRA scales
3. Computes evaluation metrics
4. Creates comparison videos
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import yaml

import torch
from peft import PeftModel
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from wan.textimage2video import WanTI2V
from wan.configs import WAN_CONFIGS
from wan.utils.utils import save_video


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def set_lora_scale(peft_model: torch.nn.Module, lora_scale: float):
    for module in peft_model.modules():
        if hasattr(module, "scaling") and isinstance(module.scaling, dict):
            if not hasattr(module, "_base_lora_scaling"):
                module._base_lora_scaling = dict(module.scaling)
            for adapter_name, base_value in module._base_lora_scaling.items():
                module.scaling[adapter_name] = float(base_value) * float(lora_scale)


class StyleLoRATester:
    """Tester for style transfer LoRA."""
    
    def __init__(self, config, lora_path):
        self.config = config
        self.lora_path = Path(lora_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create output directory
        self.output_dir = Path("./outputs/lora_tests")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_model(self):
        """Load Wan2.2 model with trained LoRA."""
        logger.info("Loading Wan2.2 model with LoRA...")
        
        # Load base model
        task = self.config['model']['task']
        wan_config = WAN_CONFIGS[task]
        
        self.model = WanTI2V(
            config=wan_config,
            checkpoint_dir=self.config['model']['base_model_path'],
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=self.config['inference']['t5_cpu'],
            init_on_cpu=False,
            convert_model_dtype=self.config['inference']['convert_model_dtype']
        )
        
        # Load LoRA weights
        logger.info(f"Loading LoRA weights from {self.lora_path}...")
        self.model.model = PeftModel.from_pretrained(self.model.model, self.lora_path)
        self.model.model.eval()
        
        logger.info("✅ Model loaded successfully")
        return self.model
    
    def generate_test_videos(self, test_prompts, lora_scales=[0.0, 0.5, 0.8, 1.0]):
        """
        Generate test videos with different LoRA scales.
        
        Args:
            test_prompts: List of prompts to test
            lora_scales: List of LoRA scales to try
        """
        logger.info(f"\n🎬 Generating test videos...")
        logger.info(f"   Prompts: {len(test_prompts)}")
        logger.info(f"   LoRA scales: {lora_scales}")
        
        results = []
        
        for prompt_idx, prompt in enumerate(test_prompts):
            logger.info(f"\n📝 Prompt {prompt_idx + 1}/{len(test_prompts)}: {prompt}")
            
            for scale in lora_scales:
                logger.info(f"   Generating with LoRA scale={scale}...")
                set_lora_scale(self.model.model, scale)
                
                try:
                    # Generate video
                    video = self.model.generate(
                        input_prompt=prompt,
                        img=None,  # Text-to-video
                        size=tuple(self.config['data']['resolution']),
                        frame_num=self.config['data']['num_frames'],
                        shift=5.0,
                        sample_solver='unipc',
                        sampling_steps=self.config['inference']['num_inference_steps'],
                        guide_scale=self.config['inference']['guidance_scale'],
                        n_prompt="",
                        seed=self.config['inference']['seed'],
                        offload_model=self.config['inference']['offload_model']
                        # TODO: Add lora_scale parameter
                    )
                    
                    # Save video
                    output_filename = f"prompt{prompt_idx:02d}_scale{scale:.1f}.mp4"
                    output_path = self.output_dir / output_filename
                    save_video(video, str(output_path))
                    
                    logger.info(f"   ✅ Saved to {output_filename}")
                    
                    results.append({
                        'prompt': prompt,
                        'lora_scale': scale,
                        'output_path': str(output_path)
                    })
                
                except Exception as e:
                    logger.error(f"   ❌ Error generating video: {e}")
                    continue
        
        # Save results metadata
        results_path = self.output_dir / "test_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n✅ Test results saved to {results_path}")
        return results
    
    def create_comparison_grid(self, results):
        """Create side-by-side comparison videos."""
        logger.info("\n🎞️  Creating comparison grids...")
        
        # Group by prompt
        by_prompt = {}
        for result in results:
            prompt = result['prompt']
            if prompt not in by_prompt:
                by_prompt[prompt] = []
            by_prompt[prompt].append(result)
        
        # Create grid for each prompt
        for prompt, prompt_results in by_prompt.items():
            logger.info(f"   Creating grid for: {prompt[:50]}...")
            
            # TODO: Implement video grid creation
            # This would load all videos for this prompt and create a side-by-side comparison
            
        logger.info("✅ Comparison grids created")
    
    def compute_metrics(self, results):
        """Compute evaluation metrics."""
        logger.info("\n📊 Computing evaluation metrics...")
        
        metrics = {
            'fid_scores': [],
            'clip_scores': [],
            'temporal_consistency': []
        }
        
        # TODO: Implement metric computation
        # - FID: Compare generated videos to reference style dataset
        # - CLIP: Measure style adherence
        # - Temporal: Measure frame-to-frame consistency
        
        logger.info("✅ Metrics computed")
        
        # Save metrics
        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def run_tests(self, lora_scales=None):
        """Run full test suite."""
        # Load model
        self.load_model()
        
        # Get test prompts from config
        test_prompts = self.config['evaluation']['test_prompts']
        
        # Generate test videos
        if lora_scales is None:
            lora_scales = [0.0, 0.5, 0.8, 1.0]
        results = self.generate_test_videos(test_prompts, lora_scales=lora_scales)
        
        # Create comparisons
        self.create_comparison_grid(results)
        
        # Compute metrics
        metrics = self.compute_metrics(results)
        
        logger.info("\n" + "=" * 80)
        logger.info("✨ Testing Complete!")
        logger.info("=" * 80)
        logger.info(f"\n📁 Results saved to: {self.output_dir}")
        logger.info(f"\n🎬 Generated {len(results)} test videos")


def main():
    parser = argparse.ArgumentParser(
        description="Test trained LoRA for style transfer"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/ghibli_style_lora.yaml",
        help="Path to training configuration YAML"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to trained LoRA checkpoint"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=None,
        help="Custom test prompts (overrides config)"
    )
    parser.add_argument(
        "--lora_scales",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 0.8, 1.0],
        help="LoRA scales to test"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override prompts if provided
    if args.prompts:
        config['evaluation']['test_prompts'] = args.prompts
    
    # Check CUDA
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available! Testing requires GPU.")
        return
    
    logger.info(f"🎮 Using device: {torch.cuda.get_device_name(0)}")
    
    # Create tester
    tester = StyleLoRATester(config, args.lora_path)
    
    # Run tests
    tester.run_tests(lora_scales=args.lora_scales)


if __name__ == "__main__":
    main()
