#!/usr/bin/env python3
"""
CLI script for running the same language cross-lingual QA experiments.
"""

import argparse
import yaml
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.same import run_same_comparison
from src.utils.device_utils import DeviceManager


def main():
    parser = argparse.ArgumentParser(description='Run the same language cross-lingual QA experiments')
    
    parser.add_argument('--config', type=str, default='config/experiment_configs.yaml',
                       help='Path to experiment configuration file')
    parser.add_argument('--model-config', type=str, default='config/model_configs.yaml',
                       help='Path to model configuration file')
    parser.add_argument('--training-config', type=str, default='config/training_configs.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--cache-dir', type=str, default='cache',
                       help='Cache directory for datasets')
    parser.add_argument('--dataset_path', type=str, default='/mnt/ljy/CrossLingualQA-Benchmark/xquad_in',
                       help='Path to the XQuAD-IN dataset')
    parser.add_argument('--languages', nargs='+', default=["as","en"],
                       help='List of languages to evaluate (default: all XQuAD languages)')
    parser.add_argument('--sampling-strategy', type=str, default='random',
                       choices=['random', 'diverse', 'stratified'],
                       help='Few-shot sampling strategy')
    parser.add_argument('--use-mps', action='store_true', default=True,
                       help='Use MPS (Apple Silicon GPU) if available')
    parser.add_argument('--use-cpu', action='store_true', default=False,
                       help='Force CPU usage')
    
    args = parser.parse_args()
    
    # Load configuration files
    with open(args.config, 'r') as f:
        experiment_config = yaml.safe_load(f)
    
    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)
    
    with open(args.training_config, 'r') as f:
        training_config = yaml.safe_load(f)
    
    # Combine configurations
    config = {
        **model_config,
        **training_config['new'],
        **experiment_config,
        'output_dir': args.output_dir,
        'cache_dir': args.cache_dir,
        'use_mps': args.use_mps,
        'use_cpu': args.use_cpu,
        'sampling_strategy': args.sampling_strategy,
        "dataset_path": args.dataset_path
    }
    
    # Override languages if specified
    if args.languages:
        config['languages'] = args.languages
    
    # Initialize device manager
    device_manager = DeviceManager(config)
    print(f"Using device: {device_manager.get_device()}")
    
    # Run experiments
    print(f"Starting the same language experiments")
    results = run_same_comparison(
        config=config,
        languages=args.languages,
        device_manager=device_manager
    )
    
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
