


import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any, Tuple
import logging
import json
from pathlib import Path
import time

from src.utils.device_utils import DeviceManager
from src.data.loaders import DifferentlangDataLoader
from src.data.preprocessors import DataPreprocessor
from src.data.same_sampler import SamplingConfig, CrossLingualSameSampler,SameSampler

logger = logging.getLogger(__name__)

class SameExperiment:
    def __init__(self,
                 model,
                 config: Dict[str, Any],
                 device_manager: Optional[DeviceManager] = None):
        self.model = model
        self.config = config
        self.device_manager = device_manager or DeviceManager()
        self.data_loader = DifferentlangDataLoader(dataset_path=config.get('dataset_path', './dataset'))
        self.preprocessor = DataPreprocessor(
            model.model_name,
            model_type=config.get('model_type', 'bert'),
            max_length=config.get('max_length', 384)
        )
        sampling_config = SamplingConfig(
            num_shots=1,  # Will be updated per experiment
            strategy=config.get('sampling_strategy', 'random'),
        )
        self.sampler = CrossLingualSameSampler(sampling_config)
    
    
    def run_experiment(self,
                       languages: List[str] = None) -> Dict[str, Any]:
        """
        Run the same language few-shot experiment.
        
        Args:
            languages: List of languages to evaluate
            
        Returns:
            Experiment results
        """
        # Load data
        print(languages)
        evaluation_data = self.data_loader.xquadin_dataloader.load_data_evaluation(languages)
        print(evaluation_data.keys())
        
        train_data = self.data_loader.xquadin_dataloader.load_data_training(languages)
        results = self._run_same_experiment(train_data,evaluation_data)
 
        
        return results
    
    def _run_shot_experiment(self,
                             train_data: Dict[str, Any],
                             eval_data: Dict[str, Any],
                             num_shots: int) -> Dict[str, Any]:
        combined_training_data = []
        for language, examples in train_data.items():
            combined_training_data.extend(examples)
        if not combined_training_data:
            logger.warning("No training data available for samesampling.")
            return {}
        train_dataset = self.preprocessor.preprocess_for_training(combined_training_data)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.get('train_batch_size', 8),
            shuffle=True,
            num_workers=self.config.get('dataloader_num_workers', 2),
            pin_memory=self.config.get('dataloader_pin_memory', True)
        )
        

def run_same_comparison(config: Dict[str, Any],
                        languages: List[str] = None,
                          device_manager: Optional[DeviceManager] = None) -> Dict[str, Any]:
    """
    Run few-shot comparison between mBERT and mT5.
    
    Args:
        config: Experiment configuration
        shots: List of shot numbers to test
        seeds: List of random seeds for reproducibility
        device_manager: Device manager instance
        
    Returns:
        Comparison results
    """
    logger.info("Starting few-shot comparison between mBERT and mT5")
    
    results = {}
    
    # Run mBERT experiment
    logger.info("Running mBERT same experiment...")
    mbert_config = {**config, 'model_type': 'mbert'}
    mbert_results = run_same_experiment(
        'mbert', 
        'bert-base-multilingual-cased', 
        mbert_config, 
        languages, 
        device_manager
    )
    results['mbert'] = mbert_results
    
    # Run mT5 experiment
    logger.info("Running mT5 same experiment...")
    mt5_config = {**config, 'model_type': 'mt5'}
    mt5_results = run_same_experiment(
        'mt5', 
        'google/mt5-base', 
        mt5_config, 
        languages, 
        device_manager
    )
    results['mt5'] = mt5_results
    
    # Save comparison results
    output_dir = config.get('output_dir', './results')
    output_path = Path(output_dir) / 'few_shot_comparison'
    output_path.mkdir(parents=True, exist_ok=True)
    
    comparison_file = output_path / 'few_shot_comparison.json'
    with open(comparison_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Saved comparison results to {comparison_file}")
    
    return results



def run_same_experiment(model_type: str,
                          model_name: str,
                          config: Dict[str, Any],
                          languages: List[str] = None,
                          device_manager: Optional[DeviceManager] = None) -> Dict[str, Any]:
    """
    Run a few-shot experiment.
    
    Args:
        model_type: Type of model ("mbert" or "mt5")
        model_name: Name of the model
        config: Experiment configuration
        shots: List of shot numbers to test
        seeds: List of random seeds for reproducibility
        device_manager: Device manager instance
        
    Returns:
        Experiment results
    """
    # Create model
    from ..models.base_model import ModelFactory
    model = ModelFactory.create_model(model_type, model_name, config)
    
    # Load model
    model.load_model()
    
    # Move to device
    if device_manager:
        model.to_device(device_manager.get_device())
    
    # Create experiment
    experiment = SameExperiment(model, config, device_manager)
    
    # Run experiment
    results = experiment.run_experiment(languages)
    
    return results