


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
from src.training.trainer import SameTrainer

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

    def _save_results(self, results: Dict[str, Any]):
        """Save experiment results."""
        output_dir = self.config.get('output_dir', './results')
        output_path = Path(output_dir) / 'few_shot_results'
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        results_file = output_path / f"{self.model.model_name.replace('/', '_')}_few_shot.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save model
        model_dir = output_path / f"{self.model.model_name.replace('/', '_')}_model"
        self.model.save_model(str(model_dir))
        
        logger.info(f"Saved results to {output_path}")
    
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
        
        final_results = {
            'same_language_results': results,
            'model_info': self.model.get_model_info(),
            'config': self.config,
            'timestamp': time.time()
        }
        self._save_results(final_results)
        return final_results
    
    def _run_same_experiment(self,
                             train_data: Dict[str, Any],
                             eval_data: Dict[str, Any]) -> Dict[str, Any]:
        combined_training_data = []
        for language, examples in train_data.items():
            combined_training_data.extend(examples['examples'])
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
        trainer = SameTrainer(self.model, self.config, self.device_manager)
        training_history = trainer.train_same(train_dataloader)
        evaluation_results = self._evaluate_all_languages(eval_data)
        
        return {
            'training_history': training_history,
            'evaluation_results': evaluation_results,
            'num_training_examples': len(combined_training_data)
        }
        
    def _evaluate_all_languages(self, evaluation_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Evaluate model on all languages.
        
        Args:
            evaluation_data: Dictionary mapping language codes to examples
            
        Returns:
            Dictionary containing evaluation results for each language
        """
        results = {}
        
        for language, examples in evaluation_data.items():
            logger.info(f"Evaluating on {language} ({len(examples)} examples)")
            
            try:
                # Preprocess evaluation data
                eval_dataset, qa_examples = self.preprocessor.preprocess_for_evaluation(examples["examples"])
                
                # Create data loader
                eval_dataloader = DataLoader(
                    eval_dataset,
                    batch_size=self.config.get('eval_batch_size', 16),
                    shuffle=False,
                    num_workers=self.config.get('dataloader_num_workers', 2),
                    pin_memory=self.config.get('dataloader_pin_memory', True)
                )
                
                # Evaluate
                eval_results = self._evaluate_language(eval_dataloader, qa_examples, language)
                results[language] = eval_results
                
            except Exception as e:
                logger.error(f"Failed to evaluate on {language}: {e}")
                results[language] = {'error': str(e)}
        
        return results
       


    def _evaluate_language(self, 
                          eval_dataloader: DataLoader,
                          qa_examples: List[Any],
                          language: str) -> Dict[str, Any]:
        """
        Evaluate model on a specific language.
        
        Args:
            eval_dataloader: Evaluation data loader
            qa_examples: Original QA examples
            language: Language code
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval_mode()
        
        predictions = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                # Move batch to device
                if self.device_manager.device is not None:
                    batch = {k: v.to(self.device_manager.device) for k, v in batch.items()}
                
                # Forward pass
                
                if hasattr(self.model, 'compute_loss'):
                    loss = self.model.compute_loss(**batch)
                    total_loss += loss.item()
                    num_batches += 1
                
                # Get predictions
                if hasattr(self.model, 'batch_predict'):
                    # Use batch prediction if available
                    batch_examples = []
                    for i in range(len(batch['input_ids'])):
                        # Extract question and context from batch
                        # This is a simplified version - in practice, you'd need to
                        # properly extract the original text
                        batch_examples.append({
                            'question': f"question_{i}",
                            'context': f"context_{i}"
                        })
                    
                    batch_predictions = self.model.batch_predict(batch_examples)
                    predictions.extend(batch_predictions)
        
        # Calculate metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # For now, return basic metrics
        # In a full implementation, you'd calculate EM, F1, etc.
        results = {
            'loss': avg_loss,
            'num_examples': len(qa_examples),
            'predictions': predictions[:10] if predictions else [],  # Sample predictions
            'language': language
        }
        
        return results
    
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
    if model_type == 'mbert' or model_type == 'bert':
        if model.qa_head:
            model.qa_head.to(device_manager.get_device())
    # Create experiment
    experiment = SameExperiment(model, config, device_manager)
    
    # Run experiment
    results = experiment.run_experiment(languages)
    
    return results