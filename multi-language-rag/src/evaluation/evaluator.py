"""
Cross-lingual evaluator for QA models.
Handles evaluation across multiple languages and generates transfer matrices.
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any, Tuple
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import json

from src.evaluation.metrics import QAMetrics, compute_qa_metrics
from src.models.base_model import BaseQAModel
from src.data.loaders import UnifiedDataLoader
from src.data.preprocessors import DataPreprocessor
from src.utils.device_utils import DeviceManager

logger = logging.getLogger(__name__)


class CrossLingualEvaluator:
    """Evaluator for cross-lingual QA models."""
    
    def __init__(self, 
                 model: BaseQAModel,
                 config: Dict[str, Any],
                 device_manager: Optional[DeviceManager] = None):
        """
        Initialize cross-lingual evaluator.
        
        Args:
            model: QA model to evaluate
            config: Evaluation configuration
            device_manager: Device manager instance
        """
        self.model = model
        self.config = config
        self.device_manager = device_manager or DeviceManager()
        
        # Initialize data components
        self.data_loader = UnifiedDataLoader(cache_dir=config.get('cache_dir', './cache'))
        self.preprocessor = DataPreprocessor(
            model.model_name,
            model_type=config.get('model_type', 'bert'),
            max_length=config.get('max_length', 384)
        )
        
        # Initialize metrics
        self.metrics = QAMetrics(
            metrics=config.get('metrics', ['exact_match', 'f1', 'bleu']),
            normalize=config.get('normalize', True)
        )
        
        logger.info(f"Initialized cross-lingual evaluator for {model.model_name}")
    
    def evaluate_all_languages(self, 
                              languages: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate model on all specified languages.
        
        Args:
            languages: List of language codes. If None, uses all XQuAD languages.
            
        Returns:
            Dictionary containing evaluation results
        """
        if languages is None:
            languages = self.data_loader.xquad_loader.languages
        
        logger.info(f"Evaluating on {len(languages)} languages: {languages}")
        
        results = {}
        
        for language in languages:
            logger.info(f"Evaluating on {language}...")
            
            try:
                # Get language data
                examples = self.data_loader.xquad_loader.get_language_examples(language)
                
                if not examples:
                    logger.warning(f"No data found for language: {language}")
                    continue
                
                # Evaluate language
                lang_results = self._evaluate_language(examples, language)
                results[language] = lang_results
                
            except Exception as e:
                logger.error(f"Failed to evaluate on {language}: {e}")
                results[language] = {'error': str(e)}
        
        # Compute summary statistics
        summary = self._compute_summary_statistics(results)
        results['summary'] = summary
        
        return results
    
    def _evaluate_language(self, 
                          examples: List[Dict[str, Any]], 
                          language: str) -> Dict[str, Any]:
        """
        Evaluate model on a specific language.
        
        Args:
            examples: List of examples for the language
            language: Language code
            
        Returns:
            Dictionary containing evaluation results
        """
        # Preprocess examples
        eval_dataset, qa_examples = self.preprocessor.preprocess_for_evaluation(examples)
        
        # Create data loader
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=self.config.get('eval_batch_size', 32),
            shuffle=False,
            num_workers=self.config.get('dataloader_num_workers', 4),
            pin_memory=self.config.get('dataloader_pin_memory', True)
        )
        
        # Get predictions
        predictions = self._get_predictions(eval_dataloader, qa_examples)
        
        # Extract ground truth answers
        references = []
        for example in qa_examples:
            if hasattr(example, 'answer'):
                references.append(example.answer)
            else:
                references.append('')
        
        # Compute metrics
        metrics = self.metrics.compute(predictions, references)
        
        # Compute per-sample metrics for analysis
        per_sample_metrics = self.metrics.compute_per_sample(predictions, references)
        
        return {
            'language': language,
            'num_examples': len(examples),
            'metrics': metrics,
            'per_sample_metrics': per_sample_metrics,
            'predictions': predictions[:10],  # Sample predictions
            'references': references[:10]     # Sample references
        }
    
    def _get_predictions(self, 
                        eval_dataloader: DataLoader,
                        qa_examples: List[Any]) -> List[str]:
        """
        Get predictions from the model.
        
        Args:
            eval_dataloader: Evaluation data loader
            qa_examples: Original QA examples
            
        Returns:
            List of predicted answers
        """
        self.model.eval_mode()
        
        predictions = []
        
        with torch.no_grad():
            for batch in eval_dataloader:
                # Move batch to device
                if self.device_manager.device is not None:
                    batch = {k: v.to(self.device_manager.device) for k, v in batch.items()}
                
                # Get predictions based on model type
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
                    predictions.extend([pred['answer'] for pred in batch_predictions])
                else:
                    # Fallback to individual predictions
                    for i in range(len(batch['input_ids'])):
                        # This would need proper implementation based on the model
                        predictions.append("")
        
        return predictions
    
    def _compute_summary_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute summary statistics across all languages.
        
        Args:
            results: Evaluation results for all languages
            
        Returns:
            Dictionary containing summary statistics
        """
        # Filter out error results
        valid_results = {lang: res for lang, res in results.items() 
                        if 'error' not in res and 'metrics' in res}
        
        if not valid_results:
            return {'error': 'No valid results found'}
        
        # Compute average metrics
        summary = {}
        for metric in ['exact_match', 'f1', 'bleu', 'rouge']:
            scores = [res['metrics'].get(metric, 0) for res in valid_results.values()]
            if scores:
                summary[f'avg_{metric}'] = np.mean(scores)
                summary[f'std_{metric}'] = np.std(scores)
                summary[f'min_{metric}'] = np.min(scores)
                summary[f'max_{metric}'] = np.max(scores)
        
        # Language-specific statistics
        summary['num_languages'] = len(valid_results)
        summary['languages'] = list(valid_results.keys())
        
        return summary
    
    def generate_transfer_matrix(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        Generate cross-lingual transfer matrix.
        
        Args:
            results: Evaluation results for all languages
            
        Returns:
            DataFrame containing transfer matrix
        """
        # Filter out error results
        valid_results = {lang: res for lang, res in results.items() 
                        if 'error' not in res and 'metrics' in res}
        
        if not valid_results:
            return pd.DataFrame()
        
        # Create transfer matrix
        languages = list(valid_results.keys())
        transfer_matrix = pd.DataFrame(index=languages, columns=languages)
        
        # Fill matrix with F1 scores
        for lang in languages:
            f1_score = valid_results[lang]['metrics'].get('f1', 0)
            transfer_matrix.loc[lang, lang] = f1_score
        
        return transfer_matrix
    
    def save_results(self, results: Dict[str, Any], output_dir: str):
        """
        Save evaluation results.
        
        Args:
            results: Evaluation results
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        results_file = output_path / f"{self.model.model_name.replace('/', '_')}_evaluation.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save transfer matrix
        if 'summary' in results:
            transfer_matrix = self.generate_transfer_matrix(results)
            if not transfer_matrix.empty:
                matrix_file = output_path / f"{self.model.model_name.replace('/', '_')}_transfer_matrix.csv"
                transfer_matrix.to_csv(matrix_file)
        
        logger.info(f"Saved evaluation results to {output_path}")


class ComparativeEvaluator:
    """Evaluator for comparing multiple models."""
    
    def __init__(self, 
                 models: Dict[str, BaseQAModel],
                 config: Dict[str, Any],
                 device_manager: Optional[DeviceManager] = None):
        """
        Initialize comparative evaluator.
        
        Args:
            models: Dictionary mapping model names to model instances
            config: Evaluation configuration
            device_manager: Device manager instance
        """
        self.models = models
        self.config = config
        self.device_manager = device_manager or DeviceManager()
        
        # Initialize evaluators for each model
        self.evaluators = {}
        for name, model in models.items():
            self.evaluators[name] = CrossLingualEvaluator(model, config, device_manager)
        
        logger.info(f"Initialized comparative evaluator for {len(models)} models")
    
    def compare_models(self, 
                      languages: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare all models on specified languages.
        
        Args:
            languages: List of language codes. If None, uses all XQuAD languages.
            
        Returns:
            Dictionary containing comparison results
        """
        logger.info("Starting model comparison")
        
        results = {}
        
        for name, evaluator in self.evaluators.items():
            logger.info(f"Evaluating {name}...")
            model_results = evaluator.evaluate_all_languages(languages)
            results[name] = model_results
        
        # Generate comparison summary
        comparison_summary = self._generate_comparison_summary(results)
        results['comparison'] = comparison_summary
        
        return results
    
    def _generate_comparison_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comparison summary across models.
        
        Args:
            results: Results for all models
            
        Returns:
            Dictionary containing comparison summary
        """
        summary = {}
        
        # Extract metrics for each model
        model_metrics = {}
        for model_name, model_results in results.items():
            if 'summary' in model_results:
                model_metrics[model_name] = model_results['summary']
        
        # Compare average metrics
        for metric in ['avg_exact_match', 'avg_f1', 'avg_bleu', 'avg_rouge']:
            metric_scores = {}
            for model_name, metrics in model_metrics.items():
                if metric in metrics:
                    metric_scores[model_name] = metrics[metric]
            
            if metric_scores:
                summary[metric] = metric_scores
                
                # Find best model
                best_model = max(metric_scores.items(), key=lambda x: x[1])
                summary[f'best_{metric}'] = best_model
        
        return summary
    
    def save_comparison_results(self, results: Dict[str, Any], output_dir: str):
        """
        Save comparison results.
        
        Args:
            results: Comparison results
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        results_file = output_path / 'model_comparison.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save individual model results
        for model_name, model_results in results.items():
            if model_name != 'comparison':
                model_file = output_path / f"{model_name}_results.json"
                with open(model_file, 'w') as f:
                    json.dump(model_results, f, indent=2, default=str)
        
        logger.info(f"Saved comparison results to {output_path}")


def evaluate_model(model: BaseQAModel,
                  config: Dict[str, Any],
                  languages: Optional[List[str]] = None,
                  device_manager: Optional[DeviceManager] = None) -> Dict[str, Any]:
    """
    Convenience function to evaluate a single model.
    
    Args:
        model: QA model to evaluate
        config: Evaluation configuration
        languages: List of language codes
        device_manager: Device manager instance
        
    Returns:
        Evaluation results
    """
    evaluator = CrossLingualEvaluator(model, config, device_manager)
    return evaluator.evaluate_all_languages(languages)


def compare_models(models: Dict[str, BaseQAModel],
                  config: Dict[str, Any],
                  languages: Optional[List[str]] = None,
                  device_manager: Optional[DeviceManager] = None) -> Dict[str, Any]:
    """
    Convenience function to compare multiple models.
    
    Args:
        models: Dictionary mapping model names to model instances
        config: Evaluation configuration
        languages: List of language codes
        device_manager: Device manager instance
        
    Returns:
        Comparison results
    """
    evaluator = ComparativeEvaluator(models, config, device_manager)
    return evaluator.compare_models(languages)
