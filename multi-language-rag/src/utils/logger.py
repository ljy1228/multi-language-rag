"""
Logging utilities for cross-lingual QA experiments.
Handles experiment tracking with Weights & Biases and local logging.
"""

import logging
import json
import wandb
from typing import Dict, Any, Optional
from pathlib import Path
import time
from datetime import datetime


class ExperimentLogger:
    """Logger for cross-lingual QA experiments."""
    
    def __init__(self, 
                 project_name: str = "cross-lingual-qa",
                 experiment_name: str = "experiment",
                 use_wandb: bool = True,
                 local_logging: bool = True,
                 log_dir: str = "./logs"):
        """
        Initialize experiment logger.
        
        Args:
            project_name: Name of the project
            experiment_name: Name of the experiment
            use_wandb: Whether to use Weights & Biases
            local_logging: Whether to use local logging
            log_dir: Directory for local logs
        """
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.use_wandb = use_wandb
        self.local_logging = local_logging
        self.log_dir = Path(log_dir)
        
        # Create log directory
        if self.local_logging:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Weights & Biases
        if self.use_wandb:
            try:
                wandb.init(
                    project=project_name,
                    name=experiment_name,
                    config={}
                )
                self.wandb_available = True
            except Exception as e:
                print(f"Warning: Failed to initialize Weights & Biases: {e}")
                self.wandb_available = False
        else:
            self.wandb_available = False
        
        # Initialize local logging
        if self.local_logging:
            self._setup_local_logging()
        
        self.start_time = time.time()
    
    def _setup_local_logging(self):
        """Setup local logging configuration."""
        log_file = self.log_dir / f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(self.experiment_name)
        self.log_file = log_file
    
    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        if self.wandb_available:
            wandb.config.update(config)
        
        if self.local_logging:
            config_file = self.log_dir / f"{self.experiment_name}_config.json"
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2, default=str)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        if self.wandb_available:
            wandb.log(metrics, step=step)
        
        if self.local_logging:
            self.logger.info(f"Metrics (step {step}): {metrics}")
    
    def log_model_info(self, model_info: Dict[str, Any]):
        """Log model information."""
        if self.wandb_available:
            wandb.config.update({"model": model_info})
        
        if self.local_logging:
            self.logger.info(f"Model info: {model_info}")
    
    def log_training_step(self, step: int, loss: float, learning_rate: float):
        """Log training step information."""
        metrics = {
            "train/loss": loss,
            "train/learning_rate": learning_rate,
            "train/step": step
        }
        self.log_metrics(metrics, step)
    
    def log_evaluation(self, 
                      metrics: Dict[str, float], 
                      language: str,
                      step: Optional[int] = None):
        """Log evaluation metrics."""
        eval_metrics = {f"eval/{language}/{k}": v for k, v in metrics.items()}
        self.log_metrics(eval_metrics, step)
    
    def log_cross_lingual_results(self, results: Dict[str, Any]):
        """Log cross-lingual evaluation results."""
        if self.wandb_available:
            # Log summary metrics
            if 'summary' in results:
                summary = results['summary']
                wandb.log({
                    "cross_lingual/avg_f1": summary.get('avg_f1', 0),
                    "cross_lingual/avg_em": summary.get('avg_exact_match', 0),
                    "cross_lingual/avg_bleu": summary.get('avg_bleu', 0),
                    "cross_lingual/num_languages": summary.get('num_languages', 0)
                })
            
            # Log per-language metrics
            for lang, lang_results in results.items():
                if lang != 'summary' and 'metrics' in lang_results:
                    metrics = lang_results['metrics']
                    wandb.log({
                        f"language/{lang}/f1": metrics.get('f1', 0),
                        f"language/{lang}/em": metrics.get('exact_match', 0),
                        f"language/{lang}/bleu": metrics.get('bleu', 0)
                    })
        
        if self.local_logging:
            self.logger.info(f"Cross-lingual results: {results}")
    
    def log_few_shot_results(self, 
                           shot: int, 
                           seed: int, 
                           results: Dict[str, Any]):
        """Log few-shot experiment results."""
        if self.wandb_available:
            wandb.log({
                f"few_shot/{shot}_shot/{seed}/avg_f1": results.get('avg_f1', 0),
                f"few_shot/{shot}_shot/{seed}/avg_em": results.get('avg_exact_match', 0),
                f"few_shot/{shot}_shot/{seed}/avg_bleu": results.get('avg_bleu', 0)
            })
        
        if self.local_logging:
            self.logger.info(f"Few-shot results ({shot}-shot, seed {seed}): {results}")
    
    def log_error(self, error: str, context: Optional[str] = None):
        """Log error information."""
        if self.local_logging:
            if context:
                self.logger.error(f"Error in {context}: {error}")
            else:
                self.logger.error(f"Error: {error}")
    
    def log_warning(self, warning: str, context: Optional[str] = None):
        """Log warning information."""
        if self.local_logging:
            if context:
                self.logger.warning(f"Warning in {context}: {warning}")
            else:
                self.logger.warning(f"Warning: {warning}")
    
    def log_info(self, info: str, context: Optional[str] = None):
        """Log information."""
        if self.local_logging:
            if context:
                self.logger.info(f"Info in {context}: {info}")
            else:
                self.logger.info(f"Info: {info}")
    
    def finish(self):
        """Finish logging session."""
        end_time = time.time()
        duration = end_time - self.start_time
        
        if self.wandb_available:
            wandb.log({"experiment/duration": duration})
            wandb.finish()
        
        if self.local_logging:
            self.logger.info(f"Experiment completed in {duration:.2f} seconds")
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """Save results to file."""
        if self.local_logging:
            results_file = self.log_dir / filename
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            self.logger.info(f"Results saved to {results_file}")


def create_logger(project_name: str = "cross-lingual-qa",
                 experiment_name: str = "experiment",
                 use_wandb: bool = True,
                 local_logging: bool = True,
                 log_dir: str = "./logs") -> ExperimentLogger:
    """
    Create an experiment logger instance.
    
    Args:
        project_name: Name of the project
        experiment_name: Name of the experiment
        use_wandb: Whether to use Weights & Biases
        local_logging: Whether to use local logging
        log_dir: Directory for local logs
        
    Returns:
        ExperimentLogger instance
    """
    return ExperimentLogger(
        project_name=project_name,
        experiment_name=experiment_name,
        use_wandb=use_wandb,
        local_logging=local_logging,
        log_dir=log_dir
    )
