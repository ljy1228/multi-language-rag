


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from transformers import get_linear_schedule_with_warmup
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
import time
import os
from pathlib import Path
import json
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed

from src.models.base_model import BaseQAModel
from src.utils.device_utils import DeviceManager, setup_mixed_precision

logger = logging.getLogger(__name__)
class QATrainer:
    """Unified trainer for QA models."""
    
    def __init__(self, 
                 model: BaseQAModel,
                 config: Dict[str, Any],
                 device_manager: Optional[DeviceManager] = None):
        """
        Initialize QA trainer.
        
        Args:
            model: QA model to train
            config: Training configuration
            device_manager: Device manager instance
        """
        self.model = model
       
        self.config = config
        self.device_manager = device_manager or DeviceManager()
        
        # Initialize accelerator for mixed precision and distributed training
        self.accelerator = Accelerator(
            mixed_precision="fp16" if setup_mixed_precision(self.device_manager) else "no",
            gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1)
        )
        
        # Set seed for reproducibility
        set_seed(config.get('seed', 42))
        
        # Training state
        self.optimizer = None
        self.scheduler = None
        self.global_step = 0
        self.epoch = 0
        self.best_metric = -float('inf')
        self.training_history = []
        
        logger.info(f"Initialized QA trainer for {model.model_name}")
    
    def setup_training(self, 
                      train_dataloader: DataLoader,
                      eval_dataloader: Optional[DataLoader] = None,
                      num_training_steps: Optional[int] = None):
        """
        Setup training components.
        
        Args:
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader
            num_training_steps: Total number of training steps
        """
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.model.parameters(),
            lr=self.config.get('learning_rate', 1e-5),
            weight_decay=self.config.get('weight_decay', 0.01),
            eps=self.config.get('adam_epsilon', 1e-8),
            betas=(self.config.get('adam_beta1', 0.9), self.config.get('adam_beta2', 0.999))
        )
        
        # Setup scheduler
        if num_training_steps is None:
            num_training_steps = len(train_dataloader) * self.config.get('num_train_epochs', 3)
        
        warmup_steps = int(num_training_steps * self.config.get('warmup_ratio', 0.1))
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Prepare for accelerator
        self.model.model, self.optimizer, train_dataloader, self.scheduler = self.accelerator.prepare(
            self.model.model, self.optimizer, train_dataloader, self.scheduler
        )
        
        if eval_dataloader is not None:
            eval_dataloader = self.accelerator.prepare(eval_dataloader)
        
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        logger.info(f"Setup training with {num_training_steps} total steps, {warmup_steps} warmup steps")
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary containing training metrics
        """
        self.model.train_mode()
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.epoch}")
        
        for batch in progress_bar:
            with self.accelerator.accumulate(self.model.model):
                if torch.isnan(batch['input_ids']).any():
                    print("Found NaN in input_ids")
                if torch.isnan(batch['labels']).any():
                    print("Found NaN in labels")
                            # Forward pass

                if hasattr(self.model, 'compute_loss'):
                    # Custom loss computation
                    loss = self.model.compute_loss(**batch)
                    print(loss)
                else:
                    # Standard forward pass
                    outputs = self.model.forward(**batch)
                    loss = outputs.get('loss', outputs.get('total_loss'))
                
                # Backward pass
                self.accelerator.backward(loss)
                
                # Gradient clipping
                if self.config.get('max_grad_norm', 0) > 0:
                    self.accelerator.clip_grad_norm_(
                        self.model.model.parameters(), 
                        self.config['max_grad_norm']
                    )
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # Update metrics
                total_loss += loss.item()
                num_batches += 1
                self.global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
                })
                
                # Logging
                if self.global_step % self.config.get('logging_steps', 100) == 0:
                    self._log_training_step(loss.item())
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'train_loss': avg_loss,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.eval_dataloader is None:
            return {}
        
        self.model.eval_mode()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                if hasattr(self.model, 'compute_loss'):
                    loss = self.model.compute_loss(**batch)
                else:
                    outputs = self.model.forward(**batch)
                    loss = outputs.get('loss', outputs.get('total_loss'))
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'eval_loss': avg_loss
        }
    
    def train(self, 
              train_dataloader: DataLoader,
              eval_dataloader: Optional[DataLoader] = None,
              num_epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader
            num_epochs: Number of training epochs
            
        Returns:
            Dictionary containing training history
        """
        num_epochs = num_epochs or self.config.get('num_train_epochs', 3)
        
        # Setup training
        self.setup_training(train_dataloader, eval_dataloader)
        
        # Training loop
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Evaluate
            eval_metrics = self.evaluate()
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **eval_metrics}
            self.training_history.append(epoch_metrics)
            
            # Log epoch results
            logger.info(f"Epoch {epoch}: {epoch_metrics}")
            
            # Save checkpoint
            # if self.config.get('save_strategy') == 'epoch':
            self._save_checkpoint(epoch_metrics)
            
            # Early stopping
            if self._should_early_stop(epoch_metrics):
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        return self._get_training_history()
    
    def _log_training_step(self, loss: float):
        """Log training step metrics."""
        if self.accelerator.is_main_process:
            logger.info(f"Step {self.global_step}: loss={loss:.4f}, lr={self.scheduler.get_last_lr()[0]:.2e}")
    
    def _save_checkpoint(self, metrics: Dict[str, float]):
        """Save model checkpoint."""
        if not self.accelerator.is_main_process:
            return
        
        output_dir = self.config.get('output_dir', './models')
        checkpoint_dir = Path(output_dir) / f"checkpoint-{self.global_step}"
        
        # Save model
        self.accelerator.save_state(checkpoint_dir)
        
        # Save metrics
        metrics_file = checkpoint_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    def _should_early_stop(self, metrics: Dict[str, float]) -> bool:
        """Check if training should stop early."""
        if not self.config.get('early_stopping', False):
            return False
        
        patience = self.config.get('early_stopping_patience', 3)
        metric_name = self.config.get('metric_for_best_model', 'eval_loss')
        
        if metric_name not in metrics:
            return False
        
        current_metric = metrics[metric_name]
        
        if self.config.get('greater_is_better', False):
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
        else:
            if current_metric < self.best_metric:
                self.best_metric = current_metric
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
        
        return self.early_stop_counter >= patience
    
    def _get_training_history(self) -> Dict[str, List[float]]:
        """Get training history."""
        if not self.training_history:
            return {}
        
        history = {}
        for key in self.training_history[0].keys():
            history[key] = [epoch[key] for epoch in self.training_history]
        
        return history
    
    def save_model(self, output_dir: str):
        """Save the final model."""
        if not self.accelerator.is_main_process:
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.accelerator.save_model(self.model.model, output_path)
        
        # Save training history
        history_file = output_path / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump(self._get_training_history(), f, indent=2)
        
        # Save config
        config_file = output_path / "training_config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Saved final model to {output_path}")


class SameTrainer(QATrainer):
    def __init__(self, model: BaseQAModel, config: Dict[str, Any], device_manager: Optional[DeviceManager] = None):
        """Initialize few-shot trainer."""
        super().__init__(model, config, device_manager)
    
    def train_same(self,
                   train_dataloader: DataLoader,
                   eval_dataloader: Optional[DataLoader] = None) -> Dict[str, List[float]]:
        same_config = self.config.get('same', {})
        training_config = {**self.config, **same_config}
        same_trainer = QATrainer(self.model, training_config, self.device_manager)
        history = same_trainer.train(train_dataloader, eval_dataloader)
        return history