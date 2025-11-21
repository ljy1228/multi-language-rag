"""
Base model interface for cross-lingual QA models.
Defines common interfaces and utilities for mBERT and mT5 models.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class BaseQAModel(ABC):
    """Abstract base class for QA models."""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        """
        Initialize base QA model.
        
        Args:
            model_name: Name of the model
            config: Model configuration
        """
        self.model_name = model_name
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None
        
        logger.info(f"Initialized base QA model: {model_name}")
    
    @abstractmethod
    def load_model(self):
        """Load the model and tokenizer."""
        pass
    
    @abstractmethod
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass of the model."""
        pass
    
    @abstractmethod
    def predict(self, question: str, context: str) -> Dict[str, Any]:
        """Make a prediction for a single question-context pair."""
        pass
    
    @abstractmethod
    def batch_predict(self, examples: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Make predictions for a batch of examples."""
        pass
    
    def to_device(self, device: torch.device):
        """Move model to specified device."""
        self.device = device
        if self.model is not None:
            self.model = self.model.to(device)
        logger.info(f"Moved model to device: {device}")
    
    def save_model(self, output_dir: str):
        """Save model and tokenizer to directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.model is not None:
            # Save model
            model_path = output_path / "pytorch_model.bin"
            torch.save(self.model.state_dict(), model_path)
            
            # Save config
            config_path = output_path / "config.json"
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"Saved model to {output_path}")
        
        if self.tokenizer is not None:
            # Save tokenizer
            self.tokenizer.save_pretrained(output_path)
            logger.info(f"Saved tokenizer to {output_path}")
    
    def load_from_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        if (checkpoint_path / "pytorch_model.bin").exists():
            model_path = checkpoint_path / "pytorch_model.bin"
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"Loaded model from {model_path}")
        
        if (checkpoint_path / "config.json").exists():
            config_path = checkpoint_path / "config.json"
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
            self.config.update(loaded_config)
            logger.info(f"Loaded config from {config_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = {
            "model_name": self.model_name,
            "config": self.config,
            "device": str(self.device) if self.device else None,
            "num_parameters": self._count_parameters() if self.model else 0
        }
        return info
    
    def _count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train_mode(self):
        """Set model to training mode."""
        if self.model is not None:
            self.model.train()
    
    def eval_mode(self):
        """Set model to evaluation mode."""
        if self.model is not None:
            self.model.eval()


class QAModelConfig:
    """Configuration class for QA models."""
    
    def __init__(self, 
                 model_name: str,
                 max_length: int = 384,
                 max_target_length: int = 64,
                 num_labels: int = 2,
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1,
                 **kwargs):
        """
        Initialize QA model configuration.
        
        Args:
            model_name: Name of the model
            max_length: Maximum sequence length
            max_target_length: Maximum target sequence length (for T5)
            num_labels: Number of labels (for BERT)
            hidden_dropout_prob: Hidden layer dropout probability
            attention_probs_dropout_prob: Attention dropout probability
            **kwargs: Additional configuration parameters
        """
        self.model_name = model_name
        self.max_length = max_length
        self.max_target_length = max_target_length
        self.num_labels = num_labels
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        
        # Add any additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "max_target_length": self.max_target_length,
            "num_labels": self.num_labels,
            "hidden_dropout_prob": self.hidden_dropout_prob,
            "attention_probs_dropout_prob": self.attention_probs_dropout_prob
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'QAModelConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)


class ModelFactory:
    """Factory class for creating QA models."""
    
    @staticmethod
    def create_model(model_type: str, model_name: str, config: Dict[str, Any]) -> BaseQAModel:
        """
        Create a QA model instance.
        
        Args:
            model_type: Type of model ("mbert" or "mt5")
            model_name: Name of the model
            config: Model configuration
            
        Returns:
            QA model instance
        """
        if model_type.lower() == "mbert":
            from .mbert_qa import MBertQAModel
            # Add model_name to config
            config_with_name = config.copy()
            config_with_name['model_name'] = model_name
            return MBertQAModel(model_name, config_with_name)
        elif model_type.lower() == "mt5":
            from .mt5_qa import MT5QAModel
            # Add model_name to config
            config_with_name = config.copy()
            config_with_name['model_name'] = model_name
            return MT5QAModel(model_name, config_with_name)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def create_from_config(config_path: str) -> BaseQAModel:
        """
        Create model from configuration file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            QA model instance
        """
        import yaml
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        model_type = config.get('model_type', 'mbert')
        model_name = config.get('model_name', 'bert-base-multilingual-cased')
        
        return ModelFactory.create_model(model_type, model_name, config)


def create_model(model_type: str, model_name: str, config: Dict[str, Any]) -> BaseQAModel:
    """
    Convenience function to create a QA model.
    
    Args:
        model_type: Type of model ("mbert" or "mt5")
        model_name: Name of the model
        config: Model configuration
        
    Returns:
        QA model instance
    """
    return ModelFactory.create_model(model_type, model_name, config)
