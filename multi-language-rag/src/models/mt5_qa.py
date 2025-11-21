"""
Multilingual T5 QA model implementation.
Handles text-to-text generation for cross-lingual question answering.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from typing import Dict, List, Optional, Any, Tuple
import logging
import numpy as np

from .base_model import BaseQAModel, QAModelConfig

logger = logging.getLogger(__name__)


class MT5QAModel(BaseQAModel):
    """Multilingual T5 model for question answering."""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        """
        Initialize mT5 QA model.
        
        Args:
            model_name: Name of the model
            config: Model configuration
        """
        super().__init__(model_name, config)
        self.model_config = QAModelConfig.from_dict(config)
        
        logger.info(f"Initialized mT5 QA model: {model_name}")
    
    def load_model(self):
        """Load the mT5 model and tokenizer."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            # Resize token embeddings if needed
            if len(self.tokenizer) != self.model.config.vocab_size:
                self.model.resize_token_embeddings(len(self.tokenizer))
            
            logger.info(f"Loaded mT5 model: {self.model_name}")
            logger.info(f"Model vocab size: {self.model.config.vocab_size}")
            
        except Exception as e:
            logger.error(f"Failed to load mT5 model: {e}")
            raise
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the mT5 QA model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels for training
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing logits and other outputs
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Get model outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True
        )
        
        return {
            "logits": outputs.logits,
            "loss": outputs.loss,
            "hidden_states": outputs.hidden_states,
            "decoder_hidden_states": outputs.decoder_hidden_states
        }
    
    def predict(self, question: str, context: str) -> Dict[str, Any]:
        """
        Make a prediction for a single question-context pair.
        
        Args:
            question: Question text
            context: Context text
            
        Returns:
            Dictionary containing prediction results
        """
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Format input as text-to-text
        input_text = f"question: {question} context: {context}"
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.model_config.max_length,
            padding=True,
            return_tensors="pt"
        )
        
        # Move to device
        if self.device is not None:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate answer
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.model_config.max_target_length,
                num_beams=self.model_config.get('num_beams', 4),
                early_stopping=self.model_config.get('early_stopping', True),
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Decode answer
        answer = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
        # Calculate confidence score (average of beam scores)
        if hasattr(outputs, 'sequences_scores') and outputs.sequences_scores is not None:
            confidence = torch.exp(outputs.sequences_scores[0]).item()
        else:
            confidence = 1.0
        
        return {
            "answer": answer,
            "confidence": confidence,
            "input_text": input_text,
            "generated_tokens": outputs.sequences[0].cpu().numpy()
        }
    
    def batch_predict(self, examples: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Make predictions for a batch of examples.
        
        Args:
            examples: List of dictionaries with 'question' and 'context' keys
            
        Returns:
            List of prediction results
        """
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        predictions = []
        
        # Process in batches to avoid memory issues
        batch_size = 4  # Smaller batch size for generation
        for i in range(0, len(examples), batch_size):
            batch_examples = examples[i:i + batch_size]
            
            # Prepare batch
            input_texts = []
            for ex in batch_examples:
                input_text = f"question: {ex['question']} context: {ex['context']}"
                input_texts.append(input_text)
            
            # Tokenize batch
            inputs = self.tokenizer(
                input_texts,
                truncation=True,
                max_length=self.model_config.max_length,
                padding=True,
                return_tensors="pt"
            )
            
            # Move to device
            if self.device is not None:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate answers
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.model_config.max_target_length,
                    num_beams=self.model_config.get('num_beams', 4),
                    early_stopping=self.model_config.get('early_stopping', True),
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            # Process predictions
            for j in range(len(batch_examples)):
                answer = self.tokenizer.decode(outputs.sequences[j], skip_special_tokens=True)
                
                # Calculate confidence score
                if hasattr(outputs, 'sequences_scores') and outputs.sequences_scores is not None:
                    confidence = torch.exp(outputs.sequences_scores[j]).item()
                else:
                    confidence = 1.0
                
                predictions.append({
                    "answer": answer,
                    "confidence": confidence,
                    "input_text": input_texts[j],
                    "generated_tokens": outputs.sequences[j].cpu().numpy()
                })
        
        return predictions
    
    def compute_loss(self, 
                    input_ids: torch.Tensor, 
                    attention_mask: torch.Tensor,
                    labels: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for training.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels
            
        Returns:
            Loss tensor
        """
        outputs = self.forward(input_ids, attention_mask, labels=labels)
        return outputs["loss"]
    
    def generate_answer(self, 
                       question: str, 
                       context: str,
                       max_length: Optional[int] = None,
                       num_beams: Optional[int] = None,
                       temperature: float = 1.0,
                       do_sample: bool = False) -> str:
        """
        Generate answer with custom generation parameters.
        
        Args:
            question: Question text
            context: Context text
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            
        Returns:
            Generated answer text
        """
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Format input
        input_text = f"question: {question} context: {context}"
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.model_config.max_length,
            padding=True,
            return_tensors="pt"
        )
        
        # Move to device
        if self.device is not None:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Set generation parameters
        gen_kwargs = {
            "max_length": max_length or self.model_config.max_target_length,
            "num_beams": num_beams or self.model_config.get('num_beams', 4),
            "early_stopping": self.model_config.get('early_stopping', True),
            "do_sample": do_sample,
            "temperature": temperature if do_sample else None
        }
        
        # Remove None values
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # Decode
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return answer
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = super().get_model_info()
        info.update({
            "model_type": "mt5",
            "vocab_size": self.model.config.vocab_size if self.model else None,
            "d_model": self.model.config.d_model if self.model else None,
            "num_layers": self.model.config.num_layers if self.model else None,
            "num_decoder_layers": self.model.config.num_decoder_layers if self.model else None,
            "num_heads": self.model.config.num_heads if self.model else None
        })
        return info
    
    def save_model(self, output_dir: str):
        """Save model and tokenizer to directory."""
        import os
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.model is not None:
            # Save model using transformers save_pretrained
            self.model.save_pretrained(output_path)
            logger.info(f"Saved mT5 model to {output_path}")
        
        if self.tokenizer is not None:
            # Save tokenizer
            self.tokenizer.save_pretrained(output_path)
            logger.info(f"Saved tokenizer to {output_path}")
    
    def load_from_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        from pathlib import Path
        
        checkpoint_path = Path(checkpoint_path)
        
        if (checkpoint_path / "config.json").exists():
            # Load using transformers from_pretrained
            self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
            logger.info(f"Loaded mT5 model from {checkpoint_path}")
        else:
            logger.error(f"No valid checkpoint found at {checkpoint_path}")
            raise FileNotFoundError(f"No valid checkpoint found at {checkpoint_path}")
