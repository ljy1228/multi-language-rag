"""
Multilingual BERT QA model implementation.
Handles span extraction for cross-lingual question answering.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import Dict, List, Optional, Any, Tuple
import logging
import numpy as np

from .base_model import BaseQAModel, QAModelConfig

logger = logging.getLogger(__name__)


class MBertQAHead(nn.Module):
    """QA head for mBERT model."""
    
    def __init__(self, hidden_size: int, num_labels: int = 2, dropout_prob: float = 0.1):
        """
        Initialize mBERT QA head.
        
        Args:
            hidden_size: Hidden size of the model
            num_labels: Number of labels (start and end positions)
            dropout_prob: Dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        
        self.dropout = nn.Dropout(dropout_prob)
        self.qa_outputs = nn.Linear(hidden_size, num_labels)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights of the QA head."""
        nn.init.xavier_uniform_(self.qa_outputs.weight)
        nn.init.constant_(self.qa_outputs.bias, 0)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the QA head.
        
        Args:
            hidden_states: Hidden states from the model
            
        Returns:
            Logits for start and end positions
        """
        hidden_states = self.dropout(hidden_states)
        logits = self.qa_outputs(hidden_states)
        
        return logits


class MBertQAModel(BaseQAModel):
    """Multilingual BERT model for question answering."""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        """
        Initialize mBERT QA model.
        
        Args:
            model_name: Name of the model
            config: Model configuration
        """
        super().__init__(model_name, config)
        self.model_config = QAModelConfig.from_dict(config)
        self.qa_head = None
        
        logger.info(f"Initialized mBERT QA model: {model_name}")
    
    def load_model(self):
        """Load the mBERT model and tokenizer."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # Create QA head
            hidden_size = self.model.config.hidden_size
            self.qa_head = MBertQAHead(
                hidden_size=hidden_size,
                num_labels=self.model_config.num_labels,
                dropout_prob=self.model_config.hidden_dropout_prob
            )
            
            logger.info(f"Loaded mBERT model: {self.model_name}")
            logger.info(f"Model hidden size: {hidden_size}")
            
        except Exception as e:
            logger.error(f"Failed to load mBERT model: {e}")
            raise
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the mBERT QA model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing logits and other outputs
        """
        if self.model is None or self.qa_head is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Get model outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Get last hidden state
        last_hidden_state = outputs.last_hidden_state
        
        # Get QA logits
        qa_logits = self.qa_head(last_hidden_state)
        
        # Split into start and end logits
        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        return {
            "start_logits": start_logits,
            "end_logits": end_logits,
            "hidden_states": outputs.hidden_states,
            "last_hidden_state": last_hidden_state
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
        
        # Tokenize input
        inputs = self.tokenizer(
            question,
            context,
            truncation=True,
            max_length=self.model_config.max_length,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=True,
            return_tensors="pt"
        )
        
        # Move to device
        if self.device is not None:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.forward(**inputs)
        
        # Get predictions
        start_logits = outputs["start_logits"]
        end_logits = outputs["end_logits"]
        
        # Get best predictions
        start_pred = torch.argmax(start_logits, dim=-1)
        end_pred = torch.argmax(end_logits, dim=-1)
        
        # Calculate confidence scores
        start_confidence = F.softmax(start_logits, dim=-1).max(dim=-1)[0]
        end_confidence = F.softmax(end_logits, dim=-1).max(dim=-1)[0]
        confidence = (start_confidence * end_confidence).mean().item()
        
        # Extract answer text
        answer_text = self._extract_answer_text(
            inputs["input_ids"],
            start_pred,
            end_pred,
            inputs.get("offset_mapping", None)
        )
        
        return {
            "answer": answer_text,
            "start_position": start_pred.item(),
            "end_position": end_pred.item(),
            "confidence": confidence,
            "start_logits": start_logits.cpu().numpy(),
            "end_logits": end_logits.cpu().numpy()
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
        batch_size = 8
        for i in range(0, len(examples), batch_size):
            batch_examples = examples[i:i + batch_size]
            
            # Prepare batch
            questions = [ex["question"] for ex in batch_examples]
            contexts = [ex["context"] for ex in batch_examples]
            
            # Tokenize batch
            inputs = self.tokenizer(
                questions,
                contexts,
                truncation=True,
                max_length=self.model_config.max_length,
                stride=128,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding=True,
                return_tensors="pt"
            )
            
            # Move to device
            if self.device is not None:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Forward pass
            with torch.no_grad():
                outputs = self.forward(**inputs)
            
            # Process predictions
            start_logits = outputs["start_logits"]
            end_logits = outputs["end_logits"]
            
            start_preds = torch.argmax(start_logits, dim=-1)
            end_preds = torch.argmax(end_logits, dim=-1)
            
            # Calculate confidence scores
            start_confidences = F.softmax(start_logits, dim=-1).max(dim=-1)[0]
            end_confidences = F.softmax(end_logits, dim=-1).max(dim=-1)[0]
            confidences = start_confidences * end_confidences
            
            # Extract answers
            for j in range(len(batch_examples)):
                answer_text = self._extract_answer_text(
                    inputs["input_ids"][j:j+1],
                    start_preds[j:j+1],
                    end_preds[j:j+1],
                    inputs.get("offset_mapping", None)[j:j+1] if inputs.get("offset_mapping") is not None else None
                )
                
                predictions.append({
                    "answer": answer_text,
                    "start_position": start_preds[j].item(),
                    "end_position": end_preds[j].item(),
                    "confidence": confidences[j].item(),
                    "start_logits": start_logits[j].cpu().numpy(),
                    "end_logits": end_logits[j].cpu().numpy()
                })
        
        return predictions
    
    def _extract_answer_text(self, 
                           input_ids: torch.Tensor, 
                           start_pos: torch.Tensor, 
                           end_pos: torch.Tensor,
                           offset_mapping: Optional[torch.Tensor] = None) -> str:
        """
        Extract answer text from token positions.
        
        Args:
            input_ids: Input token IDs
            start_pos: Start position
            end_pos: End position
            offset_mapping: Offset mapping for character positions
            
        Returns:
            Extracted answer text
        """
        if self.tokenizer is None:
            return ""
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Ensure valid positions
        start_pos = max(0, min(start_pos.item(), len(tokens) - 1))
        end_pos = max(start_pos, min(end_pos.item(), len(tokens) - 1))
        
        # Extract answer tokens
        answer_tokens = tokens[start_pos:end_pos + 1]
        
        # Clean up tokens
        answer_tokens = [token for token in answer_tokens if token not in ['[CLS]', '[SEP]', '[PAD]']]
        
        # Decode answer
        answer = self.tokenizer.convert_tokens_to_string(answer_tokens)
        
        return answer.strip()
    
    def compute_loss(self, 
                    input_ids: torch.Tensor, 
                    attention_mask: torch.Tensor,
                    start_positions: torch.Tensor,
                    end_positions: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for training.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            start_positions: Ground truth start positions
            end_positions: Ground truth end positions
            
        Returns:
            Loss tensor
        """
        outputs = self.forward(input_ids, attention_mask)
        
        start_logits = outputs["start_logits"]
        end_logits = outputs["end_logits"]
        
        # Compute loss
        start_loss = F.cross_entropy(start_logits, start_positions)
        end_loss = F.cross_entropy(end_logits, end_positions)
        
        total_loss = start_loss + end_loss
        
        return total_loss
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = super().get_model_info()
        info.update({
            "model_type": "mbert",
            "hidden_size": self.model.config.hidden_size if self.model else None,
            "num_layers": self.model.config.num_hidden_layers if self.model else None,
            "num_attention_heads": self.model.config.num_attention_heads if self.model else None
        })
        return info
