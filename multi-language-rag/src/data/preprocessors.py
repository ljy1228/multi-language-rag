"""
Data preprocessors for cross-lingual QA.
Handles tokenization, formatting, and language-specific normalization.
"""

import re
import unicodedata
from typing import Dict, List, Optional, Any, Union, Tuple
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QAExample:
    """Data class for QA examples."""
    question: str
    context: str
    answer: str
    answer_start: Optional[int] = None
    language: Optional[str] = None
    is_impossible: bool = False
    id: Optional[str] = None


class TextNormalizer:
    """Handles text normalization for different languages."""
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize whitespace in text."""
        return re.sub(r'\s+', ' ', text.strip())
    
    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalize unicode characters."""
        return unicodedata.normalize('NFKC', text)
    
    @staticmethod
    def remove_control_characters(text: str) -> str:
        """Remove control characters from text."""
        return ''.join(char for char in text if unicodedata.category(char)[0] != 'C')
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Apply all normalization steps."""
        text = TextNormalizer.normalize_unicode(text)
        text = TextNormalizer.remove_control_characters(text)
        text = TextNormalizer.normalize_whitespace(text)
        return text


class QATokenizer:
    """Tokenizer for QA tasks with support for mBERT and mT5."""
    
    def __init__(self, model_name: str, max_length: int = 384, max_target_length: int = 64):
        """
        Initialize QA tokenizer.
        
        Args:
            model_name: Name of the model/tokenizer
            max_length: Maximum sequence length
            max_target_length: Maximum target sequence length (for T5)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.max_target_length = max_target_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add special tokens if needed
        if 't5' in model_name.lower():
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Initialized tokenizer for {model_name}")
    
    def tokenize_qa_example(self, example: QAExample, model_type: str = "bert") -> Dict[str, Any]:
        """
        Tokenize a QA example for the specified model type.
        
        Args:
            example: QA example to tokenize
            model_type: Type of model ("bert" or "t5")
            
        Returns:
            Tokenized example
        """
        if model_type in ["bert", "mbert"]:
            return self._tokenize_for_bert(example)
        elif model_type in ["t5", "mt5"]:
            return self._tokenize_for_t5(example)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _tokenize_for_bert(self, example: QAExample) -> Dict[str, Any]:
        """Tokenize example for BERT-style models."""
        # Normalize text
        question = TextNormalizer.normalize_text(example.question)
        context = TextNormalizer.normalize_text(example.context)
        answer = TextNormalizer.normalize_text(example.answer)
        
        # Tokenize
        tokenized = self.tokenizer(
            question,
            context,
            truncation=True,
            max_length=self.max_length,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length"
        )
        
        # Find answer positions
        start_positions = []
        end_positions = []
        
        for i, offset_mapping in enumerate(tokenized["offset_mapping"]):
            # Find answer start and end in the context
            answer_start = example.answer_start
            if answer_start is not None:
                # Find the token that contains the answer start
                start_token = None
                end_token = None
                
                for j, (start, end) in enumerate(offset_mapping):
                    if start <= answer_start < end:
                        start_token = j
                    if start <= answer_start + len(answer) <= end:
                        end_token = j
                        break
                
                if start_token is not None and end_token is not None:
                    start_positions.append(start_token)
                    end_positions.append(end_token)
                else:
                    start_positions.append(0)
                    end_positions.append(0)
            else:
                start_positions.append(0)
                end_positions.append(0)
        
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "start_positions": start_positions,
            "end_positions": end_positions,
            "overflow_to_sample_mapping": tokenized.get("overflow_to_sample_mapping", [0] * len(tokenized["input_ids"])),
            "offset_mapping": tokenized["offset_mapping"]
        }
    
    def _tokenize_for_t5(self, example: QAExample) -> Dict[str, Any]:
        """Tokenize example for T5-style models."""
        # Normalize text
        question = TextNormalizer.normalize_text(example.question)
        context = TextNormalizer.normalize_text(example.context)
        answer = TextNormalizer.normalize_text(example.answer)
        
        # Format as text-to-text
        input_text = f"question: {question} context: {context}"
        
        # Tokenize input
        input_tokenized = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Tokenize target
        target_tokenized = self.tokenizer(
            answer,
            truncation=True,
            max_length=self.max_target_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": input_tokenized["input_ids"].squeeze(),
            "attention_mask": input_tokenized["attention_mask"].squeeze(),
            "labels": target_tokenized["input_ids"].squeeze()
        }
    
    def batch_tokenize(self, examples: List[QAExample], model_type: str = "bert") -> Dict[str, List[Any]]:
        """
        Tokenize a batch of examples.
        
        Args:
            examples: List of QA examples
            model_type: Type of model ("bert" or "t5")
            
        Returns:
            Batch tokenized data
        """
        batch_data = {
            "input_ids": [],
            "attention_mask": [],
            "start_positions": [],
            "end_positions": [],
            "labels": [],
            "overflow_to_sample_mapping": [],
            "offset_mapping": []
        }
        
        for example in examples:
            tokenized = self.tokenize_qa_example(example, model_type)
            
            for key, value in tokenized.items():
                if key in batch_data:
                    if isinstance(value, list):
                        batch_data[key].extend(value)
                    else:
                        batch_data[key].append(value)
        
        return batch_data


class DataPreprocessor:
    """Main data preprocessor for cross-lingual QA."""
    
    def __init__(self, model_name: str, model_type: str = "bert", max_length: int = 384):
        """
        Initialize data preprocessor.
        
        Args:
            model_name: Name of the model
            model_type: Type of model ("bert" or "t5")
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.model_type = model_type
        self.max_length = max_length
        self.tokenizer = QATokenizer(model_name, max_length)
        
        logger.info(f"Initialized preprocessor for {model_name} ({model_type})")
    
    def preprocess_examples(self, examples: List[Dict[str, Any]]) -> List[QAExample]:
        """
        Preprocess raw examples into QAExample objects.
        
        Args:
            examples: List of raw examples
            
        Returns:
            List of QAExample objects
        """
        processed_examples = []
        
        for example in examples:
            # Extract fields based on dataset format
            if 'question' in example and 'context' in example:
                # XQuAD/SQuAD format
                qa_example = QAExample(
                    question=example['question'],
                    context=example['context'],
                    answer=example.get('answers', {}).get('text', [''])[0] if 'answers' in example and example.get('answers', {}).get('text') else example.get('answer', ''),
                    answer_start=example.get('answers', {}).get('answer_start', [None])[0] if 'answers' in example and example.get('answers', {}).get('answer_start') else example.get('answer_start'),
                    language=example.get('language', 'en'),
                    is_impossible=example.get('is_impossible', False),
                    id=example.get('id', '')
                )
            else:
                logger.warning(f"Skipping example with missing fields: {example.keys()}")
                continue
            
            processed_examples.append(qa_example)
        
        logger.info(f"Preprocessed {len(processed_examples)} examples")
        return processed_examples
    
    def create_dataset(self, examples: List[QAExample]) -> torch.utils.data.Dataset:
        """
        Create PyTorch dataset from QA examples.
        
        Args:
            examples: List of QA examples
            
        Returns:
            PyTorch dataset
        """
        tokenized_data = self.tokenizer.batch_tokenize(examples, self.model_type)
        
        # Convert to tensors
        dataset_dict = {}
        for key, values in tokenized_data.items():
            if values:  # Only add non-empty lists
                if key in ["input_ids", "attention_mask", "labels"]:
                    dataset_dict[key] = torch.tensor(values, dtype=torch.long)
                elif key in ["start_positions", "end_positions"]:
                    dataset_dict[key] = torch.tensor(values, dtype=torch.long)
                else:
                    dataset_dict[key] = values
        
        return QADataset(dataset_dict)
    
    def preprocess_for_training(self, examples: List[Dict[str, Any]]) -> torch.utils.data.Dataset:
        """
        Preprocess examples for training.
        
        Args:
            examples: List of raw examples
            
        Returns:
            PyTorch dataset ready for training
        """
        qa_examples = self.preprocess_examples(examples)
        return self.create_dataset(qa_examples)
    
    def preprocess_for_evaluation(self, examples: List[Dict[str, Any]]) -> Tuple[torch.utils.data.Dataset, List[QAExample]]:
        """
        Preprocess examples for evaluation.
        
        Args:
            examples: List of raw examples
            
        Returns:
            Tuple of (PyTorch dataset, original QA examples)
        """
        qa_examples = self.preprocess_examples(examples)
        dataset = self.create_dataset(qa_examples)
        return dataset, qa_examples


class QADataset(torch.utils.data.Dataset):
    """PyTorch dataset for QA examples."""
    
    def __init__(self, data_dict: Dict[str, Any]):
        """
        Initialize QA dataset.
        
        Args:
            data_dict: Dictionary containing tokenized data
        """
        self.data_dict = data_dict
        self.length = len(next(iter(data_dict.values())))
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {key: values[idx] for key, values in self.data_dict.items()}


def create_preprocessor(model_name: str, model_type: str = "bert", max_length: int = 384) -> DataPreprocessor:
    """
    Create a data preprocessor instance.
    
    Args:
        model_name: Name of the model
        model_type: Type of model ("bert" or "t5")
        max_length: Maximum sequence length
        
    Returns:
        DataPreprocessor instance
    """
    return DataPreprocessor(model_name, model_type, max_length)
