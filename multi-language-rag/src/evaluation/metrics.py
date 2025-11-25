"""
Evaluation metrics for cross-lingual QA.
Implements EM, F1, BLEU, and other metrics with language-specific normalization.
"""

import re
import string
import unicodedata
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class TextNormalizer:
    """Text normalization utilities for different languages."""
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize whitespace in text."""
        return re.sub(r'\s+', ' ', text.strip())
    
    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalize unicode characters."""
        return unicodedata.normalize('NFKC', text)
    
    @staticmethod
    def remove_punctuation(text: str) -> str:
        """Remove punctuation from text."""
        return text.translate(str.maketrans('', '', string.punctuation))
    
    @staticmethod
    def normalize_text(text: str, remove_punct: bool = False) -> str:
        """
        Apply comprehensive text normalization.
        
        Args:
            text: Input text
            remove_punct: Whether to remove punctuation
            
        Returns:
            Normalized text
        """
        text = TextNormalizer.normalize_unicode(text)
        text = TextNormalizer.normalize_whitespace(text)
        
        if remove_punct:
            text = TextNormalizer.remove_punctuation(text)
        
        return text.lower()


class ExactMatch:
    """Exact Match metric for QA evaluation."""
    
    def __init__(self, normalize: bool = True):
        """
        Initialize Exact Match metric.
        
        Args:
            normalize: Whether to normalize text before comparison
        """
        self.normalize = normalize
    
    def compute(self, predictions: List[str], references: List[str]) -> float:
        """
        Compute Exact Match score.
        
        Args:
            predictions: List of predicted answers
            references: List of reference answers
            
        Returns:
            Exact Match score (0.0 to 1.0)
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")
        
        matches = 0
        for pred, ref in zip(predictions, references):
            if self._is_exact_match(pred, ref):
                matches += 1
        
        return matches / len(predictions)
    
    def _is_exact_match(self, prediction: str, reference: str) -> bool:
        """
        Check if prediction exactly matches reference.
        
        Args:
            prediction: Predicted answer
            reference: Reference answer
            
        Returns:
            True if exact match, False otherwise
        """
        if self.normalize:
            prediction = TextNormalizer.normalize_text(prediction, remove_punct=True)
            reference = TextNormalizer.normalize_text(reference, remove_punct=True)
        
        return prediction == reference


class F1Score:
    """F1 score metric for QA evaluation."""
    
    def __init__(self, normalize: bool = True):
        """
        Initialize F1 score metric.
        
        Args:
            normalize: Whether to normalize text before comparison
        """
        self.normalize = normalize
    
    def compute(self, predictions: List[str], references: List[str]) -> float:
        """
        Compute F1 score.
        
        Args:
            predictions: List of predicted answers
            references: List of reference answers
            
        Returns:
            F1 score (0.0 to 1.0)
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")
        
        f1_scores = []
        for pred, ref in zip(predictions, references):
            f1 = self._compute_f1(pred, ref)
            f1_scores.append(f1)
        
        return np.mean(f1_scores)
    
    def _compute_f1(self, prediction: str, reference: str) -> float:
        """
        Compute F1 score for a single prediction-reference pair.
        
        Args:
            prediction: Predicted answer
            reference: Reference answer
            
        Returns:
            F1 score (0.0 to 1.0)
        """
        if self.normalize:
            prediction = TextNormalizer.normalize_text(prediction, remove_punct=True)
            reference = TextNormalizer.normalize_text(reference, remove_punct=True)
        
        # Tokenize
        pred_tokens = prediction.split()
        ref_tokens = reference.split()
        
        if len(pred_tokens) == 0 and len(ref_tokens) == 0:
            return 1.0
        elif len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0
        
        # Compute precision and recall
        pred_counter = Counter(pred_tokens)
        ref_counter = Counter(ref_tokens)
        
        # Common tokens
        common_tokens = set(pred_tokens) & set(ref_tokens)
        
        # Precision: common tokens / predicted tokens
        precision = sum(min(pred_counter[token], ref_counter[token]) for token in common_tokens) / len(pred_tokens)
        
        # Recall: common tokens / reference tokens
        recall = sum(min(pred_counter[token], ref_counter[token]) for token in common_tokens) / len(ref_tokens)
        
        # F1 score
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1


class BLEUScore:
    """BLEU score metric for generated text evaluation."""
    
    def __init__(self, max_n: int = 4, weights: Optional[List[float]] = None):
        """
        Initialize BLEU score metric.
        
        Args:
            max_n: Maximum n-gram order
            weights: Weights for different n-gram orders
        """
        self.max_n = max_n
        self.weights = weights or [0.25, 0.25, 0.25, 0.25]
    
    def compute(self, predictions: List[str], references: List[str]) -> float:
        """
        Compute BLEU score.
        
        Args:
            predictions: List of predicted answers
            references: List of reference answers
            
        Returns:
            BLEU score (0.0 to 1.0)
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")
        
        bleu_scores = []
        for pred, ref in zip(predictions, references):
            bleu = self._compute_bleu(pred, ref)
            bleu_scores.append(bleu)
        
        return np.mean(bleu_scores)
    
    def _compute_bleu(self, prediction: str, reference: str) -> float:
        """
        Compute BLEU score for a single prediction-reference pair.
        
        Args:
            prediction: Predicted answer
            reference: Reference answer
            
        Returns:
            BLEU score (0.0 to 1.0)
        """
        # Normalize text
        prediction = TextNormalizer.normalize_text(prediction)
        reference = TextNormalizer.normalize_text(reference)
        
        # Tokenize
        pred_tokens = prediction.split()
        ref_tokens = reference.split()
        
        if len(pred_tokens) == 0:
            return 0.0
        
        # Compute n-gram precisions
        precisions = []
        for n in range(1, self.max_n + 1):
            precision = self._compute_ngram_precision(pred_tokens, ref_tokens, n)
            precisions.append(precision)
        
        # Compute brevity penalty
        bp = self._compute_brevity_penalty(pred_tokens, ref_tokens)
        
        # Compute BLEU score
        if any(p == 0 for p in precisions):
            return 0.0
        
        bleu = bp * np.exp(sum(w * np.log(p) for w, p in zip(self.weights, precisions)))
        return bleu
    
    def _compute_ngram_precision(self, pred_tokens: List[str], ref_tokens: List[str], n: int) -> float:
        """Compute n-gram precision."""
        pred_ngrams = self._get_ngrams(pred_tokens, n)
        ref_ngrams = self._get_ngrams(ref_tokens, n)
        
        if len(pred_ngrams) == 0:
            return 0.0
        
        # Count matches
        matches = 0
        for ngram in pred_ngrams:
            if ngram in ref_ngrams:
                matches += 1
        
        return matches / len(pred_ngrams)
    
    def _get_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """Get n-grams from tokens."""
        if len(tokens) < n:
            return []
        
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams.append(ngram)
        
        return ngrams
    
    def _compute_brevity_penalty(self, pred_tokens: List[str], ref_tokens: List[str]) -> float:
        """Compute brevity penalty."""
        pred_len = len(pred_tokens)
        ref_len = len(ref_tokens)
        
        if pred_len > ref_len:
            return 1.0
        
        return np.exp(1 - ref_len / pred_len)


class ROUGEScore:
    """ROUGE score metric for generated text evaluation."""
    
    def __init__(self, rouge_type: str = "rouge-l"):
        """
        Initialize ROUGE score metric.
        
        Args:
            rouge_type: Type of ROUGE metric ("rouge-l", "rouge-1", "rouge-2")
        """
        self.rouge_type = rouge_type
    
    def compute(self, predictions: List[str], references: List[str]) -> float:
        """
        Compute ROUGE score.
        
        Args:
            predictions: List of predicted answers
            references: List of reference answers
            
        Returns:
            ROUGE score (0.0 to 1.0)
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")
        
        rouge_scores = []
        for pred, ref in zip(predictions, references):
            rouge = self._compute_rouge(pred, ref)
            rouge_scores.append(rouge)
        
        return np.mean(rouge_scores)
    
    def _compute_rouge(self, prediction: str, reference: str) -> float:
        """
        Compute ROUGE score for a single prediction-reference pair.
        
        Args:
            prediction: Predicted answer
            reference: Reference answer
            
        Returns:
            ROUGE score (0.0 to 1.0)
        """
        # Normalize text
        prediction = TextNormalizer.normalize_text(prediction)
        reference = TextNormalizer.normalize_text(reference)
        
        # Tokenize
        pred_tokens = prediction.split()
        ref_tokens = reference.split()
        
        if len(pred_tokens) == 0 and len(ref_tokens) == 0:
            return 1.0
        elif len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0
        
        if self.rouge_type == "rouge-l":
            return self._compute_rouge_l(pred_tokens, ref_tokens)
        elif self.rouge_type == "rouge-1":
            return self._compute_rouge_n(pred_tokens, ref_tokens, 1)
        elif self.rouge_type == "rouge-2":
            return self._compute_rouge_n(pred_tokens, ref_tokens, 2)
        else:
            raise ValueError(f"Unknown ROUGE type: {self.rouge_type}")
    
    def _compute_rouge_l(self, pred_tokens: List[str], ref_tokens: List[str]) -> float:
        """Compute ROUGE-L score."""
        # Longest Common Subsequence
        lcs_length = self._compute_lcs(pred_tokens, ref_tokens)
        
        # Precision and recall
        precision = lcs_length / len(pred_tokens)
        recall = lcs_length / len(ref_tokens)
        
        # F1 score
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    def _compute_rouge_n(self, pred_tokens: List[str], ref_tokens: List[str], n: int) -> float:
        """Compute ROUGE-N score."""
        pred_ngrams = self._get_ngrams(pred_tokens, n)
        ref_ngrams = self._get_ngrams(ref_tokens, n)
        
        if len(pred_ngrams) == 0:
            return 0.0
        
        # Count matches
        matches = 0
        for ngram in pred_ngrams:
            if ngram in ref_ngrams:
                matches += 1
        
        # Precision and recall
        precision = matches / len(pred_ngrams)
        recall = matches / len(ref_ngrams)
        
        # F1 score
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    def _compute_lcs(self, seq1: List[str], seq2: List[str]) -> int:
        """Compute Longest Common Subsequence length."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]
    
    def _get_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """Get n-grams from tokens."""
        if len(tokens) < n:
            return []
        
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams.append(ngram)
        
        return ngrams


class QAMetrics:
    """Comprehensive QA evaluation metrics."""
    
    def __init__(self, 
                 metrics: Optional[List[str]] = None,
                 normalize: bool = True):
        """
        Initialize QA metrics.
        
        Args:
            metrics: List of metrics to compute
            normalize: Whether to normalize text before comparison
        """
        self.metrics = metrics or ['exact_match', 'f1', 'bleu']
        self.normalize = normalize
        
        # Initialize metric calculators
        self.exact_match = ExactMatch(normalize=normalize)
        self.f1_score = F1Score(normalize=normalize)
        self.bleu_score = BLEUScore()
        self.rouge_score = ROUGEScore()
    
    def compute(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute all specified metrics.
        
        Args:
            predictions: List of predicted answers
            references: List of reference answers
            
        Returns:
            Dictionary containing metric scores
        """
        results = {}
        
        if 'exact_match' in self.metrics:
            results['exact_match'] = self.exact_match.compute(predictions, references)
        
        if 'f1' in self.metrics:
            results['f1'] = self.f1_score.compute(predictions, references)
        
        if 'bleu' in self.metrics:
            results['bleu'] = self.bleu_score.compute(predictions, references)
        
        if 'rouge' in self.metrics:
            results['rouge'] = self.rouge_score.compute(predictions, references)
        
        return results
    
    def compute_per_sample(self, predictions: List[str], references: List[str]) -> Dict[str, List[float]]:
        """
        Compute metrics for each sample individually.
        
        Args:
            predictions: List of predicted answers
            references: List of reference answers
            
        Returns:
            Dictionary containing metric scores for each sample
        """
        results = {}
        
        if 'exact_match' in self.metrics:
            results['exact_match'] = [
                float(self.exact_match._is_exact_match(pred, ref))
                for pred, ref in zip(predictions, references)
            ]
        
        if 'f1' in self.metrics:
            results['f1'] = [
                self.f1_score._compute_f1(pred, ref)
                for pred, ref in zip(predictions, references)
            ]
        
        if 'bleu' in self.metrics:
            results['bleu'] = [
                self.bleu_score._compute_bleu(pred, ref)
                for pred, ref in zip(predictions, references)
            ]
        
        if 'rouge' in self.metrics:
            results['rouge'] = [
                self.rouge_score._compute_rouge(pred, ref)
                for pred, ref in zip(predictions, references)
            ]
        
        return results


def compute_qa_metrics(predictions: List[str], 
                      references: List[str],
                      metrics: Optional[List[str]] = None,
                      normalize: bool = True) -> Dict[str, float]:
    """
    Convenience function to compute QA metrics.
    
    Args:
        predictions: List of predicted answers
        references: List of reference answers
        metrics: List of metrics to compute
        normalize: Whether to normalize text before comparison
        
    Returns:
        Dictionary containing metric scores
    """
    qa_metrics = QAMetrics(metrics=metrics, normalize=normalize)
    return qa_metrics.compute(predictions, references)
