import random
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import logging
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class SamplingConfig:
    """Configuration for samesampling."""
    num_shots: int
    strategy: str = "random"  # Options: random, diverse, stratified
    seed: int = 42
    # max_samples_per_language: int = 50
    diversity_threshold: float = 0.3

class SameSampler:
    def __init__(self, config: SamplingConfig):
        self.config = config
        self.random_state = np.random.RandomState(config.seed)
        random.seed(config.seed)


class CrossLingualSameSampler:
    def __init__(self, config: SamplingConfig):
        """
        Initialize cross-lingual few-shot sampler.
        
        Args:
            config: Sampling configuration
        """
        self.config = config
        self.sampler = SameSampler(config)
        
        logger.info(f"Initialized cross-lingual few-shot sampler")