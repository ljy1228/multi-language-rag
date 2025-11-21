
import os
from typing import Dict, List, Optional, Union, Any
from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class XQuadINLoader:
    def __init__(self, dataset_path:Optional[str] = None):
        self.dataset_path = dataset_path
        
    def load_data_evaluation(self, languages: Optional[List[str]] = None) -> Dict[str, Dataset]:
        data_per_language = {}

        for lang in languages :
            file_path_dev = os.path.join(self.dataset_path, f'xorqa_{lang}_dev.json')
            with open(file_path_dev, "r", encoding="utf-8") as f:
                json_data_dev = json.load(f)
            print(type(json_data_dev))
            data_per_language[lang] = json_data_dev
        return data_per_language
    
    def load_data_training(self, languages: Optional[List[str]] = None) -> Dict[str, Dataset]:
        data_per_language = {}

        for lang in languages :
            file_path_train = os.path.join(self.dataset_path, f'xorqa_{lang}_train.json')
            with open(file_path_train, "r", encoding="utf-8") as f:
                json_data_train = json.load(f)
            print(type(json_data_train))
            data_per_language[lang] = json_data_train
            
        return data_per_language

class DifferentlangDataLoader:
    def __init__(self, dataset_path:Optional[str] = None):
        self.dataset_path = dataset_path
        self.xquadin_dataloader = XQuadINLoader(dataset_path=dataset_path)
        logger.info("Initialized Differentlang data loader")