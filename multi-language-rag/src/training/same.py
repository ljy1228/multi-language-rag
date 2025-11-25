

from transformers import get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any, Tuple
import logging
import json
from pathlib import Path
import time
from tqdm import tqdm
from src.utils.device_utils import DeviceManager
from src.data.loaders import DifferentlangDataLoader
from src.data.preprocessors import DataPreprocessor
from src.data.same_sampler import SamplingConfig, CrossLingualSameSampler,SameSampler
from transformers import BitsAndBytesConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from src.training.trainer import SameTrainer
from src.utils.device_utils import DeviceManager, setup_mixed_precision
from accelerate import Accelerator

logger = logging.getLogger(__name__)
class SameExperimentNew:
    def __init__(self,
                 model,
                 tokenizer,
                 model_name: str,
                 config: Dict[str, Any]):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.config = config
        self.data_loader = DifferentlangDataLoader(dataset_path=config.get('dataset_path', './dataset'))
        self.preprocessor = DataPreprocessor(
            self.model_name,
            model_type=config.get('model_type', 'bert'),
            max_length=config.get('max_length', 384)
        )
        self.device_manager = DeviceManager()
        self.optimizer = None
        self.accelerator = Accelerator(
            mixed_precision="fp16" if setup_mixed_precision(self.device_manager) else "no",
            gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1)
        )
        self.global_step = 0
        self.scheduler = None
        self.training_history = []
    
    #对于每一轮的训练
    def train_epoch(self,train_dataloader, epoch):
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
        for batch in progress_bar:
            outputs = self.model(
                input_ids=batch["input_ids"].to(self.device_manager.get_device()),
                attention_mask=batch["attention_mask"].to(self.device_manager.get_device()),
                labels=batch["labels"].to(self.device_manager.get_device()),
            )
            #自动计算交叉熵损失
            loss = outputs.loss
            print(loss)
            #反向传播loss
            loss.backward()
            if self.config.get('max_grad_norm', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                    self.model.model.parameters(), 
                    self.config['max_grad_norm']
                )
            #更新参数
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # 记录loss
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {
            'train_loss': avg_loss,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def run_experiment(self,
                       languages: List[str] = None) -> Dict[str, Any]:

        #加载eval数据
        evaluation_data = self.data_loader.xquadin_dataloader.load_data_evaluation(languages)
        #加载train数据
        train_data = self.data_loader.xquadin_dataloader.load_data_training(languages)
        
        combined_training_data = []
        combined_eval_data = []
        for language, examples in train_data.items():
            combined_training_data.extend(examples['examples'])
        for language, examples in evaluation_data.items():
            combined_eval_data.extend(examples['examples'])
        if not combined_training_data:
            logger.warning("No training data available for samesampling.")
            return {}
        
        #将训练数据和评估数据进行预处理，将input_ids,attention_mask,labels等转换为模型可接受的格式
        train_dataset = self.preprocessor.preprocess_for_training(combined_training_data)
        eval_dataset, qa_examples= self.preprocessor.preprocess_for_evaluation(combined_eval_data)
        
        #数据加载器
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.get('train_batch_size', 8),
            shuffle=True,
            num_workers=self.config.get('dataloader_num_workers', 2),
            pin_memory=self.config.get('dataloader_pin_memory', True)
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=self.config.get('eval_batch_size', 16),
            shuffle=False,      
            num_workers=self.config.get('dataloader_num_workers', 2),
            pin_memory=self.config.get('dataloader_pin_memory', True)
        )
        
        
        num_training_steps = len(train_dataloader) * self.config.get('num_train_epochs', 3)
        #warmup steps 根据比例自动计算，通常默认 warmup_ratio=0.1，warm_up是用于让模型一开始用非常低的学习率，逐步升到设定的学习率，避免初期的梯度爆炸，loss崩溃，模型不稳定/
        warmup_steps = int(num_training_steps * self.config.get('warmup_ratio', 0.1))
        lr = self.config.get("learning_rate", 1e-5)   # 之前 loss nan 的话可以先用 1e-5 或更小
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        #动态控制训练过程中的学习率的变化，让模型更加稳定，收敛更快，线性 warmup + 线性衰减
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )

        num_epochs = self.config.get("num_train_epochs", 3)
        
        device = "cuda"
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {num_epochs}")
        total_loss = 0.0
        num_batches = 0
        #循环多个epoch
        for epoch in range(num_epochs):
            self.model.train()
            train_metrics = self.train_epoch(train_dataloader, epoch)
            self.model.eval()
            eval_metrics = self.eval_same(eval_dataloader)
            epoch_metrics = {**train_metrics, **eval_metrics}
            self.training_history.append(epoch_metrics)
            self.save_checkpoint(epoch_metrics,self.global_step)
            evaluation_results = self._eval_all_languages(evaluation_data)
        return {
        'training_history': self.training_history,
        'evaluation_results': evaluation_results,
        'num_training_examples': len(combined_training_data)
    }
            
            
    def _eval_all_languages(self,evaluation_data: Dict[str, List[Dict[str, Any]]]) :
        results = {}
        for language, examples in evaluation_data.items():
            eval_dataset, qa_examples = self.preprocessor.preprocess_for_evaluation(examples["examples"])
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=self.config.get('eval_batch_size', 16),
                shuffle=False,
                num_workers=self.config.get('dataloader_num_workers', 2),
                pin_memory=self.config.get('dataloader_pin_memory', True)
            )      
            eval_results = self._evaluate_language(eval_dataloader, qa_examples, language)
            results[language] = eval_results

            return results
                
            
    
    def _evaluate_language(self,eval_dataloader: DataLoader,
                          qa_examples: List[Any],
                          language: str) -> Dict[str, Any]:
        self.model.eval()
        predictions = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                # Move batch to device
                if self.device_manager.device is not None:
                    batch = {k: v.to(self.device_manager.device) for k, v in batch.items()}
                
                
                loss = self.model(**batch).loss
                total_loss += loss.item()
                num_batches += 1

                batch_examples = []
                for i in range(len(batch['input_ids'])):
                    batch_examples.append({
                        'question': f"question_{i}",
                        'context': f"context_{i}"
                        
                    })
                    
                batch_predictions = self.predict_generation(batch_examples)
                predictions.extend(batch_predictions)
        
        
        # Calculate metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
        results = {
            'loss': avg_loss,
            'num_examples': len(qa_examples),
            'predictions': predictions[:10] if predictions else [],  # Sample predictions
            'language': language
        }
        
        return results
    
    
    def predict_generation(self,examples):
        predictions = []
        
        # Process in batches to avoid memory issues
        batch_size = 4  # Smaller batch size for generation
        for i in range(0, len(examples), batch_size):
            batch_examples = examples[i:i + batch_size]
            
            # Prepare batch
            input_texts = []
            for ex in batch_examples:
                input_text = f"question: {ex['question']} context: {ex['context']} "
                input_texts.append(input_text)
            
            # Tokenize batch
            inputs = self.tokenizer(
                input_texts,
                truncation=True,
                max_length=self.config.get('max_length', 384),
                padding=True,
                return_tensors="pt"
            )
            
            # Move to device
            if self.device_manager.device is not None:
                inputs = {k: v.to(self.device_manager.device) for k, v in inputs.items()}
            
            # Generate answers
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.config.get("max_target_length",64),
                    num_beams=self.config.get('num_beams', 4),
                    early_stopping=self.config.get('early_stopping', True),
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
    
        
            
    def save_checkpoint(self, metrics: Dict[str, float],global_step):
        output_dir = self.config.get('output_dir', './models')
        checkpoint_dir = Path(output_dir) / f"checkpoint-{global_step}"
        
        # Save model
        self.model.save_pretrained(checkpoint_dir)
        
        # Save metrics
        metrics_file = checkpoint_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
            
    def eval_same(self,eval_dataloader):
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            eval_bar = tqdm(eval_dataloader)
            for batch in eval_bar:
                if hasattr(self.model, 'compute_loss'):
                    loss = self.model.compute_loss(**batch)
                else:
                    outputs = self.model(
                        input_ids=batch["input_ids"].to("cuda"),
                        attention_mask=batch["attention_mask"].to("cuda"),
                        labels=batch["labels"].to("cuda"),
                    )
                    loss = outputs.loss
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'eval_loss': avg_loss
        }
     

def run_same_comparison(config: Dict[str, Any],
                        languages: List[str] = None,
                          device_manager: Optional[DeviceManager] = None) -> Dict[str, Any]:
    logger.info("Starting few-shot comparison between mBERT and mT5")
    
    results = {}
    
    # Run mT5 experiment
    logger.info("Running mT5 same experiment...")
    mt5_config = {**config, 'model_type': 'mt5'}
    mt5_results = run_same_experiment(
        'mt5', 
        'google/mt5-base', 
        mt5_config, 
        languages, 
        device_manager
    )
    results['mt5'] = mt5_results
    output_dir = config.get('output_dir', './results')
    output_path = Path(output_dir) / 'same_result'
    output_path.mkdir(parents=True, exist_ok=True)
    
    comparison_file = output_path / 'same_result.json'
    with open(comparison_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Saved comparison results to {comparison_file}")
    
    return results



def run_same_experiment(model_type: str,
                          model_name: str,
                          config: Dict[str, Any],
                          languages: List[str] = None,
                          device_manager: Optional[DeviceManager] = None) -> Dict[str, Any]:
    
    
    #加载模型
    model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto",
  
)
    #加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.to(device_manager.get_device())

    # 初始化实验类
    experiment = SameExperimentNew(model, tokenizer,model_name,config)
    
    # 跑实验
    results = experiment.run_experiment(languages)
    
    return results