import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    TrainerCallback
)
from transformers.trainer_utils import EvalPrediction
from datasets import load_dataset
from omegaconf import OmegaConf
from typing import Dict, Any, List, Optional, Union
import numpy as np
from flow_matching.loss import MixturePathGeneralizedKL
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from evalution.logic.flow import get_path, get_loss_function, get_source_distribution

class MyDataset(Dataset):
    """Dataset for flow matching training"""
    
    def __init__(
        self, 
        tokenizer_path: str, 
        dataset_name: str, 
        split: str, 
        chunk_size: int = 512,
        max_samples: Optional[int] = None
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.chunk_size = chunk_size
        
        # Load dataset
        raw_dataset = load_dataset(dataset_name, split=split)
        if max_samples is not None:
            raw_dataset = raw_dataset.select(range(min(max_samples, len(raw_dataset))))
        
        # Process dataset
        self.data = self._process_dataset(raw_dataset)
    
    def _process_dataset(self, raw_dataset):
        """Process raw dataset into tokenized chunks"""
        processed_data = []
        
        for item in raw_dataset:
            if 'text' in item:
                text = item['text']
            elif 'review' in item:  # For IMDB dataset
                text = item['review']
            else:
                continue
            
            # Tokenize text
            tokens = self.tokenizer(
                text,
                truncation=False,
                add_special_tokens=True,
                return_tensors="pt"
            )['input_ids'].squeeze(0)
            
            # Split into chunks
            for i in range(0, len(tokens) - self.chunk_size + 1, self.chunk_size):
                chunk = tokens[i:i + self.chunk_size]
                if len(chunk) == self.chunk_size:
                    processed_data.append(chunk)
        
        return processed_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.data[idx],
            'labels': self.data[idx].clone()  # For flow matching, labels are the same as input
        }


class MyDataCollator:
    """Data collator for flow matching training"""
    
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch_size = len(features)
        
        # Stack input_ids and labels
        input_ids = torch.stack([f['input_ids'] for f in features])
        labels = torch.stack([f['labels'] for f in features])
        
        # Create attention mask (all 1s since we're using fixed-length sequences)
        attention_mask = torch.ones_like(input_ids)
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }


class MyTrainer(Trainer):
    """Custom trainer for flow matching training"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Load flow matching configuration
        try:
            flow_cfg = OmegaConf.load("config/trainingargs.yml").trainer_args
        except:
            # Default configuration if file doesn't exist
            flow_cfg = OmegaConf.create({
                'scheduler_type': 'polynomial',
                'exponent': 2.0,
                'loss_function': 'generalized_kl',
                'source_distribution': 'mask',
                'time_epsilon': 1e-3
            })
        
        self.flow_cfg = flow_cfg
        
        # Initialize flow matching components
        self.path = get_path(
            scheduler_type=flow_cfg.scheduler_type, 
            exponent=flow_cfg.get('exponent', 2.0)
        )
        
        self.loss_fn = get_loss_function(
            loss_function=flow_cfg.loss_function, 
            path=self.path
        )
        
        self.time_epsilon = (
            flow_cfg.get('time_epsilon', 1e-3) 
            if isinstance(self.loss_fn, MixturePathGeneralizedKL) 
            else 0.0
        )
        
        # Get vocab size from model
        vocab_size = self.model.vocab_size
        self.source_distribution = get_source_distribution(
            source_distribution=flow_cfg.source_distribution,
            vocab_size=vocab_size
        )
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the loss for flow matching.
        Override the default compute_loss to implement flow matching loss.
        """
        device = inputs['input_ids'].device
        x_1 = inputs['input_ids']
        batch_size, seq_len = x_1.shape
        
        # Sample timesteps and prepare flow matching inputs
        # Everything here must be detached from any previous computation graph
        with torch.no_grad():
            t = torch.rand(batch_size, device=device) * (1.0 - self.time_epsilon)
            x_0 = self.source_distribution.sample_like(x_1)
            
            # Ensure x_0 and x_1 are completely detached
            x_0 = x_0.detach().clone()
            x_1_detached = x_1.detach().clone()
            
            # Sample from path
            path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1_detached)
            x_t = path_sample.x_t.detach().clone()
            timesteps = t.detach().clone()
        
        # Now create fresh tensors that require grad if needed
        x_t = x_t.requires_grad_(False)  # Input doesn't need grad
        timesteps = timesteps.requires_grad_(False)  # Timesteps don't need grad
        
        # Forward pass through model
        outputs = model(
            input_ids=x_t,
            timesteps=timesteps,
            attention_mask=inputs.get('attention_mask'),
            return_dict=True
        )
        logits = outputs.logits
        
        # Compute loss - ensure x_1 is the original tensor with potential gradients
        if isinstance(self.loss_fn, MixturePathGeneralizedKL):
            loss = self.loss_fn(
                logits=logits,
                x_1=x_1,  # Use original x_1, not detached version
                x_t=x_t,
                t=timesteps
            )
        else:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                x_1.view(-1),
                ignore_index=-100
            )
        
        return (loss, outputs) if return_outputs else loss
    


class DetailedProgressCallback(TrainerCallback):
    """Callback to show detailed training progress"""
    
    def __init__(self):
        self.step_count = 0
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Called when logging occurs"""
        if logs:
            print(f"Step {state.global_step}: {logs}")
    
    def on_train_step_end(self, args, state, control, model=None, **kwargs):
        """Called at the end of each training step"""
        self.step_count += 1
        if self.step_count % 100 == 0:
            print(f"Completed {self.step_count} training steps")
    
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """Called at the end of each epoch"""
        print(f"Completed epoch {state.epoch}")
    
    def on_evaluate(self, args, state, control, model=None, logs=None, **kwargs):
        """Called during evaluation"""
        if logs:
            print(f"Evaluation results: {logs}")


def load_training_args_from_yaml(yaml_path: str) -> TrainingArguments:
    """Load training arguments from YAML configuration file"""
    
    try:
        config = OmegaConf.load(yaml_path)
        training_args = config.get('training_args', {})
    except:
        print(f"Warning: Could not load {yaml_path}, using default arguments")
        training_args = {}
    
    # Default training arguments optimized for flow matching
    default_args = {
        'output_dir': './llm_cosformer_results',
        'overwrite_output_dir': True,
        'num_train_epochs': 3,
        'per_device_train_batch_size': 4,
        'per_device_eval_batch_size': 4,
        'gradient_accumulation_steps': 4,
        'eval_strategy': 'steps',
        'eval_steps': 500,
        'save_steps': 500,
        'save_total_limit': 3,
        'learning_rate': 5e-5,
        'weight_decay': 0.01,
        'warmup_steps': 500,
        'logging_steps': 100,
        'logging_dir': './logs',
        'dataloader_num_workers': 0,
        'remove_unused_columns': False,  # Important for flow matching
        'load_best_model_at_end': True,
        'metric_for_best_model': 'eval_loss',
        'greater_is_better': False,
        'report_to': None,  # Disable wandb/tensorboard by default
        'fp16': False,  # Disable mixed precision to avoid conflicts
        'gradient_checkpointing': False,  # Disable to avoid conflicts
        'dataloader_pin_memory': True,
        'ddp_find_unused_parameters': False,  # Avoid DDP issues
    }
    
    # Merge with user-provided arguments
    final_args = {**default_args, **training_args}
    
    return TrainingArguments(**final_args)


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """
    Compute metrics for evaluation (placeholder for flow matching)
    """
    # For flow matching, standard metrics like accuracy don't apply
    # We mainly rely on the loss value
    return {}


class FlowMatchingUtils:
    """Utility functions for flow matching"""
    
    @staticmethod
    def sample_timesteps(batch_size: int, device: torch.device, epsilon: float = 1e-3) -> torch.Tensor:
        """Sample random timesteps for training"""
        return torch.rand(batch_size, device=device) * (1.0 - epsilon)
    
    @staticmethod
    def prepare_flow_matching_batch(
        x_1: torch.Tensor,
        source_distribution,
        path,
        t: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Prepare a batch for flow matching training"""
        
        with torch.no_grad():
            # Sample from source distribution
            x_0 = source_distribution.sample_like(x_1)
            
            # Sample x_t from the path
            path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
            x_t = path_sample.x_t.detach().clone()
        
        return {
            'x_0': x_0.detach(),
            'x_t': x_t,
            'x_1': x_1,
            't': t.detach()
        }
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """Validate flow matching configuration"""
        required_keys = ['scheduler_type', 'loss_function', 'source_distribution']
        
        for key in required_keys:
            if key not in config:
                print(f"Warning: Missing required config key: {key}")
                return False
        
        valid_schedulers = ['polynomial']
        if config['scheduler_type'] not in valid_schedulers:
            print(f"Warning: Invalid scheduler_type: {config['scheduler_type']}")
            return False
        
        valid_losses = ['generalized_kl', 'cross_entropy']
        if config['loss_function'] not in valid_losses:
            print(f"Warning: Invalid loss_function: {config['loss_function']}")
            return False
        
        valid_distributions = ['mask', 'uniform']
        if config['source_distribution'] not in valid_distributions:
            print(f"Warning: Invalid source_distribution: {config['source_distribution']}")
            return False
        
        return True