import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset  # Added IterableDataset
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
import math
from collections import deque
import wandb
from tqdm import tqdm

# MODIFICATION 1: Changed to inherit from IterableDataset instead of Dataset
class MyDataset(IterableDataset):
    """Dataset for flow matching training - now as IterableDataset"""
    
    def __init__(
        self, 
        tokenizer_path: str, 
        dataset_name: str, 
        split: str, 
        chunk_size: int = 512,
        max_samples: Optional[int] = None,
        data_files: Optional[Union[str, List[str]]] = None
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.chunk_size = chunk_size
        self.dataset_name = dataset_name
        self.split = split
        self.max_samples = max_samples
        self.data_files = data_files
        self.is_local = self._is_local_path(dataset_name)
        
        # For IterableDataset, we don't load all data upfront
        # Instead, we'll stream it in __iter__
    def __len__(self):
        """Approximate length of the dataset"""
        # For IterableDataset, we can't know exact size upfront
        # Return a large number to ensure we don't run out of data
        return 1000000000
    
    def _is_local_path(self, path: str) -> bool:
        """Check if the dataset path is a local directory or file"""
        import os
        return os.path.exists(path) or path.startswith('/') or path.startswith('./') or path.startswith('../')
    
    def _process_text(self, text: str):
        """Process a single text into tokenized chunks"""
        chunks = []
        
        # Skip empty or very short texts
        if not text or len(text.strip()) < 10:
            return chunks
        
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
                chunks.append(chunk)
        
        return chunks
    
    def __iter__(self):
        """Iterator for streaming data"""
        import os
        
        # Determine how to load the dataset
        if self.is_local:
            print(f"Loading local dataset from: {self.dataset_name}")
            
            # Check if it's a directory with parquet/json/txt files
            if os.path.isdir(self.dataset_name):
                # Look for data files in the directory
                data_files = []
                for ext in ['*.parquet', '*.json', '*.jsonl', '*.txt', '*.csv']:
                    import glob
                    files = glob.glob(os.path.join(self.dataset_name, '**', ext), recursive=True)
                    if files:
                        data_files.extend(files)
                        break  # Use the first type of files found
                
                if not data_files:
                    raise ValueError(f"No supported data files found in {self.dataset_name}")
                
                # Determine file type from extension
                ext = os.path.splitext(data_files[0])[1].lower()
                if ext == '.parquet':
                    data_type = 'parquet'
                elif ext in ['.json', '.jsonl']:
                    data_type = 'json'
                elif ext == '.txt':
                    data_type = 'text'
                elif ext == '.csv':
                    data_type = 'csv'
                else:
                    data_type = None
                
                print(f"Found {len(data_files)} {ext} files")
                
                # Load dataset with appropriate loader
                if data_type:
                    raw_dataset = load_dataset(
                        data_type, 
                        data_files=data_files,
                        split=self.split if self.split != "test" else "train",  # Some local datasets only have train split
                        streaming=True
                    )
                else:
                    # Try loading as a local dataset directory (e.g., if it has dataset_info.json)
                    raw_dataset = load_dataset(
                        self.dataset_name,
                        split=self.split,
                        streaming=True
                    )
            else:
                # It's a single file
                ext = os.path.splitext(self.dataset_name)[1].lower()
                if ext == '.parquet':
                    raw_dataset = load_dataset('parquet', data_files=self.dataset_name, split='train', streaming=True)
                elif ext in ['.json', '.jsonl']:
                    raw_dataset = load_dataset('json', data_files=self.dataset_name, split='train', streaming=True)
                elif ext == '.txt':
                    raw_dataset = load_dataset('text', data_files=self.dataset_name, split='train', streaming=True)
                elif ext == '.csv':
                    raw_dataset = load_dataset('csv', data_files=self.dataset_name, split='train', streaming=True)
                else:
                    raise ValueError(f"Unsupported file type: {ext}")
        else:
            # Load from HuggingFace Hub
            print(f"Loading dataset from HuggingFace Hub: {self.dataset_name}")
            if self.data_files:
                raw_dataset = load_dataset(
                    self.dataset_name, 
                    data_files=self.data_files,
                    split=self.split, 
                    streaming=True
                )
            else:
                raw_dataset = load_dataset(
                    self.dataset_name, 
                    split=self.split, 
                    streaming=True
                )
        
        sample_count = 0
        chunk_count = 0
        
        for item in raw_dataset:
            # Check if we've reached max_samples
            if self.max_samples is not None and chunk_count >= self.max_samples:
                break
            
            # Extract text from item - try multiple common field names
            text = None
            for field in ['text', 'content', 'document', 'review', 'passage', 'article', 'story']:
                if field in item:
                    text = item[field]
                    break
            
            # If no text field found, skip this item
            if text is None:
                # Print available fields for the first item to help debug
                if sample_count == 0:
                    print(f"Warning: No text field found. Available fields: {list(item.keys())}")
                continue
            
            sample_count += 1
            
            # Process text into chunks
            chunks = self._process_text(text)
            
            # Yield each chunk
            for chunk in chunks:
                if self.max_samples is not None and chunk_count >= self.max_samples:
                    return
                    
                yield {
                    'input_ids': chunk,
                    'labels': chunk.clone()  # For flow matching, labels are the same as input
                }
                chunk_count += 1
        
        print(f"Processed {sample_count} samples into {chunk_count} chunks")


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
        
        # Track metrics for progress bar
        self.current_train_loss = None
        self.current_eval_loss = None
    
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
        
        # Update current loss for progress bar
        if model.training:
            self.current_train_loss = loss.detach().item()
        
        return (loss, outputs) if return_outputs else loss
    
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Override log to update current metrics
        """
        if "loss" in logs:
            self.current_train_loss = logs["loss"]
        if "eval_loss" in logs:
            self.current_eval_loss = logs["eval_loss"]
        super().log(logs, start_time=start_time)


class DetailedProgressCallback(TrainerCallback):
    """Enhanced callback with detailed metrics and generation testing"""
    
    def __init__(
        self, 
        tokenizer=None, 
        test_prefixes: Optional[List[str]] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        log_frequency: int = 10,
        use_wandb: bool = False
    ):
        """
        Initialize the detailed progress callback
        
        Args:
            tokenizer: Tokenizer for generation testing
            test_prefixes: List of prefixes to test generation on
            generation_config: Configuration for generation
            log_frequency: How often to update the progress bar (in steps)
            use_wandb: Whether to log to Weights & Biases
        """
        self.tokenizer = tokenizer
        self.test_prefixes = test_prefixes or [
            "The movie was",
            "I really enjoyed",
            "The plot of this film",
            "Overall, I would rate",
            "The acting was"
        ]
        
        self.generation_config = generation_config or {
            'max_new_tokens': 50,
            'num_steps': 30,
            'temperature': 0.8,
            'top_k': 50,
            'top_p': 0.9
        }
        
        self.log_frequency = log_frequency
        self.use_wandb = use_wandb
        
        # Metrics tracking
        self.step_count = 0
        self.train_losses = deque(maxlen=100)  # Keep last 100 losses for smoothing
        self.eval_losses = []
        self.current_epoch = 0
        
        # Progress bar
        self.pbar = None
        self.total_steps = None
        
        # Generator (will be initialized when model is available)
        self.generator = None
    
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Initialize training progress bar and generator"""
        self.total_steps = state.max_steps
        self.pbar = tqdm(total=self.total_steps, desc="Training", position=0, leave=True)
        
        # Initialize generator for testing
        if model is not None and self.tokenizer is not None:
            try:
                from eval import FlowMatchingGenerator  # Import from your eval.py
                device = next(model.parameters()).device
                self.generator = FlowMatchingGenerator(
                    model=model,
                    tokenizer=self.tokenizer,
                    device=device
                )
                print("✓ Generator initialized for prefix testing")
            except Exception as e:
                print(f"Warning: Could not initialize generator: {e}")
                self.generator = None
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Update progress bar with current metrics"""
        self.step_count += 1
        
        # Get current loss from trainer
        trainer = kwargs.get('trainer')
        if trainer and hasattr(trainer, 'current_train_loss'):
            if trainer.current_train_loss is not None:
                self.train_losses.append(trainer.current_train_loss)
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        # Update progress bar every log_frequency steps
        if self.step_count % self.log_frequency == 0:
            self.pbar.update(self.log_frequency)
            self.pbar.set_postfix(metrics)
            
            # Log to wandb if enabled
            if self.use_wandb and wandb.run is not None:
                wandb.log({
                    "train/loss": metrics.get('loss', 0),
                    "train/ppl": metrics.get('ppl', 0),
                    "train/lr": self._get_learning_rate(state),
                    "global_step": state.global_step
                })
    
    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        """Called during evaluation - perform generation testing"""
        if metrics:
            eval_loss = metrics.get('eval_loss', None)
            if eval_loss is not None:
                self.eval_losses.append(eval_loss)
                eval_ppl = math.exp(eval_loss) if eval_loss < 10 else float('inf')
                
                # Update progress bar with eval metrics
                self.pbar.set_postfix({
                    'loss': f"{eval_loss:.4f}",
                    'eval_ppl': f"{eval_ppl:.2f}",
                    'epoch': self.current_epoch
                })
                
                print(f"\n📊 Evaluation Results at Step {state.global_step}:")
                print(f"  • Eval Loss: {eval_loss:.4f}")
                print(f"  • Eval Perplexity: {eval_ppl:.2f}")
                
                # Log to wandb
                if self.use_wandb and wandb.run is not None:
                    wandb.log({
                        "eval/loss": eval_loss,
                        "eval/ppl": eval_ppl,
                        "global_step": state.global_step
                    })
        
        # Perform generation testing
        if self.generator is not None and model is not None:
            self._test_generation(model, state.global_step)
    
    def _test_generation(self, model, step):
        """Test generation with predefined prefixes"""
        print(f"\n🎯 Generation Testing at Step {step}:")
        print("-" * 50)
        
        generation_results = []
        
        # Set model to eval mode temporarily
        model.eval()
        
        try:
            # Update generator's model reference if needed
            self.generator.model = model
            
            for prefix in self.test_prefixes:
                generated = self.generator.generate(
                    prefix=prefix,
                    **self.generation_config,
                    progress_bar=True
                )
                
                generation_results.append({
                    'prefix': prefix,
                    'generated': generated
                })
                
                # Print result
                print(f"📝 Prefix: '{prefix}'")
                print(f"   Generated: {generated}")
                print()
            
            # Log to wandb if enabled
            if self.use_wandb and wandb.run is not None:
                # Create a table for wandb
                table_data = [[r['prefix'], r['generated']] for r in generation_results]
                wandb.log({
                    "generation_samples": wandb.Table(
                        data=table_data,
                        columns=["Prefix", "Generated Text"]
                    ),
                    "global_step": step
                })
                
        except Exception as e:
            print(f"⚠️ Generation testing failed: {e}")
        
        # Set model back to training mode
        model.train()
        
        print("-" * 50)
    
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """Called at the end of each epoch"""
        self.current_epoch = state.epoch
        
        # Calculate epoch metrics
        metrics = self._calculate_metrics()
        
        print(f"\n✅ Completed Epoch {self.current_epoch}")
        print(f"  • Average Train Loss: {metrics.get('loss', 'N/A')}")
        print(f"  • Average Train PPL: {metrics.get('ppl', 'N/A')}")
        
        if self.eval_losses:
            latest_eval_loss = self.eval_losses[-1]
            eval_ppl = math.exp(latest_eval_loss) if latest_eval_loss < 10 else float('inf')
            print(f"  • Latest Eval Loss: {latest_eval_loss:.4f}")
            print(f"  • Latest Eval PPL: {eval_ppl:.2f}")
    
    def on_train_end(self, args, state, control, model=None, **kwargs):
        """Clean up progress bar"""
        if self.pbar:
            self.pbar.close()
        
        print("\n🎉 Training Completed!")
        print(f"  • Total Steps: {state.global_step}")
        print(f"  • Total Epochs: {state.epoch}")
        
        # Final generation test
        if self.generator is not None and model is not None:
            print("\n🏁 Final Generation Test:")
            self._test_generation(model, state.global_step)
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Process log entries"""
        if logs:
            # Extract metrics from logs
            if 'loss' in logs:
                self.train_losses.append(logs['loss'])
            
            # Update progress bar with latest metrics
            if self.pbar and state.global_step % self.log_frequency == 0:
                metrics = self._calculate_metrics()
                if 'eval_loss' in logs:
                    eval_ppl = math.exp(logs['eval_loss']) if logs['eval_loss'] < 10 else float('inf')
                    metrics['eval_loss'] = f"{logs['eval_loss']:.4f}"
                    metrics['eval_ppl'] = f"{eval_ppl:.2f}"
                
                self.pbar.set_postfix(metrics)
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate current training metrics"""
        metrics = {}
        
        # MODIFICATION 2: Removed division by gradient_accumulation_steps
        # The loss values in self.train_losses are already the averaged loss per batch
        # Hugging Face Trainer already handles gradient accumulation internally
        if self.train_losses:
            # Simply average the losses without additional division
            avg_loss = sum(self.train_losses) / len(self.train_losses)
            metrics['loss'] = f"{avg_loss:.4f}"
            # Calculate perplexity
            ppl = math.exp(avg_loss) if avg_loss < 10 else float('inf')
            metrics['ppl'] = f"{ppl:.2f}"
        
        # Add eval metrics if available
        if self.eval_losses:
            latest_eval = self.eval_losses[-1]
            metrics['eval_loss'] = f"{latest_eval:.4f}"
            eval_ppl = math.exp(latest_eval) if latest_eval < 10 else float('inf')
            metrics['eval_ppl'] = f"{eval_ppl:.2f}"
        
        # Add epoch
        metrics['epoch'] = self.current_epoch
        
        return metrics
    
    def _get_learning_rate(self, state) -> float:
        """Get current learning rate from scheduler"""
        if hasattr(state, 'scheduler') and state.scheduler is not None:
            return state.scheduler.get_last_lr()[0]
        return 0.0


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