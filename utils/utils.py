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
import time
from collections import deque
from tqdm import tqdm


# MODIFICATION 1: Changed to inherit from IterableDataset instead of Dataset
class MyDataset(IterableDataset):
    """Dataset for flow matching training - now as IterableDataset with proper sharding"""

    def __init__(
            self,
            tokenizer_path: str,
            dataset_name: str,
            split: str,
            chunk_size: int = 512,
            max_samples: Optional[int] = None,
            data_files: Optional[Union[str, List[str]]] = None,
            seed: int = 42
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
        self.seed = seed

        # For IterableDataset, we don't load all data upfront
        # Instead, we'll stream it in __iter__

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
        """Iterator for streaming data with proper sharding for distributed training"""
        import os
        import glob

        # Get worker info for data sharding
        worker_info = torch.utils.data.get_worker_info()

        # Get distributed training info if available
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                world_size = dist.get_world_size()
                rank = dist.get_rank()
                print(f"Distributed training detected: rank={rank}, world_size={world_size}")
            else:
                world_size = 1
                rank = 0
        except:
            world_size = 1
            rank = 0

        # Load the appropriate dataset
        if self._is_slimpajama_dataset():
            print("Detected SlimPajama dataset - using optimized loading strategy")
            raw_dataset = self._load_slimpajama_dataset()
        elif self.is_local:
            print(f"Loading local dataset from: {self.dataset_name}")
            raw_dataset = self._load_local_generic_dataset()
        else:
            print(f"Loading dataset from HuggingFace Hub: {self.dataset_name}")
            raw_dataset = self._load_huggingface_dataset()

        # Apply sharding for distributed training
        if world_size > 1:
            # Use datasets library's built-in sharding if available
            if hasattr(raw_dataset, 'shard'):
                raw_dataset = raw_dataset.shard(num_shards=world_size, index=rank)
                print(f"Dataset sharded for rank {rank}/{world_size}")
            else:
                print(f"Warning: Dataset doesn't support sharding. Data might be duplicated across GPUs.")

        # Apply worker sharding if using multiple workers
        if worker_info is not None:
            # Multiple workers, split the dataset
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

            # Create a generator with worker-specific seed
            import random
            worker_seed = self.seed + worker_id
            random.seed(worker_seed)
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)

            print(f"Worker {worker_id}/{num_workers} initialized with seed {worker_seed}")

            # Skip items for this worker
            # Each worker processes every num_workers-th item starting from worker_id
            item_counter = 0
            should_process = lambda idx: idx % num_workers == worker_id
        else:
            # Single worker, process all items
            item_counter = 0
            should_process = lambda idx: True

        sample_count = 0
        chunk_count = 0

        # Determine if we should limit samples (for eval/test sets)
        is_training = 'train' in self.split.lower()

        for item_idx, item in enumerate(raw_dataset):
            # Skip items not assigned to this worker
            if not should_process(item_idx):
                continue

            # For eval/test sets, check if we've reached max_samples at sample level
            if not is_training and self.max_samples is not None and sample_count >= self.max_samples:
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
                if sample_count == 0 and (rank == 0 or world_size == 1):
                    print(f"Warning: No text field found. Available fields: {list(item.keys())}")
                continue

            sample_count += 1

            # Process text into chunks
            chunks = self._process_text(text)

            # Yield each chunk
            for chunk in chunks:
                # For training, no limit on chunks; for eval/test, check chunk count
                if not is_training and self.max_samples is not None and chunk_count >= self.max_samples:
                    if rank == 0 or world_size == 1:
                        print(f"Reached max_samples limit: {self.max_samples} chunks from {sample_count} samples")
                    return

                yield {
                    'input_ids': chunk,
                    'labels': chunk.clone()  # For flow matching, labels are the same as input
                }
                chunk_count += 1

        if rank == 0 or world_size == 1:
            print(f"Processed {sample_count} samples into {chunk_count} chunks")

    def _is_slimpajama_dataset(self) -> bool:
        """Check if this is a SlimPajama dataset based on path structure"""
        return 'SlimPajama' in self.dataset_name or 'slimpajama' in self.dataset_name.lower()

    def _load_slimpajama_dataset(self):
        """Load SlimPajama dataset with specific structure for optimal performance"""
        import os
        import glob

        print(f"Loading SlimPajama dataset")
        print(f"  Base path: {self.dataset_name}")
        print(f"  Split: {self.split}")

        # Determine the correct path based on split
        if 'train' in self.split.lower():
            # Training set: /path/to/SlimPajama-627B/train/chunk*/
            base_path = self.dataset_name
            if not base_path.endswith('/train'):
                base_path = os.path.join(base_path, 'train')

            # Find all chunk directories
            chunk_dirs = sorted(glob.glob(os.path.join(base_path, 'chunk*')))
            if not chunk_dirs:
                # Try alternative pattern
                chunk_dirs = sorted(glob.glob(os.path.join(base_path, '*')))
                chunk_dirs = [d for d in chunk_dirs if os.path.isdir(d) and 'chunk' in os.path.basename(d).lower()]

            print(f"  Found {len(chunk_dirs)} chunk directories for training")

            # Collect all .zst files from all chunks for training
            data_files = []
            for chunk_dir in chunk_dirs:
                zst_files = glob.glob(os.path.join(chunk_dir, '*.zst'))
                data_files.extend(zst_files)
                if zst_files:
                    print(f"    {os.path.basename(chunk_dir)}: {len(zst_files)} .zst files")

            if not data_files:
                # Fallback to other file types if .zst not found
                print("  No .zst files found, looking for other formats...")
                for ext in ['*.parquet', '*.json', '*.jsonl', '*.txt']:
                    for chunk_dir in chunk_dirs:
                        files = glob.glob(os.path.join(chunk_dir, ext))
                        data_files.extend(files)
                    if data_files:
                        print(f"  Found {len(data_files)} {ext} files")
                        break

            print(f"  Total training files: {len(data_files)}")

        elif 'validation' in self.split.lower() or 'val' in self.split.lower():
            # Validation set: limit data for efficiency
            base_path = self.dataset_name
            if not base_path.endswith('/validation'):
                base_path = os.path.join(base_path, 'validation')

            # Find chunk directories
            chunk_dirs = sorted(glob.glob(os.path.join(base_path, 'chunk*')))
            if not chunk_dirs:
                chunk_dirs = sorted(glob.glob(os.path.join(base_path, '*')))
                chunk_dirs = [d for d in chunk_dirs if os.path.isdir(d) and 'chunk' in os.path.basename(d).lower()]

            print(f"  Found {len(chunk_dirs)} chunk directories for validation")

            # For validation, limit the number of files
            data_files = []
            max_chunks_for_val = min(2, len(chunk_dirs))  # Use at most 2 chunks
            max_files_per_chunk = 5  # Limit files per chunk

            for chunk_dir in chunk_dirs[:max_chunks_for_val]:
                zst_files = glob.glob(os.path.join(chunk_dir, '*.zst'))[:max_files_per_chunk]
                if not zst_files:  # Fallback to other formats
                    for ext in ['*.parquet', '*.json', '*.jsonl', '*.txt']:
                        zst_files = glob.glob(os.path.join(chunk_dir, ext))[:max_files_per_chunk]
                        if zst_files:
                            break
                data_files.extend(zst_files)
                if zst_files:
                    print(f"    {os.path.basename(chunk_dir)}: using {len(zst_files)} files")

            print(f"  Total validation files: {len(data_files)} (limited for efficiency)")

        elif 'test' in self.split.lower():
            # Test set: use minimal data
            base_path = self.dataset_name
            if not base_path.endswith('/test'):
                base_path = os.path.join(base_path, 'test')

            # Find chunk directories
            chunk_dirs = sorted(glob.glob(os.path.join(base_path, 'chunk*')))
            if not chunk_dirs:
                chunk_dirs = sorted(glob.glob(os.path.join(base_path, '*')))
                chunk_dirs = [d for d in chunk_dirs if os.path.isdir(d) and 'chunk' in os.path.basename(d).lower()]

            print(f"  Found {len(chunk_dirs)} chunk directories for test")

            # For test, use minimal files
            data_files = []
            max_chunks_for_test = min(1, len(chunk_dirs))  # Use only 1 chunk
            max_files_per_chunk = 3  # Very limited files

            for chunk_dir in chunk_dirs[:max_chunks_for_test]:
                test_files = glob.glob(os.path.join(chunk_dir, '*.zst'))[:max_files_per_chunk]
                if not test_files:  # Fallback to other formats
                    for ext in ['*.parquet', '*.json', '*.jsonl', '*.txt']:
                        test_files = glob.glob(os.path.join(chunk_dir, ext))[:max_files_per_chunk]
                        if test_files:
                            break
                data_files.extend(test_files)
                if test_files:
                    print(f"    {os.path.basename(chunk_dir)}: using {len(test_files)} files")

            print(f"  Total test files: {len(data_files)} (minimal for quick evaluation)")

        else:
            raise ValueError(f"Unknown split: {self.split}")

        if not data_files:
            raise ValueError(f"No data files found for split '{self.split}' in {self.dataset_name}")

        # Determine file type and load dataset
        ext = os.path.splitext(data_files[0])[1].lower()

        # Map extensions to data types
        ext_to_type = {
            '.parquet': 'parquet',
            '.json': 'json',
            '.jsonl': 'json',
            '.txt': 'text',
            '.csv': 'csv',
            '.arrow': 'arrow',
            '.zst': 'json',  # .zst files are typically compressed JSON Lines
        }

        data_type = ext_to_type.get(ext, 'json')

        print(f"  Loading as {data_type} format...")

        try:
            raw_dataset = load_dataset(
                data_type,
                data_files=data_files,
                split='train',  # Always use 'train' for local files
                streaming=True  # Important for large datasets
            )
            print("  Successfully loaded SlimPajama dataset")
        except Exception as e:
            print(f"  Failed to load as {data_type}, trying text format: {e}")
            # Fallback to text format
            raw_dataset = load_dataset(
                'text',
                data_files=data_files,
                split='train',
                streaming=True
            )

        return raw_dataset

    def _load_local_generic_dataset(self):
        """Load a generic local dataset (original comprehensive method)"""
        import os
        import glob

        # Check if it's a directory
        if os.path.isdir(self.dataset_name):
            # First, check if it's a HuggingFace dataset directory
            if os.path.exists(os.path.join(self.dataset_name, 'dataset_info.json')) or \
                    os.path.exists(os.path.join(self.dataset_name, 'data')):
                print("  Detected HuggingFace dataset directory structure")
                try:
                    raw_dataset = load_dataset(
                        self.dataset_name,
                        split=self.split,
                        streaming=True
                    )
                    return raw_dataset
                except Exception as e:
                    print(f"  Failed to load as HuggingFace dataset: {e}")
                    # Continue to file search below

            # Look for data files in the directory
            print("  Searching for data files...")
            all_files = []
            for root, dirs, files in os.walk(self.dataset_name):
                for file in files[:10]:  # List first 10 files for debugging
                    all_files.append(os.path.join(root, file))

            if all_files:
                print(f"  Sample files found: {[os.path.basename(f) for f in all_files[:3]]}")

            # Look for data files with various extensions
            data_files = []
            supported_extensions = ['*.parquet', '*.json', '*.jsonl', '*.txt', '*.csv', '*.arrow', '*.zst']

            for ext in supported_extensions:
                # Try multiple search patterns
                patterns = [
                    os.path.join(self.dataset_name, ext),
                    os.path.join(self.dataset_name, '*', ext),
                    os.path.join(self.dataset_name, '**', ext)
                ]

                for pattern in patterns:
                    files = glob.glob(pattern, recursive=True)
                    if files:
                        data_files = files
                        print(f"  Found {len(files)} files with extension {ext}")
                        break

                if data_files:
                    break

            # If no files found with standard extensions, check for specific patterns
            if not data_files:
                special_patterns = ['train/*', 'validation/*', 'test/*', 'chunk*', '*.bin', '*.h5']
                for pattern in special_patterns:
                    files = glob.glob(os.path.join(self.dataset_name, pattern))
                    if files:
                        data_files = files
                        print(f"  Found {len(files)} files matching pattern {pattern}")
                        break

            # Last resort: try any files in the directory
            if not data_files:
                all_files = glob.glob(os.path.join(self.dataset_name, '*'))
                data_files = [f for f in all_files if os.path.isfile(f) and not os.path.basename(f).startswith('.')]
                if data_files:
                    print(f"  Found {len(data_files)} files in directory")

            if not data_files:
                raise ValueError(
                    f"No supported data files found in {self.dataset_name}\n"
                    f"Supported formats: parquet, json, jsonl, txt, csv, arrow, zst"
                )

            # Determine file type from extension
            ext = os.path.splitext(data_files[0])[1].lower()
            ext_to_type = {
                '.parquet': 'parquet',
                '.json': 'json',
                '.jsonl': 'json',
                '.txt': 'text',
                '.csv': 'csv',
                '.arrow': 'arrow',
                '.zst': 'json',
            }

            data_type = ext_to_type.get(ext, 'text')

            # Limit files for non-training splits
            if 'train' not in self.split.lower() and len(data_files) > 100:
                print(f"  Large dataset detected. Limiting to 100 files for {self.split} split")
                data_files = data_files[:100]

            print(f"  Loading {len(data_files)} {ext} files as {data_type} format")

            try:
                raw_dataset = load_dataset(
                    data_type,
                    data_files=data_files,
                    split='train',  # Always use 'train' for local files
                    streaming=True
                )
            except Exception as e:
                print(f"  Error loading with {data_type} format: {e}")
                # Fallback to text format
                raw_dataset = load_dataset(
                    'text',
                    data_files=data_files,
                    split='train',
                    streaming=True
                )
        else:
            # It's a single file
            ext = os.path.splitext(self.dataset_name)[1].lower()
            print(f"  Loading single file with extension {ext}")

            if ext == '.parquet':
                raw_dataset = load_dataset('parquet', data_files=self.dataset_name, split='train', streaming=True)
            elif ext in ['.json', '.jsonl', '.zst']:
                raw_dataset = load_dataset('json', data_files=self.dataset_name, split='train', streaming=True)
            elif ext == '.txt':
                raw_dataset = load_dataset('text', data_files=self.dataset_name, split='train', streaming=True)
            elif ext == '.csv':
                raw_dataset = load_dataset('csv', data_files=self.dataset_name, split='train', streaming=True)
            else:
                print(f"  Unknown extension {ext}, attempting as text")
                raw_dataset = load_dataset('text', data_files=self.dataset_name, split='train', streaming=True)

        return raw_dataset

    def _load_huggingface_dataset(self):
        """Load dataset from HuggingFace Hub"""
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

        return raw_dataset


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
    """Custom trainer for flow matching training with multi-GPU support"""

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

        # Get vocab size from model - handle DDP wrapped models
        if hasattr(self.model, 'module'):
            vocab_size = self.model.module.vocab_size
        else:
            vocab_size = self.model.vocab_size

        self.source_distribution = get_source_distribution(
            source_distribution=flow_cfg.source_distribution,
            vocab_size=vocab_size
        )

        # Track metrics for progress bar
        self.current_train_loss = None
        self.current_eval_loss = None

        # Set find_unused_parameters=True for DDP
        if self.args.local_rank != -1:
            self._setup_distributed_training()

    def _setup_distributed_training(self):
        """Setup for distributed training with proper DDP configuration"""
        import torch.distributed as dist
        if dist.is_initialized() and self.args.ddp_find_unused_parameters is None:
            # Force find_unused_parameters=True for flow matching models
            self.args.ddp_find_unused_parameters = True
            print(f"Setting ddp_find_unused_parameters=True for flow matching model")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the loss for flow matching.
        Override the default compute_loss to implement flow matching loss.
        """
        device = inputs['input_ids'].device
        x_1 = inputs['input_ids']
        batch_size, seq_len = x_1.shape

        # Sample timesteps uniformly
        t = torch.rand(batch_size, device=device) * (1.0 - self.time_epsilon)

        # Sample from source distribution
        x_0 = self.source_distribution.sample_like(x_1)

        # Ensure x_0 and x_1 are on the same device and detached
        x_0 = x_0.to(device).detach()
        x_1_detached = x_1.detach().clone()

        # Sample from path
        path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1_detached)
        x_t = path_sample.x_t.to(device)

        # Prepare timesteps
        timesteps = t.to(device)

        # Forward pass through model
        # Handle both regular model and DDP wrapped model
        if hasattr(model, 'module'):
            # DDP wrapped model
            outputs = model.module(
                input_ids=x_t,
                timesteps=timesteps,
                attention_mask=inputs.get('attention_mask'),
                return_dict=True
            )
        else:
            # Regular model
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

        # Ensure loss requires gradient
        if not loss.requires_grad:
            loss = loss.requires_grad_(True)

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

    def create_optimizer(self):
        """
        Setup the optimizer, ensuring all model parameters are included
        """
        # Get all model parameters
        if hasattr(self.model, 'module'):
            # DDP wrapped model
            model = self.model.module
        else:
            model = self.model

        # Ensure all parameters require gradients
        opt_model = model

        # Get all parameters that require gradients
        parameters = [p for p in opt_model.parameters() if p.requires_grad]

        if len(parameters) == 0:
            raise ValueError("No parameters require gradients!")

        # Count parameters
        total_params = sum(p.numel() for p in opt_model.parameters())
        trainable_params = sum(p.numel() for p in parameters)
        print(f"Optimizer will update {trainable_params:,}/{total_params:,} parameters")

        # Call parent's create_optimizer with our parameters
        return super().create_optimizer()


class DetailedProgressCallback(TrainerCallback):
    """Enhanced callback with detailed metrics, generation testing, and system monitoring using TensorBoard"""

    def __init__(
            self,
            tokenizer=None,
            test_prefixes: Optional[List[str]] = None,
            generation_config: Optional[Dict[str, Any]] = None,
            log_frequency: int = 10,
            use_tensorboard: bool = True,
            tensorboard_dir: str = "./logs",
            monitor_system: bool = True
    ):
        """
        Initialize the detailed progress callback with TensorBoard and system monitoring

        Args:
            tokenizer: Tokenizer for generation testing
            test_prefixes: List of prefixes to test generation on
            generation_config: Configuration for generation
            log_frequency: How often to update the progress bar (in steps)
            use_tensorboard: Whether to log to TensorBoard
            tensorboard_dir: Directory for TensorBoard logs
            monitor_system: Whether to monitor system resources
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
        self.use_tensorboard = use_tensorboard
        self.tensorboard_dir = tensorboard_dir
        self.monitor_system = monitor_system

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

        # TensorBoard writer (will be initialized in on_train_begin)
        self.tb_writer = None

        # System monitoring
        self.system_monitor = None
        if monitor_system:
            try:
                import psutil
                import GPUtil
                self.psutil = psutil
                self.GPUtil = GPUtil
                self.system_monitor_available = True
            except ImportError:
                print("Warning: psutil or GPUtil not installed. System monitoring disabled.")
                print("Install with: pip install psutil gputil")
                self.system_monitor_available = False
                self.monitor_system = False

    def _get_system_metrics(self):
        """Get current system resource usage"""
        metrics = {}

        if not self.monitor_system or not self.system_monitor_available:
            return metrics

        try:
            # CPU metrics
            metrics['system/cpu_percent'] = self.psutil.cpu_percent(interval=None)
            metrics['system/cpu_count'] = self.psutil.cpu_count()

            # Memory metrics
            memory = self.psutil.virtual_memory()
            metrics['system/memory_percent'] = memory.percent
            metrics['system/memory_used_gb'] = memory.used / (1024 ** 3)
            metrics['system/memory_available_gb'] = memory.available / (1024 ** 3)
            metrics['system/memory_total_gb'] = memory.total / (1024 ** 3)

            # Disk metrics (for output directory)
            try:
                disk = self.psutil.disk_usage('/')
                metrics['system/disk_percent'] = disk.percent
                metrics['system/disk_free_gb'] = disk.free / (1024 ** 3)
            except:
                pass

            # Network metrics
            try:
                net_io = self.psutil.net_io_counters()
                metrics['system/network_sent_mb'] = net_io.bytes_sent / (1024 ** 2)
                metrics['system/network_recv_mb'] = net_io.bytes_recv / (1024 ** 2)
            except:
                pass

            # GPU metrics
            try:
                gpus = self.GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    metrics[f'system/gpu{i}_utilization'] = gpu.load * 100
                    metrics[f'system/gpu{i}_memory_percent'] = gpu.memoryUtil * 100
                    metrics[f'system/gpu{i}_memory_used_gb'] = gpu.memoryUsed / 1024
                    metrics[f'system/gpu{i}_memory_total_gb'] = gpu.memoryTotal / 1024
                    metrics[f'system/gpu{i}_temperature'] = gpu.temperature

                # CUDA memory from PyTorch
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        metrics[f'system/cuda{i}_memory_allocated_gb'] = torch.cuda.memory_allocated(i) / (1024 ** 3)
                        metrics[f'system/cuda{i}_memory_reserved_gb'] = torch.cuda.memory_reserved(i) / (1024 ** 3)
                        if hasattr(torch.cuda, 'max_memory_allocated'):
                            metrics[f'system/cuda{i}_max_memory_allocated_gb'] = torch.cuda.max_memory_allocated(i) / (
                                        1024 ** 3)
            except Exception as e:
                pass

            # Process-specific metrics
            try:
                process = self.psutil.Process()
                metrics['system/process_memory_rss_gb'] = process.memory_info().rss / (1024 ** 3)
                metrics['system/process_memory_vms_gb'] = process.memory_info().vms / (1024 ** 3)
                metrics['system/process_cpu_percent'] = process.cpu_percent()
                metrics['system/process_num_threads'] = process.num_threads()
            except:
                pass

        except Exception as e:
            print(f"Error collecting system metrics: {e}")

        return metrics

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Initialize training progress bar, generator, TensorBoard, and system monitoring"""
        self.total_steps = state.max_steps
        self.pbar = tqdm(total=self.total_steps, desc="Training", position=0, leave=True)

        # Initialize TensorBoard writer
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                import datetime
                import os

                # Check if we're in distributed training
                try:
                    import torch.distributed as dist
                    if dist.is_initialized():
                        rank = dist.get_rank()
                        if rank == 0:  # Only main process writes to TensorBoard
                            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                            log_dir = os.path.join(self.tensorboard_dir, f"run_{timestamp}")
                            self.tb_writer = SummaryWriter(log_dir=log_dir)
                            print(f"✓ TensorBoard logging initialized at: {log_dir}")
                            print(f"  Run: tensorboard --logdir {self.tensorboard_dir}")

                            # Log system info
                            self._log_system_info()
                        else:
                            self.tb_writer = None  # Other processes don't write
                    else:
                        # Single GPU training
                        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                        log_dir = os.path.join(self.tensorboard_dir, f"run_{timestamp}")
                        self.tb_writer = SummaryWriter(log_dir=log_dir)
                        print(f"✓ TensorBoard logging initialized at: {log_dir}")
                        print(f"  Run: tensorboard --logdir {self.tensorboard_dir}")

                        # Log system info
                        self._log_system_info()
                except:
                    # Fallback for single GPU
                    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    log_dir = os.path.join(self.tensorboard_dir, f"run_{timestamp}")
                    self.tb_writer = SummaryWriter(log_dir=log_dir)
                    print(f"✓ TensorBoard logging initialized at: {log_dir}")
                    print(f"  Run: tensorboard --logdir {self.tensorboard_dir}")

                    # Log system info
                    self._log_system_info()

            except ImportError:
                print("Warning: TensorBoard not installed. Install with: pip install tensorboard")
                self.tb_writer = None
                self.use_tensorboard = False

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

        # Initialize system monitoring
        if self.monitor_system and self.system_monitor_available:
            print("✓ System resource monitoring enabled")

    def _log_system_info(self):
        """Log system information at the start of training"""
        if not self.tb_writer:
            return

        try:
            import platform
            import sys

            system_info = []
            system_info.append(f"**System Information**\n")
            system_info.append(f"- Platform: {platform.platform()}")
            system_info.append(f"- Python: {sys.version}")
            system_info.append(f"- PyTorch: {torch.__version__}")
            system_info.append(f"- CUDA Available: {torch.cuda.is_available()}")

            if torch.cuda.is_available():
                system_info.append(f"- CUDA Version: {torch.version.cuda}")
                system_info.append(f"- GPU Count: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    system_info.append(f"- GPU {i}: {torch.cuda.get_device_name(i)}")

            if self.system_monitor_available:
                import psutil
                cpu_count = psutil.cpu_count()
                memory = psutil.virtual_memory()
                system_info.append(f"- CPU Cores: {cpu_count}")
                system_info.append(f"- Total Memory: {memory.total / (1024 ** 3):.2f} GB")

            self.tb_writer.add_text('system/info', '\n'.join(system_info), 0)

        except Exception as e:
            print(f"Could not log system info: {e}")

    def _get_model_metrics(self, model):
        """Get model-related metrics including gradient and parameter norms"""
        metrics = {}

        try:
            # Get the actual model (unwrap DDP if necessary)
            if hasattr(model, 'module'):
                actual_model = model.module
            else:
                actual_model = model

            # Calculate parameter norm
            total_norm = 0.0
            param_count = 0
            for p in actual_model.parameters():
                if p.data is not None:
                    param_norm = p.data.norm(2).item()
                    total_norm += param_norm ** 2
                    param_count += 1
            total_norm = total_norm ** 0.5
            metrics['model/param_norm'] = total_norm
            metrics['model/param_count'] = param_count

            # Calculate gradient norm (only if gradients exist)
            grad_norm = 0.0
            grad_count = 0
            for p in actual_model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
                    grad_count += 1

            if grad_count > 0:
                grad_norm = grad_norm ** 0.5
                metrics['model/grad_norm'] = grad_norm
                metrics['model/grad_count'] = grad_count

                # Calculate gradient/parameter ratio
                if total_norm > 0:
                    metrics['model/grad_param_ratio'] = grad_norm / total_norm

            # Layer-wise gradient and parameter norms
            layer_grad_norms = {}
            layer_param_norms = {}

            for name, param in actual_model.named_parameters():
                if param.data is not None:
                    # Get layer prefix (e.g., "flow_blocks.0", "vocab_embed", etc.)
                    layer_name = name.split('.')[0] if '.' in name else name

                    # Parameter norm for this layer
                    param_norm = param.data.norm(2).item()
                    if layer_name not in layer_param_norms:
                        layer_param_norms[layer_name] = 0.0
                    layer_param_norms[layer_name] += param_norm ** 2

                    # Gradient norm for this layer (if exists)
                    if param.grad is not None:
                        grad_norm = param.grad.data.norm(2).item()
                        if layer_name not in layer_grad_norms:
                            layer_grad_norms[layer_name] = 0.0
                        layer_grad_norms[layer_name] += grad_norm ** 2

            # Add layer-wise norms to metrics
            for layer_name, norm_squared in layer_param_norms.items():
                metrics[f'model/param_norm/{layer_name}'] = norm_squared ** 0.5

            for layer_name, norm_squared in layer_grad_norms.items():
                metrics[f'model/grad_norm/{layer_name}'] = norm_squared ** 0.5

            # Check for gradient explosion or vanishing
            if 'model/grad_norm' in metrics:
                grad_norm_val = metrics['model/grad_norm']
                if grad_norm_val > 100:
                    metrics['model/grad_explosion_warning'] = 1.0
                elif grad_norm_val < 1e-6:
                    metrics['model/grad_vanishing_warning'] = 1.0
                else:
                    metrics['model/grad_explosion_warning'] = 0.0
                    metrics['model/grad_vanishing_warning'] = 0.0

        except Exception as e:
            print(f"Error calculating model metrics: {e}")

        return metrics

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Update progress bar with current metrics and log to TensorBoard including system and model metrics"""
        self.step_count += 1

        # Get current loss from trainer
        trainer = kwargs.get('trainer')
        if trainer and hasattr(trainer, 'current_train_loss'):
            if trainer.current_train_loss is not None:
                self.train_losses.append(trainer.current_train_loss)

        # Calculate metrics
        metrics = self._calculate_metrics(args)

        # Update progress bar every log_frequency steps
        if self.step_count % self.log_frequency == 0:
            self.pbar.update(self.log_frequency)

            # Get model metrics for display
            if model is not None:
                model_metrics = self._get_model_metrics(model)
                # Add key metrics to progress bar
                if 'model/grad_norm' in model_metrics:
                    metrics['grad_norm'] = f"{model_metrics['model/grad_norm']:.3f}"
                if 'model/param_norm' in model_metrics:
                    metrics['param_norm'] = f"{model_metrics['model/param_norm']:.3f}"

            self.pbar.set_postfix(metrics)

            # Log to TensorBoard
            if self.use_tensorboard and self.tb_writer is not None:
                # Training metrics
                try:
                    if 'loss' in metrics:
                        loss_val = float(metrics['loss'])
                        self.tb_writer.add_scalar('train/loss', loss_val, state.global_step)

                    if 'ppl' in metrics:
                        ppl_val = float(metrics['ppl'])
                        self.tb_writer.add_scalar('train/perplexity', ppl_val, state.global_step)

                    # Log learning rate
                    lr = self._get_learning_rate(state)
                    self.tb_writer.add_scalar('train/learning_rate', lr, state.global_step)

                    # Log epoch
                    self.tb_writer.add_scalar('train/epoch', self.current_epoch, state.global_step)

                    # Log training speed
                    if hasattr(state, 'num_input_tokens_seen') and state.num_input_tokens_seen > 0:
                        elapsed_time = time.time() - state.start_time if hasattr(state, 'start_time') else 1
                        tokens_per_sec = state.num_input_tokens_seen / elapsed_time
                        self.tb_writer.add_scalar('train/tokens_per_second', tokens_per_sec, state.global_step)

                except Exception as e:
                    pass

                # Log model metrics (gradients and parameters)
                if model is not None:
                    model_metrics = self._get_model_metrics(model)
                    for key, value in model_metrics.items():
                        self.tb_writer.add_scalar(key, value, state.global_step)

                    # Create a combined plot for gradient and parameter norms
                    if 'model/grad_norm' in model_metrics and 'model/param_norm' in model_metrics:
                        self.tb_writer.add_scalars('model/norms_comparison', {
                            'gradient_norm': model_metrics['model/grad_norm'],
                            'parameter_norm': model_metrics['model/param_norm']
                        }, state.global_step)

                # Log system metrics
                if self.monitor_system:
                    system_metrics = self._get_system_metrics()
                    for key, value in system_metrics.items():
                        self.tb_writer.add_scalar(key, value, state.global_step)

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        """Called during evaluation - perform generation testing and log to TensorBoard"""
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

                # Get and print model metrics
                if model is not None:
                    model_metrics = self._get_model_metrics(model)
                    if 'model/param_norm' in model_metrics:
                        print(f"  • Parameter Norm: {model_metrics['model/param_norm']:.4f}")
                    if 'model/grad_norm' in model_metrics:
                        print(f"  • Gradient Norm: {model_metrics['model/grad_norm']:.4f}")

                # Log to TensorBoard
                if self.use_tensorboard and self.tb_writer is not None:
                    self.tb_writer.add_scalar('eval/loss', eval_loss, state.global_step)
                    self.tb_writer.add_scalar('eval/perplexity', eval_ppl, state.global_step)

                    # Log train vs eval loss comparison
                    if self.train_losses:
                        avg_train_loss = sum(self.train_losses) / len(self.train_losses)
                        self.tb_writer.add_scalars('loss_comparison', {
                            'train': avg_train_loss,
                            'eval': eval_loss
                        }, state.global_step)

                    # Log model metrics during evaluation
                    if model is not None:
                        model_metrics = self._get_model_metrics(model)
                        for key, value in model_metrics.items():
                            self.tb_writer.add_scalar(f"eval_{key}", value, state.global_step)

                    # Log system metrics during evaluation
                    if self.monitor_system:
                        system_metrics = self._get_system_metrics()
                        for key, value in system_metrics.items():
                            self.tb_writer.add_scalar(f"eval_{key}", value, state.global_step)

        # Perform generation testing
        if self.generator is not None and model is not None:
            self._test_generation(model, state.global_step)

    def _test_generation(self, model, step):
        """Test generation with predefined prefixes and log to TensorBoard"""
        print(f"\n🎯 Generation Testing at Step {step}:")
        print("-" * 50)

        generation_results = []

        # Set model to eval mode temporarily
        model.eval()

        try:
            # Update generator's model reference if needed
            self.generator.model = model

            for idx, prefix in enumerate(self.test_prefixes):
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

                # Log to TensorBoard as text
                if self.use_tensorboard and self.tb_writer is not None:
                    self.tb_writer.add_text(
                        f'generation/sample_{idx}',
                        f"**Prefix:** {prefix}\n\n**Generated:** {generated}",
                        step
                    )

            # Create a combined text summary for TensorBoard
            if self.use_tensorboard and self.tb_writer is not None:
                combined_text = "\n\n".join([
                    f"**[{i}] Prefix:** {r['prefix']}\n**Generated:** {r['generated']}"
                    for i, r in enumerate(generation_results)
                ])
                self.tb_writer.add_text('generation/all_samples', combined_text, step)

        except Exception as e:
            print(f"⚠️ Generation testing failed: {e}")

        # Set model back to training mode
        model.train()

        print("-" * 50)

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """Called at the end of each epoch - log epoch metrics to TensorBoard"""
        self.current_epoch = state.epoch

        # Calculate epoch metrics
        metrics = self._calculate_metrics(args)

        print(f"\n✅ Completed Epoch {self.current_epoch}")
        print(f"  • Average Train Loss: {metrics.get('loss', 'N/A')}")
        print(f"  • Average Train PPL: {metrics.get('ppl', 'N/A')}")

        if self.eval_losses:
            latest_eval_loss = self.eval_losses[-1]
            eval_ppl = math.exp(latest_eval_loss) if latest_eval_loss < 10 else float('inf')
            print(f"  • Latest Eval Loss: {latest_eval_loss:.4f}")
            print(f"  • Latest Eval PPL: {eval_ppl:.2f}")

            # Log epoch summary to TensorBoard
            if self.use_tensorboard and self.tb_writer is not None:
                # Parse metrics for logging
                try:
                    if 'loss' in metrics:
                        train_loss = float(metrics['loss'])
                        self.tb_writer.add_scalar('epoch/train_loss', train_loss, self.current_epoch)

                    self.tb_writer.add_scalar('epoch/eval_loss', latest_eval_loss, self.current_epoch)
                    self.tb_writer.add_scalar('epoch/eval_perplexity', eval_ppl, self.current_epoch)
                except:
                    pass

    def on_train_end(self, args, state, control, model=None, **kwargs):
        """Clean up progress bar and close TensorBoard writer"""
        if self.pbar:
            self.pbar.close()

        print("\n🎉 Training Completed!")
        print(f"  • Total Steps: {state.global_step}")
        print(f"  • Total Epochs: {state.epoch}")

        # Final generation test
        if self.generator is not None and model is not None:
            print("\n🏁 Final Generation Test:")
            self._test_generation(model, state.global_step)

        # Close TensorBoard writer
        if self.tb_writer is not None:
            # Log final model metrics
            if model is not None:
                final_model_metrics = self._get_model_metrics(model)
                for key, value in final_model_metrics.items():
                    self.tb_writer.add_scalar(f"final/{key.replace('model/', '')}", value, state.global_step)

            # Log final summary
            if self.train_losses and self.eval_losses:
                final_train_loss = sum(self.train_losses) / len(self.train_losses)
                final_eval_loss = self.eval_losses[-1]

                # Get final model norms
                final_param_norm = final_model_metrics.get('model/param_norm', 0.0) if model else 0.0
                final_grad_norm = final_model_metrics.get('model/grad_norm', 0.0) if model else 0.0

                self.tb_writer.add_hparams(
                    {'final_epochs': state.epoch,
                     'total_steps': state.global_step},
                    {'hparam/final_train_loss': final_train_loss,
                     'hparam/final_eval_loss': final_eval_loss,
                     'hparam/final_param_norm': final_param_norm,
                     'hparam/final_grad_norm': final_grad_norm}
                )

            self.tb_writer.close()
            print(f"\n📈 TensorBoard logs saved. View with: tensorboard --logdir {self.tensorboard_dir}")

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Process log entries and write to TensorBoard"""
        if logs:
            # Extract metrics from logs
            if 'loss' in logs:
                self.train_losses.append(logs['loss'])

                # Log additional metrics from trainer logs to TensorBoard
                if self.use_tensorboard and self.tb_writer is not None:
                    for key, value in logs.items():
                        if isinstance(value, (int, float)):
                            # Clean up the key name for TensorBoard
                            tb_key = key.replace('_', '/')
                            if '/' not in tb_key:
                                tb_key = f'trainer/{tb_key}'
                            self.tb_writer.add_scalar(tb_key, value, state.global_step)

            # Update progress bar with latest metrics
            if self.pbar and state.global_step % self.log_frequency == 0:
                metrics = self._calculate_metrics(args)
                if 'eval_loss' in logs:
                    eval_ppl = math.exp(logs['eval_loss']) if logs['eval_loss'] < 10 else float('inf')
                    metrics['eval_loss'] = f"{logs['eval_loss']:.4f}"
                    metrics['eval_ppl'] = f"{eval_ppl:.2f}"

                self.pbar.set_postfix(metrics)

    def _calculate_metrics(self, args=None) -> Dict[str, Any]:
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
        'warmup_ratio': 0.0,  # Can use ratio instead of steps
        'lr_scheduler_type': 'cosine',  # Default to cosine with warmup
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
        'ddp_find_unused_parameters': True,  # Important for flow matching with DDP
        'ddp_backend': 'nccl',  # Use NCCL for better GPU communication
        'max_grad_norm': 1.0,  # Gradient clipping
        'adam_beta1': 0.9,
        'adam_beta2': 0.999,
        'adam_epsilon': 1e-8,
    }

    # Merge with user-provided arguments
    final_args = {**default_args, **training_args}

    # Remove num_cycles as it's not a valid TrainingArguments parameter
    # It's only used for cosine_with_restarts scheduler which isn't directly supported
    if 'num_cycles' in final_args:
        num_cycles = final_args.pop('num_cycles')
        print(f"Note: num_cycles={num_cycles} is removed (not supported by HuggingFace TrainingArguments)")
        if final_args.get('lr_scheduler_type') == 'cosine_with_restarts':
            print("  For cosine_with_restarts, consider using 'cosine' or implementing custom scheduler")
            final_args['lr_scheduler_type'] = 'cosine'  # Fallback to regular cosine

    # Handle lr_scheduler_type specifically
    scheduler_type = final_args.get('lr_scheduler_type', 'cosine')

    # Map common scheduler names to HuggingFace names
    scheduler_mapping = {
        'cosine': 'cosine',
        'cosine_with_warmup': 'cosine',
        'cosine_with_restarts': 'cosine_with_hard_restarts',
        'linear': 'linear',
        'constant': 'constant',
        'constant_with_warmup': 'constant_with_warmup',
        'polynomial': 'polynomial',
        'inverse_sqrt': 'inverse_sqrt',
    }

    if scheduler_type in scheduler_mapping:
        final_args['lr_scheduler_type'] = scheduler_mapping[scheduler_type]

    # Ensure warmup is properly configured
    if final_args.get('warmup_ratio', 0) > 0 and final_args.get('warmup_steps', 0) == 0:
        # If warmup_ratio is set but not warmup_steps, calculate warmup_steps
        # This will be done automatically by HuggingFace Trainer
        pass
    elif final_args.get('warmup_steps', 0) > 0:
        # If warmup_steps is set, use it (takes precedence over warmup_ratio)
        final_args['warmup_ratio'] = 0.0

    print(f"Learning rate scheduler: {final_args['lr_scheduler_type']}")
    print(f"  Warmup steps: {final_args.get('warmup_steps', 0)}")
    print(f"  Warmup ratio: {final_args.get('warmup_ratio', 0.0)}")
    print(f"  Base learning rate: {final_args.get('learning_rate', 5e-5)}")

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


class DistributedTrainingHelper:
    """Helper class for distributed training setup"""

    @staticmethod
    def setup_distributed():
        """Setup distributed training environment"""
        import torch.distributed as dist
        import os

        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ.get('LOCAL_RANK', 0))

            # Initialize distributed training
            if not dist.is_initialized():
                dist.init_process_group(
                    backend='nccl',
                    init_method='env://',
                    world_size=world_size,
                    rank=rank
                )

            # Set device
            torch.cuda.set_device(local_rank)

            print(f"Distributed training initialized: rank={rank}, world_size={world_size}, local_rank={local_rank}")
            return True

        return False

    @staticmethod
    def cleanup_distributed():
        """Cleanup distributed training"""
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()

    @staticmethod
    def is_main_process():
        """Check if current process is the main process"""
        import torch.distributed as dist

        if not dist.is_initialized():
            return True

        return dist.get_rank() == 0

    @staticmethod
    def synchronize():
        """Synchronize all processes"""
        import torch.distributed as dist

        if dist.is_initialized():
            dist.barrier()


def create_accelerate_config(num_gpus: int = 8, mixed_precision: str = 'no') -> Dict[str, Any]:
    """
    Create accelerate configuration for multi-GPU training

    Args:
        num_gpus: Number of GPUs to use
        mixed_precision: Mixed precision setting ('no', 'fp16', 'bf16')

    Returns:
        Dictionary with accelerate configuration
    """
    config = {
        'compute_environment': 'LOCAL_MACHINE',
        'debug': False,
        'distributed_type': 'MULTI_GPU',
        'downcast_bf16': 'no',
        'enable_cpu_affinity': False,
        'gpu_ids': 'all',
        'machine_rank': 0,
        'main_training_function': 'main',
        'mixed_precision': mixed_precision,
        'num_machines': 1,
        'num_processes': num_gpus,
        'rdzv_backend': 'static',
        'same_network': True,
        'tpu_env': [],
        'tpu_use_cluster': False,
        'tpu_use_sudo': False,
        'use_cpu': False
    }

    return config


def save_accelerate_config(config: Dict[str, Any], path: str = 'accelerate_config.yaml'):
    """
    Save accelerate configuration to file

    Args:
        config: Accelerate configuration dictionary
        path: Path to save the configuration
    """
    import yaml

    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Accelerate configuration saved to {path}")


def visualize_lr_schedule(
        total_steps: int,
        learning_rate: float = 5e-5,
        warmup_steps: int = 1000,
        warmup_ratio: float = 0.0,
        lr_scheduler_type: str = "cosine",
        num_cycles: float = 0.5,
        save_path: Optional[str] = None
):
    """
    Visualize the learning rate schedule

    Args:
        total_steps: Total number of training steps
        learning_rate: Base learning rate
        warmup_steps: Number of warmup steps
        warmup_ratio: Warmup ratio (alternative to warmup_steps)
        lr_scheduler_type: Type of scheduler
        num_cycles: Number of cycles for cosine schedule
        save_path: Path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
        from transformers import get_scheduler
        import torch.optim as optim

        # Calculate actual warmup steps
        if warmup_ratio > 0 and warmup_steps == 0:
            warmup_steps = int(total_steps * warmup_ratio)

        # Create a dummy optimizer
        model = torch.nn.Linear(1, 1)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

        # Create the scheduler
        scheduler = get_scheduler(
            name=lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=num_cycles if lr_scheduler_type == "cosine_with_restarts" else None
        )

        # Calculate learning rates for each step
        lrs = []
        for step in range(total_steps):
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()

        # Create the plot
        plt.figure(figsize=(12, 6))
        plt.plot(lrs, linewidth=2)
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.title(f'Learning Rate Schedule: {lr_scheduler_type} (warmup={warmup_steps} steps)', fontsize=14)
        plt.grid(True, alpha=0.3)

        # Add vertical line for warmup end
        if warmup_steps > 0:
            plt.axvline(x=warmup_steps, color='r', linestyle='--', alpha=0.5,
                        label=f'Warmup ends (step {warmup_steps})')
            plt.legend()

        # Add horizontal line for base learning rate
        plt.axhline(y=learning_rate, color='g', linestyle='--', alpha=0.3, label=f'Base LR ({learning_rate:.2e})')

        # Annotate key points
        plt.annotate(f'Max LR: {max(lrs):.2e}',
                     xy=(lrs.index(max(lrs)), max(lrs)),
                     xytext=(10, 10), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        plt.annotate(f'Final LR: {lrs[-1]:.2e}',
                     xy=(len(lrs) - 1, lrs[-1]),
                     xytext=(-60, 20), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.5),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Learning rate schedule plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

        # Print summary statistics
        print(f"\nLearning Rate Schedule Summary:")
        print(f"  • Scheduler Type: {lr_scheduler_type}")
        print(f"  • Total Steps: {total_steps}")
        print(f"  • Warmup Steps: {warmup_steps}")
        print(f"  • Base LR: {learning_rate:.2e}")
        print(f"  • Max LR: {max(lrs):.2e}")
        print(f"  • Min LR: {min(lrs):.2e}")
        print(f"  • Final LR: {lrs[-1]:.2e}")

        return lrs

    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return None


def create_custom_cosine_scheduler(
        optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1
):
    """
    Create a custom cosine scheduler with warmup and optional minimum learning rate

    Args:
        optimizer: The optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        num_cycles: Number of cosine cycles (0.5 = half cosine)
        min_lr_ratio: Minimum learning rate as ratio of base lr
        last_epoch: The index of last epoch

    Returns:
        torch.optim.lr_scheduler.LambdaLR scheduler
    """
    from torch.optim.lr_scheduler import LambdaLR
    import math

    def lr_lambda(current_step: int):
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine annealing phase
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))

        # Apply minimum learning rate
        decayed = (1 - min_lr_ratio) * cosine_decay + min_lr_ratio

        return decayed

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# Export all classes and functions
__all__ = [
    'MyDataset',
    'MyDataCollator',
    'MyTrainer',
    'DetailedProgressCallback',
    'FlowMatchingUtils',
    'DistributedTrainingHelper',
    'load_training_args_from_yaml',
    'compute_metrics',
    'create_accelerate_config',
    'save_accelerate_config',
    'visualize_lr_schedule',
    'create_custom_cosine_scheduler'
]