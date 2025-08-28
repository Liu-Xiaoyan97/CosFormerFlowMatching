import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from flow_matching.path import ProbPath
from flow_matching.solver import MixtureDiscreteEulerSolver
from datasets import load_dataset
from typing import Dict, Any, List, Optional, Union
import torch.nn as nn
from pathlib import Path
from flow_matching.utils import ModelWrapper
from transformers.tokenization_utils import PreTrainedTokenizer
from utils.flow import SourceDistribution


class CosformerDataset(IterableDataset):

    def __init__(
            self,
            tokenizer,
            dataset_name: str,
            split: str,
            chunk_size: int = 512,
            max_samples: Optional[int] = None,
            data_files: Optional[Union[str, List[str]]] = None,
            seed: int = 42
    ):
        self.tokenizer = tokenizer

        self.chunk_size = chunk_size
        self.dataset_name = dataset_name
        self.split = split
        self.max_samples = max_samples
        self.data_files = data_files
        self.is_local = self._is_local_path(dataset_name)
        self.seed = seed

    def _is_local_path(self, path: str) -> bool:
        """Check if the dataset path is a local directory or file"""
        import os
        return os.path.exists(path) or path.startswith('/') or path.startswith('./') or path.startswith('../')

    def _process_text(self, text: str):
        """Process a single text into tokenized chunks"""
        chunks = []

        if not text or len(text.strip()) < 10:
            return chunks

        tokens = self.tokenizer(
            text,
            truncation=False,
            add_special_tokens=True,
            return_tensors="pt"
        )['input_ids'].squeeze(0)

        for i in range(0, len(tokens) - self.chunk_size + 1, self.chunk_size):
            chunk = tokens[i:i + self.chunk_size]
            if len(chunk) == self.chunk_size:
                chunks.append(chunk)

        return chunks

    def __iter__(self):
        """Iterator for streaming data with proper sharding for distributed training"""
        if self._is_slimpajama_dataset():
            print("Detected SlimPajama dataset - using optimized loading strategy")
            raw_dataset = self._load_slimpajama_dataset()
        elif self.is_local:
            print(f"Loading local dataset from: {self.dataset_name}")
            raw_dataset = self._load_local_generic_dataset()
        else:
            print(f"Loading dataset from HuggingFace Hub: {self.dataset_name}")
            raw_dataset = self._load_huggingface_dataset()

        sample_count = 0
        chunk_count = 0
        is_training = 'train' in self.split.lower()

        for item_idx, item in enumerate(raw_dataset):

            if not is_training and self.max_samples is not None and sample_count >= self.max_samples:
                break

            text = None
            for field in ['text', 'content', 'document', 'review', 'passage', 'article', 'story']:
                if field in item:
                    text = item[field]
                    break

            if text is None:
                if sample_count == 0:
                    print(f"Warning: No text field found. Available fields: {list(item.keys())}")
                continue

            sample_count += 1

            chunks = self._process_text(text)

            for chunk in chunks:
                if not is_training and self.max_samples is not None and chunk_count >= self.max_samples:
                    print(f"Reached max_samples limit: {self.max_samples} chunks from {sample_count} samples")
                    return
                yield {
                    'input_ids': chunk,
                    'labels': chunk.clone()
                }
                chunk_count += 1


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

        if 'train' in self.split.lower():
            base_path = self.dataset_name
            if not base_path.endswith('/train'):
                base_path = os.path.join(base_path, 'train')

            chunk_dirs = sorted(glob.glob(os.path.join(base_path, 'chunk*')))
            if not chunk_dirs:
                chunk_dirs = sorted(glob.glob(os.path.join(base_path, '*')))
                chunk_dirs = [d for d in chunk_dirs if os.path.isdir(d) and 'chunk' in os.path.basename(d).lower()]

            print(f"  Found {len(chunk_dirs)} chunk directories for training")

            data_files = []
            for chunk_dir in chunk_dirs:
                zst_files = glob.glob(os.path.join(chunk_dir, '*.zst'))
                data_files.extend(zst_files)
                if zst_files:
                    print(f"    {os.path.basename(chunk_dir)}: {len(zst_files)} .zst files")

            if not data_files:
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
            base_path = self.dataset_name
            if not base_path.endswith('/validation'):
                base_path = os.path.join(base_path, 'validation')

            chunk_dirs = sorted(glob.glob(os.path.join(base_path, 'chunk*')))
            if not chunk_dirs:
                chunk_dirs = sorted(glob.glob(os.path.join(base_path, '*')))
                chunk_dirs = [d for d in chunk_dirs if os.path.isdir(d) and 'chunk' in os.path.basename(d).lower()]

            print(f"  Found {len(chunk_dirs)} chunk directories for validation")

            data_files = []
            max_chunks_for_val = min(2, len(chunk_dirs))
            max_files_per_chunk = 5

            for chunk_dir in chunk_dirs[:max_chunks_for_val]:
                zst_files = glob.glob(os.path.join(chunk_dir, '*.zst'))[:max_files_per_chunk]
                if not zst_files:
                    for ext in ['*.parquet', '*.json', '*.jsonl', '*.txt']:
                        zst_files = glob.glob(os.path.join(chunk_dir, ext))[:max_files_per_chunk]
                        if zst_files:
                            break
                data_files.extend(zst_files)
                if zst_files:
                    print(f"    {os.path.basename(chunk_dir)}: using {len(zst_files)} files")

            print(f"  Total validation files: {len(data_files)} (limited for efficiency)")

        elif 'test' in self.split.lower():
            base_path = self.dataset_name
            if not base_path.endswith('/test'):
                base_path = os.path.join(base_path, 'test')

            chunk_dirs = sorted(glob.glob(os.path.join(base_path, 'chunk*')))
            if not chunk_dirs:
                chunk_dirs = sorted(glob.glob(os.path.join(base_path, '*')))
                chunk_dirs = [d for d in chunk_dirs if os.path.isdir(d) and 'chunk' in os.path.basename(d).lower()]

            print(f"  Found {len(chunk_dirs)} chunk directories for test")

            data_files = []
            max_chunks_for_test = min(1, len(chunk_dirs))
            max_files_per_chunk = 3

            for chunk_dir in chunk_dirs[:max_chunks_for_test]:
                test_files = glob.glob(os.path.join(chunk_dir, '*.zst'))[:max_files_per_chunk]
                if not test_files:
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

        data_type = ext_to_type.get(ext, 'json')

        print(f"  Loading as {data_type} format...")

        try:
            raw_dataset = load_dataset(
                data_type,
                data_files=data_files,
                split='train',
                streaming=True
            )
            print("  Successfully loaded SlimPajama dataset")
        except Exception as e:
            print(f"  Failed to load as {data_type}, trying text format: {e}")
            raw_dataset = load_dataset(
                'text',
                data_files=data_files,
                split='train',
                streaming=True
            )

        return raw_dataset

    def _load_local_generic_dataset(self):
        """Load a generic local dataset"""
        import os
        import glob

        if os.path.isdir(self.dataset_name):
            if os.path.exists(os.path.join(self.dataset_name, 'dataset_info.json')) or \
                    os.path.exists(os.path.join(self.dataset_name, 'data')):
                try:
                    raw_dataset = load_dataset(
                        self.dataset_name,
                        split=self.split,
                        streaming=True
                    )
                    return raw_dataset
                except Exception as e:
                    raise ValueError(f"  Failed to load as HuggingFace dataset: {e}")

            all_files = []
            for root, dirs, files in os.walk(self.dataset_name):
                for file in files[:10]:
                    all_files.append(os.path.join(root, file))

            # if all_files:
            #     print(f"  Sample files found: {[os.path.basename(f) for f in all_files[:3]]}")

            data_files = []
            supported_extensions = ['*.parquet', '*.json', '*.jsonl', '*.txt', '*.csv', '*.arrow', '*.zst']

            for ext in supported_extensions:
                patterns = [
                    os.path.join(self.dataset_name, ext),
                    os.path.join(self.dataset_name, '*', ext),
                    os.path.join(self.dataset_name, '**', ext)
                ]

                for pattern in patterns:
                    files = glob.glob(pattern, recursive=True)
                    if files:
                        data_files = files
                        # print(f"  Found {len(files)} files with extension {ext}")
                        break

                if data_files:
                    break

            if not data_files:
                special_patterns = ['train/*', 'validation/*', 'test/*', 'chunk*', '*.bin', '*.h5']
                for pattern in special_patterns:
                    files = glob.glob(os.path.join(self.dataset_name, pattern))
                    if files:
                        data_files = files
                        # print(f"  Found {len(files)} files matching pattern {pattern}")
                        break

            if not data_files:
                all_files = glob.glob(os.path.join(self.dataset_name, '*'))
                data_files = [f for f in all_files if os.path.isfile(f) and not os.path.basename(f).startswith('.')]
                if data_files:
                    raise ValueError(f"  Found {len(data_files)} files in directory")

            if not data_files:
                raise ValueError(
                    f"No supported data files found in {self.dataset_name}\n"
                    f"Supported formats: parquet, json, jsonl, txt, csv, arrow, zst"
                )

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

            if 'train' not in self.split.lower() and len(data_files) > 100:
                print(f"  Large dataset detected. Limiting to 100 files for {self.split} split")
                data_files = data_files[:100]

            # print(f"  Loading {len(data_files)} {ext} files as {data_type} format")

            try:
                raw_dataset = load_dataset(
                    data_type,
                    data_files=data_files,
                    split='train',
                    streaming=True
                )
            except Exception as e:
                print(f"  Error loading with {data_type} format: {e}")
                raw_dataset = load_dataset(
                    'text',
                    data_files=data_files,
                    split='train',
                    streaming=True
                )
        else:
            ext = os.path.splitext(self.dataset_name)[1].lower()
            # print(f"  Loading single file with extension {ext}")

            if ext == '.parquet':
                raw_dataset = load_dataset('parquet', data_files=self.dataset_name, split='train', streaming=True)
            elif ext in ['.json', '.jsonl', '.zst']:
                raw_dataset = load_dataset('json', data_files=self.dataset_name, split='train', streaming=True)
            elif ext == '.txt':
                raw_dataset = load_dataset('text', data_files=self.dataset_name, split='train', streaming=True)
            elif ext == '.csv':
                raw_dataset = load_dataset('csv', data_files=self.dataset_name, split='train', streaming=True)
            else:
                # print(f"  Unknown extension {ext}, attempting as text")
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

    def __init__(self):
        pass

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([f['input_ids'] for f in features])
        labels = torch.stack([f['labels'] for f in features])

        attention_mask = torch.ones_like(input_ids)

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }


def prefix_tokens_with_noise(
        source_distribution: SourceDistribution,
        prompts: List[str],
        max_seq_len: int,
        tokenizer: PreTrainedTokenizer
    ) -> torch.Tensor:
    noise_tokens = torch.empty(1, max_seq_len)
    for prompt in prompts:
        prefix_tokens = tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt')
        noise = source_distribution.sample(
            tensor_size=(1, max_seq_len-prefix_tokens.size(-1)),
            device=prefix_tokens.device
        )
        print(f"\n{ prompt} -> {prefix_tokens.shape}, {noise.shape}")
        noise_tokens = torch.cat(
            [
                noise_tokens,
                torch.cat(
                    [prefix_tokens, noise],
                    dim=-1
                )
            ],
            dim=0
        )
    return noise_tokens.long()
class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs):
        logits = self.model.forward(input_ids=x, timesteps=t).logits
        return F.softmax(logits, dim=-1)

def generate(
        model: nn.Module,
        prompts: List[str],
        step: int,
        sampling_steps: int,
        tokenizer: PreTrainedTokenizer,
        vocab_size: int,
        path: ProbPath,
        source_distribution: SourceDistribution,
        seq_len: int,
        time_epsilon: float = 0.0,
        save_dir: Optional[Path] = None
    ) -> torch.Tensor:
    solver = MixtureDiscreteEulerSolver(
        model=model,
        path=path,
        vocabulary_size=vocab_size
    )
    x_init = prefix_tokens_with_noise(
        source_distribution,
        prompts,
        seq_len,
        tokenizer
    )
    sample = solver.sample(
        x_init=x_init,
        step_size=1 / sampling_steps,
        verbose=True,
        time_epsilon=time_epsilon,
        dtype_sample=torch.float32,
        time_grid=torch.tensor([0.0, 1.0-time_epsilon]),
    )
    sentences = tokenizer.batch_decode(sample)
    if save_dir is not None:
        file_name = save_dir / f"iter_{step}.txt"
        file_name.parents[0].mkdir(exist_ok=True, parents=True)

        with open(file_name, "w") as file:
            for sentence in sentences:
                file.write(f"{sentence}\n{'=' * 20} New sample {'=' * 20}\n")
    return sample

__all__ = [
    'CosformerDataset',
    'MyDataCollator',
    "generate",
]