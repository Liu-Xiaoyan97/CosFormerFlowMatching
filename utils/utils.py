import torch
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer, TrainerCallback, Trainer, TrainingArguments
from datasets import load_dataset
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from tqdm import tqdm
from omegaconf import OmegaConf

class DetailedLoggingCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.training_bar = None
        self.prediction_bar = None
        self.validation_bar = None

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            total_train_steps = state.max_steps
            if total_train_steps is None or total_train_steps <= 0:
                if hasattr(args, 'num_train_epochs') and hasattr(state, 'epoch') is False:
                    total_train_steps = 1000  # 默认值或根据数据集大小估算
            self.training_bar = tqdm(total=total_train_steps, desc="Training")
        
    def on_step_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero and self.training_bar is not None:
            self.training_bar.update(1)
            logs = state.log_history[-1] if state.log_history else {}
            loss = logs.get("loss", float('inf'))
            lr = logs.get("learning_rate", 0)
            try:
                perplexity = torch.exp(torch.tensor(loss)).item()
            except:
                perplexity = float('inf')
            logs['ppl'] = perplexity
            postfix_dict = {
                "loss": f"{loss:.4f}",
                "lr": f"{lr:.2e}",
                "ppl": f"{perplexity:.2f}"
            }
            if len(state.log_history) >= 2:
                for i in range(len(state.log_history) - 1, -1, -1):
                    log = state.log_history[i]
                    if "eval_loss" in log:
                        eval_loss = log["eval_loss"]
                        try:
                            eval_perplexity = torch.exp(torch.tensor(eval_loss)).item()
                        except:
                            eval_perplexity = float('inf')
                        logs["eval_ppl"] = eval_perplexity
                        postfix_dict["eval_loss"] = f"{eval_loss:.4f}"
                        postfix_dict["eval_ppl"] = f"{eval_perplexity:.2f}"
                        break
            
            self.training_bar.set_postfix(postfix_dict)

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        if state.is_local_process_zero and self.validation_bar is not None:
            self.validation_bar.update(1)

    def on_evaluate(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            if self.validation_bar is not None:
                self.validation_bar.close()
                self.validation_bar = None
            logs = state.log_history[-1] if state.log_history else {}
            eval_loss = logs.get("eval_loss", float('inf'))
            try:
                eval_perplexity = torch.exp(torch.tensor(eval_loss)).item()
            except:
                eval_perplexity = float('inf')
            if self.training_bar is not None:
                current_postfix = self.training_bar.format_dict.get('postfix', {})
                if isinstance(current_postfix, str):
                    current_postfix = {}
                elif current_postfix is None:
                    current_postfix = {}
                    
                current_postfix.update({
                    "eval_loss": f"{eval_loss:.4f}",
                    "eval_ppl": f"{eval_perplexity:.2f}"
                })
                self.training_bar.set_postfix(current_postfix)
                epoch_str = f" Epoch: {state.epoch:.2f}" if state.epoch is not None else ""
                print(f"\nEval{epoch_str} Step: {state.global_step} - "
                      f"Eval Loss: {eval_loss:.4f} - "
                      f"Eval Perplexity: {eval_perplexity:.2f}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        pass

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero and self.training_bar is not None:
            self.training_bar.close()
            self.training_bar = None

    def on_evaluate_begin(self, args, state, control, **kwargs):
        """在评估开始时调用"""
        if state.is_local_process_zero:
            self.validation_bar = tqdm(
                total=None,  
                desc="Validating",
                leave=False 
            )
            
            
class MyDataset(IterableDataset):
    def __init__(self, 
                 tokenizer_path: str = "Tokenizer_32768_v1", 
                 dataset_name: str = "stanfordnlp/imdb",
                 split: str = "train",
                 chunk_size: int = 8):
        self.data = load_dataset(dataset_name, split=split)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.bos_token_id is None:
            raise ValueError("Tokenizer must have a bos_token_id")
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must have an eos_token_id")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
            
        self.buffer = []
        self.chunk_size = chunk_size

    def __len__(self):
        return int(1e9)
    
    def __iter__(self):
        for example in self.data:
            input_ids = self.tokenizer(example['text'], return_attention_mask=False)["input_ids"]
            self.buffer.extend(input_ids)
            
            while len(self.buffer) >= self.chunk_size - 1:

                model_input_ids = [self.tokenizer.bos_token_id] + self.buffer[:self.chunk_size-1]

                targets = self.buffer[:self.chunk_size-1] + [self.tokenizer.eos_token_id]
                
                yield {
                    "input_ids": model_input_ids,
                    "labels": targets
                }

                self.buffer = self.buffer[self.chunk_size-1:]
        

        if len(self.buffer) > 0:
            num_real_tokens = len(self.buffer)
            num_pad_tokens_input = self.chunk_size - 1 - num_real_tokens
            num_pad_tokens_labels = self.chunk_size - 2 - num_real_tokens 

            model_input_ids = [self.tokenizer.pad_token_id] * num_pad_tokens_input + \
                              [self.tokenizer.bos_token_id] + self.buffer
                              
            targets = [-100] * num_pad_tokens_labels + \
                      [self.tokenizer.bos_token_id] + self.buffer + [self.tokenizer.eos_token_id]
            
            yield {
                "input_ids": model_input_ids,
                "labels": targets
            }
            self.buffer = []
            
def sample_timesteps(batch_size, device, method='uniform'):
    """采样时间步 t"""
    if method == 'uniform':
        eps = 1e-5
        return torch.rand(batch_size, device=device) * (1 - eps) + eps
    else:
        raise NotImplementedError

@dataclass
class MyDataCollator:
    """
    数据整理器，将一批样本整理成模型需要的格式。
    """
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [feature["input_ids"] for feature in features]
        labels = [feature["labels"] for feature in features]

        batch_input_ids = torch.tensor(input_ids, dtype=torch.long)
        batch_labels = torch.tensor(labels, dtype=torch.long)
        
        attention_mask = (batch_input_ids != 0).long()

        return {
            "input_ids": batch_input_ids,
            "attention_mask": attention_mask,
            "labels": batch_labels,
            "timesteps": sample_timesteps(batch_input_ids.shape[0], batch_input_ids.device)
        }

def load_training_args_from_yaml(config_path: str) -> TrainingArguments:
    """
    从 YAML 文件加载配置并创建 TrainingArguments
    """
    cfg = OmegaConf.load(config_path)
    training_args_dict = OmegaConf.to_container(cfg.training_args, resolve=True)
    training_args = TrainingArguments(**training_args_dict)
    return training_args