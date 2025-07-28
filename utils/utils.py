import torch
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer, TrainerCallback, Trainer, TrainingArguments
from datasets import load_dataset
from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass
from tqdm import tqdm
from omegaconf import OmegaConf
from transformers.modeling_outputs import CausalLMOutputWithPast
from evalution.logic.flow import get_source_distribution, get_path
from flow_matching.loss import MixturePathGeneralizedKL
from contextlib import nullcontext
import torch.nn as nn
from flow_matching.path import ProbPath
from omegaconf import OmegaConf
import math
from CosFormer.configuration_LLMFCosformer import LLMFCosformerConfig


class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    ENDC = '\033[0m'

class DetailedProgressCallback(TrainerCallback):
    """
    自定义进度条回调，在训练过程中显示详细的指标
    """
    def __init__(self):
        super().__init__()
        self.training_bar = None
        self.prediction_bar = None
        self.current_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            # 计算总步数
            total_train_steps = state.max_steps
            if total_train_steps <= 0:
                # 如果 max_steps 不可用，使用 num_train_epochs 计算
                num_update_steps_per_epoch = len(kwargs['train_dataloader']) // args.gradient_accumulation_steps
                total_train_steps = args.num_train_epochs * num_update_steps_per_epoch
                
            self.training_bar = tqdm(
                total=total_train_steps,
                desc=f"{Colors.BLUE}Training{Colors.ENDC}",
                colour="blue",
                ncols=100,
                leave=True
            )

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero and self.training_bar is not None:
            self.current_step += 1
            self.training_bar.update(1)
            # 获取最新的训练日志
            logs = state.log_history[-1] if state.log_history else {}
            
            # 提取指标
            loss = logs.get("loss", float('inf'))
            learning_rate = logs.get("learning_rate", 0)
            epoch = logs.get("epoch", 0)
            perplexity = math.exp(loss) if loss < 20 else float('inf') 
            
            # 准备显示的指标字典
            postfix_dict = {
                "loss": f"{loss:.4f}",
                "ppl": f"{perplexity:.2f}",
                "lr": f"{learning_rate:.2e}",
                "epoch": f"{epoch:.2f}"
            }
            
            # 如果有验证结果，也显示
            for i in range(len(state.log_history) - 1, -1, -1):
                log = state.log_history[i]
                if "eval_loss" in log:
                    eval_loss = log["eval_loss"]
                    try:
                        eval_ppl = math.exp(eval_loss) if eval_loss < 20 else float('inf')
                    except:
                        eval_ppl = float('inf')
                    postfix_dict["eval_loss"] = f"{eval_loss:.4f}"
                    postfix_dict["eval_ppl"] = f"{eval_ppl:.2f}"
                    break
            
            self.training_bar.set_postfix(postfix_dict)
    
    def on_evaluate(self, args, state, control, **kwargs):
        if state.is_local_process_zero and self.training_bar is not None:
            # 获取最新的评估日志
            logs = state.log_history[-1] if state.log_history else {}
            eval_loss = logs.get("eval_loss", float('inf'))

            eval_ppl = math.exp(eval_loss) if eval_loss < 20 else float('inf')
            
            # 更新进度条
            current_postfix = self.training_bar.format_dict.get('postfix', {})
            if isinstance(current_postfix, str):
                current_postfix = {}
            elif current_postfix is None:
                current_postfix = {}
                
            current_postfix.update({
                "eval_loss": f"{eval_loss:.4f}",
                "eval_ppl": f"{eval_ppl:.2f}"
            })
            self.training_bar.set_postfix(current_postfix)
            
            # 打印独立的评估信息
            epoch_str = f" Epoch: {state.epoch:.2f}" if state.epoch is not None else ""
            print(f"\nEval{epoch_str} Step: {state.global_step} - "
                  f"Eval Loss: {eval_loss:.4f} - "
                  f"Eval Perplexity: {eval_ppl:.2f}")

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero and self.training_bar is not None:
            self.training_bar.close()
            self.training_bar = None

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        if state.is_local_process_zero:
            if self.prediction_bar is None:
                self.prediction_bar = tqdm(
                    total=len(eval_dataloader),
                    desc=f"{Colors.YELLOW}Evaluating{Colors.ENDC}",
                    leave=False,
                    colour="yellow",
                    ncols=100
                )
            self.prediction_bar.update(1)

    def on_predict(self, args, state, control, metrics, **kwargs):
        if state.is_local_process_zero and self.prediction_bar is not None:
            self.prediction_bar.close()
            self.prediction_bar = None
            
            
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

def get_loss_function(loss_function: str, path: Optional[ProbPath] = None):
    if loss_function == "cross_entropy":
        return nn.CrossEntropyLoss(ignore_index=-100)
    elif loss_function == "generalized_kl":
        assert path is not None
        return MixturePathGeneralizedKL(path)
    else:
        raise ValueError(f"Unknown loss function: {loss_function}")
    
    
class MyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        flow_cfg = OmegaConf.load("config/trainingargs.yml")
        self.flow_cfg = OmegaConf.to_container(flow_cfg.trainer_args, resolve=True)
        config = LLMFCosformerConfig()
        self.source_distribution = get_source_distribution(source_distribution=self.flow_cfg["source_distribution"], vocab_size=config.vocab_size)
        self.path = get_path(scheduler_type=self.flow_cfg["scheduler_type"], exponent=self.flow_cfg["exponent"])
        
    def training_step(self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        if self.args.gradient_accumulation_steps > 1:
            loss = self.compute_loss(model, inputs) / self.args.gradient_accumulation_steps
        else:
            loss = self.compute_loss(model, inputs)
        
        if self.args.gradient_checkpointing:
            self.accelerator.backward(loss)
        else:
            loss.backward()
        
        return loss.detach()

    def compute_loss(self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], return_outputs: bool = False, num_items_in_batch: Optional[torch.Tensor] = None):
        loss_fct = get_loss_function(self.flow_cfg["loss_function"], self.path)
        time_epsilon = 1e-3 if isinstance(loss_fct, MixturePathGeneralizedKL) else 0.0
        x_1 = inputs["input_ids"]
        with torch.no_grad():
            x_0 = self.source_distribution.sample_like(x_1)
            t = torch.rand(x_1.shape[0], device=x_1.device)*(1.0 - time_epsilon)
            path_sample = self.path.sample(t=t, x_0 = x_0, x_1 = x_1)
            
        ctx = nullcontext() if self.args.gradient_checkpointing else torch.enable_grad()
        with ctx:
            inputs["input_ids"] = path_sample.x_t
            inputs["timesteps"] = path_sample.t
            outputs = model(**inputs)
            
            if isinstance(outputs, dict) and "loss" in outputs:
                loss = outputs["loss"]
            elif hasattr(outputs, "loss") and outputs.loss is not None:
                loss = outputs.loss
            else:
                logits = outputs.get("logits")
                labels = inputs.get("labels")
                if isinstance(loss_fct, nn.CrossEntropyLoss):
                    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                elif isinstance(loss_fct, MixturePathGeneralizedKL):
                    loss = loss_fct(logits=logits, x_1 = inputs["input_ids"], x_t = path_sample.x_t, t=path_sample.t).mean()
                else:
                    raise ValueError("Invalid loss function")
            
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool, ignore_keys: Optional[list[str]] = None) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()
            
            if prediction_loss_only:
                return (loss, None, None)
            logits = outputs.get("logits")
            labels = inputs.get("labels")
            
            return (loss, logits, labels)
    
