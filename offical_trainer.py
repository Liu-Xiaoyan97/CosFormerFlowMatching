import torch
from pathlib import Path
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm   
from CosFormer import LLMFCosformerForCausalLM, LLMFCosformerConfig
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from utils.utils import CosformerDataset, MyDataCollator, generate, WrappedModel
import hydra
from omegaconf import DictConfig, OmegaConf
from utils.flow import get_source_distribution, get_loss_function, get_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_config_from_yaml(config_path: str) -> DictConfig:
    config = OmegaConf.load(config_path)
    return OmegaConf.to_container(config, resolve=True)


def compute_loss(model, batch, config, model_config):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    timesteps = (
            torch.rand(
                batch["input_ids"].size(0))
            * (model_config["flow"]["sde_t"] - model_config["flow"]["time_epsilon"])
            + model_config["flow"]["time_epsilon"]
    )
    outputs = model(
        input_ids=input_ids,
        timesteps=timesteps,
        attention_mask=attention_mask,
    )
    return outputs


def train(model, train_loader, eval_loader, optimizer, scheduler, tokenizer, config, model_config):
    model.train()
    total_steps = 0
    avg_eval_loss = None
    avg_eval_ppl = None
    
    # 初始化TensorBoard writer
    writer = SummaryWriter(log_dir=config.evaluation.output_dir)
    
    while total_steps < config.train.max_steps:
        progress_bar = tqdm(train_loader, desc=f"Training")
        for batch in progress_bar:
            optimizer.zero_grad()
            outputs = compute_loss(model, batch, config, model_config)
            loss_1 = outputs.loss_1
            loss_2 = outputs.loss_2
            loss = outputs.total_loss
            loss.backward()

            if hasattr(config.train, 'max_grad_norm') and config.train.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.max_grad_norm)

            param_norm = sum(p.data.norm(2).item() ** 2 for p in model.parameters()) ** 0.5
            grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5

            if grad_norm < 1e-3 and total_steps < 100:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.data *= 1000

            train_ppl = torch.exp(loss_2).item()
            optimizer.step()
            scheduler.step()
            
            if total_steps % config.train.logging_steps == 0:
                writer.add_scalar('Train/Loss_1', loss_1.item(), total_steps)
                writer.add_scalar('Train/Loss_2', loss_2.item(), total_steps)
                writer.add_scalar('Train/Total_Loss', loss.item(), total_steps)
                writer.add_scalar('Train/Perplexity', train_ppl, total_steps)
                writer.add_scalar('Train/Param_Norm', param_norm, total_steps)
                writer.add_scalar('Train/Grad_Norm', grad_norm, total_steps)
                if scheduler.get_last_lr():
                    writer.add_scalar('Train/Learning_Rate', scheduler.get_last_lr()[0], total_steps)

            progress_bar.set_postfix({
                "loss_1": f"{loss_1.item():.4f}",
                "loss_2": f"{loss_2.item():.4f}",
                "total_loss": f"{loss:.4f}",
                "train_ppl": f"{train_ppl:.4f}" if train_ppl is not None else "N/A",
                "param_norm": f"{param_norm:.4f}",
                "grad_norm": f"{grad_norm:.4f}",
                "eval_loss": f"{avg_eval_loss:.4f}" if avg_eval_loss is not None else "N/A",
                "eval_ppl": f"{avg_eval_ppl:.4f}" if avg_eval_ppl is not None else "N/A"
            })
            total_steps += 1
            if total_steps % config.evaluation.save_steps == 0:
                model.save_pretrained(f"{config.evaluation.output_dir}/{total_steps}")
                eval_steps = 0
                total_eval_loss = 0
                total_eval_ppl = 0
                eval_bar = tqdm(eval_loader, desc=f"Evaluation")
                for batch in eval_bar:
                    model.eval()
                    outputs = compute_loss(model, batch, config, model_config)
                    eval_ppl = torch.exp(outputs.loss_2)
                    batch_eval_loss = outputs.total_loss
                    total_eval_loss += batch_eval_loss.item()
                    total_eval_ppl += eval_ppl.item()
                    eval_steps += 1
                    
                avg_eval_loss = total_eval_loss / eval_steps
                avg_eval_ppl = total_eval_ppl / eval_steps
                
                if total_steps % config.train.logging_steps == 0:
                    writer.add_scalar('Eval/Loss', avg_eval_loss, total_steps)
                    writer.add_scalar('Eval/Perplexity', avg_eval_ppl, total_steps)

            if total_steps % config.evaluation.perplexity_steps == 0:
                model.eval()
                wrapped_model = WrappedModel(model)
                samples, sentences = generate(
                    model=wrapped_model,
                    prompts=["hello", "how are", "give me you"],
                    step=total_steps,
                    sampling_steps=10,
                    tokenizer=tokenizer,
                    vocab_size=model_config["vocab_size"],
                    path=get_path(
                        scheduler_type=model_config["flow"]["scheduler_type"],
                        exponent=model_config["flow"]["exponent"]
                    ),
                    source_distribution=get_source_distribution(
                        source_distribution=model_config["flow"]["source_distribution"],
                        vocab_size=model_config["vocab_size"]
                    ),
                    seq_len=100,
                    time_epsilon=0.0,
                    save_dir=Path("outputs/")
                )
                
                # 记录生成文本示例到TensorBoard
                for i, sentence in enumerate(sentences):
                    writer.add_text(f'Generated_Text/Sample_{i}', sentence, total_steps)
                
                model.train()
    
    # 关闭writer
    writer.close()


@hydra.main(version_base=None, config_path="./config", config_name="train.yaml")
def main(config: DictConfig) -> None:
    model_config = load_model_config_from_yaml("config/model.yaml")
    cosformer_config = LLMFCosformerConfig(**model_config)
    model = LLMFCosformerForCausalLM(cosformer_config)
    tokenizer = AutoTokenizer.from_pretrained(config.dataset.tokenizer)
    train_dataset = CosformerDataset(
        tokenizer=tokenizer,
        dataset_name=config.dataset.name,
        split="train",
        chunk_size=config.dataset.chunk_size,
        max_samples=config.train.max_train_samples,
        seed=config.dataset.seed
    )
    eval_dataset = CosformerDataset(
        tokenizer=tokenizer,
        dataset_name=config.dataset.name,
        split="test",
        chunk_size=config.dataset.chunk_size,
        max_samples=config.evaluation.max_eval_samples,
        seed=config.dataset.seed
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        collate_fn=MyDataCollator(),
        num_workers=config.dataset.num_workers
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        collate_fn=MyDataCollator(),
        num_workers=config.dataset.num_workers
    )
    optimizer = AdamW(model.parameters(),
                      **config.optimizer)

    if config.scheduler.warmup_steps == 0 and config.scheduler.warmup_ratio > 0:
        warmup_steps = int(config.scheduler.warmup_ratio * config.scheduler.num_training_steps)
    else:
        warmup_steps = config.scheduler.warmup_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=config.scheduler.num_training_steps
    )
    train(model, train_loader, eval_loader, optimizer, scheduler, tokenizer, config, model_config)


if __name__ == "__main__":
    main()