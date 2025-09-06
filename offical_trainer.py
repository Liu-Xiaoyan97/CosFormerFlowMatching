import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import json
import os
import shutil
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
import warnings
import socket
from datetime import timedelta


# 分布式训练相关设置
def setup_distributed():
    """初始化分布式训练环境"""
    # 检查是否在分布式环境中
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    elif 'SLURM_PROCID' in os.environ:
        # SLURM环境
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = rank % torch.cuda.device_count()
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(local_rank)
    elif torch.cuda.device_count() > 1:
        # 单机多卡，自动设置
        print("Detected multiple GPUs, setting up single-node multi-GPU training")
        rank = 0
        local_rank = 0
        world_size = 1
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(local_rank)
        return False, rank, local_rank, world_size
    else:
        # 单机单卡
        return False, 0, 0, 1

    # 设置MASTER_ADDR和MASTER_PORT
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '29500'

    # 初始化进程组
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
        timeout=timedelta(minutes=30)
    )

    # 设置当前设备
    torch.cuda.set_device(local_rank)

    return True, rank, local_rank, world_size


def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    """检查是否是主进程"""
    return rank == 0


def print_rank0(*args, rank=0, **kwargs):
    """只在主进程打印"""
    if is_main_process(rank):
        print(*args, **kwargs)


def save_on_main_process(func):
    """装饰器：只在主进程执行保存操作"""

    def wrapper(*args, **kwargs):
        rank = kwargs.get('rank', 0)
        if is_main_process(rank):
            return func(*args, **kwargs)

    return wrapper


def load_model_config_from_yaml(config_path: str) -> DictConfig:
    config = OmegaConf.load(config_path)
    return OmegaConf.to_container(config, resolve=True)


def compute_loss(model, batch, config, model_config, device):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    timesteps = (
            torch.rand(
                batch["input_ids"].size(0), device=device)
            * (model_config["flow"]["sde_t"] - model_config["flow"]["time_epsilon"])
            + model_config["flow"]["time_epsilon"]
    )
    outputs = model(
        input_ids=input_ids,
        timesteps=timesteps,
        attention_mask=attention_mask,
    )
    return outputs


@save_on_main_process
def save_checkpoint(model, optimizer, scheduler, total_steps, config, rank=0, avg_eval_loss=None, avg_eval_ppl=None):
    """保存checkpoint，包括模型、优化器和调度器状态（只在主进程执行）"""
    checkpoint_dir = Path(f"{config.evaluation.output_dir}/checkpoint-{total_steps}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 如果是DDP模型，获取原始模型
    model_to_save = model.module if hasattr(model, 'module') else model

    # 保存模型
    model_to_save.save_pretrained(checkpoint_dir)

    # 保存优化器和调度器状态
    checkpoint_state = {
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'total_steps': total_steps,
        'eval_loss': avg_eval_loss,
        'eval_ppl': avg_eval_ppl
    }
    torch.save(checkpoint_state, checkpoint_dir / 'training_state.pt')

    # 保存训练元信息
    training_info = {
        'total_steps': total_steps,
        'eval_loss': avg_eval_loss,
        'eval_ppl': avg_eval_ppl
    }
    with open(checkpoint_dir / 'training_info.json', 'w') as f:
        json.dump(training_info, f, indent=2)

    print(f"✓ Checkpoint saved at step {total_steps} to {checkpoint_dir}")

    # 管理checkpoint数量限制
    manage_checkpoints(config.evaluation.output_dir, config.evaluation.save_total_limit)


@save_on_main_process
def manage_checkpoints(output_dir, save_total_limit, rank=0):
    """管理checkpoint数量，只保留最新的save_total_limit个（只在主进程执行）"""
    if save_total_limit is None or save_total_limit < 1:
        return

    output_path = Path(output_dir)
    checkpoint_dirs = []
    for d in output_path.glob("checkpoint-*"):
        if d.is_dir():
            try:
                step = int(d.name.split('-')[-1])
                checkpoint_dirs.append((step, d))
            except ValueError:
                continue

    checkpoint_dirs.sort(key=lambda x: x[0])

    if len(checkpoint_dirs) > save_total_limit:
        checkpoints_to_delete = checkpoint_dirs[:-save_total_limit]
        for step, dir_path in checkpoints_to_delete:
            print(f"  Removing old checkpoint at step {step}")
            shutil.rmtree(dir_path)


def find_latest_checkpoint(output_dir):
    """查找最新的checkpoint"""
    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    checkpoint_dirs = []
    for d in output_path.glob("checkpoint-*"):
        if d.is_dir() and (d / 'training_state.pt').exists():
            try:
                step = int(d.name.split('-')[-1])
                checkpoint_dirs.append((step, d))
            except ValueError:
                continue

    if not checkpoint_dirs:
        return None

    checkpoint_dirs.sort(key=lambda x: x[0])
    return checkpoint_dirs[-1][1]


def load_checkpoint(checkpoint_dir, model, optimizer, scheduler, device, is_distributed=False):
    """从checkpoint恢复训练状态"""
    checkpoint_path = Path(checkpoint_dir)

    # 加载模型
    model_to_load = LLMFCosformerForCausalLM.from_pretrained(checkpoint_path)

    if is_distributed:
        # 如果是分布式训练，需要先将模型移到设备上，然后包装成DDP
        model_to_load = model_to_load.to(device)
        model = DDP(model_to_load, device_ids=[device], output_device=device)
    else:
        model = model_to_load.to(device)

    # 加载优化器和调度器状态
    training_state_path = checkpoint_path / 'training_state.pt'
    if training_state_path.exists():
        checkpoint_state = torch.load(training_state_path, map_location=device, weights_only=False)
        optimizer.load_state_dict(checkpoint_state['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint_state['scheduler_state_dict'])
        total_steps = checkpoint_state['total_steps']
        avg_eval_loss = checkpoint_state.get('eval_loss', None)
        avg_eval_ppl = checkpoint_state.get('eval_ppl', None)

        print(f"✓ Resumed from checkpoint at step {total_steps}")
        if avg_eval_loss is not None:
            print(f"  Previous eval_loss: {avg_eval_loss:.4f}")
        if avg_eval_ppl is not None:
            print(f"  Previous eval_ppl: {avg_eval_ppl:.4f}")

        return model, total_steps, avg_eval_loss, avg_eval_ppl
    else:
        raise FileNotFoundError(f"Training state not found in {checkpoint_path}")


def train(model, train_loader, eval_loader, optimizer, scheduler, tokenizer, config, model_config,
          rank=0, world_size=1, device=None, is_distributed=False, resume_from_checkpoint=None):
    """训练函数，支持分布式训练"""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    total_steps = 0
    avg_eval_loss = None
    avg_eval_ppl = None

    # 尝试从checkpoint恢复
    if resume_from_checkpoint and is_main_process(rank):
        if resume_from_checkpoint == "auto":
            latest_checkpoint = find_latest_checkpoint(config.evaluation.output_dir)
            if latest_checkpoint:
                print_rank0(f"Found latest checkpoint: {latest_checkpoint}", rank=rank)
                model, total_steps, avg_eval_loss, avg_eval_ppl = load_checkpoint(
                    latest_checkpoint, model, optimizer, scheduler, device, is_distributed
                )
            else:
                print_rank0("No checkpoint found, starting from scratch", rank=rank)
        else:
            checkpoint_path = Path(resume_from_checkpoint)
            if checkpoint_path.exists():
                model, total_steps, avg_eval_loss, avg_eval_ppl = load_checkpoint(
                    checkpoint_path, model, optimizer, scheduler, device, is_distributed
                )
            else:
                raise FileNotFoundError(f"Checkpoint not found: {resume_from_checkpoint}")

    # 同步所有进程的total_steps
    if is_distributed:
        total_steps_tensor = torch.tensor(total_steps, device=device)
        dist.broadcast(total_steps_tensor, src=0)
        total_steps = total_steps_tensor.item()

    # 初始化TensorBoard writer（只在主进程）
    writer = None
    if is_main_process(rank):
        writer = SummaryWriter(log_dir=config.evaluation.output_dir)

    # 计算每个进程需要跳过的步数
    skip_steps = total_steps

    # 梯度累积设置
    # 保持原始的梯度累积步数，不根据world_size调整
    gradient_accumulation_steps = config.train.gradient_accumulation_steps
    effective_batch_size = config.dataset.batch_size * world_size * gradient_accumulation_steps

    print_rank0(f"Effective batch size: {effective_batch_size}", rank=rank)
    print_rank0(f"Gradient accumulation steps: {gradient_accumulation_steps}", rank=rank)
    print_rank0(f"Per-device batch size: {config.dataset.batch_size}", rank=rank)

    # 梯度累积相关变量
    accumulation_counter = 0
    accumulated_loss_1 = 0.0
    accumulated_loss_2 = 0.0
    accumulated_loss = 0.0

    while total_steps < config.train.max_steps:
        # 如果使用分布式采样器，需要设置epoch
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(total_steps // len(train_loader))

        progress_bar = tqdm(train_loader, desc=f"Training",
                            disable=not is_main_process(rank))

        for batch_idx, batch in enumerate(progress_bar):
            # 跳过已经处理过的批次
            if skip_steps > 0:
                skip_steps -= 1
                continue

            # 前向传播和反向传播
            outputs = compute_loss(model, batch, config, model_config, device)
            loss_1 = outputs.loss_1
            loss_2 = outputs.loss_2
            loss = outputs.total_loss / gradient_accumulation_steps  # 损失缩放
            loss.backward()

            # 累积损失用于日志记录（取平均值）
            accumulated_loss_1 += loss_1.item() / gradient_accumulation_steps
            accumulated_loss_2 += loss_2.item() / gradient_accumulation_steps
            accumulated_loss += outputs.total_loss.item() / gradient_accumulation_steps
            accumulation_counter += 1

            # 在完成一个梯度累积周期后更新参数
            if accumulation_counter % gradient_accumulation_steps == 0:
                # 梯度裁剪
                if hasattr(config.train, 'max_grad_norm') and config.train.max_grad_norm is not None:
                    if is_distributed:
                        torch.nn.utils.clip_grad_norm_(model.module.parameters(), config.train.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.max_grad_norm)

                # 计算梯度和参数范数
                if is_distributed:
                    param_norm = sum(p.data.norm(2).item() ** 2 for p in model.module.parameters()) ** 0.5
                    grad_norm = sum(
                        p.grad.data.norm(2).item() ** 2 for p in model.module.parameters() if p.grad is not None) ** 0.5
                else:
                    param_norm = sum(p.data.norm(2).item() ** 2 for p in model.parameters()) ** 0.5
                    grad_norm = sum(
                        p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5

                # 梯度放大（如果需要）
                if grad_norm < 1e-3 and total_steps < 100:
                    params = model.module.parameters() if is_distributed else model.parameters()
                    for p in params:
                        if p.grad is not None:
                            p.grad.data *= 1000

                # 计算困惑度
                train_ppl = torch.exp(torch.tensor(accumulated_loss_2)).item()

                # 更新参数
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # 更新步数
                total_steps += 1

                # 日志记录（只在主进程）
                if is_main_process(rank) and total_steps % config.train.logging_steps == 0:
                    writer.add_scalar('Train/Loss_1', accumulated_loss_1, total_steps)
                    writer.add_scalar('Train/Loss_2', accumulated_loss_2, total_steps)
                    writer.add_scalar('Train/Total_Loss', accumulated_loss, total_steps)
                    writer.add_scalar('Train/Perplexity', train_ppl, total_steps)
                    writer.add_scalar('Train/Param_Norm', param_norm, total_steps)
                    writer.add_scalar('Train/Grad_Norm', grad_norm, total_steps)
                    if scheduler.get_last_lr():
                        writer.add_scalar('Train/Learning_Rate', scheduler.get_last_lr()[0], total_steps)

                # 更新进度条（只在主进程）
                if is_main_process(rank):
                    progress_bar.set_postfix({
                        "step": total_steps,
                        "loss_1": f"{accumulated_loss_1:.4f}",
                        "loss_2": f"{accumulated_loss_2:.4f}",
                        "total_loss": f"{accumulated_loss:.4f}",
                        "train_ppl": f"{train_ppl:.4f}",
                        "grad_norm": f"{grad_norm:.4f}",
                        "eval_loss": f"{avg_eval_loss:.4f}" if avg_eval_loss is not None else "N/A",
                        "eval_ppl": f"{avg_eval_ppl:.4f}" if avg_eval_ppl is not None else "N/A"
                    })

                # 重置累积变量
                accumulated_loss_1 = 0.0
                accumulated_loss_2 = 0.0
                accumulated_loss = 0.0

                # 保存checkpoint和评估（只在主进程）
                if total_steps % config.evaluation.save_steps == 0:
                    if is_main_process(rank):
                        # 评估
                        eval_steps = 0
                        total_eval_loss = 0
                        total_eval_ppl = 0
                        eval_bar = tqdm(eval_loader, desc=f"Evaluation at step {total_steps}")
                        model.eval()

                        with torch.no_grad():
                            for eval_batch in eval_bar:
                                outputs = compute_loss(model, eval_batch, config, model_config, device)
                                eval_ppl = torch.exp(outputs.loss_2)
                                batch_eval_loss = outputs.total_loss
                                total_eval_loss += batch_eval_loss.item()
                                total_eval_ppl += eval_ppl.item()
                                eval_steps += 1

                        avg_eval_loss = total_eval_loss / eval_steps if eval_steps > 0 else 0
                        avg_eval_ppl = total_eval_ppl / eval_steps if eval_steps > 0 else 0

                        # 记录评估指标
                        writer.add_scalar('Eval/Loss', avg_eval_loss, total_steps)
                        writer.add_scalar('Eval/Perplexity', avg_eval_ppl, total_steps)

                        # 保存checkpoint
                        save_checkpoint(model, optimizer, scheduler, total_steps, config,
                                        rank=rank, avg_eval_loss=avg_eval_loss, avg_eval_ppl=avg_eval_ppl)

                        model.train()

                    # 同步所有进程
                    if is_distributed:
                        dist.barrier()

                # 生成样本（只在主进程）
                if total_steps % config.evaluation.perplexity_steps == 0 and is_main_process(rank):
                    model.eval()
                    model_to_use = model.module if is_distributed else model
                    wrapped_model = WrappedModel(model_to_use)
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

                # 达到最大步数，退出
                if total_steps >= config.train.max_steps:
                    print_rank0(f"\nReached maximum training steps: {config.train.max_steps}", rank=rank)
                    break

    # 训练结束，保存最终checkpoint（只在主进程）
    if is_main_process(rank):
        print("\nTraining completed. Saving final checkpoint...")
        save_checkpoint(model, optimizer, scheduler, total_steps, config,
                        rank=rank, avg_eval_loss=avg_eval_loss, avg_eval_ppl=avg_eval_ppl)
        writer.close()

    # 等待所有进程完成
    if is_distributed:
        dist.barrier()


@hydra.main(version_base=None, config_path="./config", config_name="train.yaml")
def main(config: DictConfig) -> None:
    # 设置分布式训练
    is_distributed, rank, local_rank, world_size = setup_distributed()

    # 设置设备
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    print_rank0(f"Using device: {device}", rank=rank)
    print_rank0(f"World size: {world_size}, Rank: {rank}, Local rank: {local_rank}", rank=rank)

    # 检查是否需要启动分布式训练
    if torch.cuda.device_count() > 1 and not is_distributed:
        # 单机多卡，自动启动分布式训练
        import subprocess
        import sys

        print("Launching distributed training on", torch.cuda.device_count(), "GPUs")

        # 设置环境变量
        env = os.environ.copy()
        env['MASTER_ADDR'] = 'localhost'
        env['MASTER_PORT'] = '29500'
        env['WORLD_SIZE'] = str(torch.cuda.device_count())

        # 启动多个进程
        processes = []
        for i in range(torch.cuda.device_count()):
            env['RANK'] = str(i)
            env['LOCAL_RANK'] = str(i)

            cmd = [sys.executable] + sys.argv
            process = subprocess.Popen(cmd, env=env)
            processes.append(process)

        # 等待所有进程完成
        for process in processes:
            process.wait()

        return

    # 添加resume_from_checkpoint参数支持
    resume_from_checkpoint = getattr(config, 'resume_from_checkpoint', None)
    if resume_from_checkpoint is None:
        resume_from_checkpoint = os.environ.get('RESUME_FROM_CHECKPOINT', None)

    # 加载模型配置
    model_config = load_model_config_from_yaml("config/model.yaml")
    cosformer_config = LLMFCosformerConfig(**model_config)

    # 创建模型
    model = LLMFCosformerForCausalLM(cosformer_config)
    model = model.to(device)

    # 如果是分布式训练，包装模型
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        print_rank0(f"Model wrapped with DDP on rank {rank}", rank=rank)

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.dataset.tokenizer)

    # 创建数据集
    train_dataset = CosformerDataset(
        tokenizer=tokenizer,
        dataset_name=config.dataset.name,
        split="train",
        chunk_size=config.dataset.chunk_size,
        max_samples=config.train.max_train_samples if not is_distributed else config.train.max_train_samples // world_size,
        seed=config.dataset.seed + rank  # 每个进程使用不同的seed
    )

    eval_dataset = CosformerDataset(
        tokenizer=tokenizer,
        dataset_name=config.dataset.name,
        split="test",
        chunk_size=config.dataset.chunk_size,
        max_samples=config.evaluation.max_eval_samples,
        seed=config.dataset.seed
    )

    # 创建数据加载器
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if is_distributed else None
    eval_sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank,
                                      shuffle=False) if is_distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dataset.batch_size,
        # shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=MyDataCollator(),
        num_workers=config.dataset.num_workers,
        pin_memory=config.dataset.pin_memory,
        drop_last=True
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        sampler=eval_sampler,
        collate_fn=MyDataCollator(),
        num_workers=config.dataset.num_workers,
        pin_memory=config.dataset.pin_memory
    )

    # 创建优化器（注意：如果是DDP，需要使用model.module的参数）
    model_params = model.module.parameters() if is_distributed else model.parameters()
    optimizer = AdamW(model_params, **config.optimizer)

    # 创建学习率调度器
    if config.scheduler.warmup_steps == 0 and config.scheduler.warmup_ratio > 0:
        warmup_steps = int(config.scheduler.warmup_ratio * config.scheduler.num_training_steps)
    else:
        warmup_steps = config.scheduler.warmup_steps

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=config.scheduler.num_training_steps
    )

    # 开始训练
    try:
        train(
            model, train_loader, eval_loader, optimizer, scheduler, tokenizer,
            config, model_config, rank=rank, world_size=world_size,
            device=device, is_distributed=is_distributed,
            resume_from_checkpoint=resume_from_checkpoint
        )
    finally:
        # 清理分布式环境
        cleanup_distributed()


if __name__ == "__main__":
    main()