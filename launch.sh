#!/bin/bash

# Multi-GPU training launch script for Flow Matching CosFormer

# Set environment variables for better performance
export OMP_NUM_THREADS=8
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false

# For debugging DDP issues (optional, comment out for production)
# export TORCH_DISTRIBUTED_DEBUG=INFO

# Number of GPUs to use (modify as needed)
NUM_GPUS=4

# Training script
SCRIPT="trainer.py"

# Method 1: Using accelerate (recommended)
echo "Launching multi-GPU training with accelerate..."
accelerate launch \
    --multi_gpu \
    --num_processes $NUM_GPUS \
    --num_machines 1 \
    --mixed_precision no \
    --dynamo_backend no \
    $SCRIPT \
#     >result.log 2>&1

# Method 2: Using torchrun (alternative)
# echo "Launching multi-GPU training with torchrun..."
# torchrun \
#     --nproc_per_node=$NUM_GPUS \
#     --master_port=29500 \
#     $SCRIPT

# Method 3: Using python -m torch.distributed.launch (legacy)
# echo "Launching multi-GPU training with torch.distributed.launch..."
# python -m torch.distributed.launch \
#     --nproc_per_node=$NUM_GPUS \
#     --master_port=29500 \
#     $SCRIPT