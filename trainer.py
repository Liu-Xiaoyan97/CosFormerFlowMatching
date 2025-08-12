import torch
import os
from transformers import AutoTokenizer, set_seed
from CosFormer.configuration_LLMFCosformer import LLMFCosformerConfig
from CosFormer.cosformer import LLFMCosformerForFlowMatching
from utils import (
    load_training_args_from_yaml,
    MyDataCollator,
    MyDataset,
    MyTrainer,
    DetailedProgressCallback,
    FlowMatchingUtils
)
from omegaconf import OmegaConf

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    """Main training function"""
    set_seed(42)
    # Load configuration
    config_path = "config/trainingargs.yml"
    if not os.path.exists(config_path):
        print(f"Error: Config file {config_path} not found.")
        return

    try:
        full_config = OmegaConf.load(config_path)
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    # Force disable gradient checkpointing in config
    if 'training_args' in full_config:
        full_config.training_args.gradient_checkpointing = False

    # Validate flow matching configuration
    if 'trainer_args' in full_config:
        if not FlowMatchingUtils.validate_config(dict(full_config.trainer_args)):
            print("Invalid flow matching configuration. Please check the config file.")
            return

    # Setup tokenizer
    dataset_config = full_config.get('dataset_config', {})
    tokenizer_path = dataset_config.get('tokenizer_path', 'Tokenizer_32768_v1')

    print(f"Loading tokenizer from {tokenizer_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Please ensure the tokenizer path is correct.")
        return

    # Create model config - it will automatically load from yml
    config = LLMFCosformerConfig()

    print("Model configuration (loaded from trainingargs.yml):")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Intermediate size: {config.intermediate_size}")
    print(f"  Num hidden layers (flow blocks): {config.num_hidden_layers}")
    print(f"  Num attention heads: {config.num_attention_heads}")
    print(f"  Num key-value heads: {config.num_key_value_heads}")
    print(f"  Max position embeddings: {config.max_position_embeddings}")
    print(f"  Attention dropout: {config.attention_dropout}")
    print(f"Flow matching configuration:")
    print(f"  Timestep embedding dim: {config.flow_matching['timestep_emb_dim']}")
    print(f"  Conditioning dim: {config.flow_matching['cond_dim']}")
    print(f"  Number of blocks: {config.flow_matching['n_blocks']}")
    print(f"  Number of heads: {config.flow_matching['n_heads']}")
    print(f"  MLP ratio: {config.flow_matching['mlp_ratio']}")
    print(f"  Dropout: {config.flow_matching['dropout']}")

    # Initialize model
    print("\nInitializing model...")
    masked = full_config.get('trainer_args', {}).get('source_distribution', 'mask') == 'mask'
    model = LLFMCosformerForFlowMatching(config, masked=masked)

    # Explicitly disable gradient checkpointing on the model
    model.gradient_checkpointing = False
    # Also set the attribute that some trainers check
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()

    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Setup data collator
    data_collator = MyDataCollator(pad_token_id=tokenizer.pad_token_id)

    # Setup datasets
    chunk_size = dataset_config.get('chunk_size', 512)
    dataset_name = dataset_config.get('dataset_name', 'stanfordnlp/imdb')
    max_train_samples = dataset_config.get('max_train_samples', None)
    max_eval_samples = dataset_config.get('max_eval_samples', 1000)

    print(f"\nLoading datasets from {dataset_name}...")
    print(f"  Chunk size: {chunk_size}")
    print(f"  Max train samples: {max_train_samples if max_train_samples else 'All'}")
    print(f"  Max eval samples: {max_eval_samples if max_eval_samples else 'All'}")

    try:
        train_dataset = MyDataset(
            tokenizer_path=tokenizer_path,
            dataset_name=dataset_name,
            split="train",
            chunk_size=chunk_size,
            max_samples=max_train_samples
        )

        eval_dataset = MyDataset(
            tokenizer_path=tokenizer_path,
            dataset_name=dataset_name,
            split="test",
            chunk_size=chunk_size,
            max_samples=max_eval_samples
        )

        print(f"Datasets loaded successfully")

    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    # Setup training arguments
    print("\nSetting up training arguments...")
    training_args = load_training_args_from_yaml(config_path)

    # Force disable gradient checkpointing in training args
    training_args.gradient_checkpointing = False
    training_args.gradient_checkpointing_kwargs = None

    print("Training configuration:")
    print(f"  Output directory: {training_args.output_dir}")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size per device: {training_args.per_device_train_batch_size}")
    print(f"  Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    print(
        f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Warmup steps: {training_args.warmup_steps}")
    print(f"  Weight decay: {training_args.weight_decay}")
    print(f"  Mixed precision (fp16): {training_args.fp16}")
    print(f"  Gradient checkpointing: {training_args.gradient_checkpointing}")

    # Create trainer
    print("\nCreating trainer...")

    # Create detailed callback with generation testing, TensorBoard, and system monitoring
    detailed_callback = DetailedProgressCallback(
        tokenizer=tokenizer,
        test_prefixes=[
            "The movie was",
            "I really enjoyed",
            "This film is",
            "The best part",
            "I would recommend"
        ],
        generation_config={
            'max_new_tokens': 30,
            'num_steps': 25,
            'temperature': 0.8,
            'top_k': 50,
            'top_p': 0.9
        },
        log_frequency=10,
        use_tensorboard=True,  # Enable TensorBoard logging
        tensorboard_dir="./logs",  # TensorBoard log directory
        monitor_system=True  # Enable system resource monitoring
    )

    trainer = MyTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[detailed_callback],
    )

    # Override any gradient checkpointing settings in the trainer
    trainer.args.gradient_checkpointing = False

    # Check for existing checkpoints
    if os.path.exists(training_args.output_dir):
        checkpoints = [d for d in os.listdir(training_args.output_dir) if d.startswith('checkpoint-')]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
            checkpoint_path = os.path.join(training_args.output_dir, latest_checkpoint)
            print(f"\nFound checkpoint: {checkpoint_path}")

            resume = input("Do you want to resume from this checkpoint? (y/n): ")
            if resume.lower() == 'y':
                print(f"Resuming from {checkpoint_path}")
                trainer.train(resume_from_checkpoint=checkpoint_path)
            else:
                print("Starting fresh training...")
                trainer.train()
        else:
            print("\nNo checkpoints found. Starting fresh training...")
            trainer.train()
    else:
        print("\nStarting fresh training...")
        trainer.train()

    # Save final model
    print("\nSaving final model...")
    trainer.save_model()

    # Save tokenizer with the model
    tokenizer.save_pretrained(training_args.output_dir)

    print("Training completed!")
    print(f"Model saved to: {training_args.output_dir}")

    # Run a quick evaluation
    print("\nRunning final evaluation...")
    try:
        eval_results = trainer.evaluate()
        print("Evaluation results:")
        for key, value in eval_results.items():
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    except Exception as e:
        print(f"Evaluation failed: {e}")

    print("\nTraining script completed successfully!")


if __name__ == "__main__":
    main()