import torch
import os
from transformers import AutoTokenizer
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
    
    # Load configuration
    config_path = "config/trainingargs.yml"
    if not os.path.exists(config_path):
        print(f"Warning: Config file {config_path} not found. Using default configuration.")
        # Create default config
        os.makedirs("config", exist_ok=True)
        with open(config_path, "w") as f:
            f.write("""# Flow Matching Training Configuration
trainer_args:
  scheduler_type: "polynomial"
  exponent: 2.0
  loss_function: "generalized_kl"
  source_distribution: "mask"
  time_epsilon: 1e-3

training_args:
  output_dir: "./llm_cosformer_results"
  overwrite_output_dir: true
  num_train_epochs: 3
  per_device_train_batch_size: 2
  per_device_eval_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 5e-5
  weight_decay: 0.01
  warmup_steps: 1000
  evaluation_strategy: "steps"
  eval_steps: 1000
  save_steps: 1000
  save_total_limit: 3
  logging_steps: 100
  fp16: false
  gradient_checkpointing: false
  remove_unused_columns: false

model_config:
  hidden_size: 512
  intermediate_size: 1024
  num_hidden_layers: 12
  num_attention_heads: 8
  max_position_embeddings: 4096
  vocab_size: 32768

dataset_config:
  tokenizer_path: "Tokenizer_32768_v1"
  dataset_name: "stanfordnlp/imdb"
  chunk_size: 512
""")
    
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
    
    # Setup model configuration
    model_config_dict = OmegaConf.to_container(full_config.get('model_config', {}), resolve=True)

    # Create model config with proper vocab size
    config = LLMFCosformerConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=model_config_dict.get('hidden_size', 512),
        intermediate_size=model_config_dict.get('intermediate_size', 1024),
        num_hidden_layers=model_config_dict.get('num_hidden_layers', 12),
        num_attention_heads=model_config_dict.get('num_attention_heads', 8),
        num_key_value_heads=model_config_dict.get('num_key_value_heads', 8),
        max_position_embeddings=model_config_dict.get('max_position_embeddings', 4096),
        flow_matching=model_config_dict.get('flow_matching', {
            "timestep_emb_dim": 256,
            "cond_dim": model_config_dict.get('hidden_size', 512),
            "n_blocks": 6,
            "n_heads": model_config_dict.get('num_attention_heads', 8),
            "mlp_ratio": 4,
            "dropout": 0.1
        })
    )
    
    print("Model configuration:")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num layers: {config.num_hidden_layers}")
    print(f"  Num heads: {config.num_attention_heads}")
    print(f"  Flow matching blocks: {config.flow_matching['n_blocks']}")
    
    # Initialize model
    print("Initializing model...")
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
    
    print(f"Loading datasets from {dataset_name}...")
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
        
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Eval dataset size: {len(eval_dataset)}")
        
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return
    
    # Setup training arguments
    print("Setting up training arguments...")
    training_args = load_training_args_from_yaml(config_path)
    
    # Force disable gradient checkpointing in training args
    training_args.gradient_checkpointing = False
    training_args.gradient_checkpointing_kwargs = None
    
    print("Training configuration:")
    print(f"  Output directory: {training_args.output_dir}")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size per device: {training_args.per_device_train_batch_size}")
    print(f"  Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Mixed precision: {training_args.fp16}")
    print(f"  Gradient checkpointing: {training_args.gradient_checkpointing}")
    
    # Create trainer
    print("Creating trainer...")
    
    # Create detailed callback with generation testing
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
        use_wandb=False  # Set to True if you want to use Weights & Biases
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
            print(f"Found checkpoint: {checkpoint_path}")
            
            resume = input("Do you want to resume from this checkpoint? (y/n): ")
            if resume.lower() == 'y':
                print(f"Resuming from {checkpoint_path}")
                trainer.train(resume_from_checkpoint=checkpoint_path)
            else:
                print("Starting fresh training...")
                trainer.train()
        else:
            print("No checkpoints found. Starting fresh training...")
            trainer.train()
    else:
        print("Starting fresh training...")
        trainer.train()
    
    # Save final model
    print("Saving final model...")
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
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Evaluation failed: {e}")
    
    print("\nTraining script completed successfully!")


if __name__ == "__main__":
    main()