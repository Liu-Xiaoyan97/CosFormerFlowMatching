"""
Flow Matching Generation Module
Implements generation/sampling for the trained LLFMCosformerForFlowMatching model
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Union, Tuple
import numpy as np
from transformers import AutoTokenizer
from CosFormer.configuration_LLMFCosformer import LLMFCosformerConfig
from CosFormer.cosformer import LLFMCosformerForFlowMatching
from tqdm import tqdm
import os


class FlowMatchingGenerator:
    """
    Generator class for Flow Matching Language Models
    Supports various sampling strategies including:
    - Euler sampling
    - Adaptive step size sampling
    - Temperature-based sampling
    """
    
    def __init__(
        self,
        model: LLFMCosformerForFlowMatching,
        tokenizer: AutoTokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the generator
        
        Args:
            model: Trained flow matching model
            tokenizer: Tokenizer for encoding/decoding text
            device: Device to run generation on
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.vocab_size = model.vocab_size
        
        # Set model to eval mode
        self.model.eval()
        
        # Get mask token if model uses masking
        self.mask_token_id = self.vocab_size if model.masked else None
        
    def sample_prior(self, shape: Tuple[int, int]) -> torch.LongTensor:
        """
        Sample from the prior distribution (source distribution)
        
        Args:
            shape: (batch_size, seq_len) shape for sampling
            
        Returns:
            Sampled tokens from prior distribution
        """
        if self.mask_token_id is not None:
            # Use mask tokens as prior
            return torch.full(shape, self.mask_token_id, dtype=torch.long, device=self.device)
        else:
            # Use uniform distribution as prior
            return torch.randint(0, self.vocab_size, shape, dtype=torch.long, device=self.device)
    
    # 这个地方需要修改，不能使用欧拉采样步骤，需要使用多项式采样，参考Trainer中的代码
    def euler_sample_step(
        self,
        x_t: torch.LongTensor,
        t: float,
        dt: float,
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.LongTensor:
        """
        Perform one Euler sampling step
        
        Args:
            x_t: Current tokens at time t
            t: Current time step (between 0 and 1)
            dt: Time step size
            attention_mask: Attention mask for the sequence
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            
        Returns:
            Updated tokens at time t + dt
        """
        batch_size, seq_len = x_t.shape
        
        # Create timesteps tensor
        timesteps = torch.full((batch_size,), t, dtype=torch.float32, device=self.device)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(
                input_ids=x_t,
                timesteps=timesteps,
                attention_mask=attention_mask,
                return_dict=True
            )
            logits = outputs.logits  # (batch_size, seq_len, vocab_size)
        
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Apply top-k filtering
        if top_k is not None and top_k > 0:
            indices_to_remove = probs < torch.topk(probs, top_k, dim=-1)[0][..., -1, None]
            probs[indices_to_remove] = 0
            probs = probs / probs.sum(dim=-1, keepdim=True)
        
        # Apply top-p (nucleus) filtering
        if top_p is not None and top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            probs[indices_to_remove] = 0
            probs = probs / probs.sum(dim=-1, keepdim=True)
        
        # Sample from the distribution
        if t < 0.1:  # Near the end, use argmax for stability
            x_next = torch.argmax(probs, dim=-1)
        else:
            x_next = torch.multinomial(probs.view(-1, self.vocab_size), 1).view(batch_size, seq_len)
        
        # Interpolate between current and next state based on dt
        # For discrete data, we use a probabilistic transition
        transition_prob = 1 - t + dt
        mask = torch.rand_like(x_t, dtype=torch.float32) < transition_prob
        x_t = torch.where(mask, x_next, x_t)
        
        return x_t
    
    @torch.no_grad()
    def generate(
        self,
        prefix: Union[str, List[str]],
        max_new_tokens: int = 100,
        num_steps: int = 50,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        adaptive_steps: bool = False,
        return_intermediate: bool = False,
        progress_bar: bool = True,
    ) -> Union[str, List[str], Tuple[List[str], List[torch.LongTensor]]]:
        """
        Generate text continuation given a prefix
        
        Args:
            prefix: Input text or list of texts to continue from
            max_new_tokens: Maximum number of new tokens to generate
            num_steps: Number of denoising steps in flow matching
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            adaptive_steps: Whether to use adaptive step sizes
            return_intermediate: Whether to return intermediate states
            progress_bar: Whether to show progress bar
            
        Returns:
            Generated text continuation(s), optionally with intermediate states
        """
        # Handle single string input
        if isinstance(prefix, str):
            prefix = [prefix]
            single_input = True
        else:
            single_input = False
        
        batch_size = len(prefix)
        
        # Tokenize prefix
        prefix_encoding = self.tokenizer(
            prefix,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,  # Adjust based on your model's max length
        ).to(self.device)
        
        prefix_ids = prefix_encoding["input_ids"]
        prefix_attention_mask = prefix_encoding["attention_mask"]
        prefix_len = prefix_ids.shape[1]
        
        # Calculate total sequence length
        total_len = min(prefix_len + max_new_tokens, self.model.config.max_position_embeddings)
        new_token_len = total_len - prefix_len
        
        # Initialize with random tokens for the continuation
        x_0 = self.sample_prior((batch_size, new_token_len))
        
        # Concatenate prefix with random initialization
        full_ids = torch.cat([prefix_ids, x_0], dim=1)
        
        # Create attention mask for full sequence
        continuation_mask = torch.ones((batch_size, new_token_len), dtype=torch.long, device=self.device)
        full_attention_mask = torch.cat([prefix_attention_mask, continuation_mask], dim=1)
        
        # Create a mask to keep prefix fixed during generation
        prefix_mask = torch.zeros_like(full_ids, dtype=torch.bool)
        prefix_mask[:, :prefix_len] = True
        
        # Store intermediate states if requested
        intermediate_states = [] if return_intermediate else None
        
        # Calculate step sizes
        if adaptive_steps:
            # Use smaller steps at the beginning and larger steps at the end
            ts = np.linspace(0, 1, num_steps + 1)
            ts = ts ** 2  # Quadratic spacing
            step_sizes = np.diff(ts)
        else:
            step_sizes = [1.0 / num_steps] * num_steps
        
        # Reverse time flow (from t=1 to t=0)
        current_t = 1.0
        x_t = full_ids.clone()
        
        # Generation loop
        iterator = tqdm(range(num_steps), desc="Generating", disable=not progress_bar)
        for step_idx in iterator:
            dt = step_sizes[step_idx]
            
            # Perform one sampling step
            x_t_new = self.euler_sample_step(
                x_t=x_t,
                t=current_t,
                dt=dt,
                attention_mask=full_attention_mask,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            
            # Keep prefix fixed
            x_t = torch.where(prefix_mask, full_ids, x_t_new)
            
            # Update time
            current_t -= dt
            current_t = max(current_t, 0.0)
            
            # Store intermediate state
            if return_intermediate:
                intermediate_states.append(x_t.clone())
        
        # Decode the generated sequences
        generated_texts = []
        for i in range(batch_size):
            # Get the generated part (excluding prefix)
            generated_ids = x_t[i, prefix_len:total_len]
            
            # Remove padding and special tokens if necessary
            if self.tokenizer.pad_token_id is not None:
                generated_ids = generated_ids[generated_ids != self.tokenizer.pad_token_id]
            
            # Decode to text
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Combine with prefix for full text
            full_text = prefix[i] + generated_text
            generated_texts.append(full_text)
        
        # Return based on input format and options
        if return_intermediate:
            if single_input:
                return generated_texts[0], intermediate_states
            return generated_texts, intermediate_states
        else:
            if single_input:
                return generated_texts[0]
            return generated_texts


def load_model_and_tokenizer(checkpoint_path: str, device: str = "cuda"):
    """
    Load a trained model and tokenizer from checkpoint
    
    Args:
        checkpoint_path: Path to model checkpoint directory
        device: Device to load model on
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model from {checkpoint_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = LLFMCosformerForFlowMatching.from_pretrained(
        checkpoint_path,
        masked=True,
        trust_remote_code=True
    )
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on {device}")
    return model, tokenizer


# ============== Test Examples ==============

def test_basic_generation():
    """Test basic text generation with different prefixes"""
    
    # Path to your trained model
    checkpoint_path = "./llm_cosformer_results"
    
    # Check if model exists
    if not os.path.exists(checkpoint_path):
        print(f"Model checkpoint not found at {checkpoint_path}")
        print("Please train the model first using trainer.py")
        return
    
    # Load model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(checkpoint_path, device)
    
    # Create generator
    generator = FlowMatchingGenerator(model, tokenizer, device)
    
    # Test prefixes
    test_prefixes = [
        "The movie was",
        "I really enjoyed",
        "The acting in this film",
        "Overall, I would rate",
        "The plot twist at the end",
    ]
    
    print("\n" + "="*50)
    print("Basic Generation Test")
    print("="*50)
    
    for prefix in test_prefixes:
        print(f"\nPrefix: '{prefix}'")
        
        # Generate with different settings
        # Conservative generation
        generated = generator.generate(
            prefix=prefix,
            max_new_tokens=50,
            num_steps=30,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            progress_bar=False
        )
        print(f"Conservative: {generated}")
        
        # Creative generation
        generated = generator.generate(
            prefix=prefix,
            max_new_tokens=50,
            num_steps=30,
            temperature=1.2,
            top_k=100,
            top_p=0.95,
            progress_bar=False
        )
        print(f"Creative: {generated}")


def test_batch_generation():
    """Test batch generation with multiple prefixes"""
    
    checkpoint_path = "llm_cosformer_results/checkpoint-1035"
    
    if not os.path.exists(checkpoint_path):
        print(f"Model checkpoint not found at {checkpoint_path}")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(checkpoint_path, device)
    generator = FlowMatchingGenerator(model, tokenizer, device)
    
    print("\n" + "="*50)
    print("Batch Generation Test")
    print("="*50)
    
    # Multiple prefixes at once
    prefixes = [
        "This movie is absolutely",
        "I was disappointed by",
        "The special effects were",
        "The soundtrack really",
    ]
    
    generated_texts = generator.generate(
        prefix=prefixes,
        max_new_tokens=40,
        num_steps=25,
        temperature=0.8,
        progress_bar=True
    )
    
    for prefix, generated in zip(prefixes, generated_texts):
        print(f"\nPrefix: '{prefix}'")
        print(f"Generated: {generated}")


def test_intermediate_states():
    """Test generation with intermediate states visualization"""
    
    checkpoint_path = "./llm_cosformer_results"
    
    if not os.path.exists(checkpoint_path):
        print(f"Model checkpoint not found at {checkpoint_path}")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(checkpoint_path, device)
    generator = FlowMatchingGenerator(model, tokenizer, device)
    
    print("\n" + "="*50)
    print("Intermediate States Test")
    print("="*50)
    
    prefix = "The best part about this movie"
    
    # Generate with intermediate states
    final_text, intermediate_states = generator.generate(
        prefix=prefix,
        max_new_tokens=30,
        num_steps=10,
        temperature=0.8,
        return_intermediate=True,
        progress_bar=True
    )
    
    print(f"\nPrefix: '{prefix}'")
    print(f"Final generation: {final_text}")
    
    # Show evolution at certain steps
    print("\nEvolution of generation:")
    steps_to_show = [0, len(intermediate_states)//3, 2*len(intermediate_states)//3, -1]
    
    for i, step_idx in enumerate(steps_to_show):
        state = intermediate_states[step_idx]
        # Decode just the generated part
        prefix_len = len(tokenizer.encode(prefix))
        generated_ids = state[0, prefix_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"Step {step_idx if step_idx >= 0 else len(intermediate_states)-1}: {prefix}{generated_text}")


def test_temperature_effects():
    """Test the effect of different temperature settings"""
    
    checkpoint_path = "./llm_cosformer_results"
    
    if not os.path.exists(checkpoint_path):
        print(f"Model checkpoint not found at {checkpoint_path}")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(checkpoint_path, device)
    generator = FlowMatchingGenerator(model, tokenizer, device)
    
    print("\n" + "="*50)
    print("Temperature Effects Test")
    print("="*50)
    
    prefix = "This film deserves"
    temperatures = [0.5, 0.8, 1.0, 1.5]
    
    print(f"\nPrefix: '{prefix}'")
    
    for temp in temperatures:
        generated = generator.generate(
            prefix=prefix,
            max_new_tokens=40,
            num_steps=30,
            temperature=temp,
            top_k=50,
            progress_bar=False
        )
        print(f"\nTemperature {temp}: {generated}")


def run_all_tests():
    """Run all test examples"""
    
    print("\n" + "="*60)
    print("FLOW MATCHING GENERATION TESTS")
    print("="*60)
    
    # Run all tests
    test_basic_generation()
    test_batch_generation()
    test_intermediate_states()
    test_temperature_effects()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    # Run all tests
    run_all_tests()
    
    # Or run interactive generation
    print("\n" + "="*60)
    print("INTERACTIVE GENERATION")
    print("="*60)
    
    checkpoint_path = "./llm_cosformer_results"
    if os.path.exists(checkpoint_path):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, tokenizer = load_model_and_tokenizer(checkpoint_path, device)
        generator = FlowMatchingGenerator(model, tokenizer, device)
        
        while True:
            prefix = input("\nEnter a prefix (or 'quit' to exit): ")
            if prefix.lower() == 'quit':
                break
            
            generated = generator.generate(
                prefix=prefix,
                max_new_tokens=100,
                num_steps=50,
                temperature=0.8,
                top_k=50,
                top_p=0.9
            )
            
            print(f"\nGenerated: {generated}")
    else:
        print(f"Please train the model first. Checkpoint not found at {checkpoint_path}")