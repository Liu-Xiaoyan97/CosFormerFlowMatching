"""
Flow Matching Generation Module with Polynomial Path
Implements generation/sampling for the trained LLFMCosformerForFlowMatching model
using the same polynomial path as during training
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Union, Tuple, Dict, Any
import numpy as np
from transformers import AutoTokenizer
from CosFormer.configuration_LLMFCosformer import LLMFCosformerConfig
from CosFormer.cosformer import LLFMCosformerForFlowMatching
from tqdm import tqdm
import os
from omegaconf import OmegaConf
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from evalution.logic.flow import get_path, get_source_distribution


class FlowMatchingGenerator:
    """
    Generator class for Flow Matching Language Models using Polynomial Path
    Matches the training configuration for consistent generation
    """
    
    def __init__(
        self,
        model: LLFMCosformerForFlowMatching,
        tokenizer: AutoTokenizer,
        config_path: str = "config/trainingargs.yml",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the generator with polynomial path matching training
        
        Args:
            model: Trained flow matching model
            tokenizer: Tokenizer for encoding/decoding text
            config_path: Path to training configuration file
            device: Device to run generation on
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.vocab_size = model.vocab_size
        
        # Set model to eval mode
        self.model.eval()
        
        # Load flow matching configuration to match training
        try:
            flow_cfg = OmegaConf.load(config_path).trainer_args
        except:
            # Default configuration if file doesn't exist
            flow_cfg = OmegaConf.create({
                'scheduler_type': 'polynomial',
                'exponent': 2.0,
                'source_distribution': 'mask',
                'time_epsilon': 1e-3
            })
        
        self.flow_cfg = flow_cfg
        
        # Initialize the same path as used in training
        self.path = get_path(
            scheduler_type=flow_cfg.scheduler_type,
            exponent=flow_cfg.get('exponent', 2.0)
        )
        
        # Initialize source distribution
        self.source_distribution = get_source_distribution(
            source_distribution=flow_cfg.source_distribution,
            vocab_size=self.vocab_size
        )
        
        # Get mask token if model uses masking
        self.mask_token_id = self.vocab_size if model.masked else None
        
        print(f"Generator initialized with {flow_cfg.scheduler_type} path (exponent={flow_cfg.get('exponent', 2.0)})")
    
    def sample_prior(self, shape: Tuple[int, int]) -> torch.LongTensor:
        """
        Sample from the prior distribution (source distribution)
        Uses the same distribution as training
        
        Args:
            shape: (batch_size, seq_len) shape for sampling
            
        Returns:
            Sampled tokens from prior distribution
        """
        # Create a dummy tensor for shape
        dummy = torch.zeros(shape, dtype=torch.long, device=self.device)
        # Use the source distribution's sample_like method
        return self.source_distribution.sample_like(dummy)
    
    def compute_velocity(
        self,
        x_t: torch.LongTensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the velocity field v(x_t, t) using the trained model
        
        Args:
            x_t: Current tokens at time t
            t: Current time step(s)
            attention_mask: Attention mask for the sequence
            
        Returns:
            Velocity field as probability distributions
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=x_t,
                timesteps=t,
                attention_mask=attention_mask,
                return_dict=True
            )
            logits = outputs.logits  # (batch_size, seq_len, vocab_size)
        
        return logits
    
    def polynomial_sample_step(
        self,
        x_t: torch.LongTensor,
        x_0: torch.LongTensor,
        t_start: float,
        t_end: float,
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.LongTensor:
        """
        Perform one polynomial path sampling step
        This matches the polynomial convex scheduler used in training
        
        Args:
            x_t: Current tokens at time t_start
            x_0: Source tokens (prior samples)
            t_start: Starting time step
            t_end: Ending time step
            attention_mask: Attention mask for the sequence
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            
        Returns:
            Updated tokens at time t_end
        """
        batch_size, seq_len = x_t.shape
        
        # Create time tensors
        t_start_tensor = torch.full((batch_size,), t_start, dtype=torch.float32, device=self.device)
        
        # Get model predictions (velocity field)
        logits = self.compute_velocity(x_t, t_start_tensor, attention_mask)
        
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Apply filtering techniques
        filtered_probs = self._apply_filtering(probs, top_k, top_p)
        
        # Sample predicted target tokens x_1
        x_1_pred = self._sample_from_probs(filtered_probs, t_start)
        
        # For polynomial path: x_t = (1 - sigma_t) * x_0 + sigma_t * x_1
        # where sigma_t = t^exponent
        exponent = self.flow_cfg.get('exponent', 2.0)
        
        # During generation, we go from t=1 (noise) to t=0 (clean)
        # We need to compute how much to transition from current state toward predicted clean state
        
        if t_start <= 0.001:  # Very close to the end
            # Just return the predicted tokens
            return x_1_pred
        
        # Compute sigma values for polynomial schedule
        sigma_start = t_start ** exponent
        sigma_end = max(0.0, t_end ** exponent)  # Ensure non-negative
        
        # For generation, we're denoising, so we move from noisy (x_0) toward clean (x_1)
        # The amount of transition depends on the change in sigma
        if t_start >= 0.999:  # At the very beginning
            # Start with mostly noise, small transition toward predicted
            transition_prob = 1.0 - sigma_end  # How much to move toward x_1
        elif sigma_start > 1e-6:  # Normal case
            # Calculate how much we should transition
            # As t decreases, we should move more toward x_1
            transition_prob = (sigma_start - sigma_end) / sigma_start
            transition_prob = min(1.0, max(0.0, transition_prob))  # Clamp to [0, 1]
        else:
            # Near the end, mostly use predictions
            transition_prob = 0.9
        
        # Apply stochastic transition
        # For discrete tokens, we use probabilistic replacement
        transition_mask = torch.rand_like(x_t, dtype=torch.float32) < transition_prob
        
        # Update tokens: replace some tokens with predicted clean tokens
        x_next = torch.where(transition_mask, x_1_pred, x_t)
        
        return x_next
    
    def _apply_filtering(
        self,
        probs: torch.Tensor,
        top_k: Optional[int],
        top_p: Optional[float]
    ) -> torch.Tensor:
        """
        Apply top-k and top-p filtering to probability distributions
        
        Args:
            probs: Probability distributions
            top_k: Top-k parameter
            top_p: Top-p parameter
            
        Returns:
            Filtered probabilities
        """
        # Apply top-k filtering
        if top_k is not None and top_k > 0:
            indices_to_remove = probs < torch.topk(probs, top_k, dim=-1)[0][..., -1, None]
            probs = probs.clone()
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
            probs = probs.clone()
            probs[indices_to_remove] = 0
            probs = probs / probs.sum(dim=-1, keepdim=True)
        
        return probs
    
    def _sample_from_probs(self, probs: torch.Tensor, t: float) -> torch.LongTensor:
        """
        Sample tokens from probability distributions
        
        Args:
            probs: Probability distributions
            t: Current time step (used to determine sampling strategy)
            
        Returns:
            Sampled tokens
        """
        batch_size, seq_len, vocab_size = probs.shape
        
        # Near the end of generation (t close to 0), use more deterministic sampling
        if t < 0.1:
            # Use argmax for final steps
            return torch.argmax(probs, dim=-1)
        elif t < 0.3:
            # Use top-1 from multinomial sampling for stability
            probs_reshaped = probs.view(-1, vocab_size)
            # Add small epsilon for numerical stability
            probs_reshaped = probs_reshaped + 1e-10
            probs_reshaped = probs_reshaped / probs_reshaped.sum(dim=-1, keepdim=True)
            samples = torch.multinomial(probs_reshaped, 1).view(batch_size, seq_len)
            return samples
        else:
            # Full stochastic sampling for early steps
            probs_reshaped = probs.view(-1, vocab_size)
            probs_reshaped = probs_reshaped + 1e-10
            probs_reshaped = probs_reshaped / probs_reshaped.sum(dim=-1, keepdim=True)
            samples = torch.multinomial(probs_reshaped, 1).view(batch_size, seq_len)
            return samples
    
    @torch.no_grad()
    def generate(
        self,
        prefix: Union[str, List[str]],
        max_new_tokens: int = 100,
        num_steps: int = 50,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        adaptive_steps: bool = True,
        return_intermediate: bool = False,
        progress_bar: bool = True,
    ) -> Union[str, List[str], Tuple[List[str], List[torch.LongTensor]]]:
        """
        Generate text continuation given a prefix using polynomial path
        
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
            max_length=512,
        ).to(self.device)
        
        prefix_ids = prefix_encoding["input_ids"]
        prefix_attention_mask = prefix_encoding["attention_mask"]
        prefix_len = prefix_ids.shape[1]
        
        # Calculate total sequence length
        total_len = min(prefix_len + max_new_tokens, self.model.config.max_position_embeddings)
        new_token_len = total_len - prefix_len
        
        # Initialize with samples from source distribution
        x_0_continuation = self.sample_prior((batch_size, new_token_len))
        
        # Full sequence: prefix + random initialization
        x_0_full = torch.cat([prefix_ids, x_0_continuation], dim=1)
        x_t = x_0_full.clone()
        
        # Create attention mask for full sequence
        continuation_mask = torch.ones((batch_size, new_token_len), dtype=torch.long, device=self.device)
        full_attention_mask = torch.cat([prefix_attention_mask, continuation_mask], dim=1)
        
        # Create a mask to keep prefix fixed during generation
        prefix_mask = torch.zeros_like(x_t, dtype=torch.bool)
        prefix_mask[:, :prefix_len] = True
        
        # Store intermediate states if requested
        intermediate_states = [] if return_intermediate else None
        
        # Calculate time steps for polynomial path
        if adaptive_steps:
            # Use polynomial spacing that matches the path
            exponent = self.flow_cfg.get('exponent', 2.0)
            # For generation, we go from t=1 (noise) to t=0 (clean)
            # Create more steps near t=0 where the most important denoising happens
            ts = np.linspace(0, 1, num_steps + 1)
            # Apply power transformation for better step distribution
            ts = 1.0 - (1.0 - ts) ** (1.0 / exponent)
            # Ensure we start at 1 and end at 0
            ts[0] = 1.0
            ts[-1] = 0.0
            # Reverse to go from 1 to 0
            ts = ts[::-1]
        else:
            # Uniform time steps from 1 to 0
            ts = np.linspace(1.0, 0.0, num_steps + 1)
        
        # Generation loop following polynomial path
        iterator = enumerate(zip(ts[:-1], ts[1:]))
        if progress_bar:
            iterator = tqdm(iterator, total=num_steps, desc="Generating")
        
        for step_idx, (t_start, t_end) in iterator:
            # Perform one polynomial path sampling step
            x_t_new = self.polynomial_sample_step(
                x_t=x_t,
                x_0=x_0_full,  # Pass the source samples
                t_start=t_start,
                t_end=t_end,
                attention_mask=full_attention_mask,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            
            # Keep prefix fixed - ensure dimensions match
            # x_t_new should have the same shape as x_t
            # Only update the continuation part, keep prefix unchanged
            x_t = x_t.clone()
            x_t[:, prefix_len:] = x_t_new[:, prefix_len:]
            
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
    
    def generate_with_custom_path(
        self,
        prefix: Union[str, List[str]],
        scheduler_type: str = "polynomial",
        exponent: float = 2.0,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate with a custom path configuration
        
        Args:
            prefix: Input text or list of texts
            scheduler_type: Type of scheduler to use
            exponent: Exponent for polynomial scheduler
            **kwargs: Additional arguments for generate()
            
        Returns:
            Generated text(s)
        """
        # Temporarily override path configuration
        original_path = self.path
        original_exponent = self.flow_cfg.get('exponent', 2.0)
        
        try:
            # Create new path with custom settings
            self.path = get_path(scheduler_type=scheduler_type, exponent=exponent)
            self.flow_cfg['exponent'] = exponent
            
            # Generate with custom path
            result = self.generate(prefix, **kwargs)
            
        finally:
            # Restore original path
            self.path = original_path
            self.flow_cfg['exponent'] = original_exponent
        
        return result


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
    
    # Load config
    config = LLMFCosformerConfig.from_pretrained(checkpoint_path)
    
    # Check if model was trained with masking
    config_path = "config/trainingargs.yml"
    masked = True
    if os.path.exists(config_path):
        try:
            flow_cfg = OmegaConf.load(config_path).trainer_args
            masked = flow_cfg.get('source_distribution', 'mask') == 'mask'
        except:
            pass
    
    # Load model
    model = LLFMCosformerForFlowMatching.from_pretrained(
        checkpoint_path,
        config=config,
        masked=masked,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on {device}")
    return model, tokenizer


# ============== Test Examples ==============

def test_basic_generation():
    """Test basic text generation with polynomial path"""
    
    checkpoint_path = "./llm_cosformer_results"
    
    if not os.path.exists(checkpoint_path):
        print(f"Model checkpoint not found at {checkpoint_path}")
        print("Please train the model first using trainer.py")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(checkpoint_path, device)
    
    # Create generator with polynomial path
    generator = FlowMatchingGenerator(model, tokenizer, device=device)
    
    test_prefixes = [
        "The movie was",
        "I really enjoyed",
        "The acting in this film",
        "Overall, I would rate",
        "The plot twist at the end",
    ]
    
    print("\n" + "="*50)
    print("Polynomial Path Generation Test")
    print("="*50)
    
    for prefix in test_prefixes:
        print(f"\nPrefix: '{prefix}'")
        
        # Generate with polynomial path (matching training)
        generated = generator.generate(
            prefix=prefix,
            max_new_tokens=50,
            num_steps=30,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            adaptive_steps=True,  # Use adaptive steps for polynomial path
            progress_bar=False
        )
        print(f"Generated: {generated}")


def test_different_exponents():
    """Test generation with different polynomial exponents"""
    
    checkpoint_path = "./llm_cosformer_results"
    
    if not os.path.exists(checkpoint_path):
        print(f"Model checkpoint not found at {checkpoint_path}")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(checkpoint_path, device)
    generator = FlowMatchingGenerator(model, tokenizer, device=device)
    
    print("\n" + "="*50)
    print("Different Polynomial Exponents Test")
    print("="*50)
    
    prefix = "This movie deserves"
    exponents = [1.0, 2.0, 3.0, 4.0]
    
    print(f"\nPrefix: '{prefix}'")
    
    for exp in exponents:
        generated = generator.generate_with_custom_path(
            prefix=prefix,
            scheduler_type="polynomial",
            exponent=exp,
            max_new_tokens=40,
            num_steps=30,
            temperature=0.8,
            progress_bar=False
        )
        print(f"\nExponent {exp}: {generated}")


def test_batch_generation():
    """Test batch generation with polynomial path"""
    
    checkpoint_path = "./llm_cosformer_results"
    
    if not os.path.exists(checkpoint_path):
        print(f"Model checkpoint not found at {checkpoint_path}")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(checkpoint_path, device)
    generator = FlowMatchingGenerator(model, tokenizer, device=device)
    
    print("\n" + "="*50)
    print("Batch Generation with Polynomial Path")
    print("="*50)
    
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
        adaptive_steps=True,
        progress_bar=True
    )
    
    for prefix, generated in zip(prefixes, generated_texts):
        print(f"\nPrefix: '{prefix}'")
        print(f"Generated: {generated}")


def test_intermediate_states():
    """Test generation with intermediate states using polynomial path"""
    
    checkpoint_path = "./llm_cosformer_results"
    
    if not os.path.exists(checkpoint_path):
        print(f"Model checkpoint not found at {checkpoint_path}")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(checkpoint_path, device)
    generator = FlowMatchingGenerator(model, tokenizer, device=device)
    
    print("\n" + "="*50)
    print("Polynomial Path Evolution Test")
    print("="*50)
    
    prefix = "The best part about this movie"
    
    final_text, intermediate_states = generator.generate(
        prefix=prefix,
        max_new_tokens=30,
        num_steps=10,
        temperature=0.8,
        adaptive_steps=True,
        return_intermediate=True,
        progress_bar=True
    )
    
    print(f"\nPrefix: '{prefix}'")
    print(f"Final generation: {final_text}")
    
    print("\nEvolution of generation (polynomial path):")
    steps_to_show = [0, len(intermediate_states)//3, 2*len(intermediate_states)//3, -1]
    
    for i, step_idx in enumerate(steps_to_show):
        state = intermediate_states[step_idx]
        prefix_len = len(tokenizer.encode(prefix))
        generated_ids = state[0, prefix_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        progress = (step_idx if step_idx >= 0 else len(intermediate_states)-1) / len(intermediate_states) * 100
        print(f"Step {step_idx} ({progress:.0f}%): {prefix}{generated_text}")


def test_adaptive_vs_uniform_steps():
    """Compare adaptive vs uniform time steps in polynomial path"""
    
    checkpoint_path = "./llm_cosformer_results"
    
    if not os.path.exists(checkpoint_path):
        print(f"Model checkpoint not found at {checkpoint_path}")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(checkpoint_path, device)
    generator = FlowMatchingGenerator(model, tokenizer, device=device)
    
    print("\n" + "="*50)
    print("Adaptive vs Uniform Steps Test")
    print("="*50)
    
    prefix = "In my opinion, this film"
    
    print(f"\nPrefix: '{prefix}'")
    
    # Generate with adaptive steps
    generated_adaptive = generator.generate(
        prefix=prefix,
        max_new_tokens=50,
        num_steps=30,
        temperature=0.8,
        adaptive_steps=True,
        progress_bar=False
    )
    print(f"\nAdaptive steps: {generated_adaptive}")
    
    # Generate with uniform steps
    generated_uniform = generator.generate(
        prefix=prefix,
        max_new_tokens=50,
        num_steps=30,
        temperature=0.8,
        adaptive_steps=False,
        progress_bar=False
    )
    print(f"\nUniform steps: {generated_uniform}")


def run_all_tests():
    """Run all test examples"""
    
    print("\n" + "="*60)
    print("POLYNOMIAL PATH FLOW MATCHING GENERATION TESTS")
    print("="*60)
    
    test_basic_generation()
    test_different_exponents()
    test_batch_generation()
    test_intermediate_states()
    test_adaptive_vs_uniform_steps()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    # Run all tests
    run_all_tests()
    
    # Interactive generation
    print("\n" + "="*60)
    print("INTERACTIVE GENERATION WITH POLYNOMIAL PATH")
    print("="*60)
    
    checkpoint_path = "./llm_cosformer_results"
    if os.path.exists(checkpoint_path):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, tokenizer = load_model_and_tokenizer(checkpoint_path, device)
        generator = FlowMatchingGenerator(model, tokenizer, device=device)
        
        print("\nUsing polynomial path matching your training configuration")
        print("Commands: 'quit' to exit, 'config' to show current settings")
        
        while True:
            user_input = input("\nEnter a prefix (or command): ")
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'config':
                print(f"Current configuration:")
                print(f"  Scheduler: {generator.flow_cfg.scheduler_type}")
                print(f"  Exponent: {generator.flow_cfg.get('exponent', 2.0)}")
                print(f"  Source distribution: {generator.flow_cfg.source_distribution}")
                continue
            
            generated = generator.generate(
                prefix=user_input,
                max_new_tokens=100,
                num_steps=50,
                temperature=0.8,
                top_k=50,
                top_p=0.9,
                adaptive_steps=True
            )
            
            print(f"\nGenerated: {generated}")
    else:
        print(f"Please train the model first. Checkpoint not found at {checkpoint_path}")