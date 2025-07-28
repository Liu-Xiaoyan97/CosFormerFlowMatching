from flow_matching.loss import MixturePathGeneralizedKL
from evalution.logic.flow import get_path, get_loss_function, get_source_distribution
from flow_matching.path import ProbPath
from evalution.logic.generate import generate_samples
# from logic.evalute import compute_perplexity, compute_entropy
from omegaconf import OmegaConf
import torch
from transformers import AutoTokenizer
from CosFormer.cosformer import LLFMCosformerForFlowMatching

def eval():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("Tokenizer_32768_v1")
    vocab_size = tokenizer.vocab_size
    flow_cfg = OmegaConf.load("config/trainingargs.yml").trainer_args
    path = get_path(scheduler_type=flow_cfg.scheduler_type, exponent=flow_cfg.exponent)
    loss_fn = get_loss_function(loss_function=flow_cfg.loss_function, path=path)
    time_epsilon = 1e-3 if isinstance(loss_fn, MixturePathGeneralizedKL) else 0.0
    source_distribution = get_source_distribution(
        source_distribution=flow_cfg.source_distribution,
        vocab_size=vocab_size
    )
    model = LLFMCosformerForFlowMatching.from_pretrained(
        "llm_cosformer_results/checkpoint-100",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    batch_size = 2
    sequence_length = 1024
    sampling_steps = 4
    peplexity_n_samples = 64
    samples = []
    for _ in range(peplexity_n_samples // batch_size):
        samples.append(
            generate_samples(
                model = model,
                step=0,
                rank=0,
                vocab_size=vocab_size,
                tokenizer=tokenizer,
                device=device,
                path=path,
                sample_dir="llm_cosformer_results/checkpoint-100/samples",
                source_distribution=source_distribution,
                sample_batch_size=batch_size,
                sequence_length=sequence_length,
                sampling_steps=sampling_steps,
                time_epsilon=time_epsilon,
            )
        )
    samples = torch.cat(samples, dim=0)


eval()