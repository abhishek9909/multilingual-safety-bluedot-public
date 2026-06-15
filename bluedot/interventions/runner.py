"""
Intervention runner. An Intervention compiles itself into hooks; the runner
attaches the hooks and runs `model.generate`.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Callable

import torch
from transformers import GenerationConfig
from tqdm import tqdm

from ..models.base import ModelWrapper
from ..hooks.base import add_hooks


class Intervention(ABC):
    """An intervention turns itself into (pre_hooks, fwd_hooks)."""

    @abstractmethod
    def hooks(self, model: ModelWrapper) -> Tuple[List, List]:
        ...


def generate_with_intervention(
    model: ModelWrapper,
    prompts: List[str],
    intervention: Optional[Intervention] = None,
    *,
    batch_size: int = 8,
    max_new_tokens: int = 128,
    system: Optional[str] = None,
    do_sample: bool = False,
    temperature: float = 1.0,
) -> List[dict]:
    """Generate completions, optionally with hooks installed."""
    pre_hooks, fwd_hooks = ([], [])
    if intervention is not None:
        pre_hooks, fwd_hooks = intervention.hooks(model)

    # Build a clean generation config:
    #  - eos_token_id is a LIST of every turn-terminator. Critical for Llama3,
    #    where tokenizer.eos_token is <|end_of_text|> but the instruct turn
    #    ends with <|eot_id|>. Without this the model never stops and loops.
    #  - When do_sample=False, don't pass sampling-only kwargs.
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=model.tokenizer.pad_token_id,
        eos_token_id=model.stop_token_ids(),
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
    gen_cfg = GenerationConfig(**gen_kwargs)
    # Silence the spurious "temperature, top_p may be ignored" warning that
    # comes from the model's shipped generation_config.json defaults.
    model.model.generation_config.temperature = None
    model.model.generation_config.top_p = None

    out = []
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="generate"):
            batch = prompts[i:i + batch_size]
            inputs = model.tokenize(batch, system=system)
            with add_hooks(pre_hooks=pre_hooks, fwd_hooks=fwd_hooks):
                toks = model.model.generate(
                    input_ids=inputs.input_ids.to(model.device),
                    attention_mask=inputs.attention_mask.to(model.device),
                    generation_config=gen_cfg,
                )
            gen = toks[:, inputs.input_ids.shape[-1]:]
            for p, g in zip(batch, gen):
                out.append({
                    "prompt": p,
                    "response": model.tokenizer.decode(g, skip_special_tokens=True).strip(),
                })
    return out
