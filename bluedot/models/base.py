"""
Thin wrapper around an HF causal LM + tokenizer + chat template.

Goal: every other module in bluedot consumes a `ModelWrapper`, not a raw model.
This is the only place that knows about HF specifics.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .templates import ChatTemplate


@dataclass
class ModelWrapper:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    template: ChatTemplate
    name: str = "model"

    @property
    def n_layers(self) -> int:
        return self.model.config.num_hidden_layers

    @property
    def d_model(self) -> int:
        return self.model.config.hidden_size

    @property
    def device(self):
        return self.model.device

    @property
    def block_modules(self) -> List[torch.nn.Module]:
        # llama/qwen/gemma all use `model.layers`
        return self.model.model.layers

    @property
    def attn_modules(self) -> List[torch.nn.Module]:
        return [b.self_attn for b in self.block_modules]

    @property
    def mlp_modules(self) -> List[torch.nn.Module]:
        return [b.mlp for b in self.block_modules]

    # Prefer the tokenizer's own chat template — it ships with the model and
    # already includes the right BOS / system preamble / role markers. We fall
    # back to our hand-rolled `self.template` strings only if the tokenizer
    # has no chat template.
    @property
    def use_hf_chat_template(self) -> bool:
        return getattr(self.tokenizer, "chat_template", None) is not None

    def format(self, instruction: str, system: Optional[str] = None) -> str:
        if self.use_hf_chat_template:
            msgs = []
            if system is not None:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": instruction})
            text = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            # Some chat_templates (notably several Llama3 uploads) omit
            # bos_token. With add_special_tokens=False the tokenizer can't
            # add it back, and Llama3-Instruct without <|begin_of_text|>
            # falls back to base-mode behavior (echo + loop). Prepend it
            # here if it's missing.
            bos = self.tokenizer.bos_token
            if bos and not text.startswith(bos):
                text = bos + text
            return text
        return self.template.format(instruction, system=system)

    def tokenize(self, instructions: List[str], system: Optional[str] = None,
                 padding: bool = True, return_tensors: str = "pt"):
        prompts = [self.format(i, system=system) for i in instructions]
        # add_special_tokens=False: the chat template already contains every
        # special token the model expects. With the default (True) HF would
        # prepend a second BOS for some tokenizers, breaking the instruct
        # format and causing the model to fall back to base-model continuation.
        return self.tokenizer(
            prompts, padding=padding, return_tensors=return_tensors,
            add_special_tokens=False,
        )

    def debug_prompt(self, instruction: str, system: Optional[str] = None) -> str:
        """Return the exact decoded string the model will see. Use this to
        sanity-check chat-template application when generations look wrong."""
        ids = self.tokenize([instruction], system=system).input_ids[0]
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def eoi_token_ids(self) -> List[int]:
        """Token IDs of the post-instruction string."""
        return self.tokenizer(self.template.eoi_str, add_special_tokens=False).input_ids

    @property
    def n_eoi_toks(self) -> int:
        return len(self.eoi_token_ids())

    def stop_token_ids(self) -> List[int]:
        """All token IDs that should terminate generation for this model.

        Includes the tokenizer's own eos_token_id plus every turn-terminator
        listed in the chat template (e.g. <|eot_id|> for Llama3). Pass this
        list as `eos_token_id` to model.generate().
        """
        ids = set()
        if self.tokenizer.eos_token_id is not None:
            ids.add(int(self.tokenizer.eos_token_id))
        for s in self.template.stop_token_strs:
            tid = self.tokenizer.convert_tokens_to_ids(s)
            # convert_tokens_to_ids returns the unk_id (often 0) for unknown
            # tokens — filter those out.
            if isinstance(tid, int) and tid != self.tokenizer.unk_token_id and tid >= 0:
                ids.add(int(tid))
        return sorted(ids)


def load_model(
    model_path: str,
    template_name: str,
    *,
    device_map: str = "auto",
    dtype: torch.dtype = torch.float16,
    trust_remote_code: bool = True,
    cache_dir: Optional[str] = None,
) -> ModelWrapper:
    """Load a HuggingFace causal LM + its tokenizer and bind a chat template."""
    from .templates import get_template

    tok = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=trust_remote_code, cache_dir=cache_dir
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # left-pad so that position -1 is always the last real token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
    )
    model.eval()

    return ModelWrapper(
        model=model, tokenizer=tok, template=get_template(template_name), name=model_path
    )
