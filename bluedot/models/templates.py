"""
Chat templates and post-instruction (EOI) token strings, per model family.

`eoi_str` is everything appended AFTER the user's instruction inside the chat
template — i.e. the post-instruction region whose last token == t_post_inst.
The token immediately preceding that region == t_inst (last token of user instr).
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ChatTemplate:
    name: str
    template: str          # uses {instruction} placeholder
    template_with_system: Optional[str]   # uses {system} and {instruction}
    eoi_str: str           # post-instruction string (used to find token positions)
    # Turn-terminator token strings for THIS model family. These must be added
    # to `eos_token_id` at generate() time, otherwise the model will not stop
    # at end-of-turn and will degenerate into looping (Llama3 in particular).
    stop_token_strs: tuple = ()

    def format(self, instruction: str, system: Optional[str] = None) -> str:
        if system is not None and self.template_with_system is not None:
            return self.template_with_system.format(system=system, instruction=instruction)
        return self.template.format(instruction=instruction)


QWEN2 = ChatTemplate(
    name="qwen2",
    template=(
        "<|im_start|>user\n{instruction}<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    template_with_system=(
        "<|im_start|>system\n{system}<|im_end|>\n"
        "<|im_start|>user\n{instruction}<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    eoi_str="<|im_end|>\n<|im_start|>assistant\n",
    stop_token_strs=("<|im_end|>", "<|endoftext|>"),
)

LLAMA2 = ChatTemplate(
    name="llama2",
    template="[INST] {instruction} [/INST]",
    template_with_system="[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{instruction} [/INST]",
    eoi_str=" [/INST]",
    stop_token_strs=("</s>",),
)

LLAMA3 = ChatTemplate(
    name="llama3",
    template=(
        "<|start_header_id|>user<|end_header_id|>\n\n{instruction}"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    ),
    template_with_system=(
        "<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n{instruction}"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    ),
    eoi_str="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    # CRITICAL: Llama3-Instruct emits <|eot_id|> (128009) at end-of-turn,
    # but tokenizer.eos_token is <|end_of_text|> (128001). Without including
    # <|eot_id|> here, generate() never stops and loops indefinitely.
    stop_token_strs=("<|eot_id|>", "<|end_of_text|>"),
)

GEMMA = ChatTemplate(
    name="gemma",
    template="<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n",
    template_with_system=None,
    eoi_str="<end_of_turn>\n<start_of_turn>model\n",
    stop_token_strs=("<end_of_turn>", "<eos>"),
)

REGISTRY = {t.name: t for t in [QWEN2, LLAMA2, LLAMA3, GEMMA]}


def get_template(name: str) -> ChatTemplate:
    if name not in REGISTRY:
        raise KeyError(f"Unknown template '{name}'. Available: {list(REGISTRY)}")
    return REGISTRY[name]
