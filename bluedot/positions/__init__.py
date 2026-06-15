"""
Token-position helpers.

The two positions of interest, from the "LLMs Encode Harmfulness and Refusal
Separately" paper:

    t_inst       = last token of the user's instruction
    t_post_inst  = last token of the post-instruction template
                   (the assistant turn header / end-of-instr region)

With a chat template like:

    "<|im_start|>user\\n{INSTR}<|im_end|>\\n<|im_start|>assistant\\n"
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            this is the eoi (post-instruction) region

t_post_inst is the very last token of the entire prompt (position -1).
t_inst is the token right before the eoi region begins
(position -(n_eoi + 1) or equivalently -n_eoi - 1, using left-padded inputs).
"""

from .tokens import (
    PositionSpec,
    t_inst_position,
    t_post_inst_position,
    eoi_positions,
    inst_plus_eoi_positions,
)

__all__ = [
    "PositionSpec",
    "t_inst_position",
    "t_post_inst_position",
    "eoi_positions",
    "inst_plus_eoi_positions",
]
