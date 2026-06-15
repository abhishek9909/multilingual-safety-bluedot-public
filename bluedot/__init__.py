"""
bluedot — modular toolkit for extracting, probing, and intervening with
direction vectors (t_inst harmfulness, t_post-inst refusal) in LLMs.

Each subpackage is independently usable:
    models/         model + tokenizer + template wrappers
    hooks/          forward-hook primitives (capture, act-add, ablation)
    positions/      token-position helpers (t_inst, t_post_inst)
    activations/    extract hidden states at chosen positions
    directions/     compute/save/load direction vectors
    datasets/       common-format multilingual dataset loaders
    probes/         score arbitrary directions on arbitrary datasets
    interventions/  apply a direction to a model at inference time
"""

__version__ = "0.1.0"
