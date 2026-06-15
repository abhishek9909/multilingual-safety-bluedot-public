"""Experiment scripts. Run each as a module:

    # Pilot Exp 1 — binary severity comparison
    python -m experiments.pilot_exp1_indirect --langs en ar zh

    # Pilot Exp 2 — alignment-quadrant analysis
    python -m experiments.pilot_exp2_direct --langs en ar zh

    # Validation — within-Llama persuasion attack
    python -m experiments.persuasion_attack

    # Validation — across-model deceptive paraphrase
    python -m experiments.paraphrase_attack --stage seeds \\
        --llama-art /path/to/bluedot-artifacts \\
        --qwen-art  /path/to/bluedot-artifacts-qwen

See the project README for full installation and configuration details.
"""
