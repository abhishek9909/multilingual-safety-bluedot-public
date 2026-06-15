# bluedot — multilingual safety auditing via refusal-direction ablation

Public release of the codebase used in *Refusal vs Perception: A
Mechanistic Audit of Multilingual Safety Alignment in Llama-3.1-8B-Instruct
and Qwen-2.5-7B-Instruct*. The repository contains four runnable
experiment families:

- **Pilot Exp 1** — Binary severity comparison (indirect track).
- **Pilot Exp 2** — Alignment-quadrant analysis (direct track).
- **Validation §6.1** — Within-model persuasion-bypass attack
  (PAP-style, [Zeng et al. 2024]).
- **Validation §6.2** — Across-model deceptive-paraphrase attack.

The core interpretability library lives under `bluedot/` (model wrappers,
direction extraction, residual-stream interventions). All experiments
are CLI scripts under `experiments/` and run as Python modules.

---

## 1. Installation

The project targets Python ≥ 3.10. A GPU with ≥ 24 GB VRAM is required
for the local-model stages (Llama-3.1-8B / Qwen-2.5-7B inference).

```bash
git clone https://github.com/abhishekmish9909/multilingual-safety-bluedot
cd multilingual-safety-bluedot

# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install the core library + experiment extras
pip install -e '.[experiments]'

# 3. Optional: install the attacks extra (Anthropic SDK for the rewriter)
pip install -e '.[attacks]'
```

`.[experiments]` adds matplotlib for the analysis scripts.
`.[attacks]` adds the `anthropic` package — required only for the
validation experiments under §6.

---

## 2. Configuration

Copy `.env.example` to `.env` and fill in the required values:

```bash
cp .env.example .env
$EDITOR .env
```

Required secrets:

| variable | purpose |
|---|---|
| `HF_TOKEN` | Hugging Face access (Llama-3.1 is gated; required for `bluedot.models.load_model`) |
| `ANTHROPIC_API_KEY` | Required only for the validation rewrites (`persuasion_attack`, `paraphrase_attack`) |

Optional environment overrides for the local model:

| variable | default | purpose |
|---|---|---|
| `BLUEDOT_MODEL` | `meta-llama/Meta-Llama-3-8B-Instruct` | Hugging Face model id |
| `BLUEDOT_TEMPLATE` | `llama3` | Chat-template family (`llama3`, `qwen2`) |
| `BLUEDOT_LAYER` | `14` | Mid-layer for direction extraction / ablation (use `12` for Qwen-2.5-7B) |
| `BLUEDOT_BS` | `8` | Inference batch size |
| `BLUEDOT_ARTIFACTS` | `./artifacts` | Output directory for cached directions, pilot results, analysis CSVs |
| `BLUEDOT_POLY_ROOT` | `../Multilingual-Refusal/dataset` | PolyRefuse dataset root (for `d_refuse` extraction) |

To switch to Qwen-2.5-7B, set:

```bash
export BLUEDOT_MODEL=Qwen/Qwen2.5-7B-Instruct
export BLUEDOT_TEMPLATE=qwen2
export BLUEDOT_LAYER=12
export BLUEDOT_ARTIFACTS=$PWD/artifacts-qwen
```

---

## 3. Datasets

Two datasets are needed; both are external and not vendored:

**PolyRefuse** ([Wang et al. 2025]) — refusal / harmless prompts used for
direction extraction. Clone the repository alongside this one:

```bash
cd ..
git clone https://github.com/Multilingual-Refusal Multilingual-Refusal
cd multilingual-safety-bluedot
export BLUEDOT_POLY_ROOT=$PWD/../Multilingual-Refusal/dataset
```

**LinguaSafe** ([Sun et al. 2025]) — multilingual safety-graded prompts
(L0–L3) used for the pilots and the validation seeds. Download from the
official source and place per-language JSON files at:

```
data/linguasafe/{en, ar, bn, cs, hu, ko, ms, ru, sr, th, vi, zh}.json
```

Each file follows the LinguaSafe schema: a list of objects with
`prompt`, `severity` (one of `L0/L1/L2/L3`), and `domain` keys. The
twelve language codes above are the ones used in our results; the
pilot scripts auto-discover whichever subset is present.

---

## 4. Running the pilot experiments

Pilots require a GPU and pre-downloaded model weights. The first run
will download / extract `d_refuse` from PolyRefuse-en, which takes a few
minutes; subsequent runs reuse the cached direction.

### Pilot Exp 1 — Binary severity comparison

Presents same-domain prompt pairs at known severity contrasts (L0/L2,
L1/L3, L2/L3) under both orderings and asks which is more harmful.
Reports engagement rate, position bias, and per-contrast accuracy under
`abl(d_refuse)`.

```bash
# All twelve languages, English-CoT mode, 167 pairs per contrast (≈500
# pairs per language):
python -m experiments.pilot_exp1_indirect

# Single language, native-CoT mode:
python -m experiments.pilot_exp1_indirect --langs zh --judge-language native

# Roll up post-hoc analysis:
python -m experiments.analyze_pilot_exp1
```

### Pilot Exp 2 — Alignment-quadrant analysis

Per prompt, runs two passes: a behavioural pass (raw prompt under normal
inference; record refusal) and a perception pass (severity classification
under `abl(d_refuse)`). Crosses the two into the 2×2 alignment quadrant:

| | judged harmful | judged not harmful |
|---|---|---|
| refused | `concept_deep` | `trigger_only_refusal` |
| not refused | `decoupled_perception` | `aligned_with_content` |

```bash
# All twelve languages, EN-CoT:
python -m experiments.pilot_exp2_direct

# Single language, native-CoT:
python -m experiments.pilot_exp2_direct --langs ar --judge-language native

# Roll up analysis (alignment-quadrant matrices, severity F1, domain
# breakdown of trigger-only-refusal):
python -m experiments.analyze_pilot_exp2
python -m experiments.analyze_trigger_only_by_domain
python -m experiments.analyze_pilot_summary
```

### Per-language fan-out

To parallelise across language jobs without overwriting results, use the
`--out-tag` flag and the merge script:

```bash
for L in en ar bn cs hu ko ms ru sr th vi zh; do
  python -m experiments.pilot_exp2_direct --langs $L --out-tag $L
done
python -m experiments.merge_pilot_outputs
```

---

## 5. Running the validation experiments

The validation experiments (§6 of the paper) use Claude as the rewriter
and the local model only for the attack-evaluation pass. Set
`ANTHROPIC_API_KEY` first.

### §6.1 — Persuasion attack (within Llama)

Tests whether trigger-only refusals are more easily bypassed by PAP-style
persuasion rewrites than concept-deep refusals from matched (language ×
domain) cells.

```bash
# End-to-end on Llama: seeds → rewrite (Anthropic API) → evaluate (GPU)
# → analyze. Default 50 seeds per quadrant per language; Haiku 4.5 as
# rewriter (~$8 in API costs for all 12 languages).
python -m experiments.persuasion_attack

# To run individual stages:
python -m experiments.persuasion_attack --stage seeds
python -m experiments.persuasion_attack --stage rewrite --api-model claude-haiku-4-5-20251001
python -m experiments.persuasion_attack --stage evaluate
python -m experiments.persuasion_attack --stage analyze
```

Outputs land under `{BLUEDOT_ARTIFACTS}/analysis/persuasion_attack/`:
`seeds.json`, `rewrites.json`, `responses.json`,
`compliance_by_cell.csv`, `compliance_pooled.csv`, `differential.csv`,
`_anthropic_cache.json`.

### §6.2 — Paraphrase attack (across models)

Tests whether a shared multilingual harmful-prompt set is more easily
bypassed on Qwen than on Llama after matched deceptive paraphrase.
Requires Pilot Exp 2 results from both models.

```bash
# Stage 1: seeds — needs both Llama and Qwen artifact roots
python -m experiments.paraphrase_attack --stage seeds \
    --llama-art /path/to/artifacts \
    --qwen-art  /path/to/artifacts-qwen

# Stage 2: rewrite — Anthropic API; runs once, reused for both models
python -m experiments.paraphrase_attack --stage rewrite

# Stage 3a: evaluate on Llama
BLUEDOT_MODEL=meta-llama/Meta-Llama-3-8B-Instruct BLUEDOT_TEMPLATE=llama3 BLUEDOT_LAYER=14 \
    python -m experiments.paraphrase_attack --stage evaluate --model-tag llama

# Stage 3b: evaluate on Qwen
BLUEDOT_MODEL=Qwen/Qwen2.5-7B-Instruct BLUEDOT_TEMPLATE=qwen2 BLUEDOT_LAYER=12 \
    python -m experiments.paraphrase_attack --stage evaluate --model-tag qwen

# Stage 4: cross-model differential
python -m experiments.paraphrase_attack --stage analyze
```

Outputs land under the shared cross-model directory (default sibling of
the artifact roots): `seeds.json`, `rewrites.json`,
`responses_eval-llama.json`, `responses_eval-qwen.json`,
`asr_by_cell.csv`, `asr_pooled.csv`, `differential.csv`,
`_anthropic_cache.json`.

The Anthropic rewriter is cached on disk by `(model, prompt_hash)`, so
re-running any stage with the same seeds is free.

### Conditional-on-OK follow-up notebook

The headline differentials in the paper are reported conditional on
`rewrite_status == "ok"` (rows where Claude actually produced a rewrite,
as opposed to declining and falling back to the original prompt).
`examples/attack_followup_analysis.ipynb` reproduces that conditional
analysis from the JSON outputs above, including per-strategy / per-
technique decompositions and the binomial sign tests.

---

## 6. Repository layout

```
bluedot/        Core library — model wrappers, hooks, direction
                extraction, probes, interventions, dataset registry.
experiments/    CLI scripts (run as `python -m experiments.<name>`).
data/           Place LinguaSafe per-language JSON files under
                data/linguasafe/. PolyRefuse is referenced externally via
                BLUEDOT_POLY_ROOT.
examples/       Reproduction notebook for the §6 conditional analysis.
tests/          Unit tests.
```

---

## 7. Citation

If you use this code, please cite:

> _Refusal vs Perception: A Mechanistic Audit of Multilingual Safety
> Alignment in Llama-3.1-8B-Instruct and Qwen-2.5-7B-Instruct._
> Abhishek Mishra, 2026.

Upstream work this builds on:

- Arditi, A. et al. (2024). *Refusal in Language Models Is Mediated by a
  Single Direction.* arXiv:2406.11717.
- Wang, X. et al. (2025). *Refusal Direction Is Universal Across
  Safety-Aligned Languages.* NeurIPS 2025.
- Zhao, J. et al. (2025). *LLMs Encode Harmfulness and Refusal
  Separately.* arXiv:2507.11878.
- Sun, Z. et al. (2025). *LinguaSafe: A Comprehensive Multilingual
  Safety Benchmark for Large Language Models.* arXiv:2508.12733.
- Zeng, Y. et al. (2024). *How Johnny Can Persuade LLMs to Jailbreak
  Them.* ACL 2024. arXiv:2401.06373.
