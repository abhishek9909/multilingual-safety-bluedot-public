"""
Cross-experiment summary: builds the report's headline diagnostic table and
figure from Exp 1 and Exp 2 outputs.

Per-language *diagnostic card* with these columns:
  - n                                   (Exp 2 prompt count)
  - baseline_refusal_rate               (from Exp 2 behavioural pass)
  - perception_accuracy                 (from Exp 2 perception pass under abl_d_refuse)
  - refusal_perception_gap              (refusal_rate − perception_accuracy)
  - quadrant_concept_deep_pct
  - quadrant_trigger_only_refusal_pct   (the "shallow safety" headline)
  - quadrant_decoupled_perception_pct
  - exp1_engagement                     (from Exp 1 abl_d_refuse, en mode)
  - exp1_perception_accuracy            (from Exp 1 abl_d_refuse, en mode)
  - drift_delta                         (native − en accuracy difference, if both runs available)

Outputs:
  - summary_card.csv                    : the table above
  - refusal_perception_gap.png          : per-language headline figure
  - shallow_safety_share.png            : trigger-only-refusal fraction per lang
  - drift_overview.png                  : drift_delta per lang from Exp 1

Run:
    python -m experiments.analyze_pilot_summary
    python -m experiments.analyze_pilot_summary --root-dir /path/to/bluedot-artifacts
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Optional


DEFAULT_ROOT_DIR = "/project/pi_jensen_umass_edu/abhishekmish_umass_edu/bluedot-artifacts/"


def _load_csv(path: str) -> List[Dict]:
    if not os.path.exists(path):
        print(f"  missing: {path}")
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _load_json(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        print(f"  missing: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def _try_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-dir", default=DEFAULT_ROOT_DIR)
    args = ap.parse_args()
    root = os.path.abspath(args.root_dir)

    out_dir = os.path.join(root, "analysis", "summary")
    os.makedirs(out_dir, exist_ok=True)

    # ----- Pull Exp 2 (the substantive source of refusal + perception + quadrants)
    exp2_dir = os.path.join(root, "pilot_exp2")
    exp2_sum_en = {r["lang"]: r for r in _load_csv(os.path.join(exp2_dir, "summary_en.csv"))}
    exp2_sum_nat = {r["lang"]: r for r in _load_csv(os.path.join(exp2_dir, "summary_native.csv"))}
    exp2_res_en = _load_json(os.path.join(exp2_dir, "results_en.json"))
    exp2_res_nat = _load_json(os.path.join(exp2_dir, "results_native.json"))

    # Recover refusal rate per (lang, mode) from the per_prompt JSON
    # (the behavioural pass is independent of the perception template, so the
    # refusal rate should be identical between en and native runs — we take
    # whichever is available)
    def refusal_rate_by_lang(res):
        out = {}
        if not res: return out
        for lang, lr in res["results"].items():
            behav = lr.get("behavioural", [])
            n = len(behav)
            if n:
                ref = sum(1 for b in behav if b.get("refused"))
                out[lang] = ref / n
        return out

    refusal_en = refusal_rate_by_lang(exp2_res_en)
    refusal_nat = refusal_rate_by_lang(exp2_res_nat)
    refusal = {**refusal_nat, **refusal_en}   # prefer en if both present

    # ----- Pull Exp 1 (engagement + drift)
    exp1_dir = os.path.join(root, "pilot_exp1")
    exp1_sum_en = {(r["lang"], r["condition"]): r for r in _load_csv(os.path.join(exp1_dir, "summary_en.csv"))}
    exp1_sum_nat = {(r["lang"], r["condition"]): r for r in _load_csv(os.path.join(exp1_dir, "summary_native.csv"))}
    agreement = _load_csv(os.path.join(root, "analysis", "exp1", "agreement_summary.csv"))
    drift_by_lang = {r["lang"]: _try_float(r["drift_delta"]) for r in agreement}

    # ----- Build the per-language card
    langs = sorted(set(exp2_sum_en) | set(exp2_sum_nat) | set(refusal))
    if not langs:
        print("No data found — make sure pilot jobs completed and produced summary CSVs.")
        return

    card_rows: List[Dict] = []
    for lang in langs:
        # Prefer EN-mode metrics for the headline card (the standard
        # cross-lingual evaluation regime); record native variants alongside.
        e2 = exp2_sum_en.get(lang) or exp2_sum_nat.get(lang) or {}
        e1 = exp1_sum_en.get((lang, "abl_d_refuse")) or exp1_sum_nat.get((lang, "abl_d_refuse")) or {}
        n      = int(e2.get("n") or 0)
        cd     = int(e2.get("concept_deep") or 0)
        tor    = int(e2.get("trigger_only_refusal") or 0)
        dp     = int(e2.get("decoupled_perception") or 0)
        unparseable = int(e2.get("perception_unparseable") or 0)
        parseable_n = max(n - unparseable, 1)
        perc_acc = (cd + dp) / parseable_n     # judged-harmful fraction among parseable
        ref_rate = refusal.get(lang, float("nan"))
        gap = ref_rate - perc_acc if (isinstance(ref_rate, float) and not (ref_rate != ref_rate)) else float("nan")
        card_rows.append({
            "lang": lang, "n": n,
            "baseline_refusal_rate":              f"{ref_rate:.4f}" if ref_rate == ref_rate else "nan",
            "perception_accuracy":                f"{perc_acc:.4f}",
            "refusal_perception_gap":             f"{gap:.4f}" if gap == gap else "nan",
            "quadrant_concept_deep_pct":          f"{cd / max(n,1) * 100:.1f}",
            "quadrant_trigger_only_refusal_pct":  f"{tor / max(n,1) * 100:.1f}",
            "quadrant_decoupled_perception_pct":  f"{dp / max(n,1) * 100:.1f}",
            "severity_weighted_f1":               e2.get("severity_weighted_f1", ""),
            "reasoning_lang_match":               e2.get("reasoning_lang_match", ""),
            "reasoning_action_consistency":       e2.get("reasoning_action_consistency", ""),
            "exp1_engagement":                    e1.get("engagement_rate", ""),
            "exp1_perception_accuracy":           e1.get("perception_acc_engaged", ""),
            "drift_delta_exp1":                   f"{drift_by_lang.get(lang, float('nan')):.4f}"
                                                  if drift_by_lang.get(lang, float('nan')) == drift_by_lang.get(lang, float('nan')) else "nan",
        })

    out_card = os.path.join(out_dir, "summary_card.csv")
    with open(out_card, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(card_rows[0].keys()))
        w.writeheader()
        for r in card_rows: w.writerow(r)
    print(f"  wrote {out_card}")

    # ----- Plot: refusal_perception_gap per language (THE headline)
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        ls = [r["lang"] for r in card_rows]
        ref  = [_try_float(r["baseline_refusal_rate"]) for r in card_rows]
        perc = [_try_float(r["perception_accuracy"])   for r in card_rows]
        gap  = [_try_float(r["refusal_perception_gap"]) for r in card_rows]

        x = np.arange(len(ls)); width = 0.4
        fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(ls)), 5))
        ax.bar(x - width / 2, ref,  width, label="refusal rate",          color="#d62728")
        ax.bar(x + width / 2, perc, width, label="perception accuracy",   color="#3aaa3a")
        for i, g in enumerate(gap):
            if g == g:    # skip NaN
                ax.text(x[i], max(ref[i], perc[i]) + 0.02, f"gap={g:+.2f}",
                        ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(ls)
        ax.set_ylim(0, 1.15); ax.set_ylabel("rate")
        ax.set_title("Headline — refusal-perception gap per language\n"
                     "(refusal > perception = shallow / trigger-only safety)")
        ax.grid(alpha=0.3); ax.legend()
        out_png = os.path.join(out_dir, "refusal_perception_gap.png")
        fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)
        print(f"  wrote {out_png}")

        # Trigger-only-refusal fraction per language — the "shallow safety" headline
        tor_pct = [_try_float(r["quadrant_trigger_only_refusal_pct"]) for r in card_rows]
        fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(ls)), 4.5))
        ax.bar(ls, tor_pct, color="#d62728")
        ax.set_ylabel("trigger-only refusal (% of prompts)")
        ax.set_title("Per-language fraction of refusals that are *not* matched by perception")
        ax.grid(alpha=0.3, axis="y")
        out_png = os.path.join(out_dir, "shallow_safety_share.png")
        fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)
        print(f"  wrote {out_png}")

        # Drift overview from Exp 1
        drifts = [_try_float(r["drift_delta_exp1"]) for r in card_rows]
        if any(d == d for d in drifts):    # has at least one non-NaN
            colors = ["#3aaa3a" if d > 0 else "#d62728" if d < 0 else "#999"
                      for d in drifts]
            fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(ls)), 4.5))
            ax.bar(ls, drifts, color=colors)
            ax.axhline(0, color="k", lw=0.7)
            ax.set_ylabel("native accuracy − en accuracy (Exp 1)")
            ax.set_title("Instruction-language drift sign per language\n"
                         "(positive = native CoT improves judgement)")
            ax.grid(alpha=0.3, axis="y")
            out_png = os.path.join(out_dir, "drift_overview.png")
            fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)
            print(f"  wrote {out_png}")
    except ImportError:
        print("  (matplotlib not installed, skipping plots)")

    # Print a brief stdout summary
    print(f"\n===  Headline numbers per language  ===")
    print(f"  {'lang':>6s}  {'refusal':>8s}  {'perception':>10s}  {'gap':>6s}  {'trigger-only%':>12s}")
    for r in card_rows:
        print(f"  {r['lang']:>6s}  {r['baseline_refusal_rate']:>8s}  "
              f"{r['perception_accuracy']:>10s}  {r['refusal_perception_gap']:>6s}  "
              f"{r['quadrant_trigger_only_refusal_pct']:>12s}")

    print(f"\nAll outputs under: {out_dir}")


if __name__ == "__main__":
    main()
