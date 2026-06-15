"""
Post-hoc analysis of pilot_exp1 (binary severity comparison) outputs.

Reads the EN and NATIVE judge-language outputs side by side and produces:
  - cross_lang_table.csv         : combined per-(lang, condition, judge_lang)
  - engagement_en_vs_native.png  : engagement bar chart, both modes
  - perception_en_vs_native.png  : perception score bar chart, both modes
  - by_contrast.png              : perception accuracy by severity contrast
  - audit_metrics.png            : reasoning-lang-match and reasoning-action-consistency
  - position_bias.png            : per-side accuracy split (acc_l1 vs acc_l2)
  - per_trial_agreement.csv      : per-trial EN-vs-NATIVE answer agreement
  - agreement_summary.csv        : per-lang MATCH/DIVERGE rates

Run:
    python -m experiments.analyze_pilot_exp1
    python -m experiments.analyze_pilot_exp1 --root-dir /path/to/bluedot-artifacts
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
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


def _save_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        print(f"  (skipping, no rows): {path}")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"  wrote {path}")


def _plot_bar_pair(values_en, values_native, langs, ylabel, title, out_png,
                   ylim=(0, 1.05), hline=None, hline_label=None):
    import matplotlib.pyplot as plt
    import numpy as np
    x = np.arange(len(langs))
    width = 0.4
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(langs)), 4.8))
    ax.bar(x - width / 2, values_en,     width, label="judge=en")
    ax.bar(x + width / 2, values_native, width, label="judge=native")
    if hline is not None:
        ax.axhline(hline, color="k", ls="--", lw=0.5,
                   label=hline_label or f"baseline={hline:.2f}")
    ax.set_xticks(x); ax.set_xticklabels(langs)
    ax.set_ylabel(ylabel); ax.set_ylim(*ylim); ax.set_title(title)
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)
    print(f"  wrote {out_png}")


def _aligned_langs(per_lang_en, per_lang_native):
    """Common language order — union, sorted, preserving order."""
    keys = list(per_lang_en.keys())
    for k in per_lang_native:
        if k not in keys:
            keys.append(k)
    return sorted(keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-dir", default=DEFAULT_ROOT_DIR)
    args = ap.parse_args()
    root = os.path.abspath(args.root_dir)

    exp1_dir = os.path.join(root, "pilot_exp1")
    out_dir = os.path.join(root, "analysis", "exp1")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Reading from: {exp1_dir}")
    summary_en     = _load_csv(os.path.join(exp1_dir, "summary_en.csv"))
    summary_native = _load_csv(os.path.join(exp1_dir, "summary_native.csv"))
    contrast_en    = _load_csv(os.path.join(exp1_dir, "summary_by_contrast_en.csv"))
    contrast_native= _load_csv(os.path.join(exp1_dir, "summary_by_contrast_native.csv"))
    results_en     = _load_json(os.path.join(exp1_dir, "results_en.json"))
    results_native = _load_json(os.path.join(exp1_dir, "results_native.json"))

    if not summary_en and not summary_native:
        print("No data found, run pilot_exp1 first.")
        return

    # ------------------------------------------------------------------
    # Combined cross-language table
    # ------------------------------------------------------------------
    combined: List[Dict] = []
    for r in summary_en:     combined.append({**r, "judge_language": "en"})
    for r in summary_native: combined.append({**r, "judge_language": "native"})
    _save_csv(combined, os.path.join(out_dir, "cross_lang_table.csv"))

    # Index summaries by (lang, condition)
    def index_summary(rows):
        return {(r["lang"], r["condition"]): r for r in rows}

    by_en     = index_summary(summary_en)
    by_native = index_summary(summary_native)

    # Pull the abl_d_refuse condition (the substantive comparison)
    def per_lang_under(idx, cond="abl_d_refuse"):
        out = {}
        for (l, c), r in idx.items():
            if c == cond:
                out[l] = r
        return out

    en_abl     = per_lang_under(by_en)
    native_abl = per_lang_under(by_native)
    langs = _aligned_langs(en_abl, native_abl)

    # Engagement bar chart
    if langs:
        _plot_bar_pair(
            [_try_float(en_abl.get(L, {}).get("engagement_rate", "nan")) for L in langs],
            [_try_float(native_abl.get(L, {}).get("engagement_rate", "nan")) for L in langs],
            langs, ylabel="engagement (parseable answer) rate",
            title="Pilot Exp 1 — engagement under abl(d_refuse), en vs native CoT",
            out_png=os.path.join(out_dir, "engagement_en_vs_native.png"),
        )

        _plot_bar_pair(
            [_try_float(en_abl.get(L, {}).get("perception_acc_engaged", "nan")) for L in langs],
            [_try_float(native_abl.get(L, {}).get("perception_acc_engaged", "nan")) for L in langs],
            langs, ylabel="comparison accuracy (engaged subset)",
            title="Pilot Exp 1 — intrinsic perception, en vs native CoT",
            out_png=os.path.join(out_dir, "perception_en_vs_native.png"),
            hline=0.5, hline_label="chance=0.5",
        )

        # Audit metrics: reasoning-lang-match and reasoning-action-consistency
        _plot_bar_pair(
            [_try_float(en_abl.get(L, {}).get("reasoning_lang_match", "nan")) for L in langs],
            [_try_float(native_abl.get(L, {}).get("reasoning_lang_match", "nan")) for L in langs],
            langs, ylabel="reasoning-language match rate",
            title="Pilot Exp 1 — directive efficacy (did CoT use the intended language?)",
            out_png=os.path.join(out_dir, "reasoning_lang_match.png"),
        )
        _plot_bar_pair(
            [_try_float(en_abl.get(L, {}).get("reasoning_action_consistency", "nan")) for L in langs],
            [_try_float(native_abl.get(L, {}).get("reasoning_action_consistency", "nan")) for L in langs],
            langs, ylabel="reasoning/action consistency rate",
            title="Pilot Exp 1 — CoT-faithfulness (does the reasoning's endorsement match the answer tag?)",
            out_png=os.path.join(out_dir, "reasoning_action_consistency.png"),
        )

        # Position-bias side-by-side (acc_l1 vs acc_l2 per condition)
        import matplotlib.pyplot as plt
        import numpy as np
        x = np.arange(len(langs))
        width = 0.2
        fig, ax = plt.subplots(figsize=(max(9, 1.4 * len(langs)), 5))
        ax.bar(x - 1.5 * width, [_try_float(en_abl.get(L, {}).get("acc_l1", "nan"))     for L in langs], width, label="en  | label=1")
        ax.bar(x - 0.5 * width, [_try_float(en_abl.get(L, {}).get("acc_l2", "nan"))     for L in langs], width, label="en  | label=2")
        ax.bar(x + 0.5 * width, [_try_float(native_abl.get(L, {}).get("acc_l1", "nan")) for L in langs], width, label="nat | label=1")
        ax.bar(x + 1.5 * width, [_try_float(native_abl.get(L, {}).get("acc_l2", "nan")) for L in langs], width, label="nat | label=2")
        ax.set_xticks(x); ax.set_xticklabels(langs)
        ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.05)
        ax.set_title("Pilot Exp 1 — position-bias diagnostic (acc when label=1 vs label=2)")
        ax.axhline(0.5, color="k", lw=0.5, ls="--")
        ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2)
        out_png = os.path.join(out_dir, "position_bias.png")
        fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)
        print(f"  wrote {out_png}")

    # ------------------------------------------------------------------
    # Per-contrast bar chart
    # ------------------------------------------------------------------
    if contrast_en or contrast_native:
        def key(r): return (r["lang"], r["condition"], r["contrast"])
        by_en_cont     = {key(r): r for r in contrast_en}
        by_native_cont = {key(r): r for r in contrast_native}
        contrasts = sorted({r["contrast"] for r in contrast_en + contrast_native})
        if langs and contrasts:
            import matplotlib.pyplot as plt
            import numpy as np
            fig, axes = plt.subplots(1, len(contrasts), figsize=(5 * len(contrasts), 5), sharey=True)
            if len(contrasts) == 1: axes = [axes]
            for ax, ctr in zip(axes, contrasts):
                ev = [_try_float(by_en_cont.get((L, "abl_d_refuse", ctr), {}).get("perception_acc_engaged", "nan"))     for L in langs]
                nv = [_try_float(by_native_cont.get((L, "abl_d_refuse", ctr), {}).get("perception_acc_engaged", "nan")) for L in langs]
                x = np.arange(len(langs)); width = 0.4
                ax.bar(x - width / 2, ev, width, label="judge=en")
                ax.bar(x + width / 2, nv, width, label="judge=native")
                ax.axhline(0.5, color="k", lw=0.5, ls="--")
                ax.set_xticks(x); ax.set_xticklabels(langs, rotation=0)
                ax.set_title(f"contrast = {ctr}")
                ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)
                if ax is axes[0]:
                    ax.set_ylabel("perception accuracy (engaged)")
                ax.legend(fontsize=8)
            fig.suptitle("Pilot Exp 1 — accuracy by severity contrast")
            out_png = os.path.join(out_dir, "by_contrast.png")
            fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)
            print(f"  wrote {out_png}")

    # ------------------------------------------------------------------
    # Per-trial EN vs NATIVE agreement (when both runs exist)
    # ------------------------------------------------------------------
    if results_en and results_native:
        per_trial_rows = []
        agreement_summary = defaultdict(lambda: {"both_parseable": 0, "match": 0, "diverge": 0,
                                                  "en_correct": 0, "native_correct": 0, "n": 0})
        for cond_key, r_en in results_en["conditions"].items():
            r_nat = results_native["conditions"].get(cond_key)
            if r_nat is None or "predictions" not in r_en or "predictions" not in r_nat:
                continue
            # cond_key is of form "lang__cond"; take lang
            lang = cond_key.split("__", 1)[0]
            cond = cond_key.split("__", 1)[1] if "__" in cond_key else ""
            if cond != "abl_d_refuse":
                continue
            labels = r_en.get("labels", [])
            preds_en = r_en["predictions"]
            preds_nat = r_nat["predictions"]
            n = min(len(preds_en), len(preds_nat), len(labels))
            for i in range(n):
                en, nat, lab = preds_en[i], preds_nat[i], labels[i]
                agreement_summary[lang]["n"] += 1
                en_ok = (en == lab)
                nat_ok = (nat == lab)
                if en_ok: agreement_summary[lang]["en_correct"] += 1
                if nat_ok: agreement_summary[lang]["native_correct"] += 1
                if en is not None and nat is not None:
                    agreement_summary[lang]["both_parseable"] += 1
                    if en == nat: agreement_summary[lang]["match"] += 1
                    else:         agreement_summary[lang]["diverge"] += 1
                per_trial_rows.append({
                    "lang": lang, "trial_idx": i,
                    "label": lab,
                    "en_pred": en, "native_pred": nat,
                    "en_correct": en_ok, "native_correct": nat_ok,
                    "agreement": "match" if (en is not None and nat is not None and en == nat)
                                 else ("diverge" if (en is not None and nat is not None) else "n/a"),
                })

        _save_csv(per_trial_rows, os.path.join(out_dir, "per_trial_agreement.csv"))

        # Per-language agreement summary
        summary_rows = []
        for lang, d in sorted(agreement_summary.items()):
            n = d["n"]
            bp = d["both_parseable"]
            summary_rows.append({
                "lang": lang, "n": n,
                "both_parseable": bp,
                "match_rate":     d["match"] / bp     if bp else float("nan"),
                "diverge_rate":   d["diverge"] / bp   if bp else float("nan"),
                "en_accuracy":     d["en_correct"] / n if n else float("nan"),
                "native_accuracy": d["native_correct"] / n if n else float("nan"),
                "drift_delta":     (d["native_correct"] - d["en_correct"]) / n if n else float("nan"),
            })
        _save_csv(summary_rows, os.path.join(out_dir, "agreement_summary.csv"))

        # Plot: signed drift = native_accuracy − en_accuracy per language
        if summary_rows:
            import matplotlib.pyplot as plt
            import numpy as np
            langs_s = [r["lang"] for r in summary_rows]
            deltas = [r["drift_delta"] for r in summary_rows]
            colors = ["#3aaa3a" if d > 0 else "#d62728" if d < 0 else "#999"
                      for d in deltas]
            fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(langs_s)), 4.5))
            ax.bar(langs_s, deltas, color=colors)
            ax.axhline(0, color="k", lw=0.7)
            ax.set_ylabel("native accuracy − en accuracy")
            ax.set_title("Pilot Exp 1 — instruction-language drift sign per language\n"
                         "(positive = native CoT improves judgement)")
            ax.grid(alpha=0.3, axis="y")
            out_png = os.path.join(out_dir, "drift_sign.png")
            fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)
            print(f"  wrote {out_png}")

    print(f"\nAll outputs under: {out_dir}")


if __name__ == "__main__":
    main()
