"""
Post-hoc analysis of pilot_exp2 (alignment-quadrant) outputs.

Reads EN and NATIVE judge-language outputs side by side and produces:
  - cross_lang_table.csv               : combined per-lang summary
  - quadrants_en_vs_native.png         : quadrant fractions, both modes
  - severity_f1_en_vs_native.png       : LinguaSafe-style F1 by lang & mode
  - confusion_{lang}_{mode}.png        : severity confusion heatmap per lang
  - audit_metrics.png                  : reasoning-lang-match and consistency
  - quadrant_migration.csv             : per-prompt EN-quadrant vs NATIVE-quadrant
  - quadrant_migration_summary.csv     : per-lang migration counts

Run:
    python -m experiments.analyze_pilot_exp2
    python -m experiments.analyze_pilot_exp2 --root-dir /path/to/bluedot-artifacts
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional


DEFAULT_ROOT_DIR = "/project/pi_jensen_umass_edu/abhishekmish_umass_edu/bluedot-artifacts/"

QUADRANT_ORDER = ["concept_deep", "trigger_only_refusal",
                  "decoupled_perception", "aligned_with_content",
                  "perception_unparseable"]
QUADRANT_COLORS = {
    "concept_deep":           "#3aaa3a",
    "trigger_only_refusal":   "#d62728",
    "decoupled_perception":   "#ff7f0e",
    "aligned_with_content":   "#1f77b4",
    "perception_unparseable": "#999999",
}
SEVERITY_LEVELS = ["L0", "L1", "L2", "L3"]


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


def _save_csv(rows, path):
    if not rows:
        print(f"  (skipping, no rows): {path}")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"  wrote {path}")


def _stacked_quadrants(results_json, langs, out_png, title):
    import matplotlib.pyplot as plt
    import numpy as np
    if not results_json:
        return
    x = np.arange(len(langs))
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(langs)), 5))
    bottom = np.zeros(len(langs))
    for q in QUADRANT_ORDER:
        ys = []
        for L in langs:
            counts = results_json["results"].get(L, {}).get("quadrant_counts", {})
            total = sum(counts.values()) or 1
            ys.append(counts.get(q, 0) / total)
        ax.bar(x, ys, bottom=bottom, label=q, color=QUADRANT_COLORS[q])
        bottom += np.array(ys)
    ax.set_xticks(x); ax.set_xticklabels(langs)
    ax.set_ylabel("fraction of prompts"); ax.set_ylim(0, 1.05)
    ax.set_title(title); ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)
    print(f"  wrote {out_png}")


def _side_by_side_quadrants(res_en, res_nat, langs, out_png):
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(max(14, 2.5 * len(langs)), 5), sharey=True)
    for ax, res, tag in [(axes[0], res_en, "judge=en"), (axes[1], res_nat, "judge=native")]:
        x = np.arange(len(langs))
        bottom = np.zeros(len(langs))
        for q in QUADRANT_ORDER:
            ys = []
            for L in langs:
                counts = res["results"].get(L, {}).get("quadrant_counts", {}) if res else {}
                total = sum(counts.values()) or 1
                ys.append(counts.get(q, 0) / total)
            ax.bar(x, ys, bottom=bottom, label=q, color=QUADRANT_COLORS[q])
            bottom += np.array(ys)
        ax.set_xticks(x); ax.set_xticklabels(langs)
        ax.set_ylim(0, 1.05); ax.set_title(tag)
        if ax is axes[0]: ax.set_ylabel("fraction of prompts")
    axes[1].legend(loc="lower right", fontsize=8, bbox_to_anchor=(1.6, 0))
    fig.suptitle("Pilot Exp 2 — quadrant distribution, EN vs NATIVE CoT")
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)
    print(f"  wrote {out_png}")


def _confusion_heatmaps(res, mode_tag, out_dir):
    import matplotlib.pyplot as plt
    import numpy as np
    if not res: return
    for lang, lr in res["results"].items():
        conf = lr.get("severity_confusion", {})
        if not conf: continue
        M = np.zeros((len(SEVERITY_LEVELS), len(SEVERITY_LEVELS) + 1))
        cols = SEVERITY_LEVELS + ["unparseable"]
        for i, t in enumerate(SEVERITY_LEVELS):
            for j, p in enumerate(cols):
                M[i, j] = conf.get(t, {}).get(p, 0)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        im = ax.imshow(M, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols)
        ax.set_yticks(range(len(SEVERITY_LEVELS))); ax.set_yticklabels(SEVERITY_LEVELS)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"Confusion — {lang} ({mode_tag})  F1={lr.get('severity_weighted_f1', float('nan')):.3f}")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{int(M[i, j])}", ha="center", va="center",
                        color="white" if M[i, j] > M.max() / 2 else "black",
                        fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        out_png = os.path.join(out_dir, f"confusion_{lang}_{mode_tag}.png")
        fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)
    print(f"  wrote confusion heatmaps (mode={mode_tag})")


def _quadrant_migration(res_en, res_nat, out_csv, out_summary_csv):
    """Per-prompt: EN-quadrant vs NATIVE-quadrant. Source from per_prompt."""
    if not (res_en and res_nat): return
    rows: List[Dict] = []
    per_lang: Dict[str, Counter] = defaultdict(Counter)
    for lang, lr_en in res_en["results"].items():
        lr_nat = res_nat["results"].get(lang)
        if not lr_nat: continue
        en_by_id  = {p["id"]: p for p in lr_en.get("per_prompt", [])}
        nat_by_id = {p["id"]: p for p in lr_nat.get("per_prompt", [])}
        common = set(en_by_id) & set(nat_by_id)
        for pid in sorted(common):
            en_p = en_by_id[pid]; nat_p = nat_by_id[pid]
            key = (en_p["quadrant"], nat_p["quadrant"])
            per_lang[lang][key] += 1
            rows.append({
                "lang": lang, "id": pid, "domain": en_p["domain"],
                "severity_true": en_p["severity_true"],
                "en_quadrant":     en_p["quadrant"],
                "native_quadrant": nat_p["quadrant"],
                "en_severity_pred":     en_p["severity_pred"],
                "native_severity_pred": nat_p["severity_pred"],
                "migrated": en_p["quadrant"] != nat_p["quadrant"],
            })
    _save_csv(rows, out_csv)

    summary_rows = []
    for lang, ctr in sorted(per_lang.items()):
        total = sum(ctr.values())
        migrated = sum(v for (a, b), v in ctr.items() if a != b)
        summary_rows.append({
            "lang": lang, "n": total,
            "migration_rate": migrated / total if total else float("nan"),
            "top_migration_path": ", ".join(
                f"{a}->{b}: {v}" for (a, b), v in ctr.most_common(3) if a != b
            ),
        })
    _save_csv(summary_rows, out_summary_csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-dir", default=DEFAULT_ROOT_DIR)
    args = ap.parse_args()
    root = os.path.abspath(args.root_dir)

    exp2_dir = os.path.join(root, "pilot_exp2")
    out_dir = os.path.join(root, "analysis", "exp2")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Reading from: {exp2_dir}")
    sum_en     = _load_csv(os.path.join(exp2_dir, "summary_en.csv"))
    sum_native = _load_csv(os.path.join(exp2_dir, "summary_native.csv"))
    res_en     = _load_json(os.path.join(exp2_dir, "results_en.json"))
    res_native = _load_json(os.path.join(exp2_dir, "results_native.json"))

    if not sum_en and not sum_native:
        print("No data found, run pilot_exp2 first.")
        return

    # Combined cross-language table
    combined: List[Dict] = []
    for r in sum_en:     combined.append({**r, "judge_language": "en"})
    for r in sum_native: combined.append({**r, "judge_language": "native"})
    _save_csv(combined, os.path.join(out_dir, "cross_lang_table.csv"))

    by_en = {r["lang"]: r for r in sum_en}
    by_nat = {r["lang"]: r for r in sum_native}
    langs = sorted(set(by_en) | set(by_nat))

    # Quadrant distribution side by side
    if (res_en or res_native) and langs:
        _side_by_side_quadrants(res_en or {"results": {}}, res_native or {"results": {}},
                                langs, os.path.join(out_dir, "quadrants_en_vs_native.png"))

    # Severity-weighted F1 comparison
    if langs:
        import matplotlib.pyplot as plt
        import numpy as np
        x = np.arange(len(langs)); width = 0.4
        fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(langs)), 4.5))
        ax.bar(x - width / 2, [_try_float(by_en.get(L, {}).get("severity_weighted_f1", "nan"))  for L in langs], width, label="judge=en")
        ax.bar(x + width / 2, [_try_float(by_nat.get(L, {}).get("severity_weighted_f1", "nan")) for L in langs], width, label="judge=native")
        ax.set_xticks(x); ax.set_xticklabels(langs)
        ax.set_ylabel("severity-weighted F1 (perception pass)")
        ax.set_title("Pilot Exp 2 — perception F1, en vs native CoT")
        ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend()
        out_png = os.path.join(out_dir, "severity_f1_en_vs_native.png")
        fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)
        print(f"  wrote {out_png}")

    # Audit metrics
    if langs:
        import matplotlib.pyplot as plt
        import numpy as np
        x = np.arange(len(langs)); width = 0.4
        fig, axes = plt.subplots(1, 2, figsize=(max(13, 1.7 * len(langs)), 4.5), sharey=True)
        axes[0].bar(x - width/2, [_try_float(by_en.get(L,{}).get("reasoning_lang_match","nan"))  for L in langs], width, label="judge=en")
        axes[0].bar(x + width/2, [_try_float(by_nat.get(L,{}).get("reasoning_lang_match","nan")) for L in langs], width, label="judge=native")
        axes[0].set_xticks(x); axes[0].set_xticklabels(langs)
        axes[0].set_ylim(0, 1.05); axes[0].set_ylabel("rate")
        axes[0].set_title("reasoning-language match rate"); axes[0].grid(alpha=0.3); axes[0].legend()

        axes[1].bar(x - width/2, [_try_float(by_en.get(L,{}).get("reasoning_action_consistency","nan"))  for L in langs], width, label="judge=en")
        axes[1].bar(x + width/2, [_try_float(by_nat.get(L,{}).get("reasoning_action_consistency","nan")) for L in langs], width, label="judge=native")
        axes[1].set_xticks(x); axes[1].set_xticklabels(langs)
        axes[1].set_ylim(0, 1.05); axes[1].set_title("reasoning/action consistency rate")
        axes[1].grid(alpha=0.3); axes[1].legend()

        fig.suptitle("Pilot Exp 2 — audit metrics")
        out_png = os.path.join(out_dir, "audit_metrics.png")
        fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)
        print(f"  wrote {out_png}")

    # Per-lang confusion heatmaps
    _confusion_heatmaps(res_en, "en", out_dir)
    _confusion_heatmaps(res_native, "native", out_dir)

    # Quadrant migration EN -> NATIVE
    if res_en and res_native:
        _quadrant_migration(
            res_en, res_native,
            out_csv=os.path.join(out_dir, "quadrant_migration.csv"),
            out_summary_csv=os.path.join(out_dir, "quadrant_migration_summary.csv"),
        )

    print(f"\nAll outputs under: {out_dir}")


if __name__ == "__main__":
    main()
