"""
Extract a balanced sample of currently-refused prompts per (language, domain)
cell, capped at N per cell, for persuasion-based jailbreak experiments.

By default selects from the `trigger_only_refusal` quadrant — those are the
prompts where the model refused at the behavioural pass but its own perception
(with refusal ablated) did not flag harm, which is the cell most likely to
bypass under paraphrase / persuasion. Pass `--quadrant any` to include all
refused prompts regardless of perception, or `--quadrant concept_deep` to use
prompts the model both refused AND judged harmful (the "robust refusal"
calibration set).

Reads:
    {ART}/pilot_exp2/results_{mode}.json  (default mode = en; refusal decisions
                                           are identical across modes because
                                           the behavioural pass is mode-agnostic)

Writes:
    {ART}/analysis/persuasion_seeds/
        refused_examples.csv   - flat table, sortable by (lang, domain)
        refused_examples.json  - same data grouped by language for batch work
        counts_pivot.csv       - per-(lang, domain) sample counts

Run:
    python -m experiments.extract_persuasion_seeds
    python -m experiments.extract_persuasion_seeds --cap-per-cell 10
    python -m experiments.extract_persuasion_seeds --quadrant any
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from typing import Dict, List


DEFAULT_ROOT_DIR = "/project/pi_jensen_umass_edu/abhishekmish_umass_edu/bluedot-artifacts/"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-dir", default=DEFAULT_ROOT_DIR)
    ap.add_argument(
        "--mode", default="en", choices=["en", "native"],
        help="Which results_{mode}.json to read. Default en. Refusal decisions "
             "(behavioural pass) are identical across modes; perception fields "
             "(severity_pred, quadrant) reflect the chosen mode's perception pass.",
    )
    ap.add_argument(
        "--cap-per-cell", type=int, default=10,
        help="Maximum number of examples per (lang, domain) cell (default 10).",
    )
    ap.add_argument(
        "--quadrant", default="trigger_only_refusal",
        choices=["any", "trigger_only_refusal", "concept_deep",
                 "decoupled_perception", "aligned_with_content"],
        help="Which quadrant to sample from. Default trigger_only_refusal "
             "(the cell most plausibly bypassable by paraphrase). Use 'any' "
             "for all refused prompts regardless of perception.",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = os.path.abspath(args.root_dir)
    exp2_dir = os.path.join(root, "pilot_exp2")
    out_dir = os.path.join(root, "analysis", "persuasion_seeds")
    os.makedirs(out_dir, exist_ok=True)

    results_path = os.path.join(exp2_dir, f"results_{args.mode}.json")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"{results_path} not found — run pilot_exp2 + merge first")
    with open(results_path) as f:
        results = json.load(f)

    rng = random.Random(args.seed)

    # Group eligible prompts by (lang, domain)
    candidates: Dict = defaultdict(list)
    for lang, lr in results.get("results", {}).items():
        behav_by_id = {b["id"]: b for b in lr.get("behavioural", [])}
        for pp in lr.get("per_prompt", []):
            if not pp.get("refused"):
                continue
            if args.quadrant != "any" and pp.get("quadrant") != args.quadrant:
                continue
            pid = pp["id"]
            b = behav_by_id.get(pid)
            if not b:
                continue
            candidates[(lang, pp["domain"])].append({
                "lang":              lang,
                "domain":            pp["domain"],
                "id":                pid,
                "severity_true":     pp["severity_true"],
                "severity_pred":     pp.get("severity_pred"),
                "perceived_harmful": pp.get("perceived_harmful"),
                "quadrant":          pp.get("quadrant"),
                "prompt":            b.get("prompt", ""),
                "original_refusal_response": b.get("response", ""),
            })

    if not candidates:
        print(f"No prompts matched quadrant={args.quadrant} in {results_path}")
        return

    # Cap per cell, with deterministic shuffle
    sampled: List[Dict] = []
    for cell_key, exs in candidates.items():
        rng.shuffle(exs)
        sampled.extend(exs[:args.cap_per_cell])
    sampled.sort(key=lambda r: (r["lang"], r["domain"], r["id"]))

    # Tag every output with mode + quadrant so successive runs (e.g. the
    # trigger-only set and the concept-deep control set) don't overwrite
    # each other.
    tag = f"{args.mode}_{args.quadrant}"

    # ----- CSV flat table ------------------------------------------------
    out_csv = os.path.join(out_dir, f"refused_examples_{tag}.csv")
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow([
            "lang", "domain", "id",
            "severity_true", "severity_pred_under_ablation",
            "quadrant", "perceived_harmful",
            "prompt", "original_refusal_response",
        ])
        for ex in sampled:
            w.writerow([
                ex["lang"], ex["domain"], ex["id"],
                ex["severity_true"], ex["severity_pred"],
                ex["quadrant"], ex["perceived_harmful"],
                ex["prompt"], ex["original_refusal_response"],
            ])
    print(f"Wrote {out_csv}")

    # ----- JSON grouped by language for batch paraphrasing ---------------
    by_lang: Dict[str, List[Dict]] = defaultdict(list)
    for ex in sampled:
        by_lang[ex["lang"]].append(ex)
    out_json = os.path.join(out_dir, f"refused_examples_{tag}.json")
    with open(out_json, "w") as f:
        json.dump(by_lang, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_json}")

    # ----- Counts pivot --------------------------------------------------
    cell_counts: Dict = defaultdict(int)
    for ex in sampled:
        cell_counts[(ex["lang"], ex["domain"])] += 1
    langs   = sorted({k[0] for k in cell_counts})
    domains = sorted({k[1] for k in cell_counts})

    counts_csv = os.path.join(out_dir, f"counts_pivot_{tag}.csv")
    with open(counts_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["lang"] + domains + ["row_total"])
        for lang in langs:
            row = [cell_counts.get((lang, d), 0) for d in domains]
            w.writerow([lang] + row + [sum(row)])
    print(f"Wrote {counts_csv}")

    # ----- Stdout summary ------------------------------------------------
    print(f"\nSampled {len(sampled)} prompts from quadrant={args.quadrant}, "
          f"mode={args.mode}, cap={args.cap_per_cell}/cell")
    print()
    header = "  " + f"{'lang':>5s}  " + "  ".join(f"{d[:18]:>18s}" for d in domains) + "  total"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for lang in langs:
        row = [cell_counts.get((lang, d), 0) for d in domains]
        row_str = "  ".join(f"{v:>18d}" for v in row)
        print(f"  {lang:>5s}  {row_str}  {sum(row):>5d}")

    print(f"\nAll outputs under: {out_dir}")


if __name__ == "__main__":
    main()
