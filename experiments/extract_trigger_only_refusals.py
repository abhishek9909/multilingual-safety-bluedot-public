"""
Extract every prompt that landed in the `trigger_only_refusal` quadrant from
the Exp 2 outputs, with its metadata, the actual behavioural-pass response
(the refusal), and the actual perception-pass response (the classification
that said "not harmful").

The quadrant means the model refused the user's prompt at the behavioural
pass AND judged it as not harmful (L0/L1) at the perception pass under
refusal ablation. This is the "shallow safety" cell — refusal fired without
the model recognising the harm.

Reads:
    {ART}/pilot_exp2/results_{en,native}.json

Writes:
    {ART}/analysis/trigger_only_refusal/
      examples.csv                  - one row per example, prompts truncated
      examples.json                 - same examples with full prompt + responses
      counts_by_lang_domain.csv     - count summary by (judge_lang, lang) x
                                      domain x severity_true x severity_pred
      by_lang/{mode}_{lang}.json    - per-(mode, lang) file for inspection
    plus a stdout summary

Run:
    python -m experiments.extract_trigger_only_refusals
    python -m experiments.extract_trigger_only_refusals --modes en native
    python -m experiments.extract_trigger_only_refusals \\
        --root-dir /project/.../bluedot-artifacts \\
        --max-prompt-chars 300
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional


DEFAULT_ROOT_DIR = "/project/pi_jensen_umass_edu/abhishekmish_umass_edu/bluedot-artifacts/"


def _load_results(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        print(f"  missing: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def _trunc(s: str, n: int) -> str:
    s = (s or "").replace("\n", " ").replace("\r", " ").strip()
    return s if len(s) <= n else s[:n] + "..."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-dir", default=DEFAULT_ROOT_DIR)
    ap.add_argument(
        "--modes", nargs="+", default=["en", "native"],
        help="Which judge-language modes to extract (default: both).",
    )
    ap.add_argument(
        "--max-prompt-chars", type=int, default=240,
        help="Truncate prompts and responses to this length in the combined CSV.",
    )
    args = ap.parse_args()

    root = os.path.abspath(args.root_dir)
    exp2_dir = os.path.join(root, "pilot_exp2")
    out_dir = os.path.join(root, "analysis", "trigger_only_refusal")
    os.makedirs(out_dir, exist_ok=True)
    by_lang_dir = os.path.join(out_dir, "by_lang")
    os.makedirs(by_lang_dir, exist_ok=True)

    # ----- Collect examples -----------------------------------------------
    all_examples: List[Dict] = []
    counts: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )

    for mode in args.modes:
        results = _load_results(os.path.join(exp2_dir, f"results_{mode}.json"))
        if not results:
            continue
        print(f"\n[mode = {mode}]")
        for lang, lr in results.get("results", {}).items():
            # Index the two pass-specific arrays so we can join by id
            behav_by_id = {b["id"]: b for b in lr.get("behavioural", [])}
            perc_by_id  = {p["id"]: p for p in lr.get("perception", [])}

            n_lang = 0
            for pp in lr.get("per_prompt", []):
                if pp.get("quadrant") != "trigger_only_refusal":
                    continue
                pid = pp["id"]
                b = behav_by_id.get(pid, {})
                p = perc_by_id.get(pid,  {})
                ex = {
                    "lang":            lang,
                    "judge_language":  mode,
                    "id":              pid,
                    "domain":          pp.get("domain"),
                    "severity_true":   pp.get("severity_true"),
                    "severity_pred":   pp.get("severity_pred"),
                    "perceived_harmful": pp.get("perceived_harmful"),
                    "refused":         pp.get("refused"),
                    "prompt":          b.get("prompt") or p.get("prompt") or "",
                    "behavioural_response": b.get("response", ""),
                    "perception_response":  p.get("response", ""),
                    "reasoning_lang_match":         p.get("reasoning_lang_match"),
                    "reasoning_endorses_severity":  p.get("reasoning_endorses_severity"),
                    "reasoning_action_match":       p.get("reasoning_action_match"),
                }
                all_examples.append(ex)
                n_lang += 1

                # Per-(mode, lang) count tallies for the summary CSV.
                counts[mode][lang]["total"] += 1
                if ex["domain"]:        counts[mode][lang][f"domain={ex['domain']}"]        += 1
                if ex["severity_true"]: counts[mode][lang][f"severity_true={ex['severity_true']}"] += 1
                if ex["severity_pred"]: counts[mode][lang][f"severity_pred={ex['severity_pred']}"] += 1

            print(f"  {lang:>4s}: {n_lang} trigger_only_refusal examples")

    if not all_examples:
        print("\nNo trigger_only_refusal examples found. Check that "
              "{ART}/pilot_exp2/results_{en,native}.json exist.")
        return

    # ----- Write full per-example JSON ------------------------------------
    out_json = os.path.join(out_dir, "examples.json")
    with open(out_json, "w") as f:
        json.dump(all_examples, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out_json}  ({len(all_examples)} examples)")

    # ----- Write combined CSV (truncated) ---------------------------------
    out_csv = os.path.join(out_dir, "examples.csv")
    n_chars = args.max_prompt_chars
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow([
            "judge_lang", "lang", "id", "domain",
            "severity_true", "severity_pred", "perceived_harmful", "refused",
            "prompt", "behavioural_response", "perception_response",
            "reasoning_lang_match", "reasoning_endorses_severity",
            "reasoning_action_match",
        ])
        for ex in all_examples:
            w.writerow([
                ex["judge_language"], ex["lang"], ex["id"], ex["domain"],
                ex["severity_true"], ex["severity_pred"],
                ex["perceived_harmful"], ex["refused"],
                _trunc(ex["prompt"],               n_chars),
                _trunc(ex["behavioural_response"], n_chars),
                _trunc(ex["perception_response"],  n_chars),
                ex["reasoning_lang_match"],
                ex["reasoning_endorses_severity"],
                ex["reasoning_action_match"],
            ])
    print(f"Wrote {out_csv}")

    # ----- Aggregate counts ----------------------------------------------
    all_keys = set()
    for mode_d in counts.values():
        for lang_d in mode_d.values():
            all_keys.update(lang_d)
    all_keys = sorted(all_keys, key=lambda k: (
        0 if k == "total" else 1,
        k,
    ))

    out_counts = os.path.join(out_dir, "counts_by_lang_domain.csv")
    with open(out_counts, "w") as f:
        w = csv.writer(f)
        w.writerow(["judge_lang", "lang"] + all_keys)
        for mode in args.modes:
            for lang, lang_d in sorted(counts.get(mode, {}).items()):
                w.writerow([mode, lang] + [lang_d.get(k, 0) for k in all_keys])
    print(f"Wrote {out_counts}")

    # ----- Per-(mode, lang) JSON for deep inspection ---------------------
    grouped: Dict[tuple, List[Dict]] = defaultdict(list)
    for ex in all_examples:
        grouped[(ex["judge_language"], ex["lang"])].append(ex)
    for (mode, lang), exs in grouped.items():
        with open(os.path.join(by_lang_dir, f"{mode}_{lang}.json"), "w") as f:
            json.dump(exs, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(grouped)} per-language files into {by_lang_dir}/")

    # ----- Console summary -----------------------------------------------
    print("\n=== Summary ===")
    for mode in args.modes:
        if mode not in counts:
            continue
        print(f"\n[mode = {mode}]")
        for lang in sorted(counts[mode]):
            total = counts[mode][lang]["total"]
            dom_counts = {
                k.replace("domain=", ""): v for k, v in counts[mode][lang].items()
                if k.startswith("domain=")
            }
            sev_counts = {
                k.replace("severity_true=", ""): v for k, v in counts[mode][lang].items()
                if k.startswith("severity_true=")
            }
            top_doms = sorted(dom_counts.items(), key=lambda kv: -kv[1])[:3]
            top_sev  = sorted(sev_counts.items(), key=lambda kv: -kv[1])
            doms_str = ", ".join(f"{d}={c}" for d, c in top_doms)
            sev_str  = ", ".join(f"{s}={c}" for s, c in top_sev)
            print(f"  {lang}: n={total:3d}  top domains: {doms_str}")
            print(f"         severity_true distribution: {sev_str}")

    print(f"\nAll outputs under: {out_dir}")


if __name__ == "__main__":
    main()
