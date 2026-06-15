"""
Merge per-language pilot outputs into top-level aggregates.

Each per-language slurm job writes to {ART}/pilot_exp{1,2}/by_lang/<tag>/.
This script walks those directories and concatenates the CSV summaries +
merges the results.json files into the top-level files at
{ART}/pilot_exp{1,2}/, which is what the analysis scripts read.

Safe to re-run: it overwrites the top-level aggregates.

Run:
    python -m experiments.merge_pilot_outputs
    python -m experiments.merge_pilot_outputs --root-dir /path/to/bluedot-artifacts
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Dict, List


DEFAULT_ROOT_DIR = "/project/pi_jensen_umass_edu/abhishekmish_umass_edu/bluedot-artifacts/"


# ---------------------------------------------------------------------------
# CSV merge — concat all matching per-language files (preserve header)
# ---------------------------------------------------------------------------

def merge_csvs(per_lang_paths: List[str], out_path: str) -> int:
    """Concat all matching per-language CSVs, preserving the first header."""
    if not per_lang_paths:
        return 0
    header: List[str] | None = None
    all_rows: List[List[str]] = []
    for p in sorted(per_lang_paths):
        with open(p) as f:
            reader = csv.reader(f)
            h = next(reader, None)
            if h is None:
                continue
            if header is None:
                header = h
            elif h != header:
                # Different schema between per-language files would mean a
                # mid-run code change; warn but keep going.
                print(f"  WARNING: header mismatch in {p}")
            for row in reader:
                all_rows.append(row)
    if header is None:
        return 0
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        w = csv.writer(f); w.writerow(header)
        for row in all_rows:
            w.writerow(row)
    return len(all_rows)


# ---------------------------------------------------------------------------
# JSON merge — schema differs between exp1 and exp2
# ---------------------------------------------------------------------------

def merge_jsons(per_lang_paths: List[str], out_path: str,
                container_key: str) -> int:
    """
    Merge per-language results JSON files. The shape is:
        {"meta": {...}, "<container_key>": {<sub_key>: <per-cond data>, ...}}
    where sub_key is "{lang}__{cond}" for exp1 and "{lang}" for exp2.

    Returns the number of sub-keys in the merged container.
    """
    if not per_lang_paths:
        return 0
    merged_meta: Dict = {}
    merged: Dict = {}
    seen_langs: List[str] = []
    for p in sorted(per_lang_paths):
        with open(p) as f:
            d = json.load(f)
        merged_meta = {**merged_meta, **d.get("meta", {})}
        container = d.get(container_key, {})
        for sub_key, payload in container.items():
            merged[sub_key] = payload
        # Track every language we've ingested (for the meta field)
        meta_langs = d.get("meta", {}).get("langs", [])
        for L in meta_langs:
            if L not in seen_langs:
                seen_langs.append(L)
    if seen_langs:
        merged_meta["langs"] = seen_langs

    out = {"meta": merged_meta, container_key: merged}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return len(merged)


# ---------------------------------------------------------------------------
# Per-experiment merge
# ---------------------------------------------------------------------------

def merge_exp(exp_root: str, *, container_key: str,
              csv_basenames: List[str], json_basenames: List[str]) -> None:
    by_lang_root = os.path.join(exp_root, "by_lang")
    if not os.path.isdir(by_lang_root):
        print(f"  no by_lang/ under {exp_root} — nothing to merge")
        return

    lang_dirs = [d for d in sorted(os.listdir(by_lang_root))
                 if os.path.isdir(os.path.join(by_lang_root, d))]
    if not lang_dirs:
        print(f"  no per-language directories under {by_lang_root}")
        return

    print(f"  per-language directories found: {lang_dirs}")

    for bn in csv_basenames:
        per_lang = [os.path.join(by_lang_root, d, bn) for d in lang_dirs
                    if os.path.exists(os.path.join(by_lang_root, d, bn))]
        out = os.path.join(exp_root, bn)
        n = merge_csvs(per_lang, out)
        if per_lang:
            print(f"    wrote {out}  ({n} rows from {len(per_lang)} langs)")

    for bn in json_basenames:
        per_lang = [os.path.join(by_lang_root, d, bn) for d in lang_dirs
                    if os.path.exists(os.path.join(by_lang_root, d, bn))]
        out = os.path.join(exp_root, bn)
        n = merge_jsons(per_lang, out, container_key=container_key)
        if per_lang:
            print(f"    wrote {out}  ({n} {container_key} entries "
                  f"from {len(per_lang)} langs)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-dir", default=DEFAULT_ROOT_DIR)
    args = ap.parse_args()
    root = os.path.abspath(args.root_dir)

    print(f"Merging pilot outputs under {root}\n")

    # ----- Exp 1
    print("[pilot_exp1]")
    merge_exp(
        os.path.join(root, "pilot_exp1"),
        container_key="conditions",
        csv_basenames=[
            "summary_en.csv", "summary_native.csv",
            "summary_by_contrast_en.csv", "summary_by_contrast_native.csv",
        ],
        json_basenames=["results_en.json", "results_native.json"],
    )

    print("\n[pilot_exp2]")
    merge_exp(
        os.path.join(root, "pilot_exp2"),
        container_key="results",
        csv_basenames=[
            "summary_en.csv", "summary_native.csv",
            "per_prompt_en.csv", "per_prompt_native.csv",
        ],
        json_basenames=["results_en.json", "results_native.json"],
    )

    print(f"\nDone. Analysis scripts can now run from {root}.")


if __name__ == "__main__":
    main()
