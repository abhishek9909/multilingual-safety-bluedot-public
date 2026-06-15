"""
Domain-level breakdown of trigger-only-refusal cases.

Counts every prompt in the Exp 2 outputs and groups by (mode, language,
domain), producing trigger-only rates as a fraction of *all prompts in
that cell*. That ratio is what you actually want to know — absolute counts
depend on how many prompts of each domain were sampled per language, which
varies because the proportional sampler preserves the source distribution.

Reads:
    {ART}/pilot_exp2/per_prompt_{en,native}.csv

Writes:
    {ART}/analysis/trigger_only_by_domain/
      by_mode_domain.csv         - per (mode, domain) aggregate over all langs
      by_mode_lang_domain.csv    - per (mode, lang, domain) detail
      rate_pivot_{mode}.csv      - rows=lang, cols=domain, values=rate
      severity_split.csv         - trigger-only rate split by severity_true
                                   (over-refusal vs missed-harm patterns)
    stdout: domain ranking by trigger-only rate

Run:
    python -m experiments.analyze_trigger_only_by_domain
    python -m experiments.analyze_trigger_only_by_domain --modes en
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List


DEFAULT_ROOT_DIR = "/project/pi_jensen_umass_edu/abhishekmish_umass_edu/bluedot-artifacts/"


def _load_csv(path: str) -> List[Dict]:
    if not os.path.exists(path):
        print(f"  missing: {path}")
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _save_csv(rows: List[Dict], path: str, fieldnames: List[str] | None = None) -> None:
    if not rows:
        print(f"  (no rows for {path})")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"  wrote {path}  ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-dir", default=DEFAULT_ROOT_DIR)
    ap.add_argument("--modes", nargs="+", default=["en", "native"])
    args = ap.parse_args()
    root = os.path.abspath(args.root_dir)

    exp2_dir = os.path.join(root, "pilot_exp2")
    out_dir = os.path.join(root, "analysis", "trigger_only_by_domain")
    os.makedirs(out_dir, exist_ok=True)

    # cell[(mode, lang, domain)] = {"total": n, "trigger_only": k, plus per-severity totals + trigger_only}
    cell: Dict = defaultdict(lambda: {"total": 0, "trigger_only": 0,
                                      "total_by_sev": defaultdict(int),
                                      "trigger_only_by_sev": defaultdict(int)})
    all_langs = set(); all_domains = set()

    for mode in args.modes:
        path = os.path.join(exp2_dir, f"per_prompt_{mode}.csv")
        rows = _load_csv(path)
        if not rows: continue
        print(f"[{mode}] read {len(rows)} per-prompt rows from {path}")
        for r in rows:
            lang   = r.get("lang", "")
            domain = r.get("domain", "")
            sev    = r.get("severity_true", "")
            quad   = r.get("quadrant", "")
            if not (lang and domain): continue
            all_langs.add(lang); all_domains.add(domain)
            c = cell[(mode, lang, domain)]
            c["total"] += 1
            c["total_by_sev"][sev] += 1
            if quad == "trigger_only_refusal":
                c["trigger_only"] += 1
                c["trigger_only_by_sev"][sev] += 1

    if not cell:
        print("\nNo per_prompt CSVs found. Run pilot_exp2 + merge first.")
        return

    langs   = sorted(all_langs)
    domains = sorted(all_domains)

    # -------------------------------------------------------------------
    # (mode, lang, domain) detail
    # -------------------------------------------------------------------
    detail_rows: List[Dict] = []
    for (mode, lang, dom), c in cell.items():
        t = c["total"]; k = c["trigger_only"]
        detail_rows.append({
            "judge_lang": mode, "lang": lang, "domain": dom,
            "n_prompts": t, "n_trigger_only": k,
            "trigger_only_rate": f"{(k / t):.4f}" if t else "nan",
        })
    detail_rows.sort(key=lambda r: (r["judge_lang"], r["lang"], r["domain"]))
    _save_csv(detail_rows, os.path.join(out_dir, "by_mode_lang_domain.csv"))

    # -------------------------------------------------------------------
    # (mode, domain) aggregate over all languages
    # -------------------------------------------------------------------
    agg: Dict = defaultdict(lambda: {"total": 0, "trigger_only": 0})
    for (mode, lang, dom), c in cell.items():
        a = agg[(mode, dom)]
        a["total"]        += c["total"]
        a["trigger_only"] += c["trigger_only"]

    summary_rows: List[Dict] = []
    for (mode, dom), a in agg.items():
        summary_rows.append({
            "judge_lang": mode, "domain": dom,
            "n_prompts": a["total"], "n_trigger_only": a["trigger_only"],
            "trigger_only_rate": f"{(a['trigger_only'] / a['total']):.4f}" if a["total"] else "nan",
        })
    summary_rows.sort(key=lambda r: (r["judge_lang"], -float(r["trigger_only_rate"]) if r["trigger_only_rate"] != "nan" else 0))
    _save_csv(summary_rows, os.path.join(out_dir, "by_mode_domain.csv"))

    # -------------------------------------------------------------------
    # Rate pivot per mode: rows=lang, cols=domain, values=rate
    # -------------------------------------------------------------------
    for mode in args.modes:
        pivot_path = os.path.join(out_dir, f"rate_pivot_{mode}.csv")
        with open(pivot_path, "w") as f:
            w = csv.writer(f)
            w.writerow(["lang"] + domains + ["__all__"])
            for lang in langs:
                row = [lang]
                for dom in domains:
                    c = cell.get((mode, lang, dom))
                    if c and c["total"]:
                        row.append(f"{(c['trigger_only'] / c['total']):.4f}")
                    else:
                        row.append("nan")
                # Row total
                tot_t = sum(cell.get((mode, lang, dom), {}).get("total", 0) for dom in domains)
                tot_k = sum(cell.get((mode, lang, dom), {}).get("trigger_only", 0) for dom in domains)
                row.append(f"{tot_k / tot_t:.4f}" if tot_t else "nan")
                w.writerow(row)
        print(f"  wrote {pivot_path}")

    # -------------------------------------------------------------------
    # Severity-split: is trigger-only mainly on L0/L1 (over-refusal) or
    # on L2/L3 (missed harm)?
    # -------------------------------------------------------------------
    sev_agg: Dict = defaultdict(lambda: {"total": 0, "trigger_only": 0})
    for (mode, lang, dom), c in cell.items():
        for sev, n in c["total_by_sev"].items():
            sev_agg[(mode, dom, sev)]["total"] += n
            sev_agg[(mode, dom, sev)]["trigger_only"] += c["trigger_only_by_sev"].get(sev, 0)

    sev_rows: List[Dict] = []
    for (mode, dom, sev), a in sev_agg.items():
        sev_rows.append({
            "judge_lang": mode, "domain": dom, "severity_true": sev,
            "n_prompts": a["total"], "n_trigger_only": a["trigger_only"],
            "trigger_only_rate": f"{(a['trigger_only'] / a['total']):.4f}" if a["total"] else "nan",
        })
    sev_rows.sort(key=lambda r: (r["judge_lang"], r["domain"], r["severity_true"]))
    _save_csv(sev_rows, os.path.join(out_dir, "severity_split.csv"))

    # -------------------------------------------------------------------
    # Console summary: domain ranking by trigger-only rate per mode
    # -------------------------------------------------------------------
    print("\n=== Domain ranking by trigger-only rate (aggregated across languages) ===")
    for mode in args.modes:
        per_dom = [r for r in summary_rows if r["judge_lang"] == mode]
        if not per_dom: continue
        per_dom.sort(key=lambda r: -float(r["trigger_only_rate"]) if r["trigger_only_rate"] != "nan" else 0)
        print(f"\n[mode = {mode}]")
        print(f"  {'domain':<28s}  {'rate':>8s}  {'n_trigger':>10s}  {'n_total':>8s}")
        for r in per_dom:
            print(f"  {r['domain']:<28s}  {r['trigger_only_rate']:>8s}  "
                  f"{r['n_trigger_only']:>10d}  {r['n_prompts']:>8d}")

    # Per-severity diagnostic — split over-refusal vs missed-harm
    print("\n=== Trigger-only by severity (aggregated across languages and domains) ===")
    for mode in args.modes:
        per_sev: Dict = defaultdict(lambda: {"total": 0, "trigger_only": 0})
        for r in sev_rows:
            if r["judge_lang"] != mode: continue
            per_sev[r["severity_true"]]["total"]        += r["n_prompts"]
            per_sev[r["severity_true"]]["trigger_only"] += r["n_trigger_only"]
        if not per_sev: continue
        print(f"\n[mode = {mode}]")
        print(f"  {'severity':>10s}  {'rate':>8s}  {'n_trigger':>10s}  {'n_total':>8s}  {'interpretation':<30s}")
        for sev in sorted(per_sev, key=lambda s: ["L0","L1","L2","L3"].index(s) if s in ["L0","L1","L2","L3"] else 99):
            d = per_sev[sev]
            r = d["trigger_only"] / d["total"] if d["total"] else float("nan")
            note = {
                "L0": "over-refusal: prompt is safe by LinguaSafe, model refuses + judges low-severity",
                "L1": "over-refusal-ish",
                "L2": "potentially-correct refusal but model under-rates the harm",
                "L3": "potentially-correct refusal but model under-rates the harm",
            }.get(sev, "")
            print(f"  {sev:>10s}  {r:>8.4f}  {d['trigger_only']:>10d}  {d['total']:>8d}  {note}")

    print(f"\nAll outputs under: {out_dir}")


if __name__ == "__main__":
    main()
