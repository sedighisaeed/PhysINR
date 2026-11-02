#!/usr/bin/env python3
"""Cross-check every number the manuscript quotes against the generated tables.

The risk this script guards against is a prose claim drifting from the table it
cites: §4.2 was rewritten by hand, and a digit transposed in that rewrite would
be invisible to a reader. Every numeric claim in the results and discussion is
therefore looked up in the table file it names, and the script fails if the
value is absent.

Claims are (value, metric, method, table-file). The script looks up the
method's row in that table and checks the value appears in the column for that
metric -- formatted either as the leading mean (``23.654±4.083``) or as the
delta in the ablation table (``−3.55``). Numbers that are arithmetic on table
values (``Δ = 7.95``) are excluded and checked by eye instead.

Usage:  python3 experiments/check_manuscript_numbers.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TAB = REPO / "paper" / "04_tables"

# (value_as_written, metric, method, table file)
CLAIMS = [
    # --- UIEB main comparison (Table 1), n=134
    ("23.654", "psnr", "UIEC2-Net", "comparison_UIEB.md"),
    ("16.750", "psnr", "PhysINR-Zero", "comparison_UIEB.md"),
    ("22.956", "psnr", "Five-A+", "comparison_UIEB.md"),
    ("22.712", "psnr", "Water-Net", "comparison_UIEB.md"),
    ("0.910", "ssim", "Five-A+", "comparison_UIEB.md"),
    ("0.786", "ssim", "PhysINR-Zero", "comparison_UIEB.md"),
    ("0.127", "lpips_alex", "UIEC2-Net", "comparison_UIEB.md"),
    ("0.261", "lpips_alex", "PhysINR-Zero", "comparison_UIEB.md"),
    ("0.143", "lpips_vgg", "Five-A+", "comparison_UIEB.md"),
    ("0.285", "lpips_vgg", "PhysINR-Zero", "comparison_UIEB.md"),
    ("27.416", "uciqe", "PhysINR-Zero", "comparison_UIEB.md"),
    ("31.663", "uciqe", "Retinex", "comparison_UIEB.md"),
    ("15.099", "psnr", "Retinex", "comparison_UIEB.md"),
    # --- UICM components quoted in §4.2 and §4.5
    ("0.252", "uicm", "PhysINR-Zero", "comparison_UIEB.md"),
    ("0.519", "uicm", "UDCP", "comparison_UIEB.md"),
    ("0.283", "uicm", "Retinex", "comparison_UIEB.md"),
    ("0.248", "uism", "PhysINR-Zero", "comparison_UIEB.md"),
    ("0.264", "uiconm", "PhysINR-Zero", "comparison_UIEB.md"),
    # --- LSUI (Table 2)
    ("25.611", "uciqe", "PhysINR-Zero", "comparison_LSUI.md"),
    ("17.943", "psnr", "PhysINR-Zero", "comparison_LSUI.md"),
    ("25.819", "psnr", "UIEC2-Net", "comparison_LSUI.md"),
    # --- EUVP (Table 3)
    ("29.883", "uciqe", "PhysINR-Zero", "comparison_EUVP.md"),
    ("15.695", "psnr", "PhysINR-Zero", "comparison_EUVP.md"),
    ("20.071", "psnr", "FUnIE-GAN", "comparison_EUVP.md"),
    # --- RUIE (unpaired)
    ("27.910", "uciqe", "PhysINR-Zero", "comparison_RUIE.md"),
    # --- UCCS (unpaired)
    ("20.554", "uciqe", "PhysINR-Zero", "comparison_UCCS.md"),
    # --- ablation (Table 4): deltas as written, full-model subset mean
    ("16.190", "psnr", "PhysINR-Zero", "ablation_UIEB.md"),
    # --- prior design study (Table 6): §4.5 prose values
    ("0.165", "clip (G)", "A: water-type lookup", "revision_v4.md"),
    ("29.108", "uciqe", "A: water-type lookup", "revision_v4.md"),
    ("19.557", "uciqe", "raw input", "revision_v4.md"),
    ("0.003", "clip (G)", "intermediate: content head", "revision_v4.md"),
    ("20.508", "uciqe", "intermediate: content head", "revision_v4.md"),
    ("10.483", "cast", "intermediate: content head", "revision_v4.md"),
    ("16.385", "uciqe", "intermediate: content head", "revision_v4.md",
     "UCCS, n = 12"),
    ("31.185", "cast", "D: + level, calibrated", "revision_v4.md",
     "UCCS, n = 12"),
    ("35.185", "cast", "raw input", "revision_v4.md", "UCCS, n = 12"),
    ("43.076", "cast", "A: water-type lookup", "revision_v4.md",
     "UCCS, n = 12"),
    ("0.7781", "ssim", "D: + level, calibrated", "revision_v4.md"),
    ("10.548", "cast", "D: + level, calibrated", "revision_v4.md"),
]

# Metrics whose column header in the markdown table differs from the key above.
COLUMN_FOR = {"lpips_alex": "lpips_alex", "lpips_vgg": "lpips_vgg"}


def _row_for(text: str, method: str) -> str:
    # A row starts at the beginning of a line with the method name, followed
    # by whitespace or the fallback tag. Method names may be bolded (**name**)
    # as in revision_v4.md.
    m = re.search(rf"^\|\s*\**{re.escape(method)}\**(?:\s*\[[^\]]*\])?\s*\|",
                  text, re.MULTILINE)
    return m.group(0) if m else ""


def _cell(header_text: str, rows_text: str, method: str, metric: str) -> str:
    """Extract the mean±std cell for (method, metric) from a markdown table.

    ``header_text`` is the whole table -- its first ``|...|`` line is the
    header. ``rows_text`` is where the method row is looked up, which may be a
    later block of the same table (revision_v4.md repeats the sides under
    per-dataset section labels).
    """
    header_match = re.search(r"^\|(.*)\|$", header_text, re.MULTILINE)
    if not header_match:
        return ""
    headers = [c.strip() for c in header_match.group(1).split("|")]
    want = COLUMN_FOR.get(metric, metric)
    # Header case differs across tables (psnr vs PSNR, uciqe vs UCIQE).
    names = {h.lower(): h for h in headers if h}
    if want.lower() not in names:
        return ""
    col_idx = headers.index(names[want.lower()])

    row_match = re.search(
        rf"^(\|\s*\**{re.escape(method)}\**(?:\s*\[[^\]]*\])?\s*\|.*)$",
        rows_text, re.MULTILINE)
    if not row_match:
        return ""
    # Split the whole line, pipes included, so the method name is cell 0 and
    # the metric columns line up with the header indices.
    cells = [c.strip() for c in row_match.group(1).split("|")[1:-1]]
    return cells[col_idx] if col_idx < len(cells) else ""


def main() -> int:
    cache: dict = {}
    failures: list = []
    checked = 0
    for claim in CLAIMS:
        value, metric, method, table = claim[:4]
        section = claim[4] if len(claim) > 4 else None
        if table not in cache:
            path = TAB / table
            if not path.is_file():
                failures.append(f"{table}: file missing")
                continue
            cache[table] = path.read_text(encoding="utf-8")
        full = cache[table]
        # revision_v4.md lists every side twice (UIEB block then UCCS block);
        # a section label restricts the row lookup to the block below it while
        # the header still comes from the top of the table.
        rows = full
        if section:
            m = re.search(rf"^\|\s*\**{re.escape(section)}\**.*$",
                          full, re.MULTILINE)
            if not m:
                failures.append(f"{table}: section '{section}' not found")
                continue
            rows = full[m.start():]
        if not _row_for(rows, method):
            failures.append(f"{table}: no row for '{method}'"
                            + (f" in section '{section}'" if section else ""))
            continue
        cell = _cell(full, rows, method, metric)
        if not cell:
            failures.append(f"{table}: no column '{metric}' for '{method}'")
            continue
        checked += 1
        mean = cell.split("±")[0].strip()
        if value not in mean:
            failures.append(
                f"{table}: prose says {method} {metric} = {value}, "
                f"table has {mean}")

    # --- tau claims in §3.2 and §4.5 come from the probe, not the tables ---
    probe_path = REPO / "experiments" / "tau_meta_init_probe.json"
    if not probe_path.is_file():
        failures.append("tau_meta_init_probe.json: run experiments/"
                        "probe_tau_meta_init.py to regenerate it")
    else:
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
        u100 = probe.get("uieb 100", {})
        # (value in prose, probe key, prose section)
        tau_claims = [
            ("1.11", "tau_G", "§3.2/§4.5: τ_G at meta-init"),
            ("0.34", "t_G", "§3.2/§4.5: t_G = exp(-τ_G)"),
            ("70.5", "saturate_hi_G", "§4.5: green pinned at 1.0, step 0"),
            ("1.6", "saturate_lo_R", "§4.5: red channel at 0, step 0"),
        ]
        for value, key, where in tau_claims:
            if key not in u100:
                failures.append(f"{where}: probe key '{key}' missing")
                continue
            checked += 1
            measured = f"{u100[key]:.1f}" if "saturate" in key else \
                f"{u100[key]:.2f}"
            if value != measured:
                failures.append(
                    f"{where}: prose says {value}, probe measures {measured}")

    # --- Lab metrics in §4.5 come from the v3 design-study run, not the tables
    lab_path = REPO / "experiments" / "v3_fix_results.json"
    if not lab_path.is_file():
        failures.append("v3_fix_results.json: missing (§4.5 Lab claims "
                        "unverifiable)")
    else:
        lab = json.loads(lab_path.read_text(encoding="utf-8"))["records"]
        # The run recorded 20 UIEB frames with reference images; it has no
        # "set" field, and every record is UIEB (verified by filename).
        # (value_as_written, side, field, decimals)
        lab_claims = [
            ("10.55", "v4", "chroma", 2),        # D's chroma vs reference 11.84
            ("49.76", "v4", "L", 2),             # D's L* vs reference 49.07
            ("0.401", "v4", "meanR", 3),         # D's mean red vs reference 0.413
            ("11.22", "v4", "dE", 2),            # D's Delta E (lowest of the four)
            ("0.7781", "v4", "ssim", 4),
            ("11.84", "reference", "chroma", 2),
            ("49.07", "reference", "L", 2),
            ("0.413", "reference", "meanR", 3),
            ("11.37", "v2", "chroma", 2),        # B: arrives by desaturating
            ("43.95", "v2", "L", 2),
            ("0.336", "v2", "meanR", 3),
            ("15.84", "v3", "dE", 2),            # C: worst on every metric
            ("0.659", "v3", "ssim", 3),
            ("0.7229", "input", "ssim", 4),      # the input's own SSIM
        ]
        for value, side, field, nd in lab_claims:
            if side not in lab[0] or field not in lab[0][side]:
                failures.append(f"§4.5 Lab: side '{side}' field '{field}' "
                                f"missing from v3_fix_results.json")
                continue
            checked += 1
            vals = [r[side][field] for r in lab
                    if side in r and r[side].get(field) is not None]
            measured = f"{sum(vals) / len(vals):.{nd}f}"
            if value != measured:
                failures.append(
                    f"§4.5 Lab: prose says {side} {field} = {value}, "
                    f"v3_fix_results.json mean is {measured}")

    print(f"checked {checked} numeric claim(s) against the generated tables")
    if failures:
        print(f"\n{len(failures)} mismatch(es):")
        for f in failures:
            print(f"  FAIL {f}")
        return 1
    print("every checked number matches its table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
