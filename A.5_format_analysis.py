"""
A.5_format_analysis.py

Read-only diagnostics for msstats CSV formatting/alignment.
This script does NOT modify any source files.

What it checks per msstats file:
- Header presence and required columns
- Row field-count mismatches vs header length
- Whether mismatches likely come from unquoted commas in Condition
- Non-numeric values in BioReplicate / Run
- ProteinName quality (protein-like vs numeric-like)

Outputs:
- Console summary
- data_preparation_output/format_analysis/format_analysis_summary.csv
- data_preparation_output/format_analysis/format_analysis_report.txt
"""

from pathlib import Path
import csv
import re
from collections import Counter
import pandas as pd


WORK_DIR = Path(r"G:\My Drive\Ikasketak\Postdoc\Cambridge\Cursor\blood_proteome_analysis")
MSSTATS_DIR = WORK_DIR / "msstats"
OUT_DIR = WORK_DIR / "data_preparation_output" / "format_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_COLUMNS = [
    "ProteinName",
    "PeptideSequence",
    "Condition",
    "BioReplicate",
    "Run",
    "Intensity",
]

NUMERIC_RE = re.compile(r"^[+-]?\d+(\.\d+)?(e[+-]?\d+)?$", re.IGNORECASE)
PROTEIN_LIKE_RE = re.compile(r"\b(sp|tr)\|[A-Z0-9]{6,10}\|")


def is_numeric_like(value: str) -> bool:
    if value is None:
        return False
    s = str(value).strip()
    if s == "":
        return False
    return bool(NUMERIC_RE.match(s))


def analyze_file(csv_path: Path) -> dict:
    result = {
        "dataset": csv_path.stem.replace(".sdrf_openms_design_msstats_in", ""),
        "file": csv_path.name,
        "status": "ok",
        "n_rows_scanned": 0,
        "header_cols": 0,
        "missing_required_columns": "",
        "row_len_mismatch_rows": 0,
        "row_len_gt_header_rows": 0,
        "row_len_lt_header_rows": 0,
        "condition_comma_likely_rows": 0,
        "non_numeric_bioreplicate_rows": 0,
        "non_numeric_run_rows": 0,
        "protein_numeric_like_rows": 0,
        "protein_proteinlike_rows": 0,
        "sample_issue_examples": "",
        "recommended_manual_action": "",
    }

    examples = []
    max_scan = 100_000

    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, quotechar='"')
            header = next(reader, None)
            if header is None:
                result["status"] = "empty_file"
                result["recommended_manual_action"] = "File is empty; regenerate with A."
                return result

            header = [h.strip() for h in header]
            result["header_cols"] = len(header)
            header_l = [h.lower() for h in header]
            hmap = {h.lower(): i for i, h in enumerate(header)}

            missing = [c for c in REQUIRED_COLUMNS if c.lower() not in hmap]
            result["missing_required_columns"] = ";".join(missing)
            if missing:
                result["status"] = "missing_columns"
                result["recommended_manual_action"] = (
                    "Missing required column(s): " + ", ".join(missing) + ". Re-export/fix header."
                )

            cond_idx = hmap.get("condition", None)
            biorep_idx = hmap.get("bioreplicate", None)
            run_idx = hmap.get("run", None)
            prot_idx = hmap.get("proteinname", None)

            for i, row in enumerate(reader, start=2):
                if i > max_scan + 1:
                    break
                result["n_rows_scanned"] += 1

                if len(row) != len(header):
                    result["row_len_mismatch_rows"] += 1
                    if len(row) > len(header):
                        result["row_len_gt_header_rows"] += 1
                    else:
                        result["row_len_lt_header_rows"] += 1

                    if cond_idx is not None and len(row) > len(header):
                        # Heuristic: extra columns likely belong to Condition when commas are unquoted
                        if cond_idx < len(row):
                            cond_token = str(row[cond_idx]).strip().lower()
                            if "|" in cond_token or "cell" in cond_token or "plasma" in cond_token or "serum" in cond_token:
                                result["condition_comma_likely_rows"] += 1

                if biorep_idx is not None and biorep_idx < len(row):
                    bi = str(row[biorep_idx]).strip()
                    if not is_numeric_like(bi):
                        result["non_numeric_bioreplicate_rows"] += 1
                        if len(examples) < 5:
                            examples.append(f"L{i}: BioReplicate='{bi}'")

                if run_idx is not None and run_idx < len(row):
                    runv = str(row[run_idx]).strip()
                    if not is_numeric_like(runv):
                        result["non_numeric_run_rows"] += 1
                        if len(examples) < 5:
                            examples.append(f"L{i}: Run='{runv}'")

                if prot_idx is not None and prot_idx < len(row):
                    prot = str(row[prot_idx]).strip()
                    if is_numeric_like(prot):
                        result["protein_numeric_like_rows"] += 1
                        if len(examples) < 5:
                            examples.append(f"L{i}: ProteinName='{prot}'")
                    if PROTEIN_LIKE_RE.search(prot):
                        result["protein_proteinlike_rows"] += 1

    except Exception as e:
        result["status"] = "read_error"
        result["recommended_manual_action"] = f"Could not parse file ({e}). Re-export or check encoding/CSV delimiters."
        return result

    # Decide status + recommendation
    issues = []
    if result["row_len_mismatch_rows"] > 0:
        issues.append("row_length_mismatch")
    # Run can legitimately be text sample IDs in many MSStats files.
    # Treat non-numeric BioReplicate as the stronger alignment signal.
    if result["non_numeric_bioreplicate_rows"] > 0:
        issues.append("biorep_non_numeric")
    if result["protein_numeric_like_rows"] > 0 and result["protein_proteinlike_rows"] == 0:
        issues.append("proteinname_numeric_like")

    if result["status"] == "ok" and issues:
        result["status"] = "needs_attention"

    if result["recommended_manual_action"] == "":
        if "row_length_mismatch" in issues:
            if result["condition_comma_likely_rows"] > 0:
                result["recommended_manual_action"] = (
                    "Rows have extra fields likely due to unquoted commas in Condition. "
                    "Manually quote Condition values containing commas or replace inner commas."
                )
            else:
                result["recommended_manual_action"] = (
                    "Rows have inconsistent field counts. Inspect delimiters/quotes and align to header."
                )
        elif "biorep_non_numeric" in issues:
            result["recommended_manual_action"] = (
                "BioReplicate contains text. Verify column alignment. "
                "For malformed rows, shift fragments into Condition until BioReplicate is numeric."
            )
        elif "proteinname_numeric_like" in issues:
            result["recommended_manual_action"] = (
                "ProteinName appears numeric-like. Verify column alignment; ProteinName should contain IDs like sp|...|..."
            )
        else:
            result["recommended_manual_action"] = "No format issues detected."

    result["sample_issue_examples"] = " | ".join(examples)
    return result


def main():
    files = sorted(MSSTATS_DIR.glob("*.sdrf_openms_design_msstats_in.csv"))
    if not files:
        print(f"No msstats files found in: {MSSTATS_DIR}")
        return

    print("=" * 80)
    print("A.5 FORMAT ANALYSIS (READ-ONLY)")
    print("=" * 80)
    print(f"Scanning {len(files)} files...\n")

    all_results = []
    status_counter = Counter()
    for i, fp in enumerate(files, start=1):
        dataset = fp.stem.replace(".sdrf_openms_design_msstats_in", "")
        print(f"[ {i} / {len(files)} ] {dataset}...")
        r = analyze_file(fp)
        all_results.append(r)
        status_counter[r["status"]] += 1

        if r["status"] != "ok":
            print(
                f"  -> {r['status']} | mismatches={r['row_len_mismatch_rows']}, "
                f"non-num BioRep={r['non_numeric_bioreplicate_rows']}, "
                f"non-num Run={r['non_numeric_run_rows']}, "
                f"prot numeric={r['protein_numeric_like_rows']}"
            )

    df = pd.DataFrame(all_results)
    df = df.sort_values(by=["status", "dataset"]).reset_index(drop=True)

    summary_csv = OUT_DIR / "format_analysis_summary.csv"
    df.to_csv(summary_csv, index=False, encoding="utf-8")

    report_txt = OUT_DIR / "format_analysis_report.txt"
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("A.5 FORMAT ANALYSIS REPORT (READ-ONLY)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Files scanned: {len(files)}\n")
        for k in sorted(status_counter.keys()):
            f.write(f"  {k}: {status_counter[k]}\n")
        f.write("\nFiles needing attention:\n")
        needs = df[df["status"] != "ok"]
        if len(needs) == 0:
            f.write("  None\n")
        else:
            for _, row in needs.iterrows():
                f.write(
                    f"- {row['dataset']}: status={row['status']}; "
                    f"mismatch={row['row_len_mismatch_rows']}; "
                    f"non-num BioRep={row['non_numeric_bioreplicate_rows']}; "
                    f"non-num Run={row['non_numeric_run_rows']}; "
                    f"prot numeric={row['protein_numeric_like_rows']}\n"
                )
                if row["sample_issue_examples"]:
                    f.write(f"    examples: {row['sample_issue_examples']}\n")
                if row["recommended_manual_action"]:
                    f.write(f"    action: {row['recommended_manual_action']}\n")

    print("\nDone.")
    print(f"Summary CSV: {summary_csv}")
    print(f"Report TXT : {report_txt}")
    print("=" * 80)


if __name__ == "__main__":
    main()

