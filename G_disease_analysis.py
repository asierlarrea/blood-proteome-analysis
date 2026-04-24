"""
Disease Analysis Script for Plasma/Serum Proteomes (PROTEIN-LEVEL)

This script analyzes disease-specific protein patterns in plasma/serum datasets.
Works with proteins (UniProt accessions) instead of genes.

Plots:
1. Sample and dataset composition by disease (bar plot)
2. Disease-specific protein presence (scatter plot: healthy vs disease fractions)
3. Protein sharing stacked bars (stacked bar plot: proteins shared/not shared with Normal for each disease)
4. Shared protein abundance comparison (box plot: abundance of 3 random shared proteins across diseases and healthy)

Disease labels:
- Built from **MSstats** `disease` column (when present) and/or **SDRF** TSV in `sdrf_files/`
  (columns like `characteristics[disease]`), merged per sample with caching under
  `disease_analysis/cache/` (see `sdrf_sample_disease.py`). Run **A_download_all_datasets.py**
  to fetch SDRF from the same PRIDE FTP folder as the MSstats CSV.

Methodological rules:
- Always work in fractions/percentages, never raw counts
- Do not pool samples across datasets without normalization
- Do not interpret absence as true biological absence
- Use minimum prevalence thresholds (>=10-20%) to avoid noise
- Clearly separate healthy vs disease in all plots
- Work with proteins (UniProt accessions) normalized using entry name mapping library
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict
import os
import sys
import json
import pickle
import warnings
warnings.filterwarnings('ignore')
import re
import csv

# Set working directory and add to path for imports
work_dir = Path(r"G:\My Drive\Ikasketak\Postdoc\Cambridge\Cursor\blood_proteome_analysis")
sys.path.insert(0, str(work_dir))
os.chdir(work_dir)

# Import shared utilities
from shared_utils import (
    is_decoy_entrap_protein,
    is_non_human_protein,
    find_cache_file,
    get_cache_filter_info,
    get_default_cache_parquet_suffix,
    group_condition_to_tissue,
    load_cache_config,
)
from time import perf_counter

# ---- Inlined from sdrf_sample_disease.py ----
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

# Bump when MSstats↔SDRF merge logic changes so old JSON caches are discarded
# (mtimes alone do not invalidate after code updates).
SAMPLE_DISEASE_CACHE_VERSION = 4


def _safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """
    Robust CSV reader for large/occasionally malformed MSstats exports on Windows.
    Tries C engine first, then falls back to python engine with bad-line skipping.
    """
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        fb = dict(kwargs)
        # C engine-specific option can break python engine
        fb.pop("low_memory", None)
        fb.setdefault("engine", "python")
        fb.setdefault("on_bad_lines", "skip")
        return pd.read_csv(path, **fb)


def _mtime(p: Path) -> Optional[float]:
    try:
        return p.stat().st_mtime if p.is_file() else None
    except OSError:
        return None


def normalize_match_key(s: Any) -> Optional[str]:
    """Normalize a cell value for lookup (lowercase, basename for paths)."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    t = str(s).strip()
    if not t or t.lower() in ("nan", "none", "na", ""):
        return None
    t = t.replace("\\", "/")
    if "/" in t:
        t = t.split("/")[-1]
    return t.lower()


def tokenize_cell(val: Any) -> List[str]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    s = str(val).strip()
    if not s:
        return []
    parts = re.split(r"[;|,]", s)
    return [p.strip() for p in parts if p.strip()]


def file_stem_variants(text: Any) -> List[str]:
    """
    Extract mass-spec file names and stems for joining MSstats to SDRF.

    OpenMS MSstats often puts spectrum refs in Reference like:
      'G20120724_PMI_P23_iTRAQ_basicRP_fr7.mzML_controllerType=0 ...'
    while SDRF uses the acquisition file:
      'G20120724_PMI_P23_iTRAQ_basicRP_fr7.raw'
    Stripping to the shared stem (before .mzML / .raw) links the two.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return []
    s0 = str(text).strip()
    if not s0:
        return []
    s = s0.replace("\\", "/")
    if "/" in s:
        s = s.split("/")[-1]
    sl = s.lower()
    out: List[str] = []
    seen: Set[str] = set()

    def add(x: str) -> None:
        if x and x not in seen:
            seen.add(x)
            out.append(x)

    add(sl)
    # Order matters: .mzml before .raw so both stems are added when both appear
    for ext in (".mzml", ".raw", ".mgf", ".wiff"):
        pos = sl.find(ext)
        if pos > 0:
            add(sl[:pos])
            add(sl[: pos + len(ext)])
    return out


def _add_normalized_keys(keys: Set[str], fragment: str) -> None:
    for variant in file_stem_variants(fragment):
        nk = normalize_match_key(variant)
        if nk:
            keys.add(nk)


def is_sdrf_disease_placeholder(d: str) -> bool:
    x = (d or "").strip().lower()
    return not x or x in ("not available", "na", "n/a", "unknown", "none")


def is_healthy_sdrf_label(d: str) -> bool:
    """Conservative: treat common 'normal cohort' labels as one homogeneous class."""
    x = (d or "").strip().lower()
    if not x:
        return False
    exact = {
        "healthy",
        "normal",
        "hc",
        "healthy donor",
        "healthy control",
        "normal control",
        "healthy volunteer",
        "healthy subject",
        "healthy controls",
        "normal donor",
        "negative",
        "control",
    }
    if x in exact:
        return True
    if x.startswith("healthy ") and "patient" not in x:
        return True
    if "healthy control" in x or "normal control" in x:
        return True
    return False


def infer_homogeneous_sdrf_broadcast(
    srows: List[Dict[str, str]], disease_col: str
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    If every non-placeholder SDRF disease cell is the same cohort (one disease text,
    or all healthy/normal), return that label for the whole dataset — no per-sample join.

    If two or more distinct disease cohorts appear (e.g. patient vs healthy), return
    (None, info) and caller should use per-sample SDRF matching.
    """
    info: Dict[str, Any] = {}
    cohort_tokens: List[str] = []
    rep_by_token: Dict[str, str] = {}

    for row in srows:
        raw = (row.get(disease_col) or "").strip()
        if is_sdrf_disease_placeholder(raw):
            continue
        if is_healthy_sdrf_label(raw):
            tok = "__HEALTHY__"
        else:
            tok = raw.lower()
        if tok not in rep_by_token:
            rep_by_token[tok] = raw
        cohort_tokens.append(tok)

    if not cohort_tokens:
        info["sdrf_homogeneous_check"] = "no_non_placeholder_disease_values"
        return None, info

    uniq = set(cohort_tokens)
    if len(uniq) > 1:
        info["sdrf_homogeneous_check"] = "multiple_disease_cohorts_use_per_sample_join"
        info["n_distinct_sdrf_disease_cohorts"] = len(uniq)
        return None, info

    tok = next(iter(uniq))
    label = rep_by_token[tok]
    info["sdrf_homogeneous_check"] = (
        "all_healthy_normal" if tok == "__HEALTHY__" else "single_disease_label"
    )
    info["sdrf_disease_mode"] = "homogeneous_broadcast"
    return label, info


def pick_disease_column(fieldnames: List[str]) -> Optional[str]:
    """Pick best-matching SDRF column for disease / phenotype / clinical indication."""
    best = None
    for h in fieldnames:
        hl = h.lower()
        # PRIDE SDRF often uses characteristics[disease], diagnosis, indication, etc.
        if "disease" in hl or "diagnosis" in hl:
            return h
        if "indication" in hl or "pathology" in hl or "affected" in hl:
            return h
        if "phenotype" in hl and "disease" not in (best or "").lower():
            best = h
    return best


def pick_identifier_columns(fieldnames: List[str]) -> List[str]:
    """Columns that may contain sample/run/file ids linkable to MSstats."""
    hints = (
        "source name",
        "comment[proteomics internal sample id]",
        "comment[label]",
        "comment[data file uri]",
        "comment[data file]",
        "comment[instrument file name]",
        "comment[technical replicate id]",
        "comment[sample id]",
        "comment[title]",
        "characteristics[biological replicate]",
        "comment[subject id]",
        "comment[experiment label]",
    )
    out = []
    for h in fieldnames:
        hl = h.lower().strip()
        if h in hints:
            out.append(h)
            continue
        if hl.startswith("comment[") and (
            "sample" in hl or "file" in hl or "replicate" in hl or "label" in hl
        ):
            out.append(h)
    # Always include source name if present
    for h in fieldnames:
        if h.lower() == "source name" and h not in out:
            out.insert(0, h)
    return list(dict.fromkeys(out))


def parse_sdrf_tsv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames or []
        for row in reader:
            rows.append({k: (v if v is not None else "") for k, v in row.items()})
    return list(fieldnames), rows


def build_sdrf_key_to_disease(
    rows: List[Dict[str, str]], disease_col: str, id_cols: List[str]
) -> Dict[str, str]:
    """Map normalized lookup keys -> merged disease string (| if multiple)."""
    acc: Dict[str, Set[str]] = defaultdict(set)
    for row in rows:
        d = (row.get(disease_col) or "").strip()
        if is_sdrf_disease_placeholder(d):
            continue
        keys: Set[str] = set()
        for col in id_cols:
            raw = row.get(col, "")
            for tok in tokenize_cell(raw):
                _add_normalized_keys(keys, tok)
            _add_normalized_keys(keys, raw)
        for k in keys:
            acc[k].add(d)
    return {k: " | ".join(sorted(v)) for k, v in acc.items()}


def build_homogeneous_map_from_sdrf_rows(
    rows: List[Dict[str, str]], id_cols: List[str], broadcast_label: str
) -> Dict[str, str]:
    """
    For a single-cohort SDRF, map every normalized id (source name, data file, label, …)
    to the same disease. Avoids reading the MSstats peptide table when MSstats has no
    `disease` column.
    """
    key_set: Set[str] = set()
    for row in rows:
        for col in id_cols:
            raw = row.get(col, "")
            for tok in tokenize_cell(raw):
                _add_normalized_keys(key_set, tok)
            _add_normalized_keys(key_set, raw)
    return {k: broadcast_label for k in key_set if k}


def sdrf_merge_classify_dataset(dataset_name: str, sdrf_dir: Path) -> Dict[str, Any]:
    """
    Parse SDRF only (no MSstats). For disease_list progress / planning.
    kind: homogeneous | heterogeneous | no_sdrf | no_disease_column | sdrf_error
    """
    out: Dict[str, Any] = {"dataset": dataset_name, "kind": "no_sdrf"}
    p = resolve_sdrf_path(dataset_name, sdrf_dir)
    if not p or not p.is_file():
        return out
    try:
        fieldnames, srows = parse_sdrf_tsv(p)
        dcol = pick_disease_column(fieldnames)
        if not dcol:
            out["kind"] = "no_disease_column"
            return out
        broadcast, info = infer_homogeneous_sdrf_broadcast(srows, dcol)
        out["sdrf_disease_column"] = dcol
        for k, v in info.items():
            out[k] = v
        if broadcast is not None:
            out["kind"] = "homogeneous"
            out["broadcast_label"] = broadcast
        else:
            out["kind"] = "heterogeneous"
    except Exception as e:
        out["kind"] = "sdrf_error"
        out["error"] = str(e)
    return out


def lookup_disease_for_msstats_ids(
    sdrf_map: Dict[str, str], *ids: Any
) -> Optional[str]:
    """Try Reference, Sample, Run (and path variants) against SDRF map."""
    seen: Set[str] = set()

    def try_key(nk: Optional[str]) -> Optional[str]:
        if not nk or nk in seen:
            return None
        seen.add(nk)
        return sdrf_map.get(nk)

    for raw in ids:
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            continue
        for tok in tokenize_cell(raw):
            for variant in file_stem_variants(tok):
                nk = normalize_match_key(variant)
                hit = try_key(nk)
                if hit is not None:
                    return hit
        for variant in file_stem_variants(raw):
            nk = normalize_match_key(variant)
            hit = try_key(nk)
            if hit is not None:
                return hit
    return None


def find_msstats_column(df: pd.DataFrame, name: str) -> Optional[str]:
    for c in df.columns:
        if str(c).strip().lower() == name.lower():
            return c
    return None


def resolve_sdrf_path(dataset_name: str, sdrf_dir: Path) -> Optional[Path]:
    """{dataset}.sdrf.tsv then base PXD*.sdrf.tsv."""
    candidates = [
        sdrf_dir / f"{dataset_name}.sdrf.tsv",
    ]
    if "-" in dataset_name:
        base = dataset_name.split("-")[0]
        candidates.append(sdrf_dir / f"{base}.sdrf.tsv")
    for p in candidates:
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def _read_msstats_merge_columns(
    msstats_path: Path, alias_cols: List[str], dis_c: Optional[str]
) -> pd.DataFrame:
    """Load only columns needed for sample↔disease merge (not full peptide table)."""
    cols = list(dict.fromkeys(alias_cols + ([dis_c] if dis_c else [])))
    return _safe_read_csv(msstats_path, usecols=cols, low_memory=False)


def _chunked_broadcast_aliases(
    msstats_path: Path, alias_cols: List[str], label: str
) -> Dict[str, str]:
    """
    Collect unique non-empty Reference/Sample/Run values via chunked CSV read.
    Used when SDRF homogeneous broadcast applies and MSstats has no `disease` column.
    """
    final: Dict[str, str] = {}
    try:
        reader = _safe_read_csv(
            msstats_path,
            usecols=alias_cols,
            chunksize=400_000,
            low_memory=False,
        )
        for chunk in reader:
            for c in alias_cols:
                for v in chunk[c].dropna().unique():
                    k = str(v).strip()
                    if k:
                        final[k] = label
    except Exception:
        df = _safe_read_csv(msstats_path, usecols=alias_cols, low_memory=False)
        for c in alias_cols:
            for v in df[c].dropna().unique():
                k = str(v).strip()
                if k:
                    final[k] = label
    return final


def build_sample_disease_map_merged(
    msstats_path: Path,
    dataset_name: str,
    sdrf_path: Optional[Path],
    conflict_log: Optional[Path] = None,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Build sample_id -> disease using MSstats + optional SDRF.

    Order: parse SDRF first. If the cohort is homogeneous and MSstats has no ``disease``
    column, build the map from SDRF identifiers only (no MSstats peptide rows). If the
    SDRF has multiple disease cohorts, read only narrow MSstats columns and join.
    """
    meta: Dict[str, Any] = {
        "dataset": dataset_name,
        "msstats": str(msstats_path),
        "sdrf": str(sdrf_path) if sdrf_path else None,
    }
    sdrf_map: Dict[str, str] = {}
    sdrf_broadcast: Optional[str] = None
    id_cols: List[str] = []
    srows: List[Dict[str, str]] = []

    # --- 1) SDRF first (small files): homogeneous vs per-sample join ---
    if sdrf_path and sdrf_path.is_file():
        try:
            fieldnames, srows = parse_sdrf_tsv(sdrf_path)
            dcol = pick_disease_column(fieldnames)
            if dcol:
                meta["sdrf_disease_column"] = dcol
                id_cols = pick_identifier_columns(fieldnames)
                sdrf_broadcast, hom_meta = infer_homogeneous_sdrf_broadcast(srows, dcol)
                for k, v in hom_meta.items():
                    if k not in meta:
                        meta[k] = v
                if sdrf_broadcast is not None:
                    meta["sdrf_lookup_keys"] = 0
                else:
                    sdrf_map = build_sdrf_key_to_disease(srows, dcol, id_cols)
                    meta["sdrf_lookup_keys"] = len(sdrf_map)
                    meta["sdrf_disease_mode"] = "per_sample_join"
            else:
                meta["sdrf_disease_column"] = None
        except Exception as e:
            meta["sdrf_error"] = str(e)

    # --- 2) MSstats header only (no peptide rows) ---
    peek = _safe_read_csv(msstats_path, nrows=0, low_memory=False)
    ref_c = find_msstats_column(peek, "Reference")
    samp_c = find_msstats_column(peek, "Sample")
    run_c = find_msstats_column(peek, "Run")
    dis_c = find_msstats_column(peek, "disease")
    alias_cols = [c for c in (ref_c, samp_c, run_c) if c]

    # --- 3) Homogeneous SDRF + no MSstats disease: SDRF ids only (skip MSstats table) ---
    if sdrf_broadcast is not None and not dis_c and id_cols:
        final = build_homogeneous_map_from_sdrf_rows(srows, id_cols, sdrf_broadcast)
        meta["msstats_table_read"] = False
        meta["merge_source"] = "sdrf_homogeneous"
        if len(final) == 0:
            if not alias_cols:
                meta["no_msstats_sample_columns"] = True
                return {}, meta
            final = _chunked_broadcast_aliases(msstats_path, alias_cols, sdrf_broadcast)
            meta["msstats_table_read"] = True
            meta["merge_source"] = "sdrf_homogeneous_msstats_id_fallback"
        conflicts: List[Dict[str, str]] = []
        meta["n_alias_keys"] = len(final)
        meta["n_unique_disease_labels"] = len(set(final.values())) if final else 0
        meta["n_conflicts"] = 0
        return final, meta

    if not alias_cols:
        meta["no_msstats_sample_columns"] = True
        return {}, meta

    meta["msstats_table_read"] = True
    df = _read_msstats_merge_columns(msstats_path, alias_cols, dis_c)

    ms_dis_by_tuple: Dict[Tuple[Any, ...], Optional[str]] = {}
    grouped = df.groupby(alias_cols, dropna=False)
    for key_tuple, g in grouped:
        if not isinstance(key_tuple, tuple):
            key_tuple = (key_tuple,)
        mdis = None
        if dis_c:
            svals = g[dis_c].dropna().astype(str).str.strip()
            svals = svals[svals.str.len() > 0]
            if len(svals) > 0:
                mdis = svals.iloc[0]
        ms_dis_by_tuple[key_tuple] = mdis

    final = {}
    conflicts = []

    for key_tuple, mdis in ms_dis_by_tuple.items():
        kt = list(key_tuple) if isinstance(key_tuple, tuple) else [key_tuple]
        id_by_name = {}
        for i, col in enumerate(alias_cols):
            if i < len(kt):
                id_by_name[col] = kt[i]
        ref_v = id_by_name.get(ref_c) if ref_c else None
        samp_v = id_by_name.get(samp_c) if samp_c else None
        run_v = id_by_name.get(run_c) if run_c else None
        ids = [id_by_name[c] for c in alias_cols if c in id_by_name]
        if sdrf_broadcast is not None:
            sdrf_dis = sdrf_broadcast
        else:
            sdrf_dis = lookup_disease_for_msstats_ids(sdrf_map, *ids) if sdrf_map else None

        if sdrf_dis and mdis:
            n1 = str(mdis).strip().lower()
            n2 = str(sdrf_dis).strip().lower()
            if n1 != n2 and n1 not in n2 and n2 not in n1:
                conflicts.append(
                    {
                        "dataset": dataset_name,
                        "ids": "|".join(str(x) for x in ids if pd.notna(x)),
                        "msstats_disease": str(mdis),
                        "sdrf_disease": str(sdrf_dis),
                    }
                )
            chosen = sdrf_dis  # design file wins
        elif sdrf_dis:
            chosen = sdrf_dis
        elif mdis:
            chosen = str(mdis).strip()
        else:
            chosen = None

        if not chosen:
            continue
        for col, val in ((ref_c, ref_v), (samp_c, samp_v), (run_c, run_v)):
            if col and val is not None and not (isinstance(val, float) and pd.isna(val)):
                k = str(val).strip()
                if k:
                    final[k] = chosen

    if sdrf_broadcast is not None and dis_c:
        meta["merge_source"] = "homogeneous_msstats_disease_column"
    elif sdrf_map:
        meta["merge_source"] = "heterogeneous_per_sample_join"
    else:
        meta["merge_source"] = "msstats_only_or_no_sdrf_map"

    meta["n_alias_keys"] = len(final)
    meta["n_unique_disease_labels"] = len(set(final.values()))
    meta["n_conflicts"] = len(conflicts)

    if conflict_log and conflicts:
        conflict_log.parent.mkdir(parents=True, exist_ok=True)
        cdf = pd.DataFrame(conflicts)
        if conflict_log.exists():
            cdf.to_csv(conflict_log, mode="a", header=False, index=False)
        else:
            cdf.to_csv(conflict_log, index=False)

    return final, meta


def load_cached_or_build_map(
    msstats_path: Path,
    dataset_name: str,
    sdrf_dir: Path,
    cache_dir: Path,
    conflict_log: Optional[Path] = None,
    force_rebuild: bool = False,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Return (sample_alias -> disease, meta). Uses per-dataset JSON cache.

    Cache is skipped (rebuilt) when SAMPLE_DISEASE_CACHE_VERSION in the file
    does not match — so code changes invalidate old maps without --rebuild-disease-cache.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{dataset_name}_sample_disease_cache.json"
    ms_m = _mtime(msstats_path)
    sdrf_p = resolve_sdrf_path(dataset_name, sdrf_dir)
    sd_m = _mtime(sdrf_p) if sdrf_p else None

    if not force_rebuild and cache_file.is_file():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                blob = json.load(f)
            if (
                blob.get("cache_version") == SAMPLE_DISEASE_CACHE_VERSION
                and blob.get("msstats_mtime") == ms_m
                and blob.get("sdrf_mtime") == sd_m
                and "map" in blob
            ):
                return blob["map"], {**blob.get("meta", {}), "from_cache": True}
        except Exception:
            pass

    m, meta = build_sample_disease_map_merged(
        msstats_path, dataset_name, sdrf_p, conflict_log=conflict_log
    )
    meta["from_cache"] = False
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "cache_version": SAMPLE_DISEASE_CACHE_VERSION,
                    "msstats_mtime": ms_m,
                    "sdrf_mtime": sd_m,
                    "map": m,
                    "meta": meta,
                },
                f,
                indent=2,
            )
    except Exception as e:
        meta["cache_write_error"] = str(e)
    return m, meta


# Directories
msstats_dir = work_dir / "msstats"  # Shared raw data
cache_dir = work_dir / "cache"
data_prep_output_dir = work_dir / "data_preparation_output"  # For B_ script output
output_dir = work_dir / "disease_analysis"
mapping_library_dir = work_dir / "entry_name_mapping"
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "cache").mkdir(parents=True, exist_ok=True)
sdrf_dir = work_dir / "sdrf_files"
SAMPLE_BREAKDOWN_CACHE_VERSION = 3
SAMPLE_DISEASE_MAP_CACHE_VERSION = 1

# Entry name mapping library files
library_file = mapping_library_dir / "entry_name_to_accession.json"
manual_mapping_file = mapping_library_dir / "manual_id_mapping.xlsx"

# Check for optional libraries
try:
    import pyarrow
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    print("Warning: pyarrow not available. Cannot load from cache.")

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Parameters
MIN_PREVALENCE_THRESHOLD = 0.10  # 10% minimum prevalence for protein inclusion
HEALTHY_KEYWORDS = ['normal', 'healthy', 'control', 'not available']  # Keywords for healthy samples

def is_healthy(disease_value):
    """Check if a disease value represents healthy/normal condition."""
    if pd.isna(disease_value) or disease_value == '':
        return False
    disease_str = str(disease_value).lower().strip()
    # Pipe-separated cohort strings (e.g. "CLL | normal") must not match "normal" in the tail
    if "|" in disease_str:
        return False
    return any(keyword in disease_str for keyword in HEALTHY_KEYWORDS)


def normalize_disease_label(disease_value):
    """
    Canonicalize disease labels to merge common near-duplicates.
    Examples:
      - covid 19 / covid-19 / sars-cov-2 -> COVID-19
      - obese / obesity disorder -> Obesity
      - early nsclc / late nsclc / non-small cell lung cancer -> NSCLC
    """
    if disease_value is None or pd.isna(disease_value):
        return None
    raw = str(disease_value).strip()
    if not raw:
        return None
    x = raw.lower()
    x = x.replace("_", " ")
    x = re.sub(r"[-/]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()

    # Explicit unknown/not-applicable labels first (keep as explicit categories)
    if x in {"unk", "unknown", "na", "n a"}:
        return "Not available"
    if "not available" in x:
        return "Not available"
    if "not applicable" in x:
        return "Not applicable"

    # COVID family
    if re.search(r"\bcovid\s*19\b", x) or "sars cov 2" in x or "sars-cov-2" in raw.lower():
        return "COVID-19"

    # Obesity family
    if "obes" in x:
        return "Obesity"

    # NSCLC family (merge early/late stage labels into one disease class)
    if "nsclc" in x or "non small cell lung" in x or "non-small-cell lung" in x:
        return "NSCLC"

    # Gallstone family
    if "choledocholithiasis" in x or "cholelithiasis" in x:
        return "Cholelithiasis"

    # HOCM family
    if "hypertrophic obstructive cardiomyopathy" in x:
        return "HOCM"
    if "hypertrophic trophic cardiomyopathy" in x:
        return "HOCM"
    if "hocm" in x:
        return "HOCM"

    # Requested replacement
    if "comorbid" in x:
        return "Pulmonary disease"

    # Healthy/controls (after explicit NA/NAP handling)
    if is_healthy(x):
        return "Normal"

    # Keep stable readable label for everything else
    return raw


def normalize_disease_labels(disease_value):
    """
    Return one or more canonical disease labels.
    Handles merged conflict labels like "disease A | normal" by splitting.
    """
    if disease_value is None or pd.isna(disease_value):
        return []
    raw = str(disease_value).strip()
    if not raw:
        return []
    parts = [p.strip() for p in raw.split("|")] if "|" in raw else [raw]
    out = []
    seen = set()
    for p in parts:
        if not p:
            continue
        n = normalize_disease_label(p)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out

def pick_primary_disease_label(disease_value):
    """
    Choose one label for sample-level tasks.
    If a composite label contains multiple parts, prefer a non-Normal label.
    """
    labels = normalize_disease_labels(disease_value)
    if not labels:
        return None
    non_normal = [d for d in labels if d != "Normal"]
    return non_normal[0] if non_normal else labels[0]

# ============================================
# ENTRY NAME MAPPING LIBRARY FUNCTIONS
# ============================================

def load_entry_name_mapping_library():
    """Load the entry name to accession mapping library from JSON file."""
    if library_file.exists():
        try:
            with open(library_file, 'r') as f:
                library = json.load(f)
            print(f"  Loaded {len(library)} entry name mappings from library")
            return library
        except Exception as e:
            print(f"  Warning: Could not load entry name mapping library: {e}")
    return {}

def load_manual_mapping():
    """Load manual entry name to accession mapping from Excel file."""
    if not manual_mapping_file.exists():
        return {}
    
    try:
        df = pd.read_excel(manual_mapping_file)
        
        # Try to find entry name and accession columns
        entry_col = None
        accession_col = None
        
        for col in df.columns:
            col_lower = str(col).lower()
            if entry_col is None:
                if 'entry' in col_lower and 'name' in col_lower:
                    entry_col = col
                elif col_lower == 'entry' or col_lower == 'from':
                    entry_col = col
            if accession_col is None:
                if 'accession' in col_lower or 'uniprot' in col_lower:
                    accession_col = col
                elif col_lower == 'to':
                    accession_col = col
        
        # If we found 'From' but not 'To', check if there's a second column
        if entry_col and not accession_col and len(df.columns) >= 2:
            if df.columns[0] == entry_col:
                accession_col = df.columns[1]
            elif df.columns[1] == entry_col:
                accession_col = df.columns[0]
        
        if entry_col is None or accession_col is None:
            return {}
        
        # Build mapping dictionary
        mapping = {}
        for _, row in df.iterrows():
            entry_name = str(row[entry_col]).strip() if pd.notna(row[entry_col]) else None
            accession = str(row[accession_col]).strip() if pd.notna(row[accession_col]) else None
            
            if entry_name and accession and entry_name != 'nan' and accession != 'nan':
                mapping[entry_name] = accession
        
        print(f"  Loaded {len(mapping)} manual mappings")
        return mapping
    
    except Exception as e:
        print(f"  Warning: Could not load manual mapping: {e}")
        return {}

def normalize_protein_set_for_comparison(protein_set, entry_name_library=None, return_mapping=False):
    """Normalize a set of protein identifiers to UniProt accessions for comparison.
    
    Follows specific rules for harmonizing DDA and DIA protein identifiers:
    
    DDA rules:
    - Input format: 'sp|A0A075B6P5|KV228_HUMAN;sp|P01615|KVD28_HUMAN'
    - Split on ';', keep only first entry
    - Extract UniProt accession (string between first and second '|')
    - Result: One UniProt accession per DDA row
    
    DIA rules:
    - Input format: '1433B_HUMAN' or 'ACTB_HUMAN;ACTG_HUMAN'
    - Split on ';', keep only first entry name
    - Convert entry name (e.g., 'ACTB_HUMAN') to UniProt accession using library
    - Result: One UniProt accession per DIA row
    
    Args:
        protein_set: Set of protein identifiers
        entry_name_library: Dictionary mapping entry_name -> accession (from library + manual mapping)
        return_mapping: If True, also return a mapping from normalized -> list of original proteins
    
    Returns:
        Set of normalized protein identifiers (UniProt accessions when possible)
        If return_mapping=True, also returns dict mapping normalized -> list of original proteins
    """
    from shared_utils import extract_uniprot_id
    
    normalized_set = set()
    normalized_to_original = {} if return_mapping else None
    
    # Use provided library or load it
    if entry_name_library is None:
        entry_name_library = load_entry_name_mapping_library()
        manual_mapping = load_manual_mapping()
        # Merge manual mapping into library (manual takes precedence)
        entry_name_library = {**entry_name_library, **manual_mapping}
    
    for protein_id in protein_set:
        if pd.isna(protein_id) or not protein_id:
            continue
        
        protein_str = str(protein_id).strip()
        
        # Skip decoy/entrap proteins
        if is_decoy_entrap_protein(protein_str):
            continue
        
        # Skip non-human proteins
        if is_non_human_protein(protein_str):
            continue
        
        # Handle semicolon-separated entries (take first)
        if ';' in protein_str:
            protein_str = protein_str.split(';')[0].strip()
        
        # DDA format: sp|ACCESSION|ENTRY_NAME or tr|ACCESSION|ENTRY_NAME
        if '|' in protein_str:
            uniprot_id = extract_uniprot_id(protein_str)
            if uniprot_id:
                normalized_set.add(uniprot_id)
                if return_mapping:
                    if uniprot_id not in normalized_to_original:
                        normalized_to_original[uniprot_id] = []
                    normalized_to_original[uniprot_id].append(protein_id)
            else:
                # If extraction failed, keep original (might be already normalized)
                normalized_set.add(protein_str)
                if return_mapping:
                    if protein_str not in normalized_to_original:
                        normalized_to_original[protein_str] = []
                    normalized_to_original[protein_str].append(protein_id)
        else:
            # DIA format: ENTRY_NAME_HUMAN
            # Try to convert using library
            accession = entry_name_library.get(protein_str, None)
            if accession:
                normalized_set.add(accession)
                if return_mapping:
                    if accession not in normalized_to_original:
                        normalized_to_original[accession] = []
                    normalized_to_original[accession].append(protein_id)
            else:
                # If not in library, keep original (might be already normalized)
                normalized_set.add(protein_str)
                if return_mapping:
                    if protein_str not in normalized_to_original:
                        normalized_to_original[protein_str] = []
                    normalized_to_original[protein_str].append(protein_id)
    
    if return_mapping:
        return normalized_set, normalized_to_original
    return normalized_set

# ============================================
# DATA LOADING FUNCTIONS
# ============================================

def load_all_datasets_from_cache():
    """Load all datasets from Parquet cache using configured filter level."""
    if not PARQUET_AVAILABLE:
        print("Error: pyarrow not available. Cannot load from cache.")
        print("Please run B_data_preparation_and_filtering.py first to create cache.")
        return {}
    
    # Get cache filter info
    cache_info = get_cache_filter_info(cache_dir)
    print(f"Loading datasets from cache...")
    print(f"  Cache configuration: min {cache_info['min_peptides']} peptides ({cache_info['filter_type']})")
    
    all_data = {}
    datasets_loaded = set()
    
    # First, try to get list of datasets from B_ script output
    dataset_list = []
    prep_summary_file = data_prep_output_dir / "01_dataset_summary_before_and_after_filtering.csv"
    if prep_summary_file.exists():
        try:
            prep_summary = pd.read_csv(prep_summary_file)
            if 'Dataset' in prep_summary.columns:
                dataset_list = prep_summary['Dataset'].unique().tolist()
                print(f"  Found {len(dataset_list)} datasets from data preparation output")
        except Exception as e:
            print(f"  Warning: Could not load dataset list from data preparation: {e}")
    
    # If we have a dataset list, use find_cache_file for each
    if dataset_list:
        for dataset_name in dataset_list:
            dataset_str = str(dataset_name).strip()
            cache_file = find_cache_file(dataset_str, cache_dir=cache_dir)
            
            if cache_file and cache_file.exists():
                try:
                    df = pd.read_parquet(cache_file)
                    if len(df) > 0:
                        all_data[dataset_str] = df
                        datasets_loaded.add(dataset_str)
                        print(f"  Loaded {dataset_str}: {df['Protein'].nunique()} proteins, {df['Sample'].nunique()} samples")
                except Exception as e:
                    print(f"  Warning: Error loading {dataset_str}: {e}")
    
    # Fallback: if no dataset list or some datasets not found, glob only default B_ cache (2 peptide + ENTRAP removed)
    if not dataset_list or len(datasets_loaded) == 0:
        print(f"  Falling back to glob search for cache files (default filter only)...")
        default_suffix = get_default_cache_parquet_suffix(cache_dir)
        all_cache_files = list(cache_dir.glob(f"*{default_suffix}"))
        for cache_file in all_cache_files:
            base_name = cache_file.name.replace(default_suffix, "")
            if base_name in datasets_loaded:
                continue
            try:
                df = pd.read_parquet(cache_file)
                if len(df) > 0:
                    all_data[base_name] = df
                    datasets_loaded.add(base_name)
                    print(f"  Loaded {base_name}: {df['Protein'].nunique()} proteins, {df['Sample'].nunique()} samples")
            except Exception as e:
                print(f"  Warning: Error loading {base_name}: {e}")
    
    print(f"\nLoaded {len(all_data)} datasets from cache")
    return all_data

def load_sample_disease_mapping(max_datasets=None, force_rebuild_cache=False, plasma_serum_only=True):
    """
    Build sample→disease maps from MSstats + optional SDRF (`sdrf_files`), with JSON cache
    in `disease_analysis/cache/{dataset}_sample_disease_cache.json`.
    If plasma_serum_only=True, only plasma/serum datasets are included.
    Datasets without any resolvable disease label are skipped.
    """
    print("Loading sample-to-disease mappings (MSstats + SDRF, cached)...")
    if max_datasets is not None:
        print(f"  (Limited to {max_datasets} datasets for testing)")
    else:
        print(f"  (Processing all available datasets)")
    if force_rebuild_cache:
        print("  (--rebuild-disease-cache: ignoring stale JSON caches)")

    msstats_files = list(msstats_dir.glob("*.sdrf_openms_design_msstats_in.csv"))
    map_cache_file = output_dir / "cache" / "sample_disease_map_cache.json"

    def _stat_sig(p: Path):
        try:
            st = p.stat()
            return (str(p), float(st.st_mtime), int(st.st_size))
        except OSError:
            return (str(p), None, None)

    cache_sig = {
        "cache_version": SAMPLE_DISEASE_MAP_CACHE_VERSION,
        "max_datasets": max_datasets,
        "plasma_serum_only": bool(plasma_serum_only),
        "msstats_files": sorted(_stat_sig(p) for p in msstats_files),
    }
    # Include SDRF signatures too, because disease labels come from SDRF.
    sdrf_files = list(sdrf_dir.glob("*.sdrf.tsv"))
    cache_sig["sdrf_files"] = sorted(_stat_sig(p) for p in sdrf_files)

    if not force_rebuild_cache and map_cache_file.exists():
        try:
            with open(map_cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("signature") == cache_sig:
                sample_disease_map = cached.get("sample_disease_map", {})
                dataset_conditions = cached.get("dataset_conditions", {})
                if isinstance(sample_disease_map, dict) and isinstance(dataset_conditions, dict):
                    print(
                        f"OK Loaded sample-disease mappings for {len(sample_disease_map)} datasets [global cache]"
                    )
                    return sample_disease_map, dataset_conditions
        except Exception:
            pass
    sample_disease_map = {}  # {dataset_name: {sample: disease}}
    dataset_conditions = {}  # {dataset_name: condition}
    loaded_count = 0
    cache_dir = output_dir / "cache"
    conflict_log = output_dir / "sdrf_msstats_disease_conflicts.csv"

    def _fallback_msstats_disease_map(filepath: Path) -> dict:
        """
        Fallback map builder used when SDRF+MSstats merge raises unexpected OS/read errors.
        Uses MSstats disease column only (if present), mapping disease to Sample/Reference/Run aliases.
        """
        try:
            hdr = pd.read_csv(filepath, nrows=5, low_memory=False, on_bad_lines="skip", engine="python")
            dis_c = find_msstats_column(hdr, "disease")
            ref_c = find_msstats_column(hdr, "Reference")
            samp_c = find_msstats_column(hdr, "Sample")
            run_c = find_msstats_column(hdr, "Run")
            alias_cols = [c for c in (ref_c, samp_c, run_c) if c]
            if not dis_c or not alias_cols:
                return {}
            usecols = list(dict.fromkeys(alias_cols + [dis_c]))
            df = pd.read_csv(filepath, usecols=usecols, low_memory=False, on_bad_lines="skip", engine="python")
        except Exception:
            return {}

        out = {}
        for _, row in df.iterrows():
            d_raw = row.get(dis_c, None)
            d_norm = normalize_disease_label(d_raw) if d_raw is not None else None
            if not d_norm:
                continue
            for c in alias_cols:
                v = row.get(c, None)
                if pd.isna(v):
                    continue
                k = str(v).strip()
                if k:
                    out[k] = d_norm
        return out

    for filepath in msstats_files:
        if max_datasets is not None and loaded_count >= max_datasets:
            break
        dataset_name = filepath.stem.replace(".sdrf_openms_design_msstats_in", "")

        try:
            df_sample = pd.read_csv(filepath, nrows=100)
            if "Condition" not in df_sample.columns:
                continue
            condition = df_sample["Condition"].iloc[0] if len(df_sample) > 0 else ""
            # Use shared canonical tissue grouping (same rule as B/E/F) to avoid mismatches.
            if plasma_serum_only and group_condition_to_tissue(str(condition).strip()) != "Blood Plasma/Serum":
                continue

            ref_ok = any(str(c).lower() == "reference" for c in df_sample.columns)
            samp_ok = any(str(c).lower() == "sample" for c in df_sample.columns)
            run_ok = any(str(c).lower() == "run" for c in df_sample.columns)
            if not ref_ok and not samp_ok and not run_ok:
                print(f"  Skipping {dataset_name}: no Reference / Sample / Run column")
                continue

            print(f"  {dataset_name} ({condition}) - merging MSstats + SDRF...")
            disease_dict, meta = load_cached_or_build_map(
                filepath,
                dataset_name,
                sdrf_dir,
                cache_dir,
                conflict_log=conflict_log,
                force_rebuild=force_rebuild_cache,
            )
            if not disease_dict:
                print(
                    f"    Skip: no disease labels (add SDRF to sdrf_files/ or msstats `disease` column)"
                )
                continue

            src = "cache" if meta.get("from_cache") else "built"
            sdrf_col = meta.get("sdrf_disease_column") or "-"
            print(
                f"    OK {len(disease_dict)} alias keys -> diseases [{src}; SDRF col: {sdrf_col}; "
                f"conflicts this set: {meta.get('n_conflicts', 0)}]"
            )
            # Canonicalize disease labels (merge near-duplicates)
            normalized_dict = {}
            for k, v in disease_dict.items():
                dv = normalize_disease_label(v)
                if dv and str(k).strip():
                    normalized_dict[str(k).strip()] = dv
            if not normalized_dict:
                continue
            sample_disease_map[dataset_name] = normalized_dict
            dataset_conditions[dataset_name] = condition
            loaded_count += 1

        except Exception as e:
            print(f"  Error loading {dataset_name}: {e}")
            print(f"    Trying fallback (MSstats disease-only map) for {dataset_name}...")
            try:
                fallback_map = _fallback_msstats_disease_map(filepath)
                if fallback_map:
                    sample_disease_map[dataset_name] = fallback_map
                    dataset_conditions[dataset_name] = condition
                    loaded_count += 1
                    print(f"    Fallback OK: {len(fallback_map)} alias keys")
                else:
                    print(f"    Fallback unavailable: no resolvable disease aliases")
            except Exception as e2:
                print(f"    Fallback failed for {dataset_name}: {e2}")
            continue

    print(f"OK Loaded sample-disease mappings for {len(sample_disease_map)} datasets")
    try:
        with open(map_cache_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "signature": cache_sig,
                    "sample_disease_map": sample_disease_map,
                    "dataset_conditions": dataset_conditions,
                },
                f,
                indent=2,
            )
    except Exception:
        pass
    return sample_disease_map, dataset_conditions

def load_plasma_serum_datasets(
    max_datasets=None, entry_name_library=None, force_rebuild_disease_cache=False
):
    """Load plasma/serum data from Parquet cache and build proteins_by_sample.
    
    Args:
        max_datasets: Maximum number of datasets to load (None for all). Useful for testing.
        entry_name_library: Entry name mapping library for protein normalization.
    """
    print(f"\n{'='*60}")
    print(f"STEP 1: Loading data from Parquet cache...")
    print(f"{'='*60}")
    
    # Load all datasets from Parquet cache
    all_data = load_all_datasets_from_cache()
    if not all_data:
        print("  Error: Could not load datasets from cache. Please run B_data_preparation_and_filtering.py first.")
        return None
    
    # Load sample-to-disease mappings (lightweight - only reads sample columns)
    sample_disease_map, dataset_conditions = load_sample_disease_mapping(
        max_datasets=max_datasets,
        force_rebuild_cache=force_rebuild_disease_cache,
    )
    
    if not sample_disease_map:
        print("  Error: No sample-disease mappings found")
        return None
    
    # Filter to only plasma/serum datasets that have disease info
    plasma_serum_data = {}
    for dataset_name, data in all_data.items():
        # Check if this dataset is in our sample_disease_map (meaning it's plasma/serum with disease)
        if dataset_name in sample_disease_map:
            plasma_serum_data[dataset_name] = data
    
    # Also check for split datasets (e.g., PXD004352_cd8) - these might not be in sample_disease_map
    # but we can check the base dataset
    for dataset_name, data in all_data.items():
        if dataset_name in plasma_serum_data:
            continue
        
        # Check if it's a split dataset (has underscore)
        if '_' in dataset_name:
            base_dataset = dataset_name.split('_')[0]
            # Check if base dataset is plasma/serum
            if base_dataset in sample_disease_map:
                # This is a split plasma/serum dataset - use base dataset's disease mapping
                plasma_serum_data[dataset_name] = data
                # Copy disease mapping from base dataset
                sample_disease_map[dataset_name] = sample_disease_map[base_dataset].copy()
                dataset_conditions[dataset_name] = dataset_conditions[base_dataset]
    
    print(f"OK Found {len(plasma_serum_data)} plasma/serum datasets with disease information")
    
    # Build proteins_by_sample and abundance_by_sample from Parquet cache produced by B_.
    print("  Building proteins_by_sample/abundance_by_sample from cached data...")
    intra_results = {}
    
    for dataset_name, data in plasma_serum_data.items():
        # Build proteins_by_sample dictionary and per-sample abundance table.
        proteins_by_sample = {}
        abundance_by_sample = {}
        for _, row in data.iterrows():
            sample = row['Sample']
            protein = row['Protein']
            
            if pd.isna(sample) or pd.isna(protein):
                continue
            
            sample = str(sample).strip()
            protein = str(protein).strip()
            
            # Skip decoy/entrap and non-human proteins
            if is_decoy_entrap_protein(protein) or is_non_human_protein(protein):
                continue
            
            if sample not in proteins_by_sample:
                proteins_by_sample[sample] = set()
            proteins_by_sample[sample].add(protein)
            if sample not in abundance_by_sample:
                abundance_by_sample[sample] = {}
            ab = row['PeptideCount'] if 'PeptideCount' in data.columns else 1
            try:
                ab = float(pd.to_numeric(ab, errors='coerce'))
            except Exception:
                ab = 1.0
            if pd.isna(ab):
                ab = 1.0
            abundance_by_sample[sample][protein] = max(abundance_by_sample[sample].get(protein, 0.0), ab)
        
        # Normalize proteins to UniProt accessions
        if entry_name_library:
            print(f"    Normalizing proteins for {dataset_name}...")
            normalized_proteins_by_sample = {}
            normalized_abundance_by_sample = {}
            for sample, proteins in proteins_by_sample.items():
                normalized_proteins, norm_map = normalize_protein_set_for_comparison(
                    proteins,
                    entry_name_library=entry_name_library,
                    return_mapping=True,
                )
                if normalized_proteins:
                    normalized_proteins_by_sample[sample] = normalized_proteins
                    sample_ab = abundance_by_sample.get(sample, {})
                    norm_ab = {}
                    for norm_p, originals in norm_map.items():
                        vals = [float(sample_ab.get(o, 0.0)) for o in originals]
                        norm_ab[norm_p] = max(vals) if vals else 0.0
                    normalized_abundance_by_sample[sample] = norm_ab
            proteins_by_sample = normalized_proteins_by_sample
            abundance_by_sample = normalized_abundance_by_sample
        
        intra_results[dataset_name] = {
            'proteins_by_sample': proteins_by_sample,
            'abundance_by_sample': abundance_by_sample,
        }
    
    # Create a combined structure
    combined_data = {
        'intra_results': intra_results,
        'sample_disease_map': sample_disease_map,
        'dataset_conditions': dataset_conditions
    }
    
    # Print summary
    total_samples = 0
    total_proteins = set()
    for dataset_name, results in intra_results.items():
        proteins_by_sample = results.get('proteins_by_sample', {})
        total_samples += len(proteins_by_sample)
        for proteins in proteins_by_sample.values():
            total_proteins.update(proteins)
    
    print(f"  Total unique samples: {total_samples:,}")
    print(f"  Total unique proteins (after filtering and normalization): {len(total_proteins):,}")
    
    return combined_data

def prepare_disease_data(combined_data):
    """Prepare disease data from cached results and sample-disease mappings."""
    print(f"\n{'='*60}")
    print(f"STEP 2: Preparing disease data from cached results...")
    print(f"{'='*60}")
    
    intra_results = combined_data['intra_results']
    sample_disease_map = combined_data['sample_disease_map']
    
    print("  Classifying samples as healthy or disease...")
    
    # Build sample list with disease information (single primary label per sample for
    # downstream prevalence analysis) + multi-label rows for composition summaries.
    samples_list = []
    disease_count_rows = []
    for dataset_name, results in intra_results.items():
        proteins_by_sample = results.get('proteins_by_sample', {})
        dataset_disease_map = sample_disease_map.get(dataset_name, {})
        exact_lookup, normalized_lookup = _build_disease_alias_lookups(dataset_disease_map)
        base_exact_lookup, base_normalized_lookup = {}, {}
        base_dataset = None
        if "_" in dataset_name:
            base_dataset = dataset_name.split("_")[0]
            base_disease_map = sample_disease_map.get(base_dataset, {})
            base_exact_lookup, base_normalized_lookup = _build_disease_alias_lookups(base_disease_map)
        
        for sample in proteins_by_sample.keys():
            # Get disease for this sample
            disease = _lookup_disease_alias(sample, exact_lookup, normalized_lookup)
            if disease is None:
                # Try to find in base dataset if this is a split dataset
                if base_dataset is not None:
                    disease = _lookup_disease_alias(sample, base_exact_lookup, base_normalized_lookup)
            
            disease_norm = pick_primary_disease_label(disease) if disease else None
            is_healthy_val = disease_norm == 'Normal'
            disease_category = disease_norm if disease_norm else 'Unknown'
            
            samples_list.append({
                'Sample': sample,
                'Dataset': dataset_name,
                'disease_category': disease_category,
                'is_healthy': is_healthy_val
            })

            # For composition/count reporting (plot 1), keep all disease labels found
            # in composite strings like "X | normal" so counts align with disease-list exports.
            labels = normalize_disease_labels(disease) if disease is not None else []
            if not labels:
                labels = ["Unknown"]
            for lbl in labels:
                disease_count_rows.append(
                    {"Sample": sample, "Dataset": dataset_name, "disease_category": lbl}
                )
    
    samples = pd.DataFrame(samples_list)
    
    print(f"OK Disease classification complete")
    print(f"  Total unique samples: {len(samples)}")
    print(f"  Normal samples: {samples['is_healthy'].sum()}")
    print(f"  Disease samples: {(~samples['is_healthy']).sum()}")
    
    print("  Computing disease composition...")
    # Count samples and datasets per disease (multi-label aware, one sample counts once
    # per disease label).
    disease_count_df = pd.DataFrame(disease_count_rows).drop_duplicates(
        ["Dataset", "Sample", "disease_category"]
    )
    # Keep expanded multi-label sample table for downstream plots/analysis that should
    # align with disease-list exports.
    samples_expanded = disease_count_df.copy()
    samples_expanded["is_healthy"] = samples_expanded["disease_category"] == "Normal"
    combined_data["samples_expanded"] = samples_expanded
    disease_counts = disease_count_df.groupby('disease_category').agg({
        'Sample': 'count',
        'Dataset': 'nunique'
    }).rename(columns={'Sample': 'n_samples', 'Dataset': 'n_datasets'})
    
    print("\n  Disease composition:")
    print(disease_counts)
    
    return combined_data, samples, disease_counts

def compute_protein_presence_fractions(combined_data, samples):
    """Compute protein presence fractions using cached proteins_by_sample data."""
    print(f"\n{'='*60}")
    print(f"STEP 3: Computing protein presence fractions (using cached data)...")
    print(f"{'='*60}")
    print("  Using cached proteins_by_sample - this should be much faster!")

    intra_results = combined_data['intra_results']
    samples_for_presence = combined_data.get("samples_expanded", samples)

    # Cache for this expensive step
    presence_cache_file = output_dir / "cache" / "protein_presence_fractions_cache.pkl"
    dataset_sample_counts = {
        ds: len(results.get("proteins_by_sample", {}))
        for ds, results in intra_results.items()
    }
    disease_counts_sig = (
        samples_for_presence["disease_category"].value_counts(dropna=False).sort_index().to_dict()
        if not samples_for_presence.empty
        else {}
    )
    cache_sig = {
        "min_prevalence_threshold": MIN_PREVALENCE_THRESHOLD,
        "datasets": sorted(dataset_sample_counts.items()),
        "disease_counts_sig": disease_counts_sig,
    }
    if presence_cache_file.exists():
        try:
            with open(presence_cache_file, "rb") as f:
                cached = pickle.load(f)
            if cached.get("signature") == cache_sig:
                print("  Using cached protein presence fractions")
                return cached["protein_presence"], cached["disease_categories"]
        except Exception:
            pass

    # Get all disease categories
    disease_categories = sorted([d for d in samples_for_presence["disease_category"].unique() if d != "Unknown"])
    print(f"  Processing {len(disease_categories)} disease categories...")

    # Build sample sets for each disease + normal
    disease_sample_sets = {}
    for disease in disease_categories:
        ss = samples_for_presence.loc[samples_for_presence["disease_category"] == disease, ["Dataset", "Sample"]]
        disease_sample_sets[disease] = set(zip(ss["Dataset"], ss["Sample"]))
    normal_df = samples_for_presence.loc[samples_for_presence["disease_category"] == "Normal", ["Dataset", "Sample"]]
    normal_samples = set(zip(normal_df["Dataset"], normal_df["Sample"]))

    # Build protein -> set(sample_keys) index once (major speedup)
    print("  Building protein index from cached proteins_by_sample...")
    protein_to_samples = defaultdict(set)
    for dataset_name, results in intra_results.items():
        proteins_by_sample = results.get("proteins_by_sample", {})
        for sample, proteins in proteins_by_sample.items():
            sample_key = (dataset_name, sample)
            for protein in proteins:
                if protein:
                    protein_to_samples[str(protein)].add(sample_key)
    all_proteins = sorted(protein_to_samples.keys())
    print(f"    Indexed {len(all_proteins):,} proteins")

    protein_presence = {}
    for idx, disease in enumerate(disease_categories):
        print(f"    Processing disease {idx+1}/{len(disease_categories)}: {disease}...")
        disease_samples = disease_sample_sets.get(disease, set())
        n_disease_samples = len(disease_samples)
        if n_disease_samples == 0:
            continue
        disease_fractions = {}
        for protein in all_proteins:
            n_present = len(protein_to_samples[protein] & disease_samples)
            disease_fractions[protein] = n_present / n_disease_samples
        protein_presence[disease] = disease_fractions
        print(f"      OK {disease}: {n_disease_samples} samples processed")

    print("    Processing normal samples...")
    n_normal_samples = len(normal_samples)
    if n_normal_samples > 0:
        normal_fractions = {}
        for protein in all_proteins:
            n_present = len(protein_to_samples[protein] & normal_samples)
            normal_fractions[protein] = n_present / n_normal_samples
        protein_presence["Normal"] = normal_fractions
        print(f"      OK Normal: {n_normal_samples} samples processed")

    try:
        with open(presence_cache_file, "wb") as f:
            pickle.dump(
                {
                    "signature": cache_sig,
                    "protein_presence": protein_presence,
                    "disease_categories": disease_categories,
                },
                f,
            )
        print(f"  Saved cache: {presence_cache_file}")
    except Exception:
        pass

    print("OK Protein presence fractions computed for all conditions")
    return protein_presence, disease_categories

def plot_1_sample_dataset_composition(disease_counts, output_dir):
    """Plot 1: Sample and dataset composition by disease."""
    print(f"\n{'='*60}")
    print(f"STEP 4: Creating Plot 1 - Sample and dataset composition...")
    print(f"{'='*60}")
    
    # Remove unknown/unavailable buckets from Plot 1
    excluded = {'Unknown', 'Not available', 'Not applicable'}
    disease_counts_plot = disease_counts[~disease_counts.index.isin(excluded)].copy()
    # Sort by number of samples
    disease_counts_sorted = disease_counts_plot.sort_values('n_samples', ascending=False)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x_pos = np.arange(len(disease_counts_sorted))
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, disease_counts_sorted['n_samples'], width, 
                   label='Number of Samples', alpha=0.8, color='steelblue')
    bars2 = ax.bar(x_pos + width/2, disease_counts_sorted['n_datasets'], width,
                   label='Number of Datasets', alpha=0.8, color='coral')
    
    ax.set_xlabel('Disease / Condition', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Sample and Dataset Composition by Disease\n(Plasma/Serum Datasets Only)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(disease_counts_sorted.index, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    output_file = output_dir / "plot1_sample_dataset_composition.png"
    plt.savefig(output_file, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()
    
    # Save data
    disease_counts_sorted.to_csv(output_dir / "plot1_sample_dataset_composition_data.csv")
    print(f"  Saved data: {output_dir / 'plot1_sample_dataset_composition_data.csv'}")

def plot_2_disease_specific_protein_presence(combined_data, samples, protein_presence, disease_categories, output_dir):
    """Plot 2: Disease-specific protein presence (two versions: within-dataset and all-datasets comparison)."""
    print(f"\n{'='*60}")
    print(f"STEP 5: Creating Plot 2 - Disease-specific protein presence...")
    print(f"{'='*60}")
    
    # ============================================
    # PLOT 2A: Disease vs Normal (All Normal Together)
    # ============================================
    print(f"\n  Creating Plot 2A: Disease vs Normal (All Normal Together)...")
    
    if 'Normal' not in protein_presence:
        print("  Warning: No normal samples found. Skipping plot 2A.")
    else:
        healthy_fractions = protein_presence['Normal']
        
        # Filter diseases (exclude Normal and Unknown)
        diseases_to_plot = [d for d in disease_categories if d != 'Normal' and d != 'Unknown']
        
        if diseases_to_plot:
            # Determine grid size
            n_diseases = len(diseases_to_plot)
            n_cols = 5
            n_rows = 5 if n_diseases <= 25 else (n_diseases + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
            if n_diseases == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for idx, disease in enumerate(diseases_to_plot):
                ax = axes[idx]
                
                disease_fractions = protein_presence[disease]
                
                # Get all proteins present in either healthy or disease
                all_proteins = set(healthy_fractions.keys()) | set(disease_fractions.keys())
                
                # Prepare data for scatter plot
                x_vals = []
                y_vals = []
                proteins_list = []
                
                for protein in all_proteins:
                    healthy_frac = healthy_fractions.get(protein, 0)
                    disease_frac = disease_fractions.get(protein, 0)
                    
                    # Only include proteins that meet minimum prevalence threshold in at least one group
                    if healthy_frac >= MIN_PREVALENCE_THRESHOLD or disease_frac >= MIN_PREVALENCE_THRESHOLD:
                        x_vals.append(healthy_frac * 100)  # Convert to percentage
                        y_vals.append(disease_frac * 100)
                        proteins_list.append(protein)
                
                # Scatter plot
                ax.scatter(x_vals, y_vals, alpha=0.5, s=20, edgecolors='none')
                
                # Add diagonal line (y=x)
                max_val = max(max(x_vals) if x_vals else 0, max(y_vals) if y_vals else 0, 1)
                ax.plot([0, max_val], [0, max_val], 'r--', linewidth=1, alpha=0.5, label='y=x')
                
                ax.set_xlabel('Normal Samples (All Datasets)\n(% samples where protein appears)', fontsize=10)
                ax.set_ylabel(f'{disease}\n(% samples where protein appears)', fontsize=10)
                ax.set_title(disease, fontsize=11, fontweight='bold')
                ax.set_xlim(0, 105)
                ax.set_ylim(0, 105)
                ax.grid(True, alpha=0.3)
                
                # Add count
                n_proteins = len(proteins_list)
                ax.text(0.05, 0.95, f'n = {n_proteins} proteins', 
                       transform=ax.transAxes, fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Hide unused subplots
            for idx in range(n_diseases, len(axes)):
                axes[idx].set_visible(False)
            
            plt.suptitle('Disease-Specific Protein Presence\n(Normal vs Disease Fractions - All Normal Together)', 
                         fontsize=14, fontweight='bold', y=0.995)
            plt.tight_layout()
            output_file = output_dir / "plot2a_disease_vs_all_healthy.png"
            plt.savefig(output_file, bbox_inches='tight')
            print(f"  Saved: {output_file}")
            plt.close()
    
    # ============================================
    # PLOT 2B: Disease vs Normal (Within Same Dataset)
    # ============================================
    print(f"\n  Creating Plot 2B: Disease vs Normal (Within Same Dataset)...")
    
    intra_results = combined_data['intra_results']
    samples_for_plot = combined_data.get("samples_expanded", samples)
    
    # Filter diseases (exclude Normal and Unknown)
    diseases_to_plot = [d for d in disease_categories if d != 'Normal' and d != 'Unknown']
    
    if not diseases_to_plot:
        print("  Warning: No disease conditions found. Skipping plot 2.")
        return
    
    # Build dataset-disease mapping: for each dataset, which diseases are present
    dataset_disease_map = {}  # {dataset: [disease1, disease2, ...]}
    dataset_healthy_samples = {}  # {dataset: set of healthy samples}
    dataset_disease_samples = {}  # {dataset: {disease: set of samples}}
    
    for _, row in samples_for_plot.iterrows():
        dataset = row['Dataset']
        sample = row['Sample']
        disease = row['disease_category']
        is_healthy = disease == 'Normal'
        
        if dataset not in dataset_disease_map:
            dataset_disease_map[dataset] = set()
            dataset_healthy_samples[dataset] = set()
            dataset_disease_samples[dataset] = {}
        
        if is_healthy:
            dataset_healthy_samples[dataset].add(sample)
        else:
            if disease not in dataset_disease_samples[dataset]:
                dataset_disease_samples[dataset][disease] = set()
            dataset_disease_samples[dataset][disease].add(sample)
            dataset_disease_map[dataset].add(disease)
    
    # Find datasets that have both disease and healthy samples
    datasets_with_comparison = []
    for dataset in dataset_disease_map.keys():
        if len(dataset_healthy_samples.get(dataset, set())) > 0:
            # Check if this dataset has any of the diseases we want to plot
            dataset_diseases = dataset_disease_map[dataset]
            if any(d in diseases_to_plot for d in dataset_diseases):
                datasets_with_comparison.append(dataset)
    
    if not datasets_with_comparison:
        print("  Warning: No datasets found with both disease and normal samples. Skipping plot 2.")
        return
    
    print(f"  Found {len(datasets_with_comparison)} datasets with both disease and normal samples")
    
    # Collect all comparisons: (dataset, disease) pairs
    comparisons = []
    for dataset in datasets_with_comparison:
        for disease in dataset_disease_map[dataset]:
            if disease in diseases_to_plot:
                n_disease = len(dataset_disease_samples[dataset].get(disease, set()))
                n_healthy = len(dataset_healthy_samples[dataset])
                if n_disease > 0 and n_healthy > 0:
                    comparisons.append((dataset, disease, n_disease, n_healthy))
    
    if not comparisons:
        print("  Warning: No valid comparisons found. Skipping plot 2.")
        return
    
    # Determine grid size
    n_comparisons = len(comparisons)
    n_cols = 5
    n_rows = 5 if n_comparisons <= 25 else (n_comparisons + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_comparisons == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (dataset, disease, n_disease_samples, n_healthy_samples) in enumerate(comparisons):
        ax = axes[idx]
        
        # Get proteins_by_sample for this dataset
        proteins_by_sample = intra_results[dataset]['proteins_by_sample']
        
        # Get disease and healthy sample sets
        disease_samples = dataset_disease_samples[dataset][disease]
        healthy_samples = dataset_healthy_samples[dataset]
        
        # Get all proteins in this dataset
        all_proteins = set()
        for proteins in proteins_by_sample.values():
            all_proteins.update(proteins)
        
        # Compute fractions for healthy samples in this dataset
        healthy_fractions = {}
        for protein in all_proteins:
            n_present = sum(1 for sample in healthy_samples 
                          if sample in proteins_by_sample and protein in proteins_by_sample[sample])
            fraction = n_present / n_healthy_samples if n_healthy_samples > 0 else 0
            healthy_fractions[protein] = fraction
        
        # Compute fractions for disease samples in this dataset
        disease_fractions = {}
        for protein in all_proteins:
            n_present = sum(1 for sample in disease_samples 
                          if sample in proteins_by_sample and protein in proteins_by_sample[sample])
            fraction = n_present / n_disease_samples if n_disease_samples > 0 else 0
            disease_fractions[protein] = fraction
        
        # Prepare data for scatter plot
        x_vals = []
        y_vals = []
        proteins_list = []
        
        for protein in all_proteins:
            healthy_frac = healthy_fractions.get(protein, 0)
            disease_frac = disease_fractions.get(protein, 0)
            
            # Only include proteins that meet minimum prevalence threshold in at least one group
            if healthy_frac >= MIN_PREVALENCE_THRESHOLD or disease_frac >= MIN_PREVALENCE_THRESHOLD:
                x_vals.append(healthy_frac * 100)  # Convert to percentage
                y_vals.append(disease_frac * 100)
                proteins_list.append(protein)
        
        # Scatter plot
        ax.scatter(x_vals, y_vals, alpha=0.5, s=20, edgecolors='none')
        
        # Add diagonal line (y=x)
        max_val = max(max(x_vals) if x_vals else 0, max(y_vals) if y_vals else 0, 1)
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=1, alpha=0.5, label='y=x')
        
        ax.set_xlabel('Normal Samples (Same Dataset)\n(% samples where protein appears)', fontsize=10)
        ax.set_ylabel(f'{disease}\n(% samples where protein appears)', fontsize=10)
        ax.set_title(f'{disease}\n{dataset}\n(n_healthy={n_healthy_samples}, n_disease={n_disease_samples})', 
                    fontsize=10, fontweight='bold')
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        
        # Add count
        n_proteins = len(proteins_list)
        ax.text(0.05, 0.95, f'n = {n_proteins} proteins', 
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide unused subplots
    for idx in range(n_comparisons, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Disease-Specific Protein Presence\n(Normal vs Disease Fractions - Within Same Dataset)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    output_file = output_dir / "plot2b_disease_vs_same_dataset_healthy.png"
    plt.savefig(output_file, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    print(f"  Created {n_comparisons} comparison plots (disease vs normal within same dataset)")
    plt.close()

def plot_3_protein_sharing_stacked_bars(protein_presence, disease_categories, output_dir):
    """Plot 3: Stacked bar plot showing proteins shared and not shared with Normal for each disease."""
    print(f"\n{'='*60}")
    print(f"STEP 6: Creating Plot 3 - Protein sharing stacked bars...")
    print(f"{'='*60}")
    
    if 'Normal' not in protein_presence:
        print("  Warning: No normal samples found. Skipping plot 3.")
        return
    
    # Filter diseases (exclude Normal and Unknown)
    diseases_to_plot = [d for d in disease_categories if d != 'Normal' and d != 'Unknown']
    
    if not diseases_to_plot:
        print("  Warning: No disease conditions found. Skipping plot 3.")
        return
    
    # Build disease proteomes (proteins with >=MIN_PREVALENCE_THRESHOLD)
    healthy_proteome = set()
    for protein, fraction in protein_presence['Normal'].items():
        if fraction >= MIN_PREVALENCE_THRESHOLD:
            healthy_proteome.add(protein)
    
    disease_proteomes = {}
    for disease in diseases_to_plot:
        if disease in protein_presence:
            proteome = set()
            for protein, fraction in protein_presence[disease].items():
                if fraction >= MIN_PREVALENCE_THRESHOLD:
                    proteome.add(protein)
            disease_proteomes[disease] = proteome
    
    # Calculate shared and unique proteins for each disease
    plot_data = []
    for disease in diseases_to_plot:
        if disease not in disease_proteomes:
            continue
        
        disease_proteome = disease_proteomes[disease]
        shared_with_healthy = disease_proteome & healthy_proteome
        unique_to_disease = disease_proteome - healthy_proteome
        
        plot_data.append({
            'Disease': disease,
            'Shared with Normal': len(shared_with_healthy),
            'Unique to Disease': len(unique_to_disease),
            'Total': len(disease_proteome)
        })
    
    if not plot_data:
        print("  Warning: No data to plot. Skipping plot 3.")
        return
    
    plot_df = pd.DataFrame(plot_data)
    plot_df = plot_df.sort_values('Total', ascending=False)
    
    # Create stacked bar plot
    fig, ax = plt.subplots(figsize=(max(12, len(plot_df) * 0.8), 8))
    
    x_pos = np.arange(len(plot_df))
    width = 0.7
    
    # Stacked bars
    bars1 = ax.bar(x_pos, plot_df['Shared with Normal'], width,
                   label='Shared with Normal', alpha=0.8, color='#2ecc71')
    bars2 = ax.bar(x_pos, plot_df['Unique to Disease'], width,
                   bottom=plot_df['Shared with Normal'],
                   label='Unique to Disease', alpha=0.8, color='#e74c3c')
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        # Label for shared proteins (bottom segment)
        if row['Shared with Normal'] > 0:
            ax.text(i, row['Shared with Normal'] / 2, 
                   f"{int(row['Shared with Normal'])}",
                   ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        
        # Label for unique proteins (top segment)
        if row['Unique to Disease'] > 0:
            ax.text(i, row['Shared with Normal'] + row['Unique to Disease'] / 2,
                   f"{int(row['Unique to Disease'])}",
                   ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        
        # Total label on top
        ax.text(i, row['Total'] + row['Total'] * 0.02,
               f"Total: {int(row['Total'])}",
               ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Disease', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Proteins', fontsize=12, fontweight='bold')
    ax.set_title('Protein Sharing with Normal: Stacked Bar Plot\n(Proteins with >=10% prevalence in each condition)',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(plot_df['Disease'], rotation=45, ha='right')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = output_dir / "plot3_protein_sharing_stacked_bars.png"
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"  Saved: {output_file}")
    plt.close()
    
    # Save data
    plot_df.to_csv(output_dir / "plot3_protein_sharing_stacked_bars_data.csv", index=False)
    print(f"  Saved data: {output_dir / 'plot3_protein_sharing_stacked_bars_data.csv'}")

def plot_4_shared_protein_abundance(combined_data, samples, protein_presence, disease_categories, output_dir, entry_name_library=None):
    """Plot 4: Box plot comparing abundance of shared proteins across diseases and normal control."""
    print(f"\n{'='*60}")
    print(f"STEP 7: Creating Plot 4 - Shared protein abundance comparison...")
    print(f"{'='*60}")
    
    if 'Normal' not in protein_presence:
        print("  Warning: No normal samples found. Skipping plot 4.")
        return
    
    # Filter diseases (exclude Normal and Unknown)
    diseases_to_plot = [d for d in disease_categories if d != 'Normal' and d != 'Unknown']
    
    if not diseases_to_plot:
        print("  Warning: No disease conditions found. Skipping plot 4.")
        return
    
    # Build healthy proteome (proteins with >=MIN_PREVALENCE_THRESHOLD)
    healthy_proteome = set()
    for protein, fraction in protein_presence['Normal'].items():
        if fraction >= MIN_PREVALENCE_THRESHOLD:
            healthy_proteome.add(protein)
    
    # Find shared proteins (present in healthy and at least one disease with >=MIN_PREVALENCE_THRESHOLD)
    shared_proteins = set(healthy_proteome)
    for disease in diseases_to_plot:
        if disease in protein_presence:
            disease_proteome = set()
            for protein, fraction in protein_presence[disease].items():
                if fraction >= MIN_PREVALENCE_THRESHOLD:
                    disease_proteome.add(protein)
            shared_proteins &= disease_proteome
    
    if len(shared_proteins) == 0:
        print("  Warning: No shared proteins found. Skipping plot 4.")
        return
    
    print(f"  Found {len(shared_proteins)} shared proteins")
    
    # Select 3 random shared proteins
    import random
    random.seed(42)  # For reproducibility
    selected_proteins = random.sample(list(shared_proteins), min(3, len(shared_proteins)))
    print(f"  Selected {len(selected_proteins)} random proteins: {selected_proteins}")
    
    # Load entry name mapping library if not provided
    if entry_name_library is None:
        entry_name_library = load_entry_name_mapping_library()
        manual_mapping = load_manual_mapping()
        entry_name_library = {**entry_name_library, **manual_mapping}
    
    # Load abundance data for selected proteins from curated B_ cache data.
    print("  Loading abundance data from curated cache...")
    abundance_data = []  # List of dicts: {Protein, Sample, Dataset, Disease, Abundance}
    
    intra_results = combined_data['intra_results']
    sample_disease_map = combined_data['sample_disease_map']
    
    # Build sample-to-disease mapping (including dataset info)
    sample_to_disease_full = {}  # {(dataset, sample): disease_category}
    for _, row in samples.iterrows():
        key = (row['Dataset'], row['Sample'])
        sample_to_disease_full[key] = row['disease_category']
    
    cache_file = output_dir / "cache" / "plot4_abundance_cache.pkl"
    cache_sig = {
        "cache_version": 2,
        "selected_proteins": sorted(selected_proteins),
        "datasets": sorted(intra_results.keys()),
    }

    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
            if cached.get("signature") == cache_sig:
                abundance_df = cached.get("abundance_df")
                if isinstance(abundance_df, pd.DataFrame) and not abundance_df.empty:
                    print(f"  Using cached abundance data: {cache_file}")
                else:
                    abundance_df = None
            else:
                abundance_df = None
        except Exception:
            abundance_df = None
    else:
        abundance_df = None

    if abundance_df is None:
        for dataset_name in intra_results.keys():
            try:
                abund = intra_results.get(dataset_name, {}).get("abundance_by_sample", {})
                if not abund:
                    continue
                dataset_disease_map = sample_disease_map.get(dataset_name, {})
                base_disease_map = {}
                if "_" in dataset_name:
                    base_dataset = dataset_name.split("_")[0]
                    base_disease_map = sample_disease_map.get(base_dataset, {})
                for sample, protein_map in abund.items():
                    disease = sample_to_disease_full.get((dataset_name, sample))
                    if disease is None:
                        disease = dataset_disease_map.get(sample)
                    if disease is None and base_disease_map:
                        disease = base_disease_map.get(sample)
                    if disease is None:
                        continue
                    disease_str = str(disease).strip()
                    if not disease_str or disease_str.lower() == "nan":
                        continue
                    disease_labels = normalize_disease_labels(disease_str)
                    if not disease_labels:
                        continue
                    for protein, abundance in protein_map.items():
                        if protein not in selected_proteins:
                            continue
                        if pd.isna(abundance) or float(abundance) <= 0:
                            continue
                        for disease_category in disease_labels:
                            if (
                                not disease_category
                                or str(disease_category).strip() == ""
                                or disease_category == "Unknown"
                            ):
                                continue
                            abundance_data.append(
                                {
                                    "Protein": protein,
                                    "Sample": sample,
                                    "Dataset": dataset_name,
                                    "Disease": disease_category,
                                    "Abundance": float(abundance),
                                }
                            )
            except Exception as e:
                print(f"    Warning: Error loading cached abundance for {dataset_name}: {e}")
                continue

        if len(abundance_data) == 0:
            print("  Warning: No abundance data found. Skipping plot 4.")
            return

        print(f"  Loaded abundance data for {len(abundance_data)} protein-sample combinations")
        abundance_df = pd.DataFrame(abundance_data)
        try:
            with open(cache_file, "wb") as f:
                pickle.dump({"signature": cache_sig, "abundance_df": abundance_df}, f)
            print(f"  Saved cache: {cache_file}")
        except Exception:
            pass
    
    # Filter out any rows with empty, None, or whitespace-only disease categories
    # This handles edge cases where empty strings might have slipped through
    abundance_df = abundance_df[
        abundance_df['Disease'].notna() & 
        (abundance_df['Disease'].astype(str).str.strip() != '') &
        (abundance_df['Disease'].astype(str).str.strip().str.lower() != 'nan')
    ].copy()
    
    if len(abundance_df) == 0:
        print("  Warning: No valid abundance data after filtering. Skipping plot 4.")
        return
    
    # Create box plots for each selected protein
    n_proteins = len(selected_proteins)
    fig, axes = plt.subplots(1, n_proteins, figsize=(6*n_proteins, 6))
    
    if n_proteins == 1:
        axes = [axes]
    
    # Get unique disease categories directly from the abundance data
    # This ensures we only plot diseases that actually have data
    unique_diseases = abundance_df['Disease'].unique()
    
    # Debug: Print all unique diseases to identify problematic ones
    print(f"  Debug: Found {len(unique_diseases)} unique disease categories in abundance data")
    print(f"  Debug: Disease categories: {[str(d) for d in unique_diseases]}")
    
    # Filter to only valid diseases (exclude None, empty, whitespace-only, and 'Unknown')
    diseases_with_data = []
    for d in unique_diseases:
        if d is None:
            continue
        d_str = str(d).strip()
        if d_str and d_str != '' and d_str.lower() != 'nan' and d_str != 'Unknown':
            diseases_with_data.append(d)
        else:
            print(f"  Debug: Filtered out invalid disease category: '{d}' (type: {type(d)})")
    
    # Keep a base alphabetical disease list (panel-specific sorting is applied below)
    diseases_with_data = sorted(diseases_with_data)
    
    for idx, protein in enumerate(selected_proteins):
        ax = axes[idx]
        
        # Filter data for this protein
        protein_data = abundance_df[abundance_df['Protein'] == protein].copy()
        
        if len(protein_data) == 0:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
            ax.set_title(protein, fontsize=11, fontweight='bold')
            continue
        
        # Sort diseases by abundance for this protein: Normal first, then median ascending
        disease_medians = []
        for disease in diseases_with_data:
            vals = protein_data[protein_data['Disease'] == disease]['Abundance'].values
            if len(vals) > 0:
                disease_medians.append((disease, float(np.median(vals))))
        non_normal = sorted([x for x in disease_medians if x[0] != 'Normal'], key=lambda t: t[1])
        if any(x[0] == 'Normal' for x in disease_medians):
            ordered_diseases = ['Normal'] + [d for d, _ in non_normal]
        else:
            ordered_diseases = [d for d, _ in non_normal]

        # Color palette for this panel
        colors = sns.color_palette("husl", len(ordered_diseases))
        disease_colors = {disease: colors[i] for i, disease in enumerate(ordered_diseases)}

        # Prepare data for box plot
        plot_data_list = []
        for disease in ordered_diseases:
            # Double-check disease is valid before processing
            if disease is None:
                continue
            disease_str = str(disease).strip()
            if not disease_str or disease_str.lower() == 'nan':
                continue
            
            disease_data = protein_data[protein_data['Disease'] == disease]['Abundance'].values
            if len(disease_data) > 0:
                # Truncate long disease names to 20 characters for better display
                max_label_length = 20
                disease_label = disease_str
                if len(disease_label) > max_label_length:
                    disease_label = disease_label[:max_label_length-3] + '...'
                
                if disease_label:  # Only add if label is not empty
                    plot_data_list.append({
                        'Disease': disease,
                        'Disease_Label': disease_label,
                        'Abundance': disease_data
                    })
        
        if not plot_data_list:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
            ax.set_title(protein, fontsize=11, fontweight='bold')
            continue
        
        # Create box plot - final validation of labels
        box_data = []
        box_labels = []
        box_colors = []
        
        for item in plot_data_list:
            label = str(item['Disease_Label']).strip()
            # Final check: skip if label is empty, None, or whitespace-only
            if label and label != '' and label.lower() != 'nan':
                box_data.append(item['Abundance'])
                box_labels.append(label)
                # Get color for this disease
                disease_key = item['Disease']
                if disease_key in disease_colors:
                    box_colors.append(disease_colors[disease_key])
                else:
                    # Fallback color if disease not in color map
                    box_colors.append('gray')
        
        # Final safety check: ensure we have valid data
        if not box_data or not box_labels:
            ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
            ax.set_title(protein, fontsize=11, fontweight='bold')
            continue
        
        # Debug: Print labels to identify any issues
        if idx == 0:  # Only print for first protein to avoid spam
            print(f"    Debug: Plotting {len(box_labels)} disease categories: {box_labels}")
        
        # Final validation: ensure no empty labels make it to matplotlib
        # This is critical - matplotlib will create a box even with empty label
        final_box_data = []
        final_box_labels = []
        final_box_colors = []
        
        for i, label in enumerate(box_labels):
            # Check if label is truly non-empty (not just whitespace)
            if label and isinstance(label, str) and label.strip() and label.strip() != '':
                final_box_data.append(box_data[i])
                final_box_labels.append(label.strip())
                final_box_colors.append(box_colors[i])
        
        if not final_box_data:
            ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
            ax.set_title(protein, fontsize=11, fontweight='bold')
            continue
        
        bp = ax.boxplot(final_box_data, labels=final_box_labels, patch_artist=True, 
                       showmeans=True, meanline=True)
        
        # Color boxes
        for patch, color in zip(bp['boxes'], final_box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Style other elements
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color='black', linewidth=1.5)
        
        ax.set_ylabel('Abundance (log scale)', fontsize=11, fontweight='bold')
        ax.set_title(protein, fontsize=11, fontweight='bold')
        ax.set_yscale('log')  # Set logarithmic Y-axis
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Abundance Comparison of Shared Proteins\n(Normal vs Diseases) - {len(shared_proteins)} shared proteins', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_file = output_dir / "plot4_shared_protein_abundance.png"
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"  Saved: {output_file}")
    plt.close()
    
    # Save data
    abundance_df.to_csv(output_dir / "plot4_shared_protein_abundance_data.csv", index=False)
    print(f"  Saved data: {output_dir / 'plot4_shared_protein_abundance_data.csv'}")


def _pick_msstats_sample_id_strategy(hdr: pd.DataFrame):
    """
    Choose a column set that counts **biological / multiplex samples**, not spectrum-level
    Reference rows. OpenMS MSstats often has spectrum-unique `Reference`; `Sample` may be absent.
    Priority: Sample > (Run + Channel) > Run > Reference.
    """
    s_col = find_msstats_column(hdr, "Sample")
    run_col = find_msstats_column(hdr, "Run")
    channel_col = find_msstats_column(hdr, "Channel")
    ref_col = find_msstats_column(hdr, "Reference")
    cond_col = find_msstats_column(hdr, "Condition")
    if s_col:
        return {
            "mode": "single",
            "key_col": s_col,
            "run_col": run_col,
            "ref_col": ref_col,
            "s_col": s_col,
            "condition_col": cond_col,
        }
    if run_col and channel_col:
        return {
            "mode": "run_channel",
            "run_col": run_col,
            "channel_col": channel_col,
            "ref_col": ref_col,
            "s_col": None,
            "condition_col": cond_col,
        }
    if run_col:
        return {
            "mode": "single",
            "key_col": run_col,
            "run_col": run_col,
            "ref_col": ref_col,
            "s_col": None,
            "condition_col": cond_col,
        }
    if ref_col:
        return {
            "mode": "single",
            "key_col": ref_col,
            "run_col": None,
            "ref_col": ref_col,
            "s_col": None,
            "condition_col": cond_col,
        }
    return None


def _build_disease_alias_lookups(disease_dict):
    """Build exact and normalized alias lookup tables for robust key matching."""
    exact = {}
    normalized = {}
    for k, v in disease_dict.items():
        ks = str(k).strip()
        if not ks:
            continue
        exact[ks] = v
        variants = [ks] + file_stem_variants(ks)
        for vv in variants:
            nk = normalize_match_key(vv)
            if nk and nk not in normalized:
                normalized[nk] = v
    return exact, normalized


def _lookup_disease_alias(value, exact_lookup, normalized_lookup):
    """Try exact alias first, then normalized/stem aliases."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    raw = str(value).strip()
    if not raw:
        return None
    if raw in exact_lookup:
        return exact_lookup[raw]
    variants = [raw] + file_stem_variants(raw)
    for vv in variants:
        nk = normalize_match_key(vv)
        if nk and nk in normalized_lookup:
            return normalized_lookup[nk]
    return None


def _raw_disease_for_msstats_group(
    sub,
    exact_lookup,
    normalized_lookup,
    d_col,
    ref_col,
    run_col,
    s_col,
    channel_col=None,
):
    """Resolve disease for one MSstats group row-set: MSstats column first, then alias dict."""
    d_raw = None
    if d_col and d_col in sub.columns:
        s = sub[d_col].dropna()
        if len(s) > 0:
            d_raw = s.iloc[0]
    if d_raw is None or pd.isna(d_raw) or (isinstance(d_raw, str) and not str(d_raw).strip()):
        # For multiplex data, channel aliases (e.g. TMT126 / ITRAQ114 / 1..10) are often
        # the most specific sample key; try it before broader aliases.
        for col in (s_col, channel_col, run_col, ref_col):
            if not col or col not in sub.columns:
                continue
            for val in sub[col].dropna().unique():
                hit = _lookup_disease_alias(val, exact_lookup, normalized_lookup)
                if hit is not None and not (isinstance(hit, float) and pd.isna(hit)):
                    d_raw = hit
                    break
            if d_raw is not None and not (
                isinstance(d_raw, float) and pd.isna(d_raw)
            ) and not (isinstance(d_raw, str) and not str(d_raw).strip()):
                break
    return d_raw


def export_dataset_and_tissue_disease_lists(output_dir, max_datasets=None, force_rebuild_disease_cache=False):
    """
    Export disease lists in two tabular formats:
      1) dataset-level: one row per (dataset, tissue, disease) with sample counts. Diseases
         include ``Not available`` and ``Not applicable`` when present. Counts use all unique
         MSstats sample IDs; unresolved/missing labels count as ``Not available``.
      2) tissue-level: one row per (tissue, disease) with datasets aggregated and
         ``n_samples_total`` summed across those datasets.
    """
    print(f"\n{'='*60}")
    print("STEP X: Exporting dataset/tissue disease lists...")
    print(f"{'='*60}")

    # Export over ALL tissues/cell types (not plasma-only), while core G_ analysis remains plasma-focused.
    sample_disease_map, dataset_conditions = load_sample_disease_mapping(
        max_datasets=max_datasets,
        force_rebuild_cache=force_rebuild_disease_cache,
        plasma_serum_only=False,
    )

    def _sample_breakdown_cache_path(dataset_name):
        return output_dir / "cache" / f"{dataset_name}_sample_breakdown_cache.json"

    def _cache_mtime_or_none(p):
        try:
            return p.stat().st_mtime if p.exists() else None
        except OSError:
            return None

    def _dataset_tissue_from_sdrf(dataset_name, default_tissue):
        """
        Prefer SDRF-derived tissue/cell-type when available.
        For mixed-cell datasets (e.g., PXD004352), return a generic multi-cell label.
        """
        placeholders = {
            "",
            "not available",
            "not applicable",
            "na",
            "n/a",
            "unknown",
            "unk",
            "none",
        }

        def _clean_vals(series):
            vals = []
            for v in series.dropna().astype(str):
                s = str(v).strip()
                if not s:
                    continue
                if s.lower() in placeholders:
                    continue
                vals.append(s)
            return sorted(set(vals))

        sdrf_path = sdrf_dir / f"{dataset_name}.sdrf.tsv"
        if not sdrf_path.exists() and "-" in dataset_name:
            base = dataset_name.split("-")[0]
            alt = sdrf_dir / f"{base}.sdrf.tsv"
            if alt.exists():
                sdrf_path = alt
        if not sdrf_path.exists():
            return default_tissue
        try:
            hdr = pd.read_csv(sdrf_path, sep="\t", nrows=5, low_memory=False)
            cell_col = None
            for c in hdr.columns:
                cl = str(c).strip().lower()
                if cl == "characteristics[cell type]":
                    cell_col = c
                    break
            org_col = None
            for c in hdr.columns:
                cl = str(c).strip().lower()
                if cl == "characteristics[organism part]":
                    org_col = c
                    break
            usecols = [c for c in (cell_col, org_col) if c]
            if not usecols:
                return default_tissue
            sdf = pd.read_csv(sdrf_path, sep="\t", usecols=usecols, low_memory=False)
            if cell_col and cell_col in sdf.columns:
                cell_vals = _clean_vals(sdf[cell_col])
                if len(cell_vals) == 1:
                    return cell_vals[0]
                if len(cell_vals) > 1:
                    return "Multiple cell types"
            return default_tissue
        except Exception:
            return default_tissue

    def _dataset_sample_disease_breakdown(dataset_name, disease_dict):
        """
        Per-disease sample counts on MSstats **sample-like** keys: ``Sample`` if present,
        else ``Run`` × ``Channel`` (typical TMT multiplex), else ``Run``, else ``Reference``.
        Using ``Reference`` alone would count spectrum-level rows (often ~10⁵), not samples.

        Returns:
            (n_total, counts dict disease -> n, pct_not_available_or_not_applicable)
        """
        fp = msstats_dir / f"{dataset_name}.sdrf_openms_design_msstats_in.csv"
        if not fp.exists():
            return 0, {}, np.nan
        exact_lookup, normalized_lookup = _build_disease_alias_lookups(disease_dict)
        breakdown_cache_file = _sample_breakdown_cache_path(dataset_name)
        disease_cache_file = output_dir / "cache" / f"{dataset_name}_sample_disease_cache.json"
        msstats_mtime = _cache_mtime_or_none(fp)
        disease_cache_mtime = _cache_mtime_or_none(disease_cache_file)
        if breakdown_cache_file.exists():
            try:
                with open(breakdown_cache_file, "r", encoding="utf-8") as f:
                    cobj = json.load(f)
                if (
                    cobj.get("cache_version") == SAMPLE_BREAKDOWN_CACHE_VERSION
                    and cobj.get("msstats_mtime") == msstats_mtime
                    and cobj.get("disease_cache_mtime") == disease_cache_mtime
                ):
                    return (
                        int(cobj.get("n_total", 0)),
                        {str(k): int(v) for k, v in cobj.get("disease_counts", {}).items()},
                        float(cobj.get("na_pct")) if cobj.get("na_pct") is not None else np.nan,
                    )
            except Exception:
                pass
        try:
            hdr = pd.read_csv(fp, nrows=5, low_memory=False)
            strat = _pick_msstats_sample_id_strategy(hdr)
            if not strat:
                return 0, {}, np.nan
            d_col = find_msstats_column(hdr, "disease")
            ref_col = strat.get("ref_col")
            run_col = strat.get("run_col")
            s_col = strat.get("s_col")
            cond_col = strat.get("condition_col")

            usecols = []
            if strat["mode"] == "single":
                usecols.append(strat["key_col"])
            else:
                usecols.extend([strat["run_col"], strat["channel_col"]])
            if d_col:
                usecols.append(d_col)
            for c in (ref_col, run_col, s_col, cond_col):
                if c and c not in usecols:
                    usecols.append(c)
            usecols = list(dict.fromkeys(usecols))
            df = pd.read_csv(fp, usecols=usecols, low_memory=False)
        except Exception:
            return 0, {}, np.nan

        if strat["mode"] == "run_channel":
            df = df.dropna(subset=[strat["run_col"], strat["channel_col"]])
            gb = df.groupby([strat["run_col"], strat["channel_col"]], sort=False)
        else:
            df = df.dropna(subset=[strat["key_col"]])
            gb = df.groupby(strat["key_col"], sort=False)

        # Guard against peptide-level "sample ids" (e.g., PXD004352 Run explosion).
        # If chosen key creates far more groups than disease aliases, fall back to Condition.
        if cond_col and cond_col in df.columns and len(disease_dict) > 0:
            n_groups = gb.ngroups
            if n_groups > max(5000, 10 * len(disease_dict)):
                cond_df = df.dropna(subset=[cond_col])
                cond_gb = cond_df.groupby(cond_col, sort=False)
                if cond_gb.ngroups > 1 and cond_gb.ngroups < n_groups:
                    gb = cond_gb

        n_total = gb.ngroups
        if n_total == 0:
            return 0, {}, np.nan

        counts = Counter()
        for _, sub in gb:
            d_raw = _raw_disease_for_msstats_group(
                sub,
                exact_lookup,
                normalized_lookup,
                d_col,
                ref_col,
                run_col,
                s_col,
                strat.get("channel_col"),
            )
            labels = (
                normalize_disease_labels(d_raw)
                if d_raw is not None and not pd.isna(d_raw)
                else []
            )
            if not labels:
                labels = ["Not available"]
            for d_norm in labels:
                counts[d_norm] += 1

        n_na_nap = counts.get("Not available", 0) + counts.get("Not applicable", 0)
        pct = 100.0 * n_na_nap / n_total
        try:
            payload = {
                "cache_version": SAMPLE_BREAKDOWN_CACHE_VERSION,
                "msstats_mtime": msstats_mtime,
                "disease_cache_mtime": disease_cache_mtime,
                "n_total": int(n_total),
                "disease_counts": {k: int(v) for k, v in counts.items()},
                "na_pct": float(pct),
            }
            with open(breakdown_cache_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass
        return n_total, dict(counts), pct

    dataset_rows = []
    for dataset_name, disease_dict in sample_disease_map.items():
        condition = dataset_conditions.get(dataset_name, "")
        tissue = group_condition_to_tissue(condition)
        tissue = _dataset_tissue_from_sdrf(dataset_name, tissue)
        n_total_samples, disease_counts, na_pct = _dataset_sample_disease_breakdown(
            dataset_name, disease_dict
        )
        # Only keep Not available / Not applicable as disease rows when they are the
        # sole labels for the dataset.
        if (
            n_total_samples > 0
            and "Not available" in disease_counts
            and disease_counts.get("Not available", 0) < n_total_samples
        ):
            disease_counts.pop("Not available", None)
        if (
            n_total_samples > 0
            and "Not applicable" in disease_counts
            and disease_counts.get("Not applicable", 0) < n_total_samples
        ):
            disease_counts.pop("Not applicable", None)
        for disease in sorted(disease_counts.keys()):
            n_this = disease_counts[disease]
            if n_this <= 0:
                continue
            dataset_rows.append(
                {
                    "dataset": dataset_name,
                    "tissue_cell_type": tissue,
                    "disease": disease,
                    "n_samples_this_disease": int(n_this),
                    "total_samples": int(n_total_samples),
                    "not_available_not_applicable_percentage": float(na_pct)
                    if pd.notna(na_pct)
                    else np.nan,
                }
            )

    df_dataset = pd.DataFrame(dataset_rows)
    if not df_dataset.empty:
        df_dataset = df_dataset.sort_values(
            ["tissue_cell_type", "dataset", "disease"]
        ).reset_index(drop=True)
    out_dataset = output_dir / "disease_list_by_dataset_tissue.csv"
    df_dataset.to_csv(out_dataset, index=False)
    print(f"  Saved: {out_dataset}")

    # One row per unique (tissue, disease), with datasets aggregated in the datasets column.
    tissue_disease_map = {}
    for r in dataset_rows:
        t = r["tissue_cell_type"]
        d = r["disease"]
        key = (t, d)
        if key not in tissue_disease_map:
            tissue_disease_map[key] = {"datasets": set(), "n_samples": 0}
        tissue_disease_map[key]["datasets"].add(r["dataset"])
        tissue_disease_map[key]["n_samples"] += int(r["n_samples_this_disease"])

    tissue_rows = []
    for (tissue, disease) in sorted(tissue_disease_map.keys()):
        info = tissue_disease_map[(tissue, disease)]
        ds = sorted(info["datasets"])
        tissue_rows.append(
            {
                "datasets": "; ".join(ds),
                "tissue_cell_type": tissue,
                "disease": disease,
                "n_datasets": len(ds),
                "n_samples_total": int(info["n_samples"]),
            }
        )

    df_tissue = pd.DataFrame(tissue_rows)
    out_tissue = output_dir / "disease_list_by_tissue_combined_datasets.csv"
    df_tissue.to_csv(out_tissue, index=False)
    print(f"  Saved: {out_tissue}")

def main():
    """Main function to run all analyses."""
    print("="*80)
    print("DISEASE ANALYSIS FOR PLASMA/SERUM PROTEOMES (PROTEIN-LEVEL)")
    print("="*80)
    
    # Optional: integer limit (first non-flag arg)
    max_datasets = None
    pos_args = [a for a in sys.argv[1:] if not str(a).startswith("--")]
    if pos_args:
        try:
            max_datasets = int(pos_args[0])
            print(f"\nTESTING MODE: Limited to {max_datasets} datasets")
        except ValueError:
            print(f"\nNote: Invalid limit argument '{pos_args[0]}', processing all datasets")
    else:
        print(f"\nProcessing ALL plasma/serum datasets with disease information...")

    t0 = perf_counter()
    # Load entry name mapping library (for protein normalization)
    print("\nLoading entry name mapping library...")
    entry_name_library = load_entry_name_mapping_library()
    manual_mapping = load_manual_mapping()
    entry_name_library = {**entry_name_library, **manual_mapping}  # Merge manual mapping
    print(f"  Total entry name mappings available: {len(entry_name_library)}")
    
    # Load data from Parquet cache
    combined_data = load_plasma_serum_datasets(
        max_datasets=max_datasets,
        entry_name_library=entry_name_library,
        force_rebuild_disease_cache=False,
    )
    print(f"  Timing: load_plasma_serum_datasets = {perf_counter() - t0:.1f}s")
    if combined_data is None:
        return
    
    # Prepare disease data
    t1 = perf_counter()
    combined_data, samples, disease_counts = prepare_disease_data(combined_data)
    print(f"  Timing: prepare_disease_data = {perf_counter() - t1:.1f}s")
    t2 = perf_counter()
    export_dataset_and_tissue_disease_lists(
        output_dir,
        max_datasets=max_datasets,
        force_rebuild_disease_cache=False,
    )
    print(f"  Timing: export_dataset_and_tissue_disease_lists = {perf_counter() - t2:.1f}s")
    
    # Compute protein presence fractions (using proteins_by_sample with normalized UniProt accessions)
    t3 = perf_counter()
    protein_presence, disease_categories = compute_protein_presence_fractions(combined_data, samples)
    print(f"  Timing: compute_protein_presence_fractions = {perf_counter() - t3:.1f}s")
    
    # Create plots
    print(f"\n{'='*80}")
    print("STARTING PLOT GENERATION")
    print(f"{'='*80}")
    t4 = perf_counter()
    plot_1_sample_dataset_composition(disease_counts, output_dir)
    print(f"  Timing: plot_1_sample_dataset_composition = {perf_counter() - t4:.1f}s")
    t5 = perf_counter()
    plot_2_disease_specific_protein_presence(combined_data, samples, protein_presence, disease_categories, output_dir)
    print(f"  Timing: plot_2_disease_specific_protein_presence = {perf_counter() - t5:.1f}s")
    t6 = perf_counter()
    plot_3_protein_sharing_stacked_bars(protein_presence, disease_categories, output_dir)
    print(f"  Timing: plot_3_protein_sharing_stacked_bars = {perf_counter() - t6:.1f}s")
    t7 = perf_counter()
    plot_4_shared_protein_abundance(combined_data, samples, protein_presence, disease_categories, output_dir, entry_name_library=entry_name_library)
    print(f"  Timing: plot_4_shared_protein_abundance = {perf_counter() - t7:.1f}s")
    
    print("\n" + "="*80)
    print("OK ANALYSIS COMPLETE")
    print("="*80)
    print(f"Total runtime: {perf_counter() - t0:.1f}s")
    print(f"\nAll plots saved to: {output_dir}")

if __name__ == "__main__":
    main()
