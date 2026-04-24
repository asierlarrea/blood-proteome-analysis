"""
H_sex_analysis.py

Single entrypoint for sex-related analysis:

- SDRF scan → `sex_analysis/sex_coverage_by_dataset.csv` (unless --skip-coverage)
- Prevalence volcano + one-sex protein lists + `panel_*_only_proteins.csv`
- Optional extended outputs: abundance volcanos + normal-only tables (--extended-sex-analysis)
- ML prediction → `sex_analysis/prediction_all_proteins/` and `prediction_peptide_atlas/` (unless --skip-ml)
- Root summaries: `sex_analysis_summary.json`, `sex_analysis_results.md`, `PREDICTION_PLOT_OPTIONS.md`
- Figures: `sex_analysis/prediction_plots/` (after ML)
- Caches (keep when wiping other `sex_analysis/` outputs): `sex_analysis/cache/volcano_plasma/`, `cache/ml_long_tables/`, `cache/peptide_atlas_accessions/` (all under `sex_analysis/cache/`)

Run: `python H_sex_analysis.py`  |  `python H_sex_analysis.py --skip-ml`  |  `python H_sex_analysis.py --extended-sex-analysis`
"""

from __future__ import annotations

import csv
import hashlib
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from scipy.stats import fisher_exact

import sys

# Ensure local modules import even if run outside work dir
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import E_plasma_and_cell_types as E  # noqa: E402
from shared_utils import FEATURE_COUNT_COLUMNS  # noqa: E402


def _work_dir() -> Path:
    return Path(__file__).resolve().parent


def _safe_to_csv(df: pd.DataFrame, path: Path) -> Path:
    """
    Windows/Excel often locks CSVs. If writing fails, write to a timestamped filename.
    Returns the actual path written.
    """
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt = path.with_name(f"{path.stem}_{ts}{path.suffix}")
        df.to_csv(alt, index=False)
        return alt


def normalize_match_key(s: Any) -> Optional[str]:
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
    for ext in (".mzml", ".raw", ".mgf", ".wiff"):
        pos = sl.find(ext)
        if pos > 0:
            add(sl[:pos])
            add(sl[: pos + len(ext)])
    return out


def is_sdrf_disease_placeholder(d: str) -> bool:
    x = (d or "").strip().lower()
    return not x or x in ("not available", "na", "n/a", "unknown", "none")


def is_healthy_sdrf_label(d: str) -> bool:
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
    return "healthy control" in x or "normal control" in x


def pick_disease_column(fieldnames: List[str]) -> Optional[str]:
    best = None
    for h in fieldnames:
        hl = h.lower()
        if "disease" in hl or "diagnosis" in hl:
            return h
        if "indication" in hl or "pathology" in hl or "affected" in hl:
            return h
        if "phenotype" in hl and "disease" not in (best or "").lower():
            best = h
    return best


def pick_identifier_columns(fieldnames: List[str]) -> List[str]:
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
    for h in fieldnames:
        if h.lower() == "source name" and h not in out:
            out.insert(0, h)
    return list(dict.fromkeys(out))


def resolve_sdrf_path(dataset_name: str, sdrf_dir: Path) -> Optional[Path]:
    candidates = [sdrf_dir / f"{dataset_name}.sdrf.tsv"]
    if "-" in dataset_name:
        base = dataset_name.split("-")[0]
        candidates.append(sdrf_dir / f"{base}.sdrf.tsv")
    for p in candidates:
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def _pick_sex_column(fieldnames: List[str]) -> Optional[str]:
    """
    Prefer the canonical SDRF column name characteristics[sex], but be tolerant.
    """
    if not fieldnames:
        return None
    # Exact match (case-insensitive)
    for h in fieldnames:
        if str(h).strip().lower() == "characteristics[sex]":
            return h
    # Fallback: any column containing "[sex]" or ending in "sex]"
    for h in fieldnames:
        hl = str(h).strip().lower()
        if "[sex]" in hl or hl.endswith("sex]") or hl == "sex":
            return h
    return None


def _normalize_sex_value(v: str) -> Optional[str]:
    """
    Return "male"/"female" when recognizable, else None.
    """
    x = (v or "").strip().lower()
    if not x or x in {"na", "n/a", "none", "unknown", "not available"}:
        return None
    # common variants
    male = {"m", "male", "man", "masculine"}
    female = {"f", "female", "woman", "feminine"}
    if x in male:
        return "male"
    if x in female:
        return "female"
    # sometimes embedded
    if "male" in x and "female" not in x:
        return "male"
    if "female" in x:
        return "female"
    return None


def _read_sdrf_tsv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames or []
        for row in reader:
            rows.append({k: (v if v is not None else "") for k, v in row.items()})
    return list(fieldnames), rows


def _pick_sample_id_column(fieldnames: List[str]) -> Optional[str]:
    """
    Prefer "source name" (SDRF standard). Otherwise use the first identifier-like column.
    """
    for h in fieldnames:
        if str(h).strip().lower() == "source name":
            return h
    ids = pick_identifier_columns(fieldnames)
    return ids[0] if ids else None


def analyze_sdrf_sex_coverage(sdrf_path: Path) -> Dict[str, object]:
    dataset = sdrf_path.name.replace(".sdrf.tsv", "")
    out: Dict[str, object] = {
        "dataset": dataset,
        "sdrf_path": str(sdrf_path),
        "sex_provided": "No",
        "sample_percentage_with_sex": 0.0,
        "n_samples_total": 0,
        "n_samples_with_sex": 0,
        "sex_column": "",
        "sample_id_column_used": "",
    }
    fieldnames, rows = _read_sdrf_tsv(sdrf_path)
    sex_col = _pick_sex_column(fieldnames)
    out["sex_column"] = sex_col or ""
    sample_col = _pick_sample_id_column(fieldnames)
    out["sample_id_column_used"] = sample_col or ""

    if not sex_col or not sample_col or not rows:
        return out

    # sample_id -> normalized sex (male/female); ignore non-recognized
    sample_to_sex: Dict[str, str] = {}
    all_samples: Set[str] = set()
    for r in rows:
        sid = str(r.get(sample_col, "")).strip()
        if not sid:
            continue
        all_samples.add(sid)
        sex = _normalize_sex_value(str(r.get(sex_col, "")))
        if sex is None:
            continue
        # if conflicting, keep first (coverage metric only)
        sample_to_sex.setdefault(sid, sex)

    n_total = int(len(all_samples))
    n_with = int(len(sample_to_sex))
    out["n_samples_total"] = n_total
    out["n_samples_with_sex"] = n_with
    out["sample_percentage_with_sex"] = float((100.0 * n_with / n_total) if n_total else 0.0)
    out["sex_provided"] = "Yes" if n_with > 0 else "No"
    return out


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    n = int(len(pvals))
    if n == 0:
        return pvals
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty_like(q)
    out[order] = q
    return out


def build_plasma_sex_map_for_dataset(dataset: str, sdrf_dir: Path) -> Dict[str, str]:
    """
    Return SDRF SourceName -> sex (male/female) map for a dataset.
    """
    p = resolve_sdrf_path(dataset, sdrf_dir)
    if not p:
        return {}
    fieldnames, rows = _read_sdrf_tsv(p)
    sex_col = _pick_sex_column(fieldnames)
    sample_col = _pick_sample_id_column(fieldnames)
    if not sex_col or not sample_col:
        return {}
    out: Dict[str, str] = {}
    for r in rows:
        sid = str(r.get(sample_col, "")).strip()
        if not sid:
            continue
        sex = _normalize_sex_value(str(r.get(sex_col, "")))
        if sex is None:
            continue
        out.setdefault(sid, sex)
    return out


def build_sdrf_key_to_sex(fieldnames: List[str], rows: List[Dict[str, str]]) -> Tuple[Dict[str, str], str]:
    """
    Build a robust lookup map from many SDRF identifier columns -> sex.
    Returns (lookup_map, sex_column_used).
    """
    sex_col = _pick_sex_column(fieldnames)
    if not sex_col:
        return {}, ""
    id_cols = pick_identifier_columns(fieldnames)
    # Always include source name if present
    for h in fieldnames:
        if str(h).strip().lower() == "source name" and h not in id_cols:
            id_cols.insert(0, h)
            break

    out: Dict[str, str] = {}
    for r in rows:
        sex = _normalize_sex_value(str(r.get(sex_col, "")))
        if sex is None:
            continue
        keys: Set[str] = set()
        for col in id_cols:
            raw = r.get(col, "")
            for tok in tokenize_cell(raw):
                for v in file_stem_variants(tok):
                    nk = normalize_match_key(v)
                    if nk:
                        keys.add(nk)
            for v in file_stem_variants(raw):
                nk = normalize_match_key(v)
                if nk:
                    keys.add(nk)
        for k in keys:
            out.setdefault(k, sex)
    return out, sex_col


def build_sdrf_key_to_is_normal(
    fieldnames: List[str], rows: List[Dict[str, str]]
) -> Tuple[Dict[str, bool], str]:
    """
    Build lookup map from SDRF identifiers -> is_normal (healthy/control) based on disease column.
    """
    dcol = pick_disease_column(fieldnames)
    if not dcol:
        return {}, ""
    id_cols = pick_identifier_columns(fieldnames)
    for h in fieldnames:
        if str(h).strip().lower() == "source name" and h not in id_cols:
            id_cols.insert(0, h)
            break

    out: Dict[str, bool] = {}
    for r in rows:
        raw = str(r.get(dcol, "")).strip()
        if is_sdrf_disease_placeholder(raw):
            continue
        is_norm = bool(is_healthy_sdrf_label(raw))
        keys: Set[str] = set()
        for col in id_cols:
            raw_id = r.get(col, "")
            for tok in tokenize_cell(raw_id):
                for v in file_stem_variants(tok):
                    nk = normalize_match_key(v)
                    if nk:
                        keys.add(nk)
            for v in file_stem_variants(raw_id):
                nk = normalize_match_key(v)
                if nk:
                    keys.add(nk)
        for k in keys:
            out[k] = out.get(k, False) or is_norm
    return out, dcol


def volcano_male_vs_female_plasma(out_dir: Path, sdrf_dir: Path, extended: bool = False) -> Dict[str, object]:
    """
    Build prevalence volcano (default) and sex-specific protein lists.
    If extended=True, also writes abundance volcanos and normal-only tables/plots.
    Returns basic labeled-sample counts for sex_analysis_summary.json.
    Abundance proxy uses `PeptideCount` when available, else 1 per row.
    """
    # Cache intermediates under sex_analysis/cache/ so you can delete other outputs and keep caches.
    cache_dir = out_dir / "cache" / "volcano_plasma"
    cache_dir.mkdir(parents=True, exist_ok=True)

    entry_map = E.load_entry_name_mapping_library(E.library_file)
    man = E.load_manual_mapping(E.manual_mapping_file, verbose=False)
    entry_map = {**entry_map, **man}

    # Identify plasma datasets from 02 (preferred) so list is stable.
    allowed_02 = E.get_blood_plasma_serum_datasets_from_02()
    if allowed_02 is None:
        raise RuntimeError(
            "02 plasma dataset list not found. Run B_data_preparation_and_filtering.py first "
            "or ensure 02_tissue_summary_before_and_after_filtering.csv exists."
        )

    # Pre-scan SDRFs and keep only datasets with sex info
    sex_enabled_datasets: List[str] = []
    dataset_to_keysex: Dict[str, Dict[str, str]] = {}
    dataset_to_keynormal: Dict[str, Dict[str, bool]] = {}
    for dname in sorted(allowed_02):
        sdrf_path = resolve_sdrf_path(dname, sdrf_dir)
        if not sdrf_path:
            continue
        try:
            fieldnames, srows = _read_sdrf_tsv(sdrf_path)
            key_to_sex, _ = build_sdrf_key_to_sex(fieldnames, srows)
            key_to_norm, _ = build_sdrf_key_to_is_normal(fieldnames, srows)
        except Exception:
            continue
        if not key_to_sex:
            continue
        sex_enabled_datasets.append(dname)
        dataset_to_keysex[dname] = key_to_sex
        dataset_to_keynormal[dname] = key_to_norm

    if not sex_enabled_datasets:
        raise RuntimeError("No plasma datasets with SDRF sex (male/female) found.")

    # Build sex labels and "is_normal" labels for samples per dataset using SDRF (robust key matching)
    sample_key_to_sex: Dict[str, str] = {}
    sample_key_to_is_normal: Dict[str, bool] = {}
    # NOTE: we only match to cache Sample values (dataset_sample). This is enough for volcano.
    # If needed later, we can extend this to match Run/Reference too.
    for dname in sex_enabled_datasets:
        key_to_sex = dataset_to_keysex[dname]
        key_to_norm = dataset_to_keynormal.get(dname, {})
        cache_file = E.find_cache_file(dname)
        if not cache_file:
            continue
        try:
            df = pd.read_parquet(cache_file, columns=["Sample"])
        except Exception:
            df = pd.read_parquet(cache_file)
        if "Sample" not in df.columns:
            continue
        for s in df["Sample"].dropna().astype(str).unique():
            # Try direct + normalized + stem variants
            sex_hit = None
            norm_hit_any = False
            is_norm = False
            for v in file_stem_variants(s):
                nk = normalize_match_key(v)
                if nk and nk in key_to_sex and sex_hit is None:
                    sex_hit = key_to_sex[nk]
                if nk and nk in key_to_norm:
                    norm_hit_any = True
                    is_norm = is_norm or bool(key_to_norm[nk])
            if sex_hit in ("male", "female"):
                sample_key_to_sex[f"{dname}_{s}"] = sex_hit
            if norm_hit_any:
                sample_key_to_is_normal[f"{dname}_{s}"] = bool(is_norm)

    def _build_collect_cache_key(require_normal: bool) -> str:
        """
        Cache key for the expensive aggregation step.
        Depends on:
        - which datasets are included
        - mtimes of their cache parquet files
        - mtimes of their SDRF files
        - mode (all vs normal-only)
        """
        h = hashlib.sha256()
        # Bump this tag when matching logic changes so stale caches are ignored.
        h.update(f"sex_volcano_collect_v3_add_sample_key_for_prevalence|require_normal={require_normal}".encode("utf-8"))
        for dname in sorted(sex_enabled_datasets):
            cfile = E.find_cache_file(dname)
            sfile = resolve_sdrf_path(dname, sdrf_dir)
            h.update(f"\n{dname}".encode("utf-8"))
            if cfile and cfile.exists():
                h.update(f"|cache={str(cfile)}|mtime={cfile.stat().st_mtime}".encode("utf-8"))
            if sfile and sfile.exists():
                h.update(f"|sdrf={str(sfile)}|mtime={sfile.stat().st_mtime}".encode("utf-8"))
        return h.hexdigest()[:24]


    def _collect_rows(require_normal: bool) -> pd.DataFrame:
        # Try cache first
        key = _build_collect_cache_key(require_normal=require_normal)
        stem = f"sample_protein_abundance_sex_{'normal' if require_normal else 'all'}_{key}"
        meta_path = cache_dir / f"{stem}.json"
        data_path = cache_dir / f"{stem}.parquet"
        if meta_path.exists() and data_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("cache_key") == key and meta.get("require_normal") == require_normal:
                    df_cached = pd.read_parquet(data_path)
                    if {"protein", "sex", "sample_key", "abundance"}.issubset(df_cached.columns):
                        print(f"Using cached aggregation: {data_path.name}")
                        return df_cached[["protein", "sex", "sample_key", "abundance"]].copy()
            except Exception:
                pass

        rows_local = []
        n_plasma_datasets = len(sex_enabled_datasets)
        for i_ds, dname in enumerate(sex_enabled_datasets, start=1):
            cache_file = E.find_cache_file(dname)
            if not cache_file:
                continue
            df = pd.read_parquet(cache_file)
            if "Sample" not in df.columns or "Protein" not in df.columns:
                continue
            # Keep only sex-labeled samples for this dataset
            ds_sample_to_sex = {}
            for s in df["Sample"].dropna().astype(str).unique():
                sk = f"{dname}_{s}"
                sex = sample_key_to_sex.get(sk)
                if sex not in ("male", "female"):
                    continue
                if require_normal and sample_key_to_is_normal.get(sk) is not True:
                    continue
                ds_sample_to_sex[s] = sex
            if not ds_sample_to_sex:
                continue

            tag = "normal" if require_normal else "all"
            print(f"[{tag}] [{i_ds}/{n_plasma_datasets}] {dname}: sex-labeled samples={len(ds_sample_to_sex)}")

            tmp = df.loc[df["Sample"].astype(str).isin(ds_sample_to_sex.keys()), ["Sample", "Protein"]].copy()
            if tmp.empty:
                continue
            if "PeptideCount" in df.columns:
                tmp["ab"] = pd.to_numeric(df.loc[tmp.index, "PeptideCount"], errors="coerce").fillna(0.0).astype(float)
            else:
                available_feature_cols = [c for c in FEATURE_COUNT_COLUMNS if c in df.columns]
                if available_feature_cols:
                    peptide_col = None
                    for c in df.columns:
                        cl = str(c).strip().lower()
                        if cl == "peptidesequence" or (("peptide" in cl) and ("sequence" in cl)):
                            peptide_col = c
                            break
                    cols = ["Sample", "Protein"] + ([peptide_col] if peptide_col else []) + available_feature_cols
                    tmp_feat = df.loc[tmp.index, cols].copy()
                    if peptide_col:
                        tmp_feat[peptide_col] = tmp_feat[peptide_col].astype(str).str.strip()
                        tmp_feat = tmp_feat[tmp_feat[peptide_col].ne("")]
                        tmp_feat = tmp_feat[tmp_feat[peptide_col].str.lower().ne("nan")]
                    for c in available_feature_cols:
                        tmp_feat[c] = tmp_feat[c].fillna("__NA__").astype(str).str.strip()
                    dedup_subset = ["Sample", "Protein"] + ([peptide_col] if peptide_col else []) + available_feature_cols
                    tmp_feat = tmp_feat.drop_duplicates(subset=dedup_subset)
                    feat_counts = tmp_feat.groupby(["Sample", "Protein"]).size().to_dict()
                    tmp["ab"] = [
                        float(feat_counts.get((str(s), str(p)), 0))
                        for s, p in zip(tmp["Sample"].astype(str), tmp["Protein"].astype(str))
                    ]
                else:
                    tmp["ab"] = 1.0
            tmp = tmp.dropna(subset=["Sample", "Protein"])

            raw_prots = set(tmp["Protein"].astype(str).unique())
            if not raw_prots:
                continue
            _norm_all, norm_map = E.normalize_protein_set_for_comparison(
                raw_prots, entry_name_library=entry_map, return_mapping=True
            )
            orig_to_norm = {}
            for norm_p, originals in norm_map.items():
                for o in originals:
                    orig_to_norm[o] = norm_p
            tmp["protein_norm"] = tmp["Protein"].astype(str).map(lambda p: orig_to_norm.get(p, p))

            g = tmp.groupby(["Sample", "protein_norm"], as_index=False)["ab"].sum()
            g["sex"] = g["Sample"].astype(str).map(ds_sample_to_sex)
            g = g[g["sex"].isin(["male", "female"])]
            if g.empty:
                continue
            g["sample_key"] = g["Sample"].astype(str).map(lambda s: f"{dname}_{s}")
            rows_local.extend(
                {
                    "protein": str(r["protein_norm"]),
                    "sex": str(r["sex"]),
                    "sample_key": str(r["sample_key"]),
                    "abundance": float(r["ab"]),
                }
                for _, r in g.iterrows()
            )
        df_out = pd.DataFrame(rows_local)
        # Write cache
        try:
            df_out.to_parquet(data_path, index=False)
            meta_path.write_text(
                json.dumps(
                    {
                        "cache_key": key,
                        "require_normal": require_normal,
                        "n_rows": int(len(df_out)),
                        "n_datasets": int(len(sex_enabled_datasets)),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"Wrote cached aggregation: {data_path.name}")
        except Exception:
            pass
        return df_out

    def _write_volcano(adf: pd.DataFrame, stem: str, title: str) -> None:
        if adf.empty:
            raise RuntimeError(f"No data rows for {stem}.")
        prots = sorted(adf["protein"].unique())
        out_rows = []
        eps = 1e-6
        for p in prots:
            x = adf[adf["protein"] == p]
            male_vals = x.loc[x["sex"] == "male", "abundance"].to_numpy(dtype=float)
            female_vals = x.loc[x["sex"] == "female", "abundance"].to_numpy(dtype=float)
            if len(male_vals) < 2 or len(female_vals) < 2:
                continue
            m_mean = float(np.mean(male_vals))
            f_mean = float(np.mean(female_vals))
            log2fc = float(np.log2((m_mean + eps) / (f_mean + eps)))
            pval = float(ttest_ind(male_vals, female_vals, equal_var=False, nan_policy="omit").pvalue)
            out_rows.append(
                {
                    "protein": p,
                    "n_male_samples": int(len(male_vals)),
                    "n_female_samples": int(len(female_vals)),
                    "mean_male_abundance": m_mean,
                    "mean_female_abundance": f_mean,
                    "log2fc_male_over_female": log2fc,
                    "p_value": pval,
                }
            )
        res = pd.DataFrame(out_rows)
        if res.empty:
            raise RuntimeError(f"Not enough proteins with >=2 male and >=2 female samples for {stem}.")
        res["fdr_bh"] = _bh_fdr(res["p_value"].to_numpy(dtype=float))
        res["neglog10_p"] = res["p_value"].map(lambda v: -math.log10(v) if v and v > 0 else np.nan)
        res = res.sort_values(["fdr_bh", "p_value"], ascending=[True, True]).reset_index(drop=True)

        out_csv = _safe_to_csv(res, out_dir / f"{stem}.csv")
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.scatter(res["log2fc_male_over_female"], res["neglog10_p"], s=14, alpha=0.5, color="#4C78A8", edgecolors="none")
        ax.axvline(0, color="black", linewidth=1, alpha=0.6)
        ax.axhline(-math.log10(0.05), color="black", linewidth=1, alpha=0.4, linestyle="--")
        ax.set_xlabel("log2 fold-change (male / female)")
        ax.set_ylabel("-log10(p-value) (Welch t-test)")
        ax.set_title(title)
        fig.tight_layout()
        out_png = out_dir / f"{stem}.png"
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_csv}")
        print(f"Saved: {out_png}")

    def _write_volcano_prevalence(adf: pd.DataFrame, stem: str, title: str) -> None:
        """
        Volcano based on prevalence (% samples detected) instead of abundance magnitude.
        Uses Fisher exact test on 2x2 present/absent x sex.
        """
        if adf.empty:
            raise RuntimeError(f"No data rows for {stem}.")
        if "sample_key" not in adf.columns:
            raise RuntimeError("Prevalence volcano requires 'sample_key' in aggregated table.")
        x = adf[["protein", "sex", "sample_key", "abundance"]].copy()
        x["present"] = (x["abundance"] > 0).astype(int)

        male_total = int(x.loc[x["sex"] == "male", "sample_key"].nunique())
        female_total = int(x.loc[x["sex"] == "female", "sample_key"].nunique())
        if male_total == 0 or female_total == 0:
            raise RuntimeError(f"No male/female samples available for prevalence volcano: {stem}")

        pres = x[x["present"] > 0]
        male_present_by_p = pres[pres["sex"] == "male"].groupby("protein")["sample_key"].nunique()
        female_present_by_p = pres[pres["sex"] == "female"].groupby("protein")["sample_key"].nunique()

        prots = sorted(x["protein"].unique())
        out_rows = []
        eps = 1e-9
        for p in prots:
            mp = int(male_present_by_p.get(p, 0))
            fp = int(female_present_by_p.get(p, 0))
            # Volcano prevalence plot should include proteins detected in both groups.
            if mp == 0 or fp == 0:
                continue
            ma = male_total - mp
            fa = female_total - fp
            male_pct = 100.0 * mp / male_total
            female_pct = 100.0 * fp / female_total
            log2fc = float(np.log2((male_pct + eps) / (female_pct + eps)))
            try:
                _odds, pval = fisher_exact([[mp, ma], [fp, fa]], alternative="two-sided")
                pval = float(pval)
            except Exception:
                pval = 1.0
            out_rows.append(
                {
                    "protein": p,
                    "male_present": mp,
                    "male_total": male_total,
                    "female_present": fp,
                    "female_total": female_total,
                    "male_pct_present": male_pct,
                    "female_pct_present": female_pct,
                    "log2fc_male_over_female_prevalence": log2fc,
                    "p_value": pval,
                }
            )

        res = pd.DataFrame(out_rows)
        if res.empty:
            raise RuntimeError(f"No prevalence rows for {stem}.")
        res["fdr_bh"] = _bh_fdr(res["p_value"].to_numpy(dtype=float))
        res["neglog10_p"] = res["p_value"].map(lambda v: -math.log10(v) if v and v > 0 else np.nan)
        res = res.sort_values(["fdr_bh", "p_value"], ascending=[True, True]).reset_index(drop=True)

        out_csv = _safe_to_csv(res, out_dir / f"{stem}.csv")
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.scatter(
            res["log2fc_male_over_female_prevalence"],
            res["neglog10_p"],
            s=14,
            alpha=0.55,
            color="#4C78A8",
            edgecolors="none",
        )
        ax.axvline(0, color="black", linewidth=1, alpha=0.6)
        ax.axhline(-math.log10(0.05), color="black", linewidth=1, alpha=0.4, linestyle="--")
        ax.set_xlabel("log2 fold-change of prevalence (male% / female%)")
        ax.set_ylabel("-log10(p-value) (Fisher exact test)")
        ax.set_title(title)
        fig.tight_layout()
        out_png = out_dir / f"{stem}.png"
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_csv}")
        print(f"Saved: {out_png}")

    def _write_sex_specific_only_file(adf: pd.DataFrame, stem: str) -> None:
        """
        Write proteins present in only one sex (male-only or female-only).
        """
        if "sample_key" not in adf.columns:
            return
        x = adf[["protein", "sex", "sample_key", "abundance"]].copy()
        x["present"] = (x["abundance"] > 0).astype(int)
        male_total = int(x.loc[x["sex"] == "male", "sample_key"].nunique())
        female_total = int(x.loc[x["sex"] == "female", "sample_key"].nunique())
        if male_total == 0 or female_total == 0:
            return
        pres = x[x["present"] > 0]
        male_present_by_p = pres[pres["sex"] == "male"].groupby("protein")["sample_key"].nunique()
        female_present_by_p = pres[pres["sex"] == "female"].groupby("protein")["sample_key"].nunique()

        proteins = sorted(x["protein"].unique())
        rows = []
        for p in proteins:
            mp = int(male_present_by_p.get(p, 0))
            fp = int(female_present_by_p.get(p, 0))
            if mp > 0 and fp == 0:
                rows.append(
                    {
                        "protein": p,
                        "sex_specificity": "male_only",
                        "male_present": mp,
                        "male_total": male_total,
                        "male_pct_present": 100.0 * mp / male_total,
                        "female_present": fp,
                        "female_total": female_total,
                        "female_pct_present": 0.0,
                    }
                )
            elif fp > 0 and mp == 0:
                rows.append(
                    {
                        "protein": p,
                        "sex_specificity": "female_only",
                        "male_present": mp,
                        "male_total": male_total,
                        "male_pct_present": 0.0,
                        "female_present": fp,
                        "female_total": female_total,
                        "female_pct_present": 100.0 * fp / female_total,
                    }
                )
        out = pd.DataFrame(rows)
        out_csv = _safe_to_csv(out, out_dir / f"{stem}.csv")
        print(f"Saved: {out_csv}")

    adf_all = _collect_rows(require_normal=False)
    if adf_all.empty:
        raise RuntimeError("No plasma samples with SDRF sex (male/female) matched to cache data.")
    male_n = int(adf_all.loc[adf_all["sex"] == "male", "sample_key"].nunique())
    female_n = int(adf_all.loc[adf_all["sex"] == "female", "sample_key"].nunique())
    stats_out: Dict[str, object] = {
        "n_plasma_samples_sex_labeled_male": male_n,
        "n_plasma_samples_sex_labeled_female": female_n,
        "n_plasma_samples_sex_labeled_total": male_n + female_n,
    }

    if extended:
        _write_volcano(
            adf_all,
            "volcano_male_vs_female_plasma",
            "Plasma proteins: male vs female (sex-labeled samples; abundance)",
        )
    _write_volcano_prevalence(
        adf_all,
        "volcano_male_vs_female_plasma_prevalence",
        "Plasma proteins: male vs female (prevalence; sex-labeled samples)",
    )
    _write_sex_specific_only_file(adf_all, "proteins_only_one_sex_plasma")
    _write_sex_panel_tables(out_dir / "proteins_only_one_sex_plasma.csv", out_dir)

    if extended:
        adf_norm = _collect_rows(require_normal=True)
        if adf_norm.empty:
            print(
                "Warning: No sex-labeled plasma samples could be matched AND classified as normal/healthy. "
                "Skipping normal-only volcano outputs."
            )
        else:
            _write_volcano(
                adf_norm,
                "volcano_male_vs_female_plasma_normal_only",
                "Plasma proteins: male vs female (normal-only; abundance)",
            )
            _write_volcano_prevalence(
                adf_norm,
                "volcano_male_vs_female_plasma_normal_only_prevalence",
                "Plasma proteins: male vs female (prevalence; normal-only)",
            )
            _write_sex_specific_only_file(adf_norm, "proteins_only_one_sex_plasma_normal_only")
    return stats_out


def _write_sex_panel_tables(one_sex_csv: Path, out_dir: Path) -> None:
    """Split proteins_only_one_sex_plasma.csv into panel_male_only_proteins.csv / panel_female_only_proteins.csv."""
    if not one_sex_csv.is_file():
        return
    try:
        df = pd.read_csv(one_sex_csv)
    except Exception:
        return
    if "sex_specificity" not in df.columns:
        return
    m = df[df["sex_specificity"].astype(str).str.lower() == "male_only"].copy()
    f = df[df["sex_specificity"].astype(str).str.lower() == "female_only"].copy()
    _safe_to_csv(m, out_dir / "panel_male_only_proteins.csv")
    _safe_to_csv(f, out_dir / "panel_female_only_proteins.csv")
    print(f"Saved: {out_dir / 'panel_male_only_proteins.csv'}")
    print(f"Saved: {out_dir / 'panel_female_only_proteins.csv'}")


def write_prediction_plot_options_md(out_dir: Path) -> None:
    """Documents prediction figures (generated under `prediction_plots/` when ML runs)."""
    text = """# Prediction diagnostic plots

Generated when you run `H_sex_analysis.py` **without** `--skip-ml`.

| # | Files | Description |
|---|--------|-------------|
| 01 | `01_prob_male_histogram_all_proteins.png`, `01_prob_male_histogram_peptide_atlas.png` | P(male) histogram + counts per confidence band (unknown samples) |
| 02 | `02_confidence_histogram_*.png` | Confidence 0–1 with 0.6 / 0.8 reference lines |
| 03 | `03_predicted_sex_counts_*.png` | Predicted male/female counts by confidence band |
| 04 | `04_validation_roc_prediction_all_proteins.png`, `04_validation_roc_prediction_peptide_atlas.png` | **ROC curves** for all four benchmark models on the 20% stratified validation split (male = positive); legend shows **AUROC** per model |
| 05 | `05_sex_skewed_depth_*.png`, `05_sex_skewed_depth_*.csv` | Unknowns: **detection depth** (proteins with abundance > 0) vs **fraction** of those proteins in the prevalence **male-only ∪ female-only** list (`proteins_only_one_sex_plasma.csv`); color = P(male); dashed line = median fraction by depth bin. Second panel: depth vs P(male), color = fraction. |

Numeric AUROC also appears in `prediction_*/model_validation_metrics.csv` (`roc_auc` column).

### More plot ideas (not generated by default)

- **Precision–recall curves** on the same validation split (often more informative if classes are imbalanced).
- **Confusion matrix** heatmap for the chosen model on validation data.
- **Threshold sweep**: balanced accuracy or F1 vs. classification threshold on P(male).
- **Calibration** (reliability diagram) once you have true sex labels for a subset of “unknown” samples.
"""
    (out_dir / "PREDICTION_PLOT_OPTIONS.md").write_text(text, encoding="utf-8")
    print(f"Saved: {out_dir / 'PREDICTION_PLOT_OPTIONS.md'}")


def write_sex_analysis_results(work: Path, volcano_stats: Optional[Dict[str, object]] = None) -> None:
    """Root-level summary JSON + Markdown after ML folders are populated."""
    sa = work / "sex_analysis"
    summ = {
        "sex_coverage_csv": str(sa / "sex_coverage_by_dataset.csv"),
        "prevalence_volcano_csv": str(sa / "volcano_male_vs_female_plasma_prevalence.csv"),
        "prevalence_volcano_png": str(sa / "volcano_male_vs_female_plasma_prevalence.png"),
        "proteins_only_one_sex_plasma": str(sa / "proteins_only_one_sex_plasma.csv"),
        "panel_male_only_proteins": str(sa / "panel_male_only_proteins.csv"),
        "panel_female_only_proteins": str(sa / "panel_female_only_proteins.csv"),
        "prediction_all_proteins_dir": str(sa / "prediction_all_proteins"),
        "prediction_peptide_atlas_dir": str(sa / "prediction_peptide_atlas"),
        "cache_root": str(sa / "cache"),
        "cache_volcano_plasma": str(sa / "cache" / "volcano_plasma"),
        "cache_ml_long_tables": str(sa / "cache" / "ml_long_tables"),
        "cache_peptide_atlas_accessions": str(sa / "cache" / "peptide_atlas_accessions"),
        "prediction_plots_dir": str(sa / "prediction_plots"),
    }
    if volcano_stats:
        summ["volcano_labeled_sample_counts"] = volcano_stats

    pa_json = sa / "prediction_all_proteins" / "model_validation_summary.json"
    pat_json = sa / "prediction_peptide_atlas" / "model_validation_summary.json"
    for label, path in [("all_proteins_ml", pa_json), ("peptide_atlas_ml", pat_json)]:
        if path.is_file():
            try:
                summ[label] = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                summ[label] = {"error": "could not read"}

    (sa / "sex_analysis_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(f"Saved: {sa / 'sex_analysis_summary.json'}")

    lines = [
        "# Sex analysis — results summary",
        "",
        "## Sample coverage (SDRF)",
        f"- Table: `sex_coverage_by_dataset.csv`",
        "",
        "## Plasma samples used for male vs female comparison (volcano / panels)",
    ]
    if volcano_stats:
        lines.append(f"- Sex-labeled male: **{volcano_stats.get('n_plasma_samples_sex_labeled_male', '?')}**")
        lines.append(f"- Sex-labeled female: **{volcano_stats.get('n_plasma_samples_sex_labeled_female', '?')}**")
        lines.append(f"- Total sex-labeled: **{volcano_stats.get('n_plasma_samples_sex_labeled_total', '?')}**")
    lines.extend(
        [
            "",
            "## Core outputs (under `sex_analysis/`)",
            "- `volcano_male_vs_female_plasma_prevalence.png` (+ `.csv`)",
            "- `proteins_only_one_sex_plasma.csv`",
            "- `panel_male_only_proteins.csv`, `panel_female_only_proteins.csv`",
            "",
            "## ML prediction (subfolders)",
            "- `prediction_all_proteins/` — all proteins (cap after prevalence filter, see `max_features_cap` in JSON)",
            "- `prediction_peptide_atlas/` — restricted to `external_databases/peptideatlas.csv` (same as F_database_comparison)",
            "- `prediction_plots/` — histograms, scatter comparison, PCA/UMAP, feature-importance figures (see `PREDICTION_PLOT_OPTIONS.md`)",
            "",
            "## Caches (optional to keep)",
            "- Heavy intermediates live under `sex_analysis/cache/` (`volcano_plasma/`, `ml_long_tables/`, `peptide_atlas_accessions/`).",
            "- You can delete everything else under `sex_analysis/` and rerun; keeping `cache/` speeds up the next run.",
            "",
            "See `sex_analysis_summary.json` for numeric details and validation metrics paths.",
            "",
            "## Prediction plots",
            "See `PREDICTION_PLOT_OPTIONS.md` and the `prediction_plots/` folder.",
        ]
    )
    (sa / "sex_analysis_results.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {sa / 'sex_analysis_results.md'}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Sex analysis: SDRF coverage, prevalence volcano, panels, ML prediction.")
    parser.add_argument(
        "--extended-sex-analysis",
        action="store_true",
        help="Also write abundance volcanos and normal-only prevalence/abundance outputs.",
    )
    parser.add_argument("--skip-ml", action="store_true", help="Skip prediction_all_proteins and prediction_peptide_atlas.")
    parser.add_argument("--skip-coverage", action="store_true", help="Skip sex_coverage_by_dataset.csv scan.")
    args = parser.parse_args()

    work = _work_dir()
    sdrf_dir = work / "sdrf_files"
    out_dir = work / "sex_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_coverage:
        sdrf_files = sorted(sdrf_dir.glob("*.sdrf.tsv"))
        if not sdrf_files:
            raise FileNotFoundError(f"No SDRF files found in {sdrf_dir}")

        rows = []
        for p in sdrf_files:
            try:
                rows.append(analyze_sdrf_sex_coverage(p))
            except Exception as e:
                rows.append(
                    {
                        "dataset": p.name.replace(".sdrf.tsv", ""),
                        "sdrf_path": str(p),
                        "sex_provided": "No",
                        "sample_percentage_with_sex": 0.0,
                        "n_samples_total": 0,
                        "n_samples_with_sex": 0,
                        "sex_column": "",
                        "sample_id_column_used": "",
                        "error": str(e),
                    }
                )

        df = pd.DataFrame(rows)
        if "sample_percentage_with_sex" in df.columns:
            df = df.sort_values(
                ["sex_provided", "sample_percentage_with_sex", "dataset"], ascending=[False, False, True]
            )
        out_csv = _safe_to_csv(df, out_dir / "sex_coverage_by_dataset.csv")
        print(f"Saved: {out_csv}")

    vstats = volcano_male_vs_female_plasma(out_dir=out_dir, sdrf_dir=sdrf_dir, extended=args.extended_sex_analysis)

    write_prediction_plot_options_md(out_dir)

    if not args.skip_ml:
        run_all_proteins_ml(work, sdrf_dir)
        run_peptide_atlas_ml(work, sdrf_dir)
        generate_all_prediction_plots(work, sdrf_dir)

    write_sex_analysis_results(work, volcano_stats=vstats)


if __name__ == "__main__":
    main()

