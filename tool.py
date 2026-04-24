"""
Interactive Protein Filtering Exploration Tool

This tool allows dynamic exploration of how protein inclusion changes as filtering 
thresholds are modified. It visualizes proteins as points and allows interactive 
filtering based on:
- Number of datasets a protein appears in
- Minimum number of samples per dataset
- Minimum number of unique peptides per dataset

Use the sidebar **form**: edit thresholds, then **Apply filters** (no full rerun while typing).
UniProt normalization is **cached** (recomputed only when Parquet data change).
Optional **Automatic filter search** explores many (d,s,p) combinations to trade off **z-score r** vs gene counts.

-------------------------------------------------------------------------------
HOW TO RUN (Streamlit app — do not use: python tool.py)
-------------------------------------------------------------------------------
1. Open a terminal (cmd or PowerShell).

2. Go to this project folder (same folder that contains tool.py), e.g.:
       cd /d "G:\\My Drive\\Ikasketak\\Postdoc\\Cambridge\\Cursor\\blood_proteome_analysis"

3. Install dependencies once if needed:
       pip install streamlit plotly pyarrow pandas numpy openpyxl

4. Start the app:
       streamlit run tool.py

5. Your browser opens to http://localhost:8501 (or follow the URL printed in the terminal).

Tip: Peptide counts come from B_ Parquet (PeptideCount column) — seconds, not msstats CSVs.
-------------------------------------------------------------------------------
"""

# Quick reminder: from project root →  streamlit run tool.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import os
# Add parent directory to path to import shared_utils
work_dir = Path(r"G:\My Drive\Ikasketak\Postdoc\Cambridge\Cursor\blood_proteome_analysis")
sys.path.insert(0, str(work_dir))
os.chdir(work_dir)

from shared_utils import (
    extract_uniprot_id,
    convert_entry_name_to_accession,
    load_protein_mapping_cache,
    get_default_cache_parquet_suffix,
    group_condition_to_tissue,
)
import json

DATABASE_COMPARISON_DIR = work_dir / "database_comparison"

# Directories
cache_dir = work_dir / "cache"
mapping_library_dir = work_dir / "entry_name_mapping"

# Entry name mapping library files
library_file = mapping_library_dir / "entry_name_to_accession.json"
manual_mapping_file = mapping_library_dir / "manual_id_mapping.xlsx"

# Check if Parquet is available
try:
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    st.error("pyarrow not available. Please install with 'pip install pyarrow'")
    st.stop()

# ============================================
# PROTEIN NORMALIZATION FUNCTIONS
# ============================================

@st.cache_data
def load_entry_name_mapping_library():
    """Load the entry name to accession mapping library from JSON file."""
    if library_file.exists():
        try:
            with open(library_file, 'r') as f:
                library = json.load(f)
            return library
        except Exception as e:
            st.warning(f"Could not load entry name mapping library: {e}")
    return {}

@st.cache_data
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
        
        return mapping
    
    except Exception as e:
        st.warning(f"Could not load manual mapping: {e}")
        return {}

def normalize_protein_set_for_comparison(protein_set, entry_name_library=None, return_mapping=False):
    """Normalize a set of protein identifiers to UniProt accessions for comparison."""
    normalized_set = set()
    normalized_to_original = {} if return_mapping else None
    
    # Use provided library or load it
    if entry_name_library is None:
        entry_name_library = load_entry_name_mapping_library()
        manual_mapping = load_manual_mapping()
        entry_name_library = {**entry_name_library, **manual_mapping}
    
    for protein_id in protein_set:
        protein_str = str(protein_id).strip()
        
        # Check if it's DDA format (contains '|')
        if '|' in protein_str:
            # DDA format: sp|ACCESSION|ENTRY_NAME;sp|ACCESSION2|ENTRY_NAME2
            first_entry = protein_str.split(';')[0].strip()
            uniprot_id = extract_uniprot_id(first_entry)
            if uniprot_id:
                normalized_set.add(uniprot_id)
                if return_mapping:
                    if uniprot_id not in normalized_to_original:
                        normalized_to_original[uniprot_id] = []
                    normalized_to_original[uniprot_id].append(protein_id)
            else:
                normalized_set.add(protein_id)
                if return_mapping:
                    if protein_id not in normalized_to_original:
                        normalized_to_original[protein_id] = []
                    normalized_to_original[protein_id].append(protein_id)
        else:
            # DIA format or direct UniProt accession
            uniprot_id = extract_uniprot_id(protein_str)
            if uniprot_id:
                normalized_set.add(uniprot_id)
                if return_mapping:
                    if uniprot_id not in normalized_to_original:
                        normalized_to_original[uniprot_id] = []
                    normalized_to_original[uniprot_id].append(protein_id)
            else:
                # Try to convert entry name to accession (DIA format)
                first_entry = protein_str.split(';')[0].strip()
                accession = convert_entry_name_to_accession(first_entry, entry_name_library)
                if accession:
                    normalized_set.add(accession)
                    if return_mapping:
                        if accession not in normalized_to_original:
                            normalized_to_original[accession] = []
                        normalized_to_original[accession].append(protein_id)
                else:
                    normalized_set.add(first_entry)
                    if return_mapping:
                        if first_entry not in normalized_to_original:
                            normalized_to_original[first_entry] = []
                        normalized_to_original[first_entry].append(protein_id)
    
    if return_mapping:
        return normalized_set, normalized_to_original
    return normalized_set

# ============================================
# DATA LOADING FUNCTIONS
# ============================================

@st.cache_data
def load_all_datasets():
    """Load all datasets from default Parquet cache (B_ filter: default min peptides + ENTRAP removed), plasma/serum only."""
    default_suffix = get_default_cache_parquet_suffix(cache_dir)
    cache_files = list(cache_dir.glob(f"*{default_suffix}"))
    
    if not cache_files:
        st.error(f"No cache files found in {cache_dir}")
        st.stop()
    
    all_data = []
    for cache_file in cache_files:
        try:
            df = pd.read_parquet(cache_file)
            if len(df) > 0:
                # Check if it's plasma or serum
                if 'Condition' in df.columns and len(df) > 0:
                    condition = str(df["Condition"].iloc[0])
                    if group_condition_to_tissue(condition) == "Blood Plasma/Serum":
                        dataset_name = cache_file.name.replace(default_suffix, "")
                        df['Dataset'] = dataset_name
                        all_data.append(df)
        except Exception as e:
            st.warning(f"Error loading {cache_file.name}: {e}")
            continue
    
    if not all_data:
        st.error("No plasma/serum datasets could be loaded")
        st.stop()
    
    combined = pd.concat(all_data, ignore_index=True)
    return combined

@st.cache_data(show_spinner="Loading peptide counts from B_ Parquet cache…")
def load_peptide_data(dataset_names_filter: tuple):
    """
    Unique peptides per protein per dataset — read from B_'s *_processed.parquet only.

    B_ already computed PeptideCount = n distinct peptide sequences per protein (same basis
    as the old msstats path) after decoy/ENTRAP/non-human filtering. No msstats CSV scan.
    """
    peptide_data = {}
    suffix = get_default_cache_parquet_suffix(cache_dir)

    for dataset_name in dataset_names_filter:
        cache_path = cache_dir / f"{dataset_name}{suffix}"
        if not cache_path.is_file():
            continue
        try:
            df = pd.read_parquet(cache_path, columns=["Protein", "PeptideCount"])
            if len(df) == 0 or "PeptideCount" not in df.columns:
                continue
            # One PeptideCount per protein (constant across samples in B_ cache)
            agg = df.groupby("Protein", as_index=False)["PeptideCount"].max()
            agg.columns = ["Protein", "n_unique_peptides"]
            agg["Dataset"] = dataset_name
            peptide_data[dataset_name] = agg
        except Exception:
            continue

    return peptide_data

@st.cache_data
def create_protein_dataset_summary(df, peptide_data_dict=None):
    """
    Create protein-dataset-sample summary table.
    
    One row per (protein_id, dataset_id) combination with:
    - protein_id
    - dataset_id
    - n_samples_detected: number of samples in this dataset where the protein is detected
    - n_unique_peptides: number of unique peptides supporting this protein in this dataset
    
    Also calculates global protein-level metrics:
    - protein_abundance: fraction of samples across all datasets where protein appears
    """
    # Group by protein and dataset to get per-dataset metrics
    protein_dataset_summary = df.groupby(['Protein', 'Dataset']).agg({
        'Sample': 'nunique'  # Number of unique samples where protein is detected
    }).reset_index()
    
    protein_dataset_summary.columns = ['protein_id', 'dataset_id', 'n_samples_detected']
    
    # Get unique peptide counts from peptide data if available
    if peptide_data_dict:
        peptide_summary_list = []
        for dataset_name, peptide_df in peptide_data_dict.items():
            peptide_summary_list.append(peptide_df[['Protein', 'Dataset', 'n_unique_peptides']])
        
        if peptide_summary_list:
            peptide_summary = pd.concat(peptide_summary_list, ignore_index=True)
            peptide_summary.columns = ['protein_id', 'dataset_id', 'n_unique_peptides']
            
            # Merge with protein_dataset_summary
            protein_dataset_summary = protein_dataset_summary.merge(
                peptide_summary,
                on=['protein_id', 'dataset_id'],
                how='left'
            )
            
            # Missing peptide counts: assume B_-like floor (2+ peptides) so rows aren’t silently treated as 1-peptide
            protein_dataset_summary['n_unique_peptides'] = protein_dataset_summary['n_unique_peptides'].fillna(2).astype(int)
        else:
            protein_dataset_summary['n_unique_peptides'] = 2
    else:
        protein_dataset_summary['n_unique_peptides'] = 2
    
    # Calculate protein abundance (fraction of samples across all datasets)
    protein_abundance = df.groupby('Protein').agg({
        'Sample': 'nunique'
    }).reset_index()
    protein_abundance.columns = ['protein_id', 'total_samples_all_datasets']
    
    total_samples_all_datasets = df['Sample'].nunique()
    protein_abundance['protein_abundance'] = protein_abundance['total_samples_all_datasets'] / total_samples_all_datasets
    
    # Merge abundance back to summary
    protein_dataset_summary = protein_dataset_summary.merge(
        protein_abundance[['protein_id', 'protein_abundance']],
        on='protein_id',
        how='left'
    )
    
    return protein_dataset_summary


@st.cache_data(show_spinner="Normalizing proteins to UniProt accessions (cached; runs once per cache snapshot)…")
def normalize_plasma_dataframe_proteins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map Protein IDs in the combined plasma/serum Parquet frame to UniProt accessions.
    Cached on the contents of `df` (from load_all_datasets) so this is not repeated every rerun.
    """
    entry_name_library = {**load_entry_name_mapping_library(), **load_manual_mapping()}
    normalized_proteins_list = []
    for dataset_name in sorted(df["Dataset"].astype(str).unique()):
        dataset_df = df[df["Dataset"] == dataset_name].copy()
        unique_proteins = set(dataset_df["Protein"].unique())
        protein_mapping = {}
        for orig_protein in unique_proteins:
            normalized = normalize_protein_set_for_comparison(
                {orig_protein}, entry_name_library=entry_name_library
            )
            if normalized:
                protein_mapping[orig_protein] = list(normalized)[0]
            else:
                protein_mapping[orig_protein] = orig_protein
        dataset_df["Protein"] = dataset_df["Protein"].map(protein_mapping)
        normalized_proteins_list.append(dataset_df)
    return pd.concat(normalized_proteins_list, ignore_index=True)


def apply_filtering(protein_dataset_summary, dataset_threshold, sample_threshold, peptide_threshold):
    """
    Apply filtering logic in the specified order:
    
    Step 1: Dataset-level filtering
    - Remove rows where n_samples_detected < sample_threshold
    - Remove rows where n_unique_peptides < peptide_threshold
    
    Step 2: Recompute protein-level metrics
    - n_datasets_filtered: number of remaining datasets
    - total_samples_filtered: sum of n_samples_detected of remaining datasets
    
    Step 3: Protein-level filtering
    - Remove proteins where n_datasets_filtered < dataset_threshold
    
    Returns:
        Filtered summary table with added columns for filtered metrics
    """
    # Step 1: Dataset-level filtering
    filtered = protein_dataset_summary[
        (protein_dataset_summary['n_samples_detected'] >= sample_threshold) &
        (protein_dataset_summary['n_unique_peptides'] >= peptide_threshold)
    ].copy()
    
    # Step 2: Recompute protein-level metrics
    protein_metrics = filtered.groupby('protein_id').agg({
        'dataset_id': 'nunique',
        'n_samples_detected': 'sum',
        'protein_abundance': 'first'  # Keep original abundance
    }).reset_index()
    
    protein_metrics.columns = ['protein_id', 'n_datasets_filtered', 'total_samples_filtered', 'protein_abundance']
    
    # Step 3: Protein-level filtering
    protein_metrics = protein_metrics[
        protein_metrics['n_datasets_filtered'] >= dataset_threshold
    ].copy()
    
    # Add max values for tooltips
    max_values = filtered.groupby('protein_id').agg({
        'n_unique_peptides': 'max',
        'n_samples_detected': 'max'
    }).reset_index()
    max_values.columns = ['protein_id', 'max_peptides', 'max_samples_single_dataset']
    
    protein_metrics = protein_metrics.merge(max_values, on='protein_id', how='left')
    
    return protein_metrics


def _protein_metrics_after_sp_only(
    protein_dataset_summary: pd.DataFrame,
    sample_threshold: int,
    p_thr: int,
):
    """
    Steps 1–2 of apply_filtering: per-(s,p) aggregate before the dataset-count threshold **d**.
    Enables scanning all **d** without repeating the heavy groupby on the long summary table.
    """
    p_thr = max(2, int(p_thr))
    s_th = int(sample_threshold)
    filtered = protein_dataset_summary[
        (protein_dataset_summary["n_samples_detected"] >= s_th)
        & (protein_dataset_summary["n_unique_peptides"] >= p_thr)
    ].copy()
    if len(filtered) == 0:
        return None
    protein_metrics = filtered.groupby("protein_id", sort=False).agg(
        {
            "dataset_id": "nunique",
            "n_samples_detected": "sum",
            "protein_abundance": "first",
        }
    ).reset_index()
    protein_metrics.columns = [
        "protein_id",
        "n_datasets_filtered",
        "total_samples_filtered",
        "protein_abundance",
    ]
    max_values = filtered.groupby("protein_id", sort=False).agg(
        {"n_unique_peptides": "max", "n_samples_detected": "max"}
    ).reset_index()
    max_values.columns = ["protein_id", "max_peptides", "max_samples_single_dataset"]
    protein_metrics = protein_metrics.merge(max_values, on="protein_id", how="left")
    return protein_metrics


def _evaluate_from_sp_metrics(
    protein_metrics_pre_d: pd.DataFrame,
    dataset_threshold: int,
    sample_threshold: int,
    p_thr: int,
    df_quantm_pa: pd.DataFrame,
    protein_to_gene: dict,
):
    """Apply dataset threshold **d** to precomputed protein metrics; same outputs as evaluate_filter_thresholds."""
    fm = protein_metrics_pre_d[
        protein_metrics_pre_d["n_datasets_filtered"] >= int(dataset_threshold)
    ].copy()
    if len(fm) == 0:
        return None
    fm = _attach_genes_with_map(fm, protein_to_gene)
    is_minimal = (
        int(dataset_threshold) == 1
        and int(sample_threshold) == 1
        and int(p_thr) == 2
    )
    if is_minimal:
        merged = df_quantm_pa.copy()
    else:
        merged = fm[["gene"]].drop_duplicates().merge(df_quantm_pa, on="gene", how="inner")
    n_quantm_genes = int(fm["gene"].nunique())
    n_shared = int(merged["gene"].nunique())
    r, n_pairs = pearson_r_z_scores_like_f(merged)
    return {
        "d": int(dataset_threshold),
        "s": int(sample_threshold),
        "p": int(p_thr),
        "n_quantm_genes": n_quantm_genes,
        "n_shared_peptideatlas": n_shared,
        "r_z": r,
        "n_z_pairs": n_pairs,
    }


def pearson_r_z_scores_like_f(df: pd.DataFrame):
    """
    Pearson r on quantile-normalized z-scores — same definition as
    F_database_comparison.py → density_plot_quantm_vs_peptideatlas.png
    (finite pairs only; subtracting a constant in F_ for plotting does not change r).
    """
    xcol, ycol = "z_score_quantm", "z_score_peptideatlas"
    if xcol not in df.columns or ycol not in df.columns:
        return float("nan"), 0
    xz = pd.to_numeric(df[xcol], errors="coerce")
    yz = pd.to_numeric(df[ycol], errors="coerce")
    mask = np.isfinite(xz.to_numpy()) & np.isfinite(yz.to_numpy())
    n = int(mask.sum())
    if n < 2:
        return float("nan"), n
    r = float(xz[mask].corr(yz[mask]))
    return r, n


def _attach_genes_with_map(pdf: pd.DataFrame, protein_to_gene: dict) -> pd.DataFrame:
    o = pdf.copy()
    o["gene"] = o["protein_id"].map(protein_to_gene)
    o["gene"] = o["gene"].fillna(o["protein_id"])
    return o


def evaluate_filter_thresholds(
    protein_dataset_summary: pd.DataFrame,
    df_quantm_pa: pd.DataFrame,
    protein_to_gene: dict,
    dataset_threshold: int,
    sample_threshold: int,
    peptide_threshold: int,
):
    """
    Returns metrics for one (d, s, p) triple, or None if no proteins pass.
    Merged table / r logic matches the main app (including minimal → full F_ CSV).
    """
    p_thr = max(2, int(peptide_threshold))
    pre = _protein_metrics_after_sp_only(
        protein_dataset_summary, int(sample_threshold), p_thr
    )
    if pre is None:
        return None
    return _evaluate_from_sp_metrics(
        pre,
        int(dataset_threshold),
        int(sample_threshold),
        p_thr,
        df_quantm_pa,
        protein_to_gene,
    )


def _search_grid_values(max_s: int, max_p: int, grain: str):
    """Finite grid for (samples, peptides) to keep search tractable."""
    if grain == "coarse":
        s_vals = sorted(
            {1, 2, 3, max(1, max_s // 4), max(1, max_s // 2), max_s}
        )
    elif grain == "medium":
        s_vals = list(range(1, min(max_s, 15) + 1))
        if max_s > 15:
            s_vals.append(max_s)
        s_vals = sorted(set(s_vals))
    elif grain == "aggressive":
        # Stricter cuts; bounded width (fast path: one groupby per (s,p), then scan d).
        cap_s = min(max_s, 32)
        s_vals = list(range(1, cap_s + 1))
        if max_s > cap_s:
            s_vals.extend(
                [max_s, max(1, max_s - 1), max(1, max_s - 3), max(1, max_s // 2)]
            )
        s_vals = sorted(set(x for x in s_vals if 1 <= x <= max_s))
        cap_p = min(max_p, 18)
        p_vals = list(range(2, cap_p + 1))
        if max_p > cap_p:
            p_vals.extend(
                [
                    max_p,
                    max(2, max_p - 1),
                    max(2, max_p // 2),
                    max(2, (2 * max_p) // 3),
                ]
            )
        p_vals = sorted(set(x for x in p_vals if 2 <= x <= max_p))
    else:
        # fine
        s_vals = list(range(1, min(max_s, 25) + 1))
        if max_s > 25:
            s_vals.extend(
                sorted(
                    {
                        max_s,
                        max(1, max_s * 3 // 4),
                        max(1, max_s // 2),
                    }
                )
            )
        s_vals = sorted(set(s_vals))
        p_vals = list(range(2, min(max_p, 12) + 1))
        if max_p > 12:
            p_vals.append(max_p)
        p_vals = sorted(set(p_vals))
        return [x for x in s_vals if 1 <= x <= max_s], [x for x in p_vals if 2 <= x <= max_p]

    if grain != "aggressive":
        p_vals = list(range(2, min(max_p, 12) + 1))
        if max_p > 12:
            p_vals.append(max_p)
        p_vals = sorted(set(p_vals))
    return [x for x in s_vals if 1 <= x <= max_s], [x for x in p_vals if 2 <= x <= max_p]


def estimate_search_work(max_d: int, max_s: int, max_p: int, grain: str):
    """(n_s, n_p, n_groupby_heavy, n_scored_triples)."""
    s_vals, p_vals = _search_grid_values(max_s, max_p, grain)
    ns, np_ = len(s_vals), len(p_vals)
    return ns, np_, ns * np_, ns * np_ * max_d


def run_filter_search(
    protein_dataset_summary: pd.DataFrame,
    df_quantm_pa: pd.DataFrame,
    protein_to_gene: dict,
    max_d: int,
    max_s: int,
    max_p: int,
    grain: str,
    min_shared: int,
    min_quantm: int,
    weight_r: float,
    objective: str = "weighted",
    retention_gamma: float = 1.0,
):
    """
    Enumerate (d,s,p), keep rows meeting min_shared & min_quantm, rank by score.

    objective:
      - "weighted": score = w*r_z + (1-w) * retention^γ  (γ<1 → down-weight gene loss = more aggressive toward r)
      - "max_r": sort by r_z only, tie-break Shared PA
    """
    s_vals, p_vals = _search_grid_values(max_s, max_p, grain)
    relaxed = evaluate_filter_thresholds(
        protein_dataset_summary, df_quantm_pa, protein_to_gene, 1, 1, 2
    )
    if relaxed is None:
        return pd.DataFrame()
    base_q = max(1, relaxed["n_quantm_genes"])
    base_sh = max(1, relaxed["n_shared_peptideatlas"])

    rows = []
    gamma = float(np.clip(retention_gamma, 0.05, 1.0))
    w = float(np.clip(weight_r, 0.0, 1.0))

    # One heavy groupby per (s,p), then cheap scans over d (was repeating groupby for every d).
    for s in s_vals:
        for p in p_vals:
            p_thr = max(2, int(p))
            pre = _protein_metrics_after_sp_only(protein_dataset_summary, int(s), p_thr)
            if pre is None:
                continue
            for d in range(1, max_d + 1):
                ev = _evaluate_from_sp_metrics(
                    pre,
                    d,
                    int(s),
                    p_thr,
                    df_quantm_pa,
                    protein_to_gene,
                )
                if ev is None:
                    continue
                if ev["n_shared_peptideatlas"] < min_shared:
                    continue
                if ev["n_quantm_genes"] < min_quantm:
                    continue
                if not np.isfinite(ev["r_z"]):
                    continue
                fq = ev["n_quantm_genes"] / base_q
                fsh = ev["n_shared_peptideatlas"] / base_sh
                f_ret = float(np.sqrt(max(fq, 1e-9) * max(fsh, 1e-9)))
                if objective == "max_r":
                    score = float(ev["r_z"])
                else:
                    score = w * float(ev["r_z"]) + (1.0 - w) * (f_ret**gamma)
                rows.append({**ev, "score": score, "retention_geom": f_ret})

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    if objective == "max_r":
        out = out.sort_values(
            ["r_z", "n_shared_peptideatlas"], ascending=[False, False]
        ).reset_index(drop=True)
    else:
        out = out.sort_values("score", ascending=False).reset_index(drop=True)
    return out


# ============================================
# STREAMLIT APP
# ============================================

st.set_page_config(
    page_title="QuantM vs PeptideAtlas – Interactive Filtering",
    layout="wide"
)

st.title("QuantM vs PeptideAtlas – Interactive scatter (log abundances)")

st.markdown("""
Compare **QuantM** plasma to **PeptideAtlas** using the same gene-level table as **F_** (`density_plot_quantm_vs_peptideatlas_data.csv`).
Data are already **2+ peptides** and filtered in **B_**; here you apply **extra** rules on top.

**Interactive filters** (edit in the sidebar form, then **Apply filters** — the form avoids a full rerun on every keystroke)

**Optional:** **Automatic filter search** explores (datasets / samples / peptides); use **max r_z** ranking, **γ < 1**, or **aggressive** grid to favour correlation over gene counts.
- Minimum **datasets** — distinct **plasma/serum Parquet cache datasets** (same names as in the peptide-count line) where a protein is seen  
- Minimum **samples per dataset**  
- Minimum **unique peptides per dataset** (floor **2**, same as B_ — cannot go lower)

**Stats shown**
- **QuantM genes** and **overlap with PeptideAtlas**: *after / before* these interactive filters  
- **Correlation**: Pearson **r on z-scores** (same as **`density_plot_quantm_vs_peptideatlas.png`** in F_), not on raw log abundances  
- At **minimum** thresholds (1 dataset, 1 sample, 2 peptides), **r** and **n** use the **full F_ CSV** so they match that figure exactly  
""")

# Load data (Parquet from B_ is already decoy/ENTRAP/non-human filtered — no need to filter again)
with st.spinner("Loading QuantM plasma/serum datasets from cache..."):
    df = load_all_datasets()

with st.spinner("Loading QuantM vs PeptideAtlas abundance table..."):
    quantm_vs_pa_path = DATABASE_COMPARISON_DIR / "density_plot_quantm_vs_peptideatlas_data.csv"
    if not quantm_vs_pa_path.exists():
        st.error(f"File not found: {quantm_vs_pa_path}")
        st.stop()
    df_quantm_pa = pd.read_csv(quantm_vs_pa_path)

_z_required = {"z_score_quantm", "z_score_peptideatlas", "gene"}
if not _z_required.issubset(df_quantm_pa.columns):
    st.error(
        f"CSV missing required columns {_z_required - set(df_quantm_pa.columns)}. "
        "Re-run **F_database_comparison.py** to regenerate `density_plot_quantm_vs_peptideatlas_data.csv`."
    )
    st.stop()

# Normalize proteins to UniProt accessions (cached; not rerun on every Streamlit interaction)
df = normalize_plasma_dataframe_proteins(df)

with st.spinner("Loading peptide counts from B_ Parquet cache (PeptideCount column)…"):
    # Tuple = hashable cache key for @st.cache_data (do not pass dicts — they break caching)
    _peptide_filter = tuple(sorted(df["Dataset"].astype(str).unique()))
    peptide_data_dict = load_peptide_data(dataset_names_filter=_peptide_filter)

n_parquet_datasets = len(_peptide_filter)
st.caption(
    f"Peptide counts from Parquet for **{len(peptide_data_dict)}** / **{n_parquet_datasets}** plasma/serum "
    f"cache datasets (same **PeptideCount** as B_; missing counts default to **2** in the summary)."
)

with st.spinner("Creating QuantM protein–dataset summary table..."):
    protein_dataset_summary = create_protein_dataset_summary(df, peptide_data_dict=peptide_data_dict)

# Filter limits: "minimum datasets" counts distinct dataset_id rows (each Parquet / cache name), not PXD-only groups
max_datasets = int(protein_dataset_summary["dataset_id"].nunique())
max_samples = int(protein_dataset_summary["n_samples_detected"].max())
max_peptides = max(2, int(protein_dataset_summary["n_unique_peptides"].max()))

# Protein → gene (needed for “before filter” stats and plot)
with st.spinner("Loading protein→gene mapping…"):
    _protein_to_gene = load_protein_mapping_cache()


def _attach_genes(pdf: pd.DataFrame) -> pd.DataFrame:
    return _attach_genes_with_map(pdf, _protein_to_gene)


# --- Baseline (before interactive filters): same universe as sidebar sliders start from ---
_summary_genes = _attach_genes(protein_dataset_summary)
n_quantm_genes_before = int(_summary_genes["gene"].nunique())
_merged_before = _summary_genes[["gene"]].drop_duplicates().merge(df_quantm_pa, on="gene", how="inner")
n_shared_pepa_before = int(_merged_before["gene"].nunique())
_corr_before, _n_corr_before = pearson_r_z_scores_like_f(_merged_before)
# Reference: full F_ merged table (same genes / r as density_plot_quantm_vs_peptideatlas.png, “All proteins”)
_corr_f_full_csv, _n_f_full_csv = pearson_r_z_scores_like_f(df_quantm_pa)

# --- Automatic filter search (main column) ---
_relax = evaluate_filter_thresholds(
    protein_dataset_summary, df_quantm_pa, _protein_to_gene, 1, 1, 2
)
if _relax is None:
    _relax = {"n_shared_peptideatlas": 1, "n_quantm_genes": 1}
_default_min_shared = int(max(1, round(0.7 * _relax["n_shared_peptideatlas"])))
_default_min_quantm = int(max(1, round(0.7 * _relax["n_quantm_genes"])))

with st.expander("**Automatic filter search** (balance r vs gene counts)", expanded=False):
    st.markdown(
        """
        Searches a grid of **(d, s, p)** and ranks combinations.

        - **Weighted** mode: `score = w × r_z + (1−w) × retention^γ`  
          **retention** = geometric mean of (QuantM genes / baseline) and (Shared PA / baseline); baseline = **(1,1,2)**.  
          **γ < 1** makes gene loss “cheaper” → rankings favour **higher r_z** more aggressively.
        - **Max r_z only** mode: sort purely by **r_z**, then by **Shared PA** as tie-breaker (strongest push for correlation).

        **To chase higher r:** use **aggressive** grid density, set **Max r_z only** or **w = 1**, lower **γ**, and **lower** the two minimum-gene inputs so stricter (s,p,d) combinations are allowed.

        **r_z** is Pearson on z-scores (F_ density plot). There is **no guarantee** of reaching e.g. 0.7 without large gene loss.
        """
    )
    c_a, c_b = st.columns(2)
    with c_a:
        search_grain = st.selectbox(
            "Search density",
            ["coarse", "medium", "fine", "aggressive"],
            index=1,
            help="**aggressive** = many more sample/peptide thresholds (slower, explores stricter filters).",
        )
    with c_b:
        search_objective = st.selectbox(
            "Ranking objective",
            [
                "weighted",
                "max_r",
            ],
            format_func=lambda x: {
                "weighted": "Weighted score (w and γ)",
                "max_r": "Maximize r_z only (tie-break: Shared PA)",
            }[x],
            index=0,
            key="search_objective",
        )

    c_w, c_g, c_n = st.columns(3)
    with c_w:
        weight_r = st.slider(
            "Weight on r_z (**w**), weighted mode",
            min_value=0.0,
            max_value=1.0,
            value=0.65,
            step=0.05,
            disabled=(search_objective == "max_r"),
            help="Set **1.0** for correlation-only blending. Ignored in “Max r_z only”.",
            key="search_weight_r",
        )
    with c_g:
        retention_gamma = st.slider(
            "Retention exponent **γ** (weighted only)",
            min_value=0.05,
            max_value=1.0,
            value=1.0,
            step=0.05,
            disabled=(search_objective == "max_r"),
            help="**Lower γ** → retention term stays high even when genes drop → more aggressive toward r_z.",
            key="search_retention_gamma",
        )
    with c_n:
        top_n_show = st.number_input("Show top N rows", min_value=5, max_value=100, value=20, step=5)

    c_d, c_e = st.columns(2)
    with c_d:
        min_shared_search = st.number_input(
            "Require ≥ this many genes shared with PeptideAtlas",
            min_value=1,
            max_value=max(int(_relax["n_shared_peptideatlas"]), 1),
            value=min(_default_min_shared, max(int(_relax["n_shared_peptideatlas"]), 1)),
            step=50,
            help="Filters out combinations that keep too few overlapping genes.",
        )
    with c_e:
        min_quantm_search = st.number_input(
            "Require ≥ this many QuantM genes",
            min_value=1,
            max_value=max(int(_relax["n_quantm_genes"]), 1),
            value=min(_default_min_quantm, max(int(_relax["n_quantm_genes"]), 1)),
            step=50,
        )

    _ns, _np, _nheavy, _nscore = estimate_search_work(
        max_datasets, max_samples, max_peptides, search_grain
    )
    st.caption(
        f"Search size: **{_ns}×{_np}** sample/peptide pairs → **{_nheavy}** aggregate passes, "
        f"**{_nscore}** scored (d,s,p) (after optimization this should finish in **seconds to a few minutes**, not tens of minutes)."
    )

    if st.button("Run search", type="secondary"):
        with st.spinner("Searching filter grid…"):
            res_df = run_filter_search(
                protein_dataset_summary,
                df_quantm_pa,
                _protein_to_gene,
                max_datasets,
                max_samples,
                max_peptides,
                search_grain,
                int(min_shared_search),
                int(min_quantm_search),
                float(weight_r),
                objective=str(search_objective),
                retention_gamma=float(retention_gamma),
            )
        st.session_state["filter_search_results"] = res_df

    if "filter_search_results" in st.session_state and len(st.session_state["filter_search_results"]) > 0:
        res = st.session_state["filter_search_results"]
        st.success(f"**{len(res)}** combinations passed your constraints (sorted by score).")
        show = res.head(int(top_n_show)).copy()
        show.columns = [
            "d",
            "s",
            "p",
            "QuantM genes",
            "Shared PA",
            "r (z)",
            "n z-pairs",
            "Score",
            "Retention geom.",
        ]
        st.dataframe(show, use_container_width=True, hide_index=True)

        sub = res.head(int(top_n_show))
        labels = []
        for i, (_, row) in enumerate(sub.iterrows()):
            labels.append(
                f"#{i+1}  d={int(row['d'])}  s={int(row['s'])}  p={int(row['p'])}  |  "
                f"r={float(row['r_z']):.4f}  shared={int(row['n_shared_peptideatlas'])}"
            )
        pick = st.selectbox(
            "Apply one suggestion to the sidebar filters",
            options=list(range(len(labels))),
            format_func=lambda i: labels[i],
        )
        if st.button("Apply selected combination", type="primary"):
            row = sub.iloc[int(pick)]
            st.session_state.applied_filters = {
                "d": int(row["d"]),
                "s": int(row["s"]),
                "p": max(2, int(row["p"])),
            }
            st.rerun()
    elif "filter_search_results" in st.session_state:
        st.warning("No combination satisfied your minimum gene constraints. Lower the two minimums or use a coarser search.")

# Sidebar: **st.form** so editing numbers does not rerun the app (no grey overlay until Submit)
st.sidebar.header("Filtering Controls")
st.sidebar.caption(
    "Values below only apply after **Apply filters** inside the form (Streamlit does not rerun on each keystroke)."
)

if "applied_filters" not in st.session_state:
    st.session_state.applied_filters = {"d": 1, "s": 1, "p": 2}

with st.sidebar.form("filter_form", clear_on_submit=False):
    st.subheader("Minimum number of datasets")
    st.caption(
        f"**Scale:** **1**–**{max_datasets}** — distinct **cache dataset** names (Parquet files). "
        f"You have **{n_parquet_datasets}** plasma/serum datasets with peptide metadata."
    )
    form_d = st.number_input(
        "Datasets (min)",
        min_value=1,
        max_value=max_datasets,
        value=int(st.session_state.applied_filters["d"]),
        step=1,
        help="Protein kept only if it meets sample & peptide rules in at least this many distinct cache datasets.",
    )

    st.subheader("Minimum samples per dataset")
    st.caption(
        f"**Scale:** **1**–**{max_samples}** — distinct samples per cache dataset."
    )
    form_s = st.number_input(
        "Samples (min)",
        min_value=1,
        max_value=max_samples,
        value=int(st.session_state.applied_filters["s"]),
        step=1,
        help="Drop (protein, dataset) pairs with fewer supporting samples before counting datasets.",
    )

    st.subheader("Minimum unique peptides per dataset")
    st.caption(f"**Scale:** **2**–**{max_peptides}** — B_ floor is 2.")
    form_p = st.number_input(
        "Peptides (min)",
        min_value=2,
        max_value=max_peptides,
        value=max(2, int(st.session_state.applied_filters["p"])),
        step=1,
        help="Minimum distinct peptides (Parquet **PeptideCount**) per protein in that dataset.",
    )

    submitted_filters = st.form_submit_button("Apply filters", type="primary")

if submitted_filters:
    st.session_state.applied_filters = {
        "d": int(form_d),
        "s": int(form_s),
        "p": max(2, int(form_p)),
    }
    st.rerun()

if st.sidebar.button("Reset to minimum thresholds"):
    st.session_state.applied_filters = {"d": 1, "s": 1, "p": 2}
    st.rerun()

st.sidebar.caption(
    f"**Currently applied:** datasets ≥ **{st.session_state.applied_filters['d']}**, "
    f"samples ≥ **{st.session_state.applied_filters['s']}**, peptides ≥ **{st.session_state.applied_filters['p']}**."
)

dataset_threshold = int(st.session_state.applied_filters["d"])
sample_threshold = int(st.session_state.applied_filters["s"])
peptide_threshold = max(2, int(st.session_state.applied_filters["p"]))

# Apply filtering
filtered_metrics = apply_filtering(
    protein_dataset_summary,
    dataset_threshold,
    sample_threshold,
    peptide_threshold
)

if len(filtered_metrics) == 0:
    st.warning("No proteins meet the current filtering criteria. Try relaxing the thresholds.")
    st.info(
        f"**Before** your interactive filters: **{n_quantm_genes_before}** QuantM genes, "
        f"**{n_shared_pepa_before}** also in PeptideAtlas."
    )
else:
    filtered_metrics = _attach_genes(filtered_metrics)

    # At minimum thresholds, use the full F_ table so r and n match density_plot_quantm_vs_peptideatlas.png
    is_minimal_filters = (
        dataset_threshold == 1 and sample_threshold == 1 and peptide_threshold == 2
    )
    if is_minimal_filters:
        merged = df_quantm_pa.copy()
    else:
        merged = filtered_metrics[["gene"]].drop_duplicates().merge(
            df_quantm_pa, on="gene", how="inner"
        )

    n_quantm_genes_after = int(filtered_metrics["gene"].nunique())
    n_shared_pepa_after = int(merged["gene"].nunique())
    corr_after, n_corr_pairs = pearson_r_z_scores_like_f(merged)

    st.subheader("Summary (interactive filters: after / before)")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "QuantM genes (after / before tool filters)",
            f"{n_quantm_genes_after} / {n_quantm_genes_before}",
        )
    with c2:
        st.metric(
            "Shared with PeptideAtlas (after / before)",
            f"{n_shared_pepa_after} / {n_shared_pepa_before}",
        )
    with c3:
        st.metric(
            "Pearson r (z-scores, as F_ plot)",
            f"{corr_after:.4f}" if np.isfinite(corr_after) else "N/A",
            help=f"Finite z-score pairs used: n = {n_corr_pairs}",
        )
    if is_minimal_filters and np.isfinite(corr_after):
        st.caption(
            f"**Matches `density_plot_quantm_vs_peptideatlas.png`:** Pearson r = **{corr_after:.4f}**, "
            f"**n = {n_corr_pairs}** genes (full F_ merged table; z-scores, same as F_)."
        )
        if n_quantm_genes_after != n_shared_pepa_after:
            st.caption(
                "The first metric counts **unique genes** in your plasma/serum Parquet summary; "
                "the scatter / r use **every gene** in the F_ CSV (can be larger if F_ included more genes)."
            )
    elif np.isfinite(_corr_before):
        st.caption(
            f"Correlation on **parquet-derived** gene ∩ F_ table (before stricter sliders): "
            f"**r = {_corr_before:.4f}**, n = {_n_corr_before}. "
            f"**F_ full CSV** (PNG reference): **r = {_corr_f_full_csv:.4f}**, n = {_n_f_full_csv}."
        )
    elif np.isfinite(_corr_f_full_csv):
        st.caption(
            f"**F_ full CSV** (always matches PNG when that file is current): **r = {_corr_f_full_csv:.4f}**, n = {_n_f_full_csv}."
        )

    st.subheader("Scatter: QuantM vs PeptideAtlas (filtered genes)")

    if len(merged) > 0:
        fig = px.scatter(
            merged,
            x="log_abundance_quantm",
            y="log_abundance_peptideatlas",
            opacity=0.35,
            marginal_x="histogram",
            marginal_y="histogram",
            labels={
                "log_abundance_quantm": "log10 QuantM abundance",
                "log_abundance_peptideatlas": "log10 PeptideAtlas abundance",
            },
        )
        # Only the main scatter supports marker.size; marginals are histograms
        fig.update_traces(
            selector=dict(type="scatter"),
            marker=dict(size=5, line=dict(width=0)),
        )
        fig.update_layout(height=700, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        _gene_meta = (
            filtered_metrics.groupby("gene", as_index=False)
            .agg(
                n_datasets_filtered=("n_datasets_filtered", "max"),
                total_samples_filtered=("total_samples_filtered", "max"),
                max_peptides=("max_peptides", "max"),
            )
        )
        merged_table = merged.merge(_gene_meta, on="gene", how="left")

        with st.expander("View merged table (filtered genes)"):
            st.dataframe(
                merged_table[
                    [
                        "gene",
                        "log_abundance_quantm",
                        "log_abundance_peptideatlas",
                        "n_datasets_filtered",
                        "total_samples_filtered",
                        "max_peptides",
                    ]
                ].sort_values("log_abundance_quantm", ascending=False),
                use_container_width=True,
            )
            csv = merged_table.to_csv(index=False)
            st.download_button(
                label="Download merged QuantM–PeptideAtlas data (filtered)",
                data=csv,
                file_name=f"quantm_peptideatlas_filtered_d{dataset_threshold}_s{sample_threshold}_p{peptide_threshold}.csv",
                mime="text/csv",
                key="download_filtered_quantm_pepa",
            )
