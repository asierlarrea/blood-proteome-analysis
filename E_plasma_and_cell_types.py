"""
Plasma and Cell Types Comparison Script
Compares protein presence between plasma/serum datasets and cell type datasets.
Creates scatter plots and UpSet plots showing overlaps and differences.

This script loads data from cache (created by 2_data_analysis_cache_and_filtering.py).
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path
import os
import sys
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

# Set working directory and add to path for imports
work_dir = Path(r"G:\My Drive\Ikasketak\Postdoc\Cambridge\Cursor\blood_proteome_analysis")
sys.path.insert(0, str(work_dir))
os.chdir(work_dir)

# Import shared utilities
from shared_utils import (
    load_protein_mapping_cache,
    convert_protein_to_gene,
    extract_uniprot_id,
    convert_entry_name_to_accession,
    compute_protein_feature_counts,
    get_default_cache_parquet_suffix,
    group_condition_to_tissue,
    load_entry_name_mapping_library,
    load_manual_mapping,
)
import json

# Directories
cache_dir = work_dir / "cache"
data_prep_output_dir = work_dir / "data_preparation_output"
msstats_dir = work_dir / "msstats"  # Shared raw data
plots_dir = work_dir / "tissue_data_and_comparison"
tables_dir = work_dir / "tables"
mapping_library_dir = work_dir / "entry_name_mapping"
plots_dir.mkdir(parents=True, exist_ok=True)
tables_dir.mkdir(parents=True, exist_ok=True)

# Deepdive reduced outputs (merged from E_cell_contamination_deepdive.py)
deepdive_dir = work_dir / "contamination_deepdive"
deepdive_dir.mkdir(parents=True, exist_ok=True)


def _safe_name(s: str) -> str:
    return str(s).replace("/", "_").replace("\\", "_").replace(" ", "_").replace(":", "_")


def _tokenize_protein_group_text(text: str) -> set:
    s = str(text).strip()
    if not s:
        return set()
    tokens = set()
    for t in re.split(r"[;,|/\\\s]+", s):
        tt = t.strip()
        if tt:
            tokens.add(tt)
    for t in re.findall(r"[A-Z0-9]{5,10}(?:-\d+)?", s):
        tokens.add(t.strip())
    return tokens


def _minmax_01(series: pd.Series) -> pd.Series:
    """Min-max normalize a numeric series to [0,1]. Returns 0 for constant/invalid series."""
    vals = pd.to_numeric(series, errors="coerce").astype(float)
    finite = vals.replace([np.inf, -np.inf], np.nan)
    vmin = finite.min(skipna=True)
    vmax = finite.max(skipna=True)
    if pd.isna(vmin) or pd.isna(vmax) or vmax <= vmin:
        return pd.Series(np.zeros(len(series), dtype=float), index=series.index)
    return (finite - vmin) / (vmax - vmin)


def _signed_log1p(series: pd.Series) -> pd.Series:
    """Signed log1p transform preserving direction around zero."""
    vals = pd.to_numeric(series, errors="coerce").astype(float)
    return np.sign(vals) * np.log1p(np.abs(vals))


def _compute_feature_abundance_by_protein(df: pd.DataFrame, protein_col: str = "Protein") -> dict:
    """Return protein abundance using feature-count definition.

    Priority:
    1) `PeptideCount` column from B_ cache (already feature-based after update),
    2) compute from raw feature columns if available,
    3) fallback handled by `compute_protein_feature_counts` (row counts).
    """
    if df is None or len(df) == 0 or protein_col not in df.columns:
        return {}
    if "PeptideCount" in df.columns:
        vals = pd.to_numeric(df["PeptideCount"], errors="coerce")
        tmp = pd.DataFrame({protein_col: df[protein_col].astype(str), "pc": vals})
        tmp = tmp[tmp[protein_col].str.strip().ne("")]
        tmp = tmp[tmp[protein_col].str.lower().ne("nan")]
        if len(tmp) > 0:
            return tmp.groupby(protein_col)["pc"].max().fillna(0).astype(float).to_dict()
    return {k: float(v) for k, v in compute_protein_feature_counts(df, protein_col=protein_col).items()}


def _load_geyer_list(entry_name_library: dict, xlsx_path: Path) -> set:
    """
    Load external Geyer contaminant list where each cell can contain a protein group.
    Returns normalized protein identifiers (UniProt accessions).
    """
    if not xlsx_path.exists():
        return set()
    raw = pd.read_excel(xlsx_path, dtype=str)
    out = set()
    for _, r in raw.iterrows():
        for c in raw.columns:
            v = r.get(c, None)
            if pd.isna(v):
                continue
            cell = str(v).strip()
            if not cell:
                continue
            toks = _tokenize_protein_group_text(cell)
            norm = normalize_protein_set_for_comparison(toks, entry_name_library=entry_name_library)
            out |= set(norm)
    return out


def _write_list_csv(out_dir: Path, filename: str, proteins: set) -> None:
    pd.DataFrame({"protein": sorted(set(proteins))}).to_csv(out_dir / filename, index=False)


def write_contamination_deepdive_outputs(
    entry_name_library: dict,
    normalized_plasma_serum_proteins_by_sample: dict,
    plasma_sample_to_dataset: dict,
    normalized_plasma_serum_proteins: set,
    scatter_data_dict: dict,
    proteins_only_plasma_celltype: dict,
    valid_cell_types: list,
) -> None:
    """
    Reduced deepdive outputs per cell type:
    - all_{cell}_protein_list.csv (shared plasma+cell type)
    - unique_{cell}_protein_list.csv (only plasma+cell type)
    - contaminant_candidates_unique_{cell}.csv with columns:
        protein_name, datasets_where_it_appeared, n_samples_present, co_detected_with_same_list
    - For Platelet/Erythrocyte: contaminant_candidates_geyer_{cell}.csv (same columns) and overlay plot with yellow dots.
      Others: overlay plot with blue+orange only.
    """
    if not valid_cell_types:
        return

    # Precompute union of unique proteins across cell types (keeps indexing work small).
    union_unique = set()
    shared_by_ct = {}
    unique_by_ct = {}
    for ct in valid_cell_types:
        df_sc = scatter_data_dict.get(ct)
        if df_sc is None or len(df_sc) == 0:
            continue
        shared = set(df_sc["Protein"].astype(str).tolist())
        unique = set(proteins_only_plasma_celltype.get(ct, set()))
        shared_by_ct[ct] = shared
        unique_by_ct[ct] = unique
        union_unique |= unique

    # Build per-sample: which union_unique proteins are present.
    sample_unique_present = {}
    for s, pset in normalized_plasma_serum_proteins_by_sample.items():
        present = set(pset) & union_unique
        if present:
            sample_unique_present[s] = present

    # Index unique protein -> list of plasma samples containing it.
    unique_to_samples = {p: [] for p in union_unique}
    for s, present in sample_unique_present.items():
        for p in present:
            unique_to_samples[p].append(s)

    # Load Geyer lists (only used for platelet/erythrocyte).
    geyer_platelet_xlsx = deepdive_dir / "geyer_contaminant_list" / "geyer_platelet_contaminants.xlsx"
    geyer_ery_xlsx = deepdive_dir / "geyer_contaminant_list" / "geyer_erythrocyte_contaminants.xlsx"
    geyer_platelet = _load_geyer_list(entry_name_library, geyer_platelet_xlsx)
    geyer_ery = _load_geyer_list(entry_name_library, geyer_ery_xlsx)

    for ct in valid_cell_types:
        safe_ct = _safe_name(ct)
        out_dir = deepdive_dir / safe_ct
        out_dir.mkdir(parents=True, exist_ok=True)

        shared = shared_by_ct.get(ct, set())
        unique = unique_by_ct.get(ct, set())
        _write_list_csv(out_dir, f"all_{safe_ct}_protein_list.csv", shared)
        _write_list_csv(out_dir, f"unique_{safe_ct}_protein_list.csv", unique)

        # Candidate table for unique proteins (no scoring; direct support table).
        rows = []
        for p in sorted(unique):
            present_samples = unique_to_samples.get(p, [])
            datasets = sorted({plasma_sample_to_dataset.get(s, "Unknown") for s in present_samples})
            partners = set()
            for s in present_samples:
                partners |= sample_unique_present.get(s, set())
            partners.discard(p)
            rows.append(
                {
                    "protein_name": p,
                    "datasets_where_it_appeared": "; ".join(datasets),
                    "n_samples_present": int(len(present_samples)),
                    "co_detected_with_same_list": "; ".join(sorted(partners)),
                }
            )
        pd.DataFrame(rows).to_csv(out_dir / f"contaminant_candidates_unique_{safe_ct}.csv", index=False)

        # Geyer candidate table (only for platelet/erythrocyte; intersect with shared proteins).
        geyer_set = set()
        plot_name = f"plasma_vs_{safe_ct}_unique_scatter.png"
        if ct.lower() == "platelet":
            geyer_set = geyer_platelet & set(shared)
            plot_name = "platelet_geyer_vs_plasma_scatter.png"
        elif ct.lower() == "erythrocyte":
            geyer_set = geyer_ery & set(shared)
            plot_name = "erythrocyte_geyer_vs_plasma_scatter.png"

        if geyer_set:
            # Build quick index for geyer proteins in plasma samples.
            geyer_to_samples = {p: [] for p in geyer_set}
            for s, pset in normalized_plasma_serum_proteins_by_sample.items():
                present = set(pset) & geyer_set
                for p in present:
                    geyer_to_samples[p].append(s)
            g_rows = []
            for p in sorted(geyer_set):
                present_samples = geyer_to_samples.get(p, [])
                datasets = sorted({plasma_sample_to_dataset.get(s, "Unknown") for s in present_samples})
                partners = set()
                for s in present_samples:
                    partners |= (set(normalized_plasma_serum_proteins_by_sample.get(s, set())) & geyer_set)
                partners.discard(p)
                g_rows.append(
                    {
                        "protein_name": p,
                        "datasets_where_it_appeared": "; ".join(datasets),
                        "n_samples_present": int(len(present_samples)),
                        "co_detected_with_same_list": "; ".join(sorted(partners)),
                    }
                )
            pd.DataFrame(g_rows).to_csv(out_dir / f"contaminant_candidates_geyer_{safe_ct}.csv", index=False)

        # Plot overlay using already computed scatter table for this cell type.
        df_sc = scatter_data_dict.get(ct)
        if df_sc is None or len(df_sc) == 0:
            continue
        dfp = df_sc.copy()
        dfp["Protein"] = dfp["Protein"].astype(str)
        dfp["In_Geyer_List"] = dfp["Protein"].isin(geyer_set) if geyer_set else False

        # Blue: other shared proteins; Orange: unique shared; Yellow: geyer subset (if any)
        other = dfp[(~dfp["Only_Plasma_CellType"]) & (~dfp["In_Geyer_List"])]
        orange = dfp[(dfp["Only_Plasma_CellType"]) & (~dfp["In_Geyer_List"])]
        yellow = dfp[dfp["In_Geyer_List"]]

        fig, ax = plt.subplots(figsize=(8.5, 7.0))
        if len(other):
            ax.scatter(
                other["Plasma_Serum_Pct"], other["Cell_Type_Pct"],
                s=13, alpha=0.45, color="#4C78A8", edgecolors="none",
                label=f"Other shared proteins (n={len(other)})",
            )
        if len(orange):
            ax.scatter(
                orange["Plasma_Serum_Pct"], orange["Cell_Type_Pct"],
                s=26, alpha=0.95, color="orange", edgecolors="black", linewidth=0.25,
                label=f"Unique shared proteins (n={len(orange)})",
            )
        if len(yellow):
            ax.scatter(
                yellow["Plasma_Serum_Pct"], yellow["Cell_Type_Pct"],
                s=38, alpha=0.95, color="yellow", edgecolors="black", linewidth=0.4,
                label=f"Geyer list (n={len(yellow)})",
            )
        ax.plot([0, 100], [0, 100], "r--", alpha=0.5, linewidth=1)
        ax.set_xlim(0, 102)
        ax.set_ylim(0, 102)
        ax.set_xlabel("Plasma/Serum (% samples)")
        ax.set_ylabel(f"{ct} (% samples)")
        ax.set_title(f"Plasma vs {ct} proteins (shared proteins only)")
        ax.legend(loc="lower right", fontsize=8, frameon=True)
        fig.tight_layout()
        fig.savefig(out_dir / plot_name, dpi=300, bbox_inches="tight")
        plt.close(fig)

# Entry name mapping library files
library_file = mapping_library_dir / "entry_name_to_accession.json"
manual_mapping_file = mapping_library_dir / "manual_id_mapping.xlsx"

NORMALIZATION_STATUS_FILE = cache_dir / "normalization_status.json"
_normalization_skip_logged = False

def cache_is_normalized():
    """True if B_ certified that cache Protein column is already normalized (UniProt accessions)."""
    if not NORMALIZATION_STATUS_FILE.exists():
        return False
    try:
        with open(NORMALIZATION_STATUS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("status") == "ok"
    except Exception:
        return False

# Check for optional libraries
try:
    import pyarrow
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    print("Warning: pyarrow not available. Cannot load from cache.")

try:
    from upsetplot import UpSet, from_contents
    UPSET_AVAILABLE = True
except ImportError:
    UPSET_AVAILABLE = False
    print("Note: upsetplot not available. Install with 'pip install upsetplot' for UpSet plots.")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("Note: networkx not available. Install with 'pip install networkx' for network plots.")

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Note: scikit-learn not available. Install with 'pip install scikit-learn' for PCA plots.")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Note: UMAP not available. Install with 'pip install umap-learn' for UMAP plots.")

from scipy import stats

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# ============================================
# HELPER FUNCTIONS
# ============================================

# Canonical label for plasma/serum in plots and CSVs (matches 02's Tissue_CellType)
CANONICAL_PLASMA_SERUM_LABEL = 'Blood Plasma/Serum'

def _is_plasma_serum_label(label):
    """True if label represents plasma/serum (so we don't show it as a separate 'cell type' in comparison plots)."""
    if label is None:
        return False
    return 'plasma' in str(label).lower() or 'serum' in str(label).lower()

def get_blood_plasma_serum_datasets_from_02():
    """Return the set of dataset names for Blood Plasma/Serum from 02_tissue_summary (B_ output).
    Used so E_ uses the same dataset list as 02 and protein counts match (e.g. 5864)."""
    path_02 = data_prep_output_dir / "02_tissue_summary_before_and_after_filtering.csv"
    if not path_02.exists():
        return None
    try:
        df = pd.read_csv(path_02)
        if 'Tissue_CellType' not in df.columns or 'Datasets' not in df.columns:
            return None
        row = df[df['Tissue_CellType'] == 'Blood Plasma/Serum']
        if len(row) == 0:
            return None
        datasets_str = row['Datasets'].iloc[0]
        if pd.isna(datasets_str):
            return None
        return set(d.strip() for d in str(datasets_str).split(','))
    except Exception:
        return None


def _tissue_to_02_safe_name(tissue_name):
    """Convert tissue/cell type name to 02 filename suffix (e.g. 'T Cell' -> 'T_Cell')."""
    return str(tissue_name).replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '').replace(':', '_').strip()


def _cell_type_to_02_tissue_name(cell_type):
    """Map E_ get_cell_type return value (lowercase) to 02 Tissue_CellType for file lookup."""
    c = (cell_type or '').strip().lower()
    mapping = {
        'plasma': 'Blood Plasma/Serum', 'serum': 'Blood Plasma/Serum',
        't cell': 'T Cell', 'cd4': 'T Cell', 'cd8': 'T Cell',
        'nk cell': 'NK Cell', 'platelet': 'Platelet', 'monocyte': 'Monocyte',
        'b cell': 'B Cell', 'erythrocyte': 'Erythrocyte', 'neutrophil': 'Neutrophil',
        'dendritic cell': 'Dendritic Cell', 'granulocyte': 'Granulocyte',
        'macrophage': 'Macrophage', 'eosinophil': 'Erythrocyte', 'basophil': 'Granulocyte',
    }
    return mapping.get(c, cell_type)  # fallback: use as-is (e.g. title case)


def get_02_protein_set_for_tissue(tissue_name):
    """Load the set of correct protein IDs for a tissue from B_ export (02_proteins_{safe}.txt).
    tissue_name: 02-style e.g. 'Blood Plasma/Serum', 'T Cell', or E_-style e.g. 't cell'."""
    if not tissue_name:
        return None
    # Map E_ cell type to 02 name if needed
    name_02 = _cell_type_to_02_tissue_name(tissue_name) if tissue_name == tissue_name.lower() else tissue_name
    safe = _tissue_to_02_safe_name(name_02)
    path_txt = (data_prep_output_dir / f"02_proteins_{safe}.txt").resolve()
    if not path_txt.exists():
        return None
    try:
        with open(path_txt, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f if line.strip())
    except Exception:
        return None


def get_02_blood_plasma_serum_protein_set():
    """Load the set of correct protein IDs for Blood Plasma/Serum (legacy path or 02_proteins_*)."""
    path_legacy = (data_prep_output_dir / "02_blood_plasma_serum_proteins.txt").resolve()
    if path_legacy.exists():
        try:
            with open(path_legacy, 'r', encoding='utf-8') as f:
                return set(line.strip() for line in f if line.strip())
        except Exception:
            pass
    return get_02_protein_set_for_tissue('Blood Plasma/Serum')


def get_02_correct_proteins_after_plasma_serum():
    """Return Correct_Proteins_After for Blood Plasma/Serum from 02, or None."""
    path_02 = data_prep_output_dir / "02_tissue_summary_before_and_after_filtering.csv"
    if not path_02.exists():
        return None
    try:
        df = pd.read_csv(path_02)
        if 'Tissue_CellType' not in df.columns or 'Correct_Proteins_After' not in df.columns:
            return None
        row = df[df['Tissue_CellType'] == 'Blood Plasma/Serum']
        if len(row) == 0:
            return None
        val = row['Correct_Proteins_After'].iloc[0]
        if pd.isna(val):
            return None
        return int(val)
    except Exception:
        return None


def _raw_condition_for_tissue_grouping(dataset_name, data):
    """Raw Condition string (B-style: mode from column, then | split) for group_condition_to_tissue."""
    raw = None
    if data is not None and len(data) > 0 and 'Condition' in data.columns:
        vals = data['Condition'].dropna().astype(str).str.strip()
        vals = vals[vals != '']
        vals = vals[~vals.str.lower().isin(['nan', 'none', 'na'])]
        if len(vals) > 0:
            mode_val = vals.mode()
            if len(mode_val) > 0:
                raw = str(mode_val.iloc[0]).strip()
    if not raw:
        parts = re.split(r'[-_]', str(dataset_name))
        for i in range(len(parts) - 1, -1, -1):
            candidate = (parts[i] or '').strip()
            if candidate and not candidate.isdigit():
                raw = candidate
                break
        if not raw:
            raw = 'unknown'
    if '|' in raw:
        raw = raw.split('|')[0].strip()
    return raw


def is_plasma_serum(dataset_name, data):
    """True if this dataset is Blood Plasma/Serum in B's sense (plasma, serum, blood, etc.)."""
    raw = _raw_condition_for_tissue_grouping(dataset_name, data)
    return group_condition_to_tissue(raw) == CANONICAL_PLASMA_SERUM_LABEL

def get_cell_type(dataset_name, data):
    """Tissue/cell-type label matching B's 02 CSV (shared_utils.group_condition_to_tissue)."""
    raw = _raw_condition_for_tissue_grouping(dataset_name, data)
    tissue = group_condition_to_tissue(raw)
    if tissue == CANONICAL_PLASMA_SERUM_LABEL:
        return CANONICAL_PLASMA_SERUM_LABEL
    return tissue

def find_cache_file(dataset_name):
    """Find cache file for a dataset (default B_ filter only: 2 peptide + ENTRAP removed)."""
    default_suffix = get_default_cache_parquet_suffix(cache_dir)
    # Try exact match with default suffix first
    cache_file = cache_dir / f"{dataset_name}{default_suffix}"
    if cache_file.exists():
        return cache_file
    # Try with common suffixes (for datasets split by condition)
    for suffix in ['-plasma', '-serum', '-erythrocyte', '-DDA', '-DIA', '-blood_serum', '-blood_plasma']:
        cache_file = cache_dir / f"{dataset_name}{suffix}{default_suffix}"
        if cache_file.exists():
            return cache_file
    # Try to find any cache file that starts with the dataset name and uses default suffix
    base_name = dataset_name.split('-')[0]
    cache_files = list(cache_dir.glob(f"{base_name}*{default_suffix}"))
    if cache_files:
        return cache_files[0]
    return None

def load_metadata(dataset_name):
    """Load metadata JSON file for a dataset if available."""
    metadata_file = cache_dir / f"{dataset_name}_metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"  Warning: Could not load metadata for {dataset_name}: {e}")
    return None

def load_all_datasets_from_cache():
    """Load all datasets from the default Parquet cache only (B_ default: 2 peptide filter + ENTRAP removed).
    Uses get_default_cache_parquet_suffix() so we do not load _processed_unfiltered.parquet or other
    variants; counts then match 02_tissue_summary_before_and_after_filtering."""
    if not PARQUET_AVAILABLE:
        print("Error: pyarrow not available. Cannot load from cache.")
        print("Please run B_data_preparation_and_filtering.py first to create cache.")
        return {}
    
    suffix = get_default_cache_parquet_suffix(cache_dir)
    all_data = {}
    cache_files = list(cache_dir.glob(f"*{suffix}"))
    
    print(f"Loading datasets from cache (default filter only: *{suffix})...")
    print(f"Found {len(cache_files)} cached datasets")
    
    for cache_file in cache_files:
        dataset_name = cache_file.name.replace(suffix, "")
        try:
            df = pd.read_parquet(cache_file)
            if len(df) > 0:
                all_data[dataset_name] = df
                print(f"  Loaded {dataset_name}: {df['Protein'].nunique()} proteins, {df['Sample'].nunique()} samples")
        except Exception as e:
            print(f"  Warning: Error loading {dataset_name}: {e}")
    
    return all_data

# ============================================
# INTRA-DATASET ANALYSIS FUNCTION
# ============================================

def intra_dataset_analysis(data, dataset_name):
    """Perform intra-dataset analysis.
    Uses pre-calculated metadata from B_data_preparation if available for faster processing.
    """
    print(f"Analyzing {dataset_name}...")
    
    results = {}
    
    # Try to load metadata first (pre-calculated by B_)
    metadata = load_metadata(dataset_name)
    
    # 1. Proteins per sample
    valid_data = data[data['Sample'].notna() & (data['Sample'].astype(str).str.strip() != '')].copy()
    
    if len(valid_data) == 0:
        print(f"  WARNING: No valid samples found for {dataset_name}")
        results['proteins_per_sample'] = pd.DataFrame(columns=['Sample', 'ProteinCount'])
        results['proteins_by_sample'] = {}
        return results
    
    # Use metadata if available, otherwise calculate
    if metadata and 'proteins_per_sample' in metadata:
        proteins_per_sample_data = []
        for sample, count in metadata['proteins_per_sample'].items():
            proteins_per_sample_data.append({'Sample': sample, 'ProteinCount': count})
        proteins_per_sample = pd.DataFrame(proteins_per_sample_data)
        proteins_per_sample = proteins_per_sample.sort_values('ProteinCount')
        print(f"  Using pre-calculated proteins per sample from metadata")
    else:
        proteins_per_sample = valid_data.groupby('Sample')['Protein'].nunique().reset_index()
        proteins_per_sample.columns = ['Sample', 'ProteinCount']
    
    if (proteins_per_sample['ProteinCount'] < 0).any():
        negative_samples = proteins_per_sample[proteins_per_sample['ProteinCount'] < 0]
        print(f"  WARNING: {dataset_name} has negative protein counts for samples: {negative_samples['Sample'].tolist()}")
        proteins_per_sample = proteins_per_sample[proteins_per_sample['ProteinCount'] >= 0].copy()
    
    proteins_per_sample = proteins_per_sample.sort_values('ProteinCount')
    
    results['proteins_per_sample'] = proteins_per_sample
    
    # 2. Shared proteins across all samples
    # Use metadata if available, otherwise calculate
    if metadata and 'protein_sets_per_sample' in metadata:
        proteins_by_sample = {}
        for sample, protein_list in metadata['protein_sets_per_sample'].items():
            proteins_by_sample[sample] = set(protein_list)
        print(f"  Using pre-calculated protein sets from metadata")
    else:
        # One groupby pass instead of filtering once per sample (much faster for many samples).
        # Use data (same as original) so result is identical: dict[sample] = set(proteins in that sample).
        proteins_by_sample = data.groupby('Sample')['Protein'].apply(lambda x: set(x.unique())).to_dict()
    
    if len(proteins_by_sample) > 0:
        sample_list = list(proteins_by_sample.values())
        shared_proteins = sample_list[0].copy()
        for sample_proteins in sample_list[1:]:
            shared_proteins &= sample_proteins
            if len(shared_proteins) == 0:
                break
        results['shared_proteins'] = list(shared_proteins)
        results['n_shared_proteins'] = len(shared_proteins)
    else:
        results['shared_proteins'] = []
        results['n_shared_proteins'] = 0
    
    results['proteins_by_sample'] = proteins_by_sample
    
    # Print summary statistics
    mean_prot_sample = proteins_per_sample['ProteinCount'].mean()
    sd_prot_sample = proteins_per_sample['ProteinCount'].std()
    
    print(f"  Proteins per sample - Mean: {mean_prot_sample:.2f}, SD: {sd_prot_sample:.2f}")
    print(f"  Shared proteins across all samples: {results.get('n_shared_proteins', 0)}")
    print()
    
    return results

# ============================================
# PLOTTING FUNCTIONS (from proteomics_analysis.py)
# ============================================

def create_intra_dataset_plots(all_data, intra_results, plots_dir, tables_dir):
    """Create intra-dataset plots: violin plots, protein presence plots, dataset count plots."""
    print("=" * 50)
    print("CREATING INTRA-DATASET PLOTS")
    print("=" * 50)
    
    # Helper function to categorize conditions for coloring
    def categorize_condition_for_dotplot(condition):
        condition_lower = condition.lower().strip()
        if 'plasma' in condition_lower or 'serum' in condition_lower:
            return 'Plasma/Serum'
        elif 'erythrocyte' in condition_lower or 'red blood cell' in condition_lower:
            return 'Erythrocytes'
        elif 'platelet' in condition_lower:
            return 'Platelets'
        else:
            return 'Immune Cells'
    
    # Helper function to group conditions
    def group_condition_to_tissue_for_violin(condition):
        condition_lower = condition.lower().strip()
        if 'plasma' in condition_lower or 'serum' in condition_lower:
            return 'Blood Plasma/Serum'
        elif 'erythrocyte' in condition_lower or 'red blood cell' in condition_lower:
            return 'Erythrocyte'
        elif 'platelet' in condition_lower:
            return 'Platelet'
        elif 'cd4' in condition_lower or 'cd8' in condition_lower or 't cell' in condition_lower:
            return 'T Cell'
        elif 'b cell' in condition_lower or 'cd19' in condition_lower:
            return 'B Cell'
        elif 'blood' in condition_lower and ('plasma' not in condition_lower and 'serum' not in condition_lower):
            return 'Blood (other)'
        else:
            return condition
    
    # 1. Missing fraction box plot
    print("Creating missing fraction box plot per dataset...")
    
    # Calculate missing_fraction for each sample in each dataset
    missing_fraction_data = []
    
    for dataset_name, results in intra_results.items():
        proteins_by_sample = results.get('proteins_by_sample', {})
        proteins_per_sample_df = results.get('proteins_per_sample', pd.DataFrame())
        
        if not proteins_by_sample or len(proteins_per_sample_df) == 0:
            continue
        
        # Get total proteins in dataset
        all_proteins_in_dataset = set()
        for sample_proteins in proteins_by_sample.values():
            all_proteins_in_dataset.update(sample_proteins)
        n_proteins_total = len(all_proteins_in_dataset)
        
        # Pre-filtering: exclude datasets with <3 samples
        if len(proteins_by_sample) < 3:
                    continue
        
        # Pre-filtering: ensure n_proteins_total > 0
        if n_proteins_total == 0:
            continue
        
        # Get condition for coloring
        if dataset_name in all_data:
            condition = str(all_data[dataset_name]['Condition'].iloc[0])
            if "|" in condition:
                condition = condition.split("|")[0].strip()
            condition_group = categorize_condition_for_dotplot(condition)
        else:
            condition_group = 'Unknown'
        
        # Calculate missing_fraction for each sample
        for sample, sample_proteins in proteins_by_sample.items():
            n_proteins_sample = len(sample_proteins)
            
            # Pre-filtering: exclude samples where n_proteins_sample is missing/NaN
            if pd.isna(n_proteins_sample) or n_proteins_sample < 0:
                continue
            
            # Calculate missing_fraction
            missing_fraction = 1 - (n_proteins_sample / n_proteins_total) if n_proteins_total > 0 else 1.0
            
            missing_fraction_data.append({
                'dataset_id': dataset_name,
                'missing_fraction': missing_fraction,
                'group': condition_group
            })
    
    if len(missing_fraction_data) == 0:
        print("  Warning: No valid data for missing fraction plot")
        return
    
    df_missing = pd.DataFrame(missing_fraction_data)
    
    # Calculate median missing_fraction per dataset and order datasets
    dataset_medians = df_missing.groupby('dataset_id')['missing_fraction'].median().sort_values(ascending=True)
    dataset_order = dataset_medians.index.tolist()
    df_missing['dataset_id'] = pd.Categorical(df_missing['dataset_id'], categories=dataset_order, ordered=True)
    df_missing = df_missing.sort_values('dataset_id')
    
    # Get color mapping
    group_colors = {
        'Plasma/Serum': '#FFD700',
        'Immune Cells': '#0066FF',
        'Erythrocytes': '#FF0000',
        'Platelets': '#00CC00'
    }
    
    # Create box plot
    n_datasets = len(dataset_order)
    fig_width = max(15, n_datasets * 0.3)  # Increase width for many datasets
    fig, ax = plt.subplots(figsize=(fig_width, 8))
    
    # Prepare data for boxplot
    box_data = []
    box_positions = []
    box_colors_list = []
    
    for idx, dataset_id in enumerate(dataset_order):
        dataset_data = df_missing[df_missing['dataset_id'] == dataset_id]['missing_fraction'].values
        if len(dataset_data) > 0:
            box_data.append(dataset_data)
            box_positions.append(idx + 1)
            # Get color for this dataset
            group = df_missing[df_missing['dataset_id'] == dataset_id]['group'].iloc[0]
            box_colors_list.append(group_colors.get(group, '#808080'))
    
    # Create boxplot
    bp = ax.boxplot(box_data, positions=box_positions, widths=0.6, patch_artist=True, 
                   showmeans=False, showfliers=True)
    
    # Color boxes
    for patch, color in zip(bp['boxes'], box_colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Set labels: dataset identifier on X axis
    ax.set_xticks(box_positions)
    ax.set_xticklabels(dataset_order, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Dataset (ordered by median missing fraction)', fontsize=12)
    ax.set_ylabel('Fraction of proteins missing per sample', fontsize=12)
    ax.set_title('Within-dataset protein missingness across plasma samples', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)  # Fixed range: 0 to 1
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=group_colors[group], label=group, edgecolor='black', linewidth=1) 
                      for group in ['Plasma/Serum', 'Immune Cells', 'Erythrocytes', 'Platelets'] 
                      if group in df_missing['group'].unique()]
    ax.legend(handles=legend_elements, title='Group', loc='best', fontsize=10)
    
    # Optional text annotation
    ax.text(0.02, 0.98, 'Each box represents one dataset', 
           transform=ax.transAxes, fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    output_file = plots_dir / "protein_missing_fraction_per_dataset.png"
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.savefig(plots_dir / "protein_missing_fraction_per_dataset.pdf", bbox_inches='tight')
    plt.close()
    print(f"  Saved: protein_missing_fraction_per_dataset.png and .pdf ({n_datasets} datasets)")
    
    # 3. Dataset count plot
    print("\nCreating dataset count plot (main groups + sub-groups)...")
    
    def categorize_dataset(condition):
        condition_lower = condition.lower().strip()
        if 'plasma' in condition_lower and 'serum' not in condition_lower:
            return ('Plasma/Serum', 'Plasma')
        elif 'serum' in condition_lower:
            return ('Plasma/Serum', 'Serum')
        elif 'erythrocyte' in condition_lower or 'red blood cell' in condition_lower:
            return ('Erythrocytes', 'Erythrocyte')
        elif 'platelet' in condition_lower:
            return ('Platelets', 'Platelet')
        elif 'cd4' in condition_lower:
            return ('Immune Cells', 'CD4')
        elif 'cd8' in condition_lower:
            return ('Immune Cells', 'CD8')
        elif 'dendritic' in condition_lower:
            return ('Immune Cells', 'Dendritic Cell')
        elif 'granulocyte' in condition_lower:
            return ('Immune Cells', 'Granulocyte')
        elif 'b cell' in condition_lower or 'cd19' in condition_lower:
            return ('Immune Cells', 'B Cell')
        elif 'monocyte' in condition_lower:
            return ('Immune Cells', 'Monocyte')
        elif 'natural killer' in condition_lower or 'nk' in condition_lower:
            return ('Immune Cells', 'Natural Killer')
        elif 't cell' in condition_lower and 'cd4' not in condition_lower and 'cd8' not in condition_lower:
            return ('Immune Cells', 'T Cell (other)')
        else:
            return ('Other', condition)
    
    main_group_counts = {}
    sub_group_counts = {}
    
    for dataset_name, data in all_data.items():
        condition = str(data['Condition'].iloc[0])
        if "|" in condition:
            condition = condition.split("|")[0].strip()
        main_group, sub_group = categorize_dataset(condition)
        if main_group not in main_group_counts:
            main_group_counts[main_group] = 0
        main_group_counts[main_group] += 1
        if sub_group not in sub_group_counts:
            sub_group_counts[sub_group] = 0
        sub_group_counts[sub_group] += 1
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    main_groups_sorted = sorted(main_group_counts.items(), key=lambda x: x[1], reverse=True)
    main_group_names = [g[0] for g in main_groups_sorted]
    main_group_values = [g[1] for g in main_groups_sorted]
    
    bars1 = ax1.bar(main_group_names, main_group_values, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Main Group', fontsize=12)
    ax1.set_ylabel('Number of Datasets', fontsize=12)
    ax1.set_title('Dataset Count by Main Group', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=45)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    sub_groups_sorted = sorted(sub_group_counts.items(), key=lambda x: x[1], reverse=True)
    sub_group_names = [g[0] for g in sub_groups_sorted]
    sub_group_values = [g[1] for g in sub_groups_sorted]
    
    bars2 = ax2.bar(sub_group_names, sub_group_values, color='darkgreen', alpha=0.7)
    ax2.set_xlabel('Sub-Group', fontsize=12)
    ax2.set_ylabel('Number of Datasets', fontsize=12)
    ax2.set_title('Dataset Count by Sub-Group', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(plots_dir / "dataset_count_by_tissue.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("  Saved: dataset_count_by_tissue.png")
    
    dataset_count_df = pd.DataFrame({
        'Main_Group': [g[0] for g in main_groups_sorted],
        'Count': [g[1] for g in main_groups_sorted]
    })
    dataset_count_df.to_csv(tables_dir / "dataset_count_main_groups.csv", index=False)
    
    sub_group_count_df = pd.DataFrame({
        'Sub_Group': [g[0] for g in sub_groups_sorted],
        'Count': [g[1] for g in sub_groups_sorted]
    })
    sub_group_count_df.to_csv(tables_dir / "dataset_count_sub_groups.csv", index=False)
    print("  Saved: dataset_count_main_groups.csv and dataset_count_sub_groups.csv")
    
    # Save summary tables
    for dataset_name, results in intra_results.items():
        results.get('proteins_per_sample', pd.DataFrame(columns=['Sample', 'ProteinCount'])).to_csv(
            tables_dir / f"{dataset_name}_proteins_per_sample.csv", index=False
        )
        if len(results.get('shared_proteins', [])) > 0:
            pd.DataFrame({'Protein': results.get('shared_proteins', [])}).to_csv(
                tables_dir / f"{dataset_name}_shared_proteins.csv", index=False
            )

def create_inter_dataset_plots(all_data, plots_dir, tables_dir):
    """Create inter-dataset analysis plots grouped by condition."""
    print("=" * 50)
    print("INTER-DATASET ANALYSIS (Grouped by Condition)")
    print("=" * 50)
    
    # 1. Dot plot: Protein counts per dataset
    print("\nCreating overall inter-dataset plots...")
    print("Creating dot plot: Protein counts per dataset...")
    
    def categorize_condition_for_dotplot(condition):
        condition_lower = condition.lower().strip()
        if 'plasma' in condition_lower or 'serum' in condition_lower:
            return 'Plasma/Serum'
        elif 'erythrocyte' in condition_lower or 'red blood cell' in condition_lower:
            return 'Erythrocytes'
        elif 'platelet' in condition_lower:
            return 'Platelets'
        else:
            return 'Immune Cells'
    
    all_dataset_protein_counts = []
    for dataset_name, data in all_data.items():
        n_proteins = data['Protein'].nunique()
        condition = str(data['Condition'].iloc[0])
        if "|" in condition:
            condition = condition.split("|")[0].strip()
        condition_group = categorize_condition_for_dotplot(condition)
        all_dataset_protein_counts.append({
            'Dataset': dataset_name,
            'Proteins': n_proteins,
            'Condition': condition,
            'Group': condition_group
        })
    
    df_protein_counts = pd.DataFrame(all_dataset_protein_counts)
    df_protein_counts = df_protein_counts.sort_values('Proteins')
    
    fig, ax = plt.subplots(figsize=(max(12, len(df_protein_counts) * 0.5), 8))
    
    group_colors = {
        'Plasma/Serum': '#FFD700',
        'Immune Cells': '#0066FF',
        'Erythrocytes': '#FF0000',
        'Platelets': '#00CC00'
    }
    
    plot_colors = [group_colors[group] for group in df_protein_counts['Group']]
    
    ax.scatter(range(len(df_protein_counts)), df_protein_counts['Proteins'], 
              c=plot_colors, s=100, alpha=0.8, edgecolors='black', linewidth=1.5)
    ax.set_xticks([])  # Remove dataset names from X-axis
    ax.set_xlabel('Dataset (sorted by protein count)', fontsize=12)
    ax.set_ylabel('Number of Proteins', fontsize=12)
    ax.set_title('Protein Counts per Dataset', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=group_colors[group], label=group, edgecolor='black', linewidth=1) 
                       for group in ['Plasma/Serum', 'Immune Cells', 'Erythrocytes', 'Platelets']]
    ax.legend(handles=legend_elements, title='Group', loc='best', fontsize=10)
    
    plt.tight_layout()
    output_file = plots_dir / "protein_number_per_dataset.png"
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved: protein_number_per_dataset.png (with grouped colors, no dataset names)")
    
    # Split plot: Plasma/Serum only
    print("Creating dot plot: Protein counts per dataset (Plasma/Serum only)...")
    df_plasma_serum = df_protein_counts[df_protein_counts['Group'] == 'Plasma/Serum'].copy()
    df_plasma_serum = df_plasma_serum.sort_values('Proteins')
    
    if len(df_plasma_serum) > 0:
        fig, ax = plt.subplots(figsize=(max(12, len(df_plasma_serum) * 0.5), 8))
        ax.scatter(range(len(df_plasma_serum)), df_plasma_serum['Proteins'], 
                  c=group_colors['Plasma/Serum'], s=100, alpha=0.8, edgecolors='black', linewidth=1.5)
        ax.set_xticks([])  # Remove dataset names from X-axis
        ax.set_xlabel('Dataset (sorted by protein count)', fontsize=12)
        ax.set_ylabel('Number of Proteins', fontsize=12)
        ax.set_title('Protein Counts per Dataset - Plasma/Serum', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        output_file = plots_dir / "protein_number_per_dataset_plasma_serum.png"
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  Saved: protein_number_per_dataset_plasma_serum.png")
    
    # Split plot: Cell Types only
    print("Creating dot plot: Protein counts per dataset (Cell Types only)...")
    df_cell_types = df_protein_counts[df_protein_counts['Group'] != 'Plasma/Serum'].copy()
    df_cell_types = df_cell_types.sort_values('Proteins')
    
    if len(df_cell_types) > 0:
        fig, ax = plt.subplots(figsize=(max(12, len(df_cell_types) * 0.5), 8))
        plot_colors_cell = [group_colors[group] for group in df_cell_types['Group']]
        ax.scatter(range(len(df_cell_types)), df_cell_types['Proteins'], 
                  c=plot_colors_cell, s=100, alpha=0.8, edgecolors='black', linewidth=1.5)
        ax.set_xticks([])  # Remove dataset names from X-axis
        ax.set_xlabel('Dataset (sorted by protein count)', fontsize=12)
        ax.set_ylabel('Number of Proteins', fontsize=12)
        ax.set_title('Protein Counts per Dataset - Cell Types', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Legend for cell types only
        cell_type_groups = df_cell_types['Group'].unique()
        legend_elements_cell = [Patch(facecolor=group_colors[group], label=group, edgecolor='black', linewidth=1) 
                                for group in cell_type_groups]
        ax.legend(handles=legend_elements_cell, title='Group', loc='best', fontsize=10)
        
        plt.tight_layout()
        output_file = plots_dir / "protein_number_per_dataset_cell_types.png"
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  Saved: protein_number_per_dataset_cell_types.png")
    
    # Methodology metadata map (for plasma/serum UMAP coloring variants)
    methodology_map = {}
    try:
        methodology_file = work_dir / "methodology_analysis" / "methodology_dataset_summary.csv"
        if methodology_file.exists():
            mdf = pd.read_csv(methodology_file)
            for _, r in mdf.iterrows():
                ds = str(r.get("Dataset", "")).strip()
                if ds:
                    methodology_map[ds] = {
                        "Acquisition": str(r.get("Acquisition", "Unknown")).strip() or "Unknown",
                        "Fractionation": str(r.get("Fractionation", "Unknown")).strip() or "Unknown",
                        "Depletion": str(r.get("Depletion", "Unknown")).strip() or "Unknown",
                    }
    except Exception as e:
        print(f"  Warning: could not load methodology metadata for UMAP variants: {e}")

    protein_count_map = {
        str(r["Dataset"]): int(r["Proteins"]) for _, r in df_protein_counts.iterrows()
    }

    def _protein_count_bin(n):
        if n <= 500:
            return "1-500"
        if n <= 1000:
            return "500-1000"
        if n <= 2000:
            return "1000-2000"
        return ">2000"

    # Group datasets by Condition (same mapping as B / 02: blood, plasma, serum -> Blood Plasma/Serum)
    def canonical_condition(condition):
        c = str(condition).strip()
        if "|" in c:
            c = c.split("|")[0].strip()
        return group_condition_to_tissue(c)
    datasets_by_condition = {}
    for dataset_name, data in all_data.items():
        condition = str(data['Condition'].iloc[0])
        if "|" in condition:
            condition = condition.split("|")[0].strip()
        key = canonical_condition(condition)
        if key not in datasets_by_condition:
            datasets_by_condition[key] = {}
        datasets_by_condition[key][dataset_name] = data
    
    print(f"Found {len(datasets_by_condition)} unique conditions (plasma/serum merged as '{CANONICAL_PLASMA_SERUM_LABEL}'):")
    for condition, datasets in datasets_by_condition.items():
        print(f"  {condition}: {len(datasets)} dataset(s) - {list(datasets.keys())}")
    print()
    
    # Perform inter-dataset analysis for each condition group
    for condition, condition_datasets in datasets_by_condition.items():
        if len(condition_datasets) < 2:
            print(f"Skipping {condition}: only {len(condition_datasets)} dataset(s) (need at least 2 for comparison)")
            continue
        
        condition_lower = condition.lower()
        if ('plasma' in condition_lower or 'serum' in condition_lower) and condition != 'Blood Plasma/Serum':
            continue
        if 'terminally' in condition_lower and 'differentiated' in condition_lower:
            continue
        
        print("=" * 50)
        print(f"ANALYZING CONDITION: {condition}")
        print("=" * 50)
        print(f"Number of datasets: {len(condition_datasets)}")
        
        proteins_by_dataset = {}
        for dataset_name, data in condition_datasets.items():
            proteins_by_dataset[dataset_name] = set(data['Protein'].unique())
        
        n_datasets = len(proteins_by_dataset)
        all_proteins = set()
        for protein_set in proteins_by_dataset.values():
            all_proteins.update(protein_set)
        
        print(f"Total unique proteins across all datasets: {len(all_proteins)}\n")
        
        presence_matrix = pd.DataFrame({'Protein': list(all_proteins)})
        for dataset_name in proteins_by_dataset.keys():
            presence_matrix[dataset_name] = presence_matrix['Protein'].isin(proteins_by_dataset[dataset_name])
        
        dataset_cols = list(proteins_by_dataset.keys())
        presence_matrix['n_datasets'] = presence_matrix[dataset_cols].sum(axis=1)
        
        bin_distribution = presence_matrix['n_datasets'].value_counts().sort_index()
        print("Protein distribution by number of datasets:")
        print(bin_distribution)
        print()
        
        bin_percentages = (bin_distribution / len(presence_matrix) * 100).round(2)
        print("Percentage distribution:")
        print(bin_percentages)
        print()
        
        shared_all = presence_matrix[presence_matrix['n_datasets'] == n_datasets]['Protein'].tolist()
        print(f"Proteins shared in ALL datasets: {len(shared_all)}")
        
        threshold_80 = int(np.ceil(n_datasets * 0.8))
        shared_80plus = presence_matrix[presence_matrix['n_datasets'] >= threshold_80]['Protein'].tolist()
        print(f"Proteins shared in 80%+ datasets ({threshold_80}+ datasets): {len(shared_80plus)}\n")
        
        # Pairwise comparisons
        print("Pairwise protein overlap:")
        dataset_names = list(proteins_by_dataset.keys())
        pairwise_overlap = np.zeros((n_datasets, n_datasets), dtype=int)
        
        for i, dataset1 in enumerate(dataset_names):
            for j, dataset2 in enumerate(dataset_names):
                if i == j:
                    pairwise_overlap[i, j] = len(proteins_by_dataset[dataset1])
                else:
                    overlap = len(proteins_by_dataset[dataset1].intersection(proteins_by_dataset[dataset2]))
                    pairwise_overlap[i, j] = overlap
        
        pairwise_df = pd.DataFrame(pairwise_overlap, index=dataset_names, columns=dataset_names)
        print(pairwise_df)
        print()
        
        condition_safe = condition.replace('/', '_').replace('\\', '_').replace(' ', '_').replace(':', '_')
        condition_plot_dir = plots_dir / condition_safe
        condition_plot_dir.mkdir(exist_ok=True)
        
        # Plot 1: Core proteome
        core_proteome_counts = {}
        for n_present in range(1, n_datasets + 1):
            count = len(presence_matrix[presence_matrix['n_datasets'] == n_present])
            core_proteome_counts[n_present] = count
        
        fig, ax = plt.subplots(figsize=(10, 6))
        n_datasets_present = sorted(core_proteome_counts.keys())
        protein_counts = [core_proteome_counts[n] for n in n_datasets_present]
        ax.plot(n_datasets_present, protein_counts, marker='o', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Datasets Where Protein Appears', fontsize=12)
        ax.set_ylabel('Number of Proteins', fontsize=12)
        ax.set_title(f'Core Proteome - Shared Proteins Across Datasets - {condition}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(condition_plot_dir / "core_proteome.png", bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  Saved: {condition_safe}/core_proteome.png")
        
        # UMAP visualization (optional)
        if UMAP_AVAILABLE and n_datasets >= 3:
            binary_matrix = presence_matrix[dataset_cols].astype(int).values
            dataset_binary = binary_matrix.T
            n_components_umap = min(2, n_datasets - 1) if n_datasets > 2 else 1
            n_neighbors_umap = min(15, max(2, n_datasets - 1))
            try:
                reducer = umap.UMAP(n_components=n_components_umap, n_neighbors=n_neighbors_umap,
                                  random_state=42, metric='jaccard')
                embedding = reducer.fit_transform(dataset_binary)
            except (TypeError, ValueError) as e:
                print(f"  Warning: UMAP failed for {condition}: {e}. Skipping UMAP plot.")
                embedding = None
            
            if embedding is not None:
                fig, ax = plt.subplots(figsize=(10, 8))
                if n_components_umap == 2:
                    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], s=100, alpha=0.7)
                    for i, dataset_name in enumerate(dataset_names):
                        ax.annotate(dataset_name, (embedding[i, 0], embedding[i, 1]), fontsize=9, ha='center', va='bottom')
                    ax.set_xlabel('UMAP 1', fontsize=12)
                    ax.set_ylabel('UMAP 2', fontsize=12)
                else:
                    scatter = ax.scatter(embedding[:, 0], [0] * len(embedding), s=100, alpha=0.7)
                    for i, dataset_name in enumerate(dataset_names):
                        ax.annotate(dataset_name, (embedding[i, 0], 0), fontsize=9, ha='center', va='bottom')
                    ax.set_xlabel('UMAP 1', fontsize=12)
                    ax.set_ylabel('', fontsize=12)
                    ax.set_yticks([])
                ax.set_title(f'Dataset Similarity (UMAP) - {condition}', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(condition_plot_dir / "umap.png", bbox_inches='tight', dpi=300)
                plt.close()
                print(f"  Saved: {condition_safe}/umap.png")

                # Additional UMAPs for plasma/serum using different color encodings.
                if condition == CANONICAL_PLASMA_SERUM_LABEL and n_components_umap == 2:
                    def _plot_colored_umap(label_name, label_values, filename):
                        uniq = sorted(list(dict.fromkeys(label_values)))
                        colors = sns.color_palette("husl", len(uniq))
                        cmap = {u: colors[i] for i, u in enumerate(uniq)}
                        fig2, ax2 = plt.subplots(figsize=(10, 8))
                        for i, dsn in enumerate(dataset_names):
                            lbl = label_values[i]
                            ax2.scatter(
                                embedding[i, 0], embedding[i, 1],
                                s=120, alpha=0.85, c=[cmap[lbl]],
                                edgecolors='black', linewidth=0.5
                            )
                            ax2.annotate(dsn, (embedding[i, 0], embedding[i, 1]), fontsize=8, ha='center', va='bottom')
                        from matplotlib.patches import Patch
                        handles = [Patch(facecolor=cmap[u], edgecolor='black', label=u) for u in uniq]
                        ax2.legend(handles=handles, title=label_name, loc='best', fontsize=9)
                        ax2.set_xlabel('UMAP 1', fontsize=12)
                        ax2.set_ylabel('UMAP 2', fontsize=12)
                        ax2.set_title(f'Dataset Similarity (UMAP) - {condition}\nColored by {label_name}', fontsize=13, fontweight='bold')
                        ax2.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(condition_plot_dir / filename, bbox_inches='tight', dpi=300)
                        plt.close()
                        print(f"  Saved: {condition_safe}/{filename}")

                    protein_bin_labels = [
                        _protein_count_bin(protein_count_map.get(d, 0)) for d in dataset_names
                    ]
                    _plot_colored_umap("Protein Count Bin", protein_bin_labels, "umap_colored_by_protein_count_bin.png")

                    # Sample-level PCA: each dot is a sample, colored by dataset.
                    if SKLEARN_AVAILABLE:
                        sample_rows = []
                        all_ps_proteins = set()
                        for dsn in dataset_names:
                            ddf = condition_datasets[dsn]
                            for s, sdf in ddf.groupby('Sample'):
                                prots = set(sdf['Protein'].dropna().astype(str).unique())
                                sample_rows.append((dsn, str(s), prots))
                                all_ps_proteins.update(prots)
                        if len(sample_rows) >= 3 and len(all_ps_proteins) >= 5:
                            proteins_list = sorted(all_ps_proteins)
                            p_index = {p: i for i, p in enumerate(proteins_list)}
                            X = np.zeros((len(sample_rows), len(proteins_list)), dtype=float)
                            sample_dataset = []
                            sample_names = []
                            for i, (dsn, sname, prots) in enumerate(sample_rows):
                                sample_dataset.append(dsn)
                                sample_names.append(sname)
                                for p in prots:
                                    j = p_index.get(p)
                                    if j is not None:
                                        X[i, j] = 1.0
                            Xs = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
                            pca = PCA(n_components=2, random_state=42)
                            emb = pca.fit_transform(Xs)
                            uniq_ds = sorted(list(dict.fromkeys(sample_dataset)))
                            colors = sns.color_palette("husl", len(uniq_ds))
                            cmap_ds = {u: colors[i] for i, u in enumerate(uniq_ds)}
                            figp, axp = plt.subplots(figsize=(11, 8))
                            for i in range(len(sample_rows)):
                                axp.scatter(
                                    emb[i, 0], emb[i, 1], s=20, alpha=0.7,
                                    c=[cmap_ds[sample_dataset[i]]], edgecolors='none'
                                )
                            from matplotlib.patches import Patch
                            handles = [Patch(facecolor=cmap_ds[u], edgecolor='black', label=u) for u in uniq_ds]
                            axp.legend(handles=handles, title="Dataset", loc='best', fontsize=8, ncol=2)
                            axp.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
                            axp.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
                            axp.set_title(f'Sample-level PCA - {condition}\n(Binary protein presence; colored by dataset)', fontsize=13, fontweight='bold')
                            axp.grid(True, alpha=0.3)
                            plt.tight_layout()
                            plt.savefig(condition_plot_dir / "pca_samples_by_dataset.png", bbox_inches='tight', dpi=300)
                            plt.close()
                            print(f"  Saved: {condition_safe}/pca_samples_by_dataset.png")
                            pd.DataFrame({
                                "Dataset": sample_dataset,
                                "Sample": sample_names,
                                "PC1": emb[:, 0],
                                "PC2": emb[:, 1],
                            }).to_csv(tables_dir / f"pca_samples_by_dataset_{condition_safe}.csv", index=False)
        
        # Save results
        presence_matrix.to_csv(tables_dir / f"inter_dataset_protein_presence_{condition_safe}.csv", index=False)
        bin_df = pd.DataFrame({
            'n_datasets': bin_distribution.index,
            'count': bin_distribution.values,
            'percentage': bin_percentages.values
        })
        bin_df.to_csv(tables_dir / f"inter_dataset_bin_distribution_{condition_safe}.csv", index=False)
        pairwise_df.to_csv(tables_dir / f"inter_dataset_pairwise_overlap_{condition_safe}.csv")
        
        if len(shared_all) > 0:
            pd.DataFrame({'Protein': shared_all}).to_csv(
                tables_dir / f"inter_dataset_shared_all_{condition_safe}.csv", index=False
            )
        
        if len(shared_80plus) > 0:
            pd.DataFrame({'Protein': shared_80plus}).to_csv(
                tables_dir / f"inter_dataset_shared_80plus_{condition_safe}.csv", index=False
            )
        
        print()

def create_tissue_grouped_plots(all_data, intra_results, plots_dir, tables_dir, entry_name_library=None):
    """Create tissue-grouped protein distribution analysis plots."""
    print("=" * 50)
    print("TISSUE-GROUPED PROTEIN DISTRIBUTION ANALYSIS")
    print("=" * 50)
    
    # Load entry name library if not provided
    if entry_name_library is None:
        print("  Loading entry name mapping library...")
        entry_name_library = load_entry_name_mapping_library(library_file)
        manual_mapping = load_manual_mapping(manual_mapping_file, verbose=False)
        entry_name_library = {**entry_name_library, **manual_mapping}
    
    # Use same raw condition + group_condition_to_tissue as B_ / get_cell_type (mode, | split)
    tissue_groups = {}
    for dataset_name, data in all_data.items():
        raw = _raw_condition_for_tissue_grouping(dataset_name, data)
        tissue = group_condition_to_tissue(raw)
        if tissue not in tissue_groups:
            tissue_groups[tissue] = {}
        tissue_groups[tissue][dataset_name] = data
    
    print(f"Grouped into {len(tissue_groups)} tissue types:")
    for tissue, datasets in tissue_groups.items():
        print(f"  {tissue}: {len(datasets)} dataset(s) - {list(datasets.keys())}")
    print()
    
    for tissue, tissue_datasets in tissue_groups.items():
        if len(tissue_datasets) < 1:
            continue
        
        tissue_lower = tissue.lower()
        if ('plasma' in tissue_lower or 'serum' in tissue_lower) and tissue != 'Blood Plasma/Serum':
            continue
        if 'terminally' in tissue_lower and 'differentiated' in tissue_lower:
            continue
        
        print("=" * 50)
        print(f"ANALYZING TISSUE: {tissue}")
        print("=" * 50)
        print(f"Number of datasets: {len(tissue_datasets)}")
        
        # Normalize proteins for consistent counting (normalize union once, then map per dataset)
        print("  Normalizing protein identifiers to UniProt accessions for consistent counting...")
        all_raw_tissue = set()
        for data in tissue_datasets.values():
            all_raw_tissue.update(data['Protein'].unique())
        normalized_tissue_all, tissue_norm_mapping = normalize_protein_set_for_comparison(all_raw_tissue, entry_name_library=entry_name_library, return_mapping=True)
        orig_to_norm_tissue = {}
        for norm_p, originals in tissue_norm_mapping.items():
            for o in originals:
                orig_to_norm_tissue[o] = norm_p
        normalized_protein_sets_by_dataset = {}
        for dataset_name, data in tissue_datasets.items():
            raw_proteins = set(data['Protein'].unique())
            normalized_protein_sets_by_dataset[dataset_name] = {orig_to_norm_tissue.get(p, p) for p in raw_proteins}
        # Total unique proteins = normalized set from union
        all_proteins = set(normalized_tissue_all)
        for normalized_proteins in normalized_protein_sets_by_dataset.values():
            all_proteins.update(normalized_proteins)
        
        # Build presence matrix from full (unrestricted) data for frequency_bins_stacked_bar so per-dataset counts match 01
        presence_matrix_full = pd.DataFrame({'Protein': list(all_proteins)})
        for dataset_name in normalized_protein_sets_by_dataset.keys():
            presence_matrix_full[dataset_name] = presence_matrix_full['Protein'].isin(normalized_protein_sets_by_dataset[dataset_name])
        dataset_cols = list(normalized_protein_sets_by_dataset.keys())
        n_datasets = len(dataset_cols)
        presence_matrix_full['n_datasets'] = presence_matrix_full[dataset_cols].sum(axis=1)
        
        # Restrict to 02 list for this tissue (for totals and distribution; frequency plot uses full counts to match 01)
        proteins_02_tissue = get_02_protein_set_for_tissue(tissue)
        if proteins_02_tissue is not None:
            before_t = len(all_proteins)
            all_proteins = all_proteins & proteins_02_tissue
            for dn in normalized_protein_sets_by_dataset:
                normalized_protein_sets_by_dataset[dn] = normalized_protein_sets_by_dataset[dn] & proteins_02_tissue
            if before_t != len(all_proteins):
                print(f"  Restricted to 02 list: {len(all_proteins)} proteins (from {before_t})")
        
        total_proteins = len(all_proteins)
        print(f"Total unique proteins across all datasets (normalized): {total_proteins}")
        
        # Create presence matrix using 02-restricted proteins (for distribution stats and other plots)
        presence_matrix = pd.DataFrame({'Protein': list(all_proteins)})
        for dataset_name in normalized_protein_sets_by_dataset.keys():
            presence_matrix[dataset_name] = presence_matrix['Protein'].isin(normalized_protein_sets_by_dataset[dataset_name])
        presence_matrix['n_datasets'] = presence_matrix[dataset_cols].sum(axis=1)
        
        distribution = presence_matrix['n_datasets'].value_counts().sort_index()
        print("\nDistribution of proteins by number of datasets:")
        print(distribution)
        print()
        
        tissue_safe = tissue.replace('/', '_').replace('\\', '_').replace(' ', '_').replace(':', '_').replace('(', '').replace(')', '')
        tissue_plot_dir = plots_dir / tissue_safe
        tissue_plot_dir.mkdir(exist_ok=True)
        
        # Histogram/Density Plot - Split by DDA/DIA
        print(f"Creating histogram/density plot for {tissue} (split by DDA/DIA)...")
        
        # Get acquisition method for each dataset
        datasets_dda = []
        datasets_dia = []
        for dataset_name in tissue_datasets.keys():
            acquisition = get_acquisition_method(dataset_name)
            if acquisition == 'DDA':
                datasets_dda.append(dataset_name)
            elif acquisition == 'DIA':
                datasets_dia.append(dataset_name)
        
        # Create separate distributions for DDA and DIA using normalized proteins
        def create_distribution_for_datasets(dataset_list):
            """Create distribution for a subset of datasets using normalized proteins."""
            if len(dataset_list) == 0:
                return None, 0
            
            # Use normalized protein sets
            subset_proteins = set()
            for dataset_name in dataset_list:
                if dataset_name in normalized_protein_sets_by_dataset:
                    subset_proteins.update(normalized_protein_sets_by_dataset[dataset_name])
            
            # Count how many datasets each protein appears in (within this subset)
            protein_dataset_counts = {}
            for protein in subset_proteins:
                count = sum(1 for ds in dataset_list if ds in normalized_protein_sets_by_dataset and protein in normalized_protein_sets_by_dataset[ds])
                protein_dataset_counts[protein] = count
            
            # Create distribution
            dist = {}
            for protein, count in protein_dataset_counts.items():
                if count not in dist:
                    dist[count] = 0
                dist[count] += 1
            
            total = len(subset_proteins)
            return dist, total
        
        dist_dda, total_dda = create_distribution_for_datasets(datasets_dda)
        dist_dia, total_dia = create_distribution_for_datasets(datasets_dia)
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # DDA plot
        if dist_dda and total_dda > 0:
            n_datasets_present_dda = sorted(dist_dda.keys())
            protein_counts_dda = [dist_dda[n] for n in n_datasets_present_dda]

            ax1.bar(n_datasets_present_dda, protein_counts_dda, alpha=0.7, color='#3498db', edgecolor='black')
            ax1.set_xlabel('Number of Datasets Where Protein Appears', fontsize=12)
            ax1.set_ylabel('Number of Proteins', fontsize=12)
            ax1.set_title(
                f'Protein Distribution - {tissue} (DDA)\nTotal: {total_dda} proteins',
                fontsize=13,
                fontweight='bold',
            )
            ax1.grid(True, alpha=0.3, axis='y')
            ax1.set_xticks(n_datasets_present_dda)

            for x, y in zip(n_datasets_present_dda, protein_counts_dda):
                ax1.text(x, y, str(y), ha='center', va='bottom', fontsize=9)
        else:
            ax1.text(0.5, 0.5, 'No DDA datasets', ha='center', va='center', transform=ax1.transAxes, fontsize=12)
            ax1.set_title(f'Protein Distribution - {tissue} (DDA)', fontsize=13, fontweight='bold')

        # DIA plot
        if dist_dia and total_dia > 0:
            n_datasets_present_dia = sorted(dist_dia.keys())
            protein_counts_dia = [dist_dia[n] for n in n_datasets_present_dia]

            ax2.bar(n_datasets_present_dia, protein_counts_dia, alpha=0.7, color='#e74c3c', edgecolor='black')
            ax2.set_xlabel('Number of Datasets Where Protein Appears', fontsize=12)
            ax2.set_ylabel('Number of Proteins', fontsize=12)
            ax2.set_title(
                f'Protein Distribution - {tissue} (DIA)\nTotal: {total_dia} proteins',
                fontsize=13,
                fontweight='bold',
            )
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.set_xticks(n_datasets_present_dia)

            for x, y in zip(n_datasets_present_dia, protein_counts_dia):
                ax2.text(x, y, str(y), ha='center', va='bottom', fontsize=9)
        else:
            ax2.text(0.5, 0.5, 'No DIA datasets', ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title(f'Protein Distribution - {tissue} (DIA)', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(tissue_plot_dir / "distribution_histogram.png", bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  Saved: {tissue_safe}/distribution_histogram.png (DDA: {total_dda} proteins, DIA: {total_dia} proteins)")
        
        # Combined DDA+DIA plot (for Plasma/Serum only)
        if tissue == 'Blood Plasma/Serum':
            print(f"Creating combined distribution histogram (DDA+DIA) for {tissue}...")
            # Use all datasets together
            all_datasets_combined = list(tissue_datasets.keys())
            dist_combined, total_combined = create_distribution_for_datasets(all_datasets_combined)
            
            if dist_combined and total_combined > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                n_datasets_present_combined = sorted(dist_combined.keys())
                protein_counts_combined = [dist_combined[n] for n in n_datasets_present_combined]
                
                ax.bar(n_datasets_present_combined, protein_counts_combined, alpha=0.7, color='steelblue', edgecolor='black')
                ax.set_xlabel('Number of Datasets Where Protein Appears', fontsize=12)
                ax.set_ylabel('Number of Proteins', fontsize=12)
                ax.set_title(f'Protein Distribution - {tissue} (DDA + DIA Combined)\nTotal: {total_combined} proteins', fontsize=13, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                ax.set_xticks(n_datasets_present_combined)
                
                for x, y in zip(n_datasets_present_combined, protein_counts_combined):
                    ax.text(x, y, str(y), ha='center', va='bottom', fontsize=9)
                
                plt.tight_layout()
                plt.savefig(tissue_plot_dir / "distribution_histogram_combined.png", bbox_inches='tight', dpi=300)
                plt.close()
                print(f"  Saved: {tissue_safe}/distribution_histogram_combined.png (Total: {total_combined} proteins)")
        
        # Distribution by Samples (for Plasma/Serum only) - Split by DDA/DIA
        if tissue == 'Blood Plasma/Serum':
            print(f"Creating distribution histogram by samples for {tissue} (split by DDA/DIA)...")
            
            # Get acquisition method for each dataset
            datasets_dda_samples = []
            datasets_dia_samples = []
            for dataset_name in tissue_datasets.keys():
                acquisition = get_acquisition_method(dataset_name)
                if acquisition == 'DDA':
                    datasets_dda_samples.append(dataset_name)
                elif acquisition == 'DIA':
                    datasets_dia_samples.append(dataset_name)
            
            def create_sample_distribution_for_datasets(dataset_list):
                """Sample-level distribution: unique normalized proteins that appear in ≥1 sample (matches cache after B_)."""
                if len(dataset_list) == 0:
                    return None, 0
                
                # First pass: collect all (sample_name, protein_set) and union of all protein IDs
                sample_to_proteins = {}
                all_proteins_union = set()
                for dataset_name in dataset_list:
                    if dataset_name in intra_results:
                        proteins_by_sample = intra_results[dataset_name].get('proteins_by_sample', {})
                        if proteins_by_sample:
                            for sample, proteins in proteins_by_sample.items():
                                combined_sample_name = f"{dataset_name}_{sample}"
                                sample_to_proteins[combined_sample_name] = set(proteins)
                                all_proteins_union.update(proteins)
                
                if len(sample_to_proteins) == 0:
                    return None, 0
                
                # Normalize once over the full union and get mapping original -> normalized
                if all_proteins_union:
                    norm_set, norm_to_orig = normalize_protein_set_for_comparison(
                        all_proteins_union, entry_name_library=entry_name_library, return_mapping=True
                    )
                    original_to_normalized = {orig: norm for norm, origs in norm_to_orig.items() for orig in origs}
                else:
                    original_to_normalized = {}
                
                all_proteins_by_sample_subset = {
                    name: {original_to_normalized.get(p, p) for p in prots}
                    for name, prots in sample_to_proteins.items()
                }
                
                protein_sample_counts = {}
                for sample, proteins in all_proteins_by_sample_subset.items():
                    for protein in proteins:
                        if protein not in protein_sample_counts:
                            protein_sample_counts[protein] = 0
                        protein_sample_counts[protein] += 1
                
                def assign_sample_bin(n_samples):
                    if n_samples == 1:
                        return '1'
                    elif 2 <= n_samples <= 4:
                        return '2-4'
                    elif 5 <= n_samples <= 10:
                        return '5-10'
                    elif 11 <= n_samples <= 50:
                        return '11-50'
                    elif 51 <= n_samples <= 100:
                        return '51-100'
                    else:
                        return '100+'
                
                bin_order = ['1', '2-4', '5-10', '11-50', '51-100', '100+']
                bin_counts = {bin_label: 0 for bin_label in bin_order}
                
                for protein, n_samples in protein_sample_counts.items():
                    bin_label = assign_sample_bin(n_samples)
                    bin_counts[bin_label] += 1
                
                total = len(protein_sample_counts)
                return bin_counts, total
            
            bin_counts_dda, total_dda_samples = create_sample_distribution_for_datasets(datasets_dda_samples)
            bin_counts_dia, total_dia_samples = create_sample_distribution_for_datasets(datasets_dia_samples)
            
            if (bin_counts_dda and total_dda_samples > 0) or (bin_counts_dia and total_dia_samples > 0):
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                bin_order = ['1', '2-4', '5-10', '11-50', '51-100', '100+']
                x_pos = range(len(bin_order))
                
                # DDA plot
                if bin_counts_dda and total_dda_samples > 0:
                    protein_counts_dda = [bin_counts_dda[label] for label in bin_order]
                    ax1.bar(x_pos, protein_counts_dda, alpha=0.7, color='#3498db', edgecolor='black')
                    ax1.set_xlabel('Number of Samples Where Protein Appears', fontsize=12)
                    ax1.set_ylabel('Number of Proteins', fontsize=12)
                    ax1.set_title(
                        f'Protein Distribution by Samples - {tissue} (DDA)\nTotal: {total_dda_samples} proteins',
                        fontsize=13,
                        fontweight='bold',
                    )
                    ax1.grid(True, alpha=0.3, axis='y')
                    ax1.set_xticks(x_pos)
                    ax1.set_xticklabels(bin_order, rotation=0)

                    for x, y in zip(x_pos, protein_counts_dda):
                        ax1.text(x, y, str(y), ha='center', va='bottom', fontsize=9)
                else:
                    ax1.set_xlabel('Number of Samples Where Protein Appears', fontsize=12)
                    ax1.set_ylabel('Number of Proteins', fontsize=12)
                    ax1.set_title(f'Protein Distribution by Samples - {tissue} (DDA)', fontsize=13, fontweight='bold')
                    ax1.grid(True, alpha=0.3, axis='y')

                # DIA plot
                if bin_counts_dia and total_dia_samples > 0:
                    protein_counts_dia = [bin_counts_dia[label] for label in bin_order]
                    ax2.bar(x_pos, protein_counts_dia, alpha=0.7, color='#e74c3c', edgecolor='black')
                    ax2.set_xlabel('Number of Samples Where Protein Appears', fontsize=12)
                    ax2.set_ylabel('Number of Proteins', fontsize=12)
                    ax2.set_title(
                        f'Protein Distribution by Samples - {tissue} (DIA)\nTotal: {total_dia_samples} proteins',
                        fontsize=13,
                        fontweight='bold',
                    )
                    ax2.grid(True, alpha=0.3, axis='y')
                    ax2.set_xticks(x_pos)
                    ax2.set_xticklabels(bin_order, rotation=0)

                    for x, y in zip(x_pos, protein_counts_dia):
                        ax2.text(x, y, str(y), ha='center', va='bottom', fontsize=9)
                else:
                    ax2.set_xlabel('Number of Samples Where Protein Appears', fontsize=12)
                    ax2.set_ylabel('Number of Proteins', fontsize=12)
                    ax2.set_title(f'Protein Distribution by Samples - {tissue} (DIA)', fontsize=13, fontweight='bold')
                    ax2.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                plt.savefig(tissue_plot_dir / "distribution_histogram_by_samples.png", bbox_inches='tight', dpi=300)
                plt.close()
                print(f"  Saved: {tissue_safe}/distribution_histogram_by_samples.png (DDA: {total_dda_samples} proteins, DIA: {total_dia_samples} proteins)")
                
                # Combined DDA+DIA plot for samples (total = proteins observed in ≥1 sample; matches B_ 02 after global ENTRAP fix)
                print(f"Creating combined distribution histogram by samples (DDA+DIA) for {tissue}...")
                all_datasets_samples_combined = list(tissue_datasets.keys())
                bin_counts_combined, total_combined_samples = create_sample_distribution_for_datasets(
                    all_datasets_samples_combined
                )
                
                if bin_counts_combined and total_combined_samples > 0:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bin_order_samples_combined = ['1', '2-4', '5-10', '11-50', '51-100', '100+']
                    x_pos = range(len(bin_order_samples_combined))
                    protein_counts_combined = [bin_counts_combined[label] for label in bin_order_samples_combined]
                    
                    ax.bar(x_pos, protein_counts_combined, alpha=0.7, color='steelblue', edgecolor='black')
                    ax.set_xlabel('Number of Samples Where Protein Appears', fontsize=12)
                    ax.set_ylabel('Number of Proteins', fontsize=12)
                    ax.set_title(
                        f'Protein Distribution by Samples - {tissue} (DDA + DIA Combined)\n'
                        f'Total: {total_combined_samples} proteins (unique across samples; default 2-peptide cache)',
                        fontsize=13,
                        fontweight='bold',
                    )
                    ax.grid(True, alpha=0.3, axis='y')
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(bin_order_samples_combined, rotation=0)
                    
                    for x, y in zip(x_pos, protein_counts_combined):
                        ax.text(x, y, str(y), ha='center', va='bottom', fontsize=9)
                    
                    plt.tight_layout()
                    plt.savefig(tissue_plot_dir / "distribution_histogram_by_samples_combined.png", bbox_inches='tight', dpi=300)
                    plt.close()
                    print(f"  Saved: {tissue_safe}/distribution_histogram_by_samples_combined.png (Total: {total_combined_samples} proteins)")
        
        # Stacked Bar Plot with Frequency Bins (use full counts so per-dataset totals match 01 Correct_Proteins_After)
        print(f"Creating stacked bar plot with frequency bins for {tissue}...")
        
        def assign_frequency_bin(n_datasets):
            if n_datasets == 1:
                return 'Unique'
            elif 2 <= n_datasets <= 3:
                return 'Rare'
            elif 4 <= n_datasets <= 10:
                return 'Intermediate'
            else:
                return 'Core'
        
        presence_matrix_full['frequency_bin'] = presence_matrix_full['n_datasets'].apply(assign_frequency_bin)
        
        dataset_bin_counts = {}
        dataset_total_counts = {}
        
        for dataset_name in dataset_cols:
            dataset_proteins = presence_matrix_full[presence_matrix_full[dataset_name] == True]
            bin_counts = dataset_proteins['frequency_bin'].value_counts()
            dataset_bin_counts[dataset_name] = {
                'Unique': bin_counts.get('Unique', 0),
                'Rare': bin_counts.get('Rare', 0),
                'Intermediate': bin_counts.get('Intermediate', 0),
                'Core': bin_counts.get('Core', 0)
            }
            dataset_total_counts[dataset_name] = len(dataset_proteins)
        
        sorted_datasets = sorted(dataset_total_counts.items(), key=lambda x: x[1], reverse=True)
        dataset_names_sorted = [d[0] for d in sorted_datasets]
        
        unique_counts = [dataset_bin_counts[name]['Unique'] for name in dataset_names_sorted]
        rare_counts = [dataset_bin_counts[name]['Rare'] for name in dataset_names_sorted]
        intermediate_counts = [dataset_bin_counts[name]['Intermediate'] for name in dataset_names_sorted]
        core_counts = [dataset_bin_counts[name]['Core'] for name in dataset_names_sorted]
        
        fig, ax = plt.subplots(figsize=(max(12, len(dataset_names_sorted) * 0.3), 8))
        x_pos = np.arange(len(dataset_names_sorted))
        width = 0.8
        
        colors_bins = {
            'Unique': '#d62728',
            'Rare': '#ff7f0e',
            'Intermediate': '#2ca02c',
            'Core': '#1f77b4'
        }
        
        p1 = ax.bar(x_pos, unique_counts, width, label='Unique (1 dataset)', color=colors_bins['Unique'], alpha=0.8)
        p2 = ax.bar(x_pos, rare_counts, width, bottom=unique_counts, label='Rare (2-3 datasets)', color=colors_bins['Rare'], alpha=0.8)
        p3 = ax.bar(x_pos, intermediate_counts, width, bottom=np.array(unique_counts) + np.array(rare_counts), 
                   label='Intermediate (4-10 datasets)', color=colors_bins['Intermediate'], alpha=0.8)
        p4 = ax.bar(x_pos, core_counts, width,
                   bottom=np.array(unique_counts) + np.array(rare_counts) + np.array(intermediate_counts),
                   label='Core (>10 datasets)', color=colors_bins['Core'], alpha=0.8)
        
        ax.set_xlabel('Dataset (sorted by total proteins)', fontsize=12)
        ax.set_ylabel('Number of Proteins', fontsize=12)
        ax.set_title(f'Protein Frequency Bins per Dataset - {tissue}', fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(dataset_names_sorted, rotation=45, ha='right', fontsize=8)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(tissue_plot_dir / "frequency_bins_stacked_bar.png", bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  Saved: {tissue_safe}/frequency_bins_stacked_bar.png")
        
        # Unique-Protein Burden vs Dataset Size
        print(f"Creating unique burden vs size plot for {tissue}...")
        dataset_unique_counts = {}
        for dataset_name in dataset_cols:
            unique_proteins = presence_matrix[
                (presence_matrix[dataset_name] == True) & 
                (presence_matrix['n_datasets'] == 1)
            ]
            dataset_unique_counts[dataset_name] = len(unique_proteins)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        total_proteins_list = [dataset_total_counts[name] for name in dataset_cols]
        unique_proteins_list = [dataset_unique_counts[name] for name in dataset_cols]
        
        # Color by DDA/DIA
        colors_list = []
        unknown_acquisition_datasets = []
        for name in dataset_cols:
            acquisition = get_acquisition_method(name)
            if acquisition == 'DDA':
                colors_list.append('#3498db')  # Blue for DDA
            elif acquisition == 'DIA':
                colors_list.append('#e74c3c')  # Red for DIA
            else:
                colors_list.append('#95a5a6')  # Gray for unknown
                unknown_acquisition_datasets.append(name)
        if unknown_acquisition_datasets:
            print(f"  Note: Datasets with unknown acquisition (grey in plot): {', '.join(unknown_acquisition_datasets)}")
        
        # Plot all points with their respective colors
        for total, unique, color in zip(total_proteins_list, unique_proteins_list, colors_list):
            ax.scatter(total, unique, s=150, alpha=0.7, edgecolors='black', linewidth=1, zorder=3, c=[color])
        
        for name, total, unique in zip(dataset_cols, total_proteins_list, unique_proteins_list):
            ax.annotate(name, (total, unique), fontsize=8, alpha=0.7, xytext=(5, 5), textcoords='offset points')
        
        if len(total_proteins_list) > 1:
            x_all = np.array(total_proteins_list)
            y_all = np.array(unique_proteins_list)
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_all, y_all)
            x_line = np.linspace(0, x_all.max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, '--', color='red', linewidth=2, alpha=0.7,
                   label=f'Regression: y = {slope:.4f}x + {intercept:.2f} (R² = {r_value**2:.3f})')
            ax.legend(loc='best', fontsize=9)
        
        ax.set_xlabel('Total Proteins in Dataset', fontsize=12)
        ax.set_ylabel('Number of Unique Proteins', fontsize=12)
        ax.set_title(f'Unique-Protein Burden vs Dataset Size - {tissue}\n(Diagnostic: Is uniqueness driven by size?)', 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add legend for DDA/DIA
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#3498db', label='DDA', alpha=0.7),
            Patch(facecolor='#e74c3c', label='DIA', alpha=0.7),
            Patch(facecolor='#95a5a6', label='Unknown', alpha=0.7)
        ]
        ax.legend(handles=legend_elements, loc='best', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(tissue_plot_dir / "unique_burden_vs_size.png", bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  Saved: {tissue_safe}/unique_burden_vs_size.png")
        
        # Create logarithmic version
        fig, ax = plt.subplots(figsize=(10, 8))
        for total, unique, color in zip(total_proteins_list, unique_proteins_list, colors_list):
            ax.scatter(total, unique, s=150, alpha=0.7, edgecolors='black', linewidth=1, zorder=3, c=[color])
        
        for name, total, unique in zip(dataset_cols, total_proteins_list, unique_proteins_list):
            ax.annotate(name, (total, unique), fontsize=8, alpha=0.7, xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('Total Proteins in Dataset', fontsize=12)
        ax.set_ylabel('Number of Unique Proteins (log scale)', fontsize=12)
        ax.set_yscale('log')
        ax.set_title(f'Unique-Protein Burden vs Dataset Size - {tissue} (Log Scale)\n(Diagnostic: Is uniqueness driven by size?)', 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add legend for DDA/DIA
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#3498db', label='DDA', alpha=0.7),
            Patch(facecolor='#e74c3c', label='DIA', alpha=0.7),
            Patch(facecolor='#95a5a6', label='Unknown', alpha=0.7)
        ]
        ax.legend(handles=legend_elements, loc='best', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(tissue_plot_dir / "unique_burden_vs_size_log.png", bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  Saved: {tissue_safe}/unique_burden_vs_size_log.png")
        
        distribution_df = pd.DataFrame({
            'n_datasets': distribution.index,
            'n_proteins': distribution.values,
            'proportion': (distribution.values / total_proteins * 100).round(2)
        })
        distribution_df.to_csv(tissue_plot_dir / "distribution.csv", index=False)
        print()

def get_acquisition_method(dataset_name):
    """Get acquisition method (DDA/DIA) for a dataset.
    
    Priority order:
    1. Data preparation output (B_data_preparation_and_filtering.py) - most reliable
    2. Methodology analysis output
    3. SDRF file parsing
    4. Dataset name checking
    """
    # FIRST: Try to load from data preparation output (most reliable)
    prep_summary_file = data_prep_output_dir / "01_dataset_summary_before_and_after_filtering.csv"
    if prep_summary_file.exists():
        try:
            prep_summary = pd.read_csv(prep_summary_file)
            if 'Dataset' in prep_summary.columns and 'Acquisition' in prep_summary.columns:
                # Try exact match first
                dataset_match = prep_summary[prep_summary['Dataset'] == dataset_name]
                if len(dataset_match) > 0:
                    acquisition = dataset_match['Acquisition'].iloc[0]
                    if pd.notna(acquisition) and acquisition in ['DDA', 'DIA']:
                        return acquisition
                
                # Try with common suffixes
                for suffix in ['-plasma', '-serum', '-erythrocyte', '-DDA', '-DIA', '-blood_serum', '-blood_plasma']:
                    dataset_name_with_suffix = f"{dataset_name}{suffix}"
                    dataset_match = prep_summary[prep_summary['Dataset'] == dataset_name_with_suffix]
                    if len(dataset_match) > 0:
                        acquisition = dataset_match['Acquisition'].iloc[0]
                        if pd.notna(acquisition) and acquisition in ['DDA', 'DIA']:
                            return acquisition
        except Exception as e:
            pass  # Fall through to other methods
    
    # SECOND: Try to load from methodology CSV
        methodology_csv = work_dir / "methodology_analysis" / "methodology_dataset_summary.csv"
    if methodology_csv.exists():
        try:
            methodology_df = pd.read_csv(methodology_csv)
            if 'Dataset' in methodology_df.columns and 'Acquisition' in methodology_df.columns:
                dataset_match = methodology_df[methodology_df['Dataset'] == dataset_name]
                if len(dataset_match) > 0:
                    acquisition = dataset_match['Acquisition'].iloc[0]
                    if pd.notna(acquisition) and acquisition in ['DDA', 'DIA']:
                        return acquisition
        except Exception as e:
            pass  # Fall through to SDRF parsing
    
    # Try to parse SDRF file (same logic as 4_methodology_analysis.py)
    import csv
    
    # Extract base dataset name (remove suffixes)
    base_name = dataset_name
    for suffix in ['-DDA', '-DIA', '-LFQ', '-plasma', '-serum', '-erythrocyte']:
        if base_name.endswith(suffix):
            base_name = base_name[:-len(suffix)]
            break
    
    sdrf_dir = work_dir / "sdrf_files"
    sdrf_file = sdrf_dir / f"{base_name}.sdrf.tsv"
    
    # Also try with full dataset name
    if not sdrf_file.exists():
        sdrf_file = sdrf_dir / f"{dataset_name}.sdrf.tsv"
    
    if sdrf_file.exists():
        try:
            with open(sdrf_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                rows = list(reader)
            
            if len(rows) > 0:
                acquisition_values = set()
                for row in rows:
                    acquisition = row.get('comment[proteomics data acquisition method]', '').strip()
                    if acquisition:
                        acquisition_values.add(acquisition)
                
                # Determine acquisition method (same logic as 4_methodology_analysis.py)
                acquisition_method = 'DDA'  # Default
                if acquisition_values:
                    for val in acquisition_values:
                        val_upper = val.upper()
                        if 'DIA' in val_upper or 'DATA-INDEPENDENT' in val_upper:
                            acquisition_method = 'DIA'
                            break
                        elif 'DDA' in val_upper or 'DATA-DEPENDENT' in val_upper:
                            acquisition_method = 'DDA'
                            break
                
                return acquisition_method
        except Exception as e:
            pass  # Fall through to name checking
    
    # Fallback: check dataset name
    dataset_name_lower = dataset_name.lower()
    if '-dia' in dataset_name_lower or '_dia' in dataset_name_lower:
        return 'DIA'
    elif '-dda' in dataset_name_lower or '_dda' in dataset_name_lower:
        return 'DDA'
    # TMT / iTRAQ (e.g. MSV000079033-Blood-Plasma-TMT10) are typically DDA
    elif 'tmt' in dataset_name_lower or 'itraq' in dataset_name_lower:
        return 'DDA'
    else:
        return 'Unknown'

# ============================================
# PROTEIN NORMALIZATION FOR COMPARISON
# ============================================

def normalize_protein_set_for_comparison(protein_set, entry_name_library=None, return_mapping=False):
    """Normalize a set of protein identifiers to UniProt accessions for comparison.
    
    If B_ certified cache as normalized (normalization_status.json status=='ok'), returns
    the set as-is to avoid redundant work. Otherwise follows DDA/DIA rules below.
    
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
    global _normalization_skip_logged
    if cache_is_normalized():
        if not _normalization_skip_logged:
            print("  Using cache normalization (B_ certified); skipping re-normalization.")
            _normalization_skip_logged = True
        normalized_set = set(protein_set)
        if return_mapping:
            return normalized_set, {p: [p] for p in normalized_set}
        return normalized_set

    normalized_set = set()
    normalized_to_original = {} if return_mapping else None
    
    # Use provided library or load it (same paths as B_ via shared_utils)
    if entry_name_library is None:
        entry_name_library = load_entry_name_mapping_library(library_file)
        manual_mapping = load_manual_mapping(manual_mapping_file, verbose=False)
        # Merge manual mapping into library (manual takes precedence)
        entry_name_library = {**entry_name_library, **manual_mapping}
    
    n_total = len(protein_set)
    report_every = max(50000, n_total // 10) if n_total > 20000 else n_total + 1  # progress for large sets
    for idx, protein_id in enumerate(protein_set):
        if report_every <= n_total and (idx + 1) % report_every == 0:
            print(f"      Normalized {idx + 1}/{n_total} proteins...", flush=True)
        protein_str = str(protein_id).strip()
        
        # Check if it's DDA format (contains '|')
        if '|' in protein_str:
            # DDA format: sp|ACCESSION|ENTRY_NAME;sp|ACCESSION2|ENTRY_NAME2
            # Split on ';', take first entry
            first_entry = protein_str.split(';')[0].strip()
            # Extract UniProt accession (between first and second '|')
            uniprot_id = extract_uniprot_id(first_entry)
            if uniprot_id:
                normalized_set.add(uniprot_id)
                if return_mapping:
                    if uniprot_id not in normalized_to_original:
                        normalized_to_original[uniprot_id] = []
                    normalized_to_original[uniprot_id].append(protein_id)
            else:
                # Fallback: keep original if extraction fails
                normalized_set.add(protein_id)
                if return_mapping:
                    if protein_id not in normalized_to_original:
                        normalized_to_original[protein_id] = []
                    normalized_to_original[protein_id].append(protein_id)
        else:
            # DIA format or direct UniProt accession
            # First, try to extract as direct UniProt accession
            uniprot_id = extract_uniprot_id(protein_str)
            if uniprot_id:
                normalized_set.add(uniprot_id)
                if return_mapping:
                    if uniprot_id not in normalized_to_original:
                        normalized_to_original[uniprot_id] = []
                    normalized_to_original[uniprot_id].append(protein_id)
            else:
                # Try to convert entry name to accession (DIA format: ENTRY_NAME_HUMAN)
                # Split on ';', take first entry name (DIA rule: keep only first entry)
                first_entry = protein_str.split(';')[0].strip()
                accession = convert_entry_name_to_accession(first_entry, entry_name_library)
                if accession:
                    normalized_set.add(accession)
                    if return_mapping:
                        if accession not in normalized_to_original:
                            normalized_to_original[accession] = []
                        normalized_to_original[accession].append(protein_id)
                else:
                    # If we can't convert to accession, use the first entry (split result)
                    # This follows the DIA rule: split on ';' and keep only first entry
                    normalized_set.add(first_entry)
                    if return_mapping:
                        if first_entry not in normalized_to_original:
                            normalized_to_original[first_entry] = []
                        normalized_to_original[first_entry].append(protein_id)
    
    if return_mapping:
        return normalized_set, normalized_to_original
    return normalized_set

# ============================================
# CONNECTIVITY ANALYSIS FUNCTIONS (from 8_connectivity_analysis.py)
# ============================================

def compute_shared_fraction(proteins_i, proteins_j):
    """Compute shared proteome fraction between two datasets.
    
    shared_fraction(i, j) = |Pi ∩ Pj| / min(|Pi|, |Pj|)
    """
    intersection = len(proteins_i.intersection(proteins_j))
    min_size = min(len(proteins_i), len(proteins_j))
    if min_size == 0:
        return 0.0
    return intersection / min_size

def compute_pairwise_similarities(all_data, entry_name_library=None):
    """Compute pairwise similarity for all dataset pairs."""
    print("\nComputing pairwise similarities...")
    print("  Normalizing protein identifiers to UniProt accessions for comparison...")
    dataset_names = list(all_data.keys())
    n_datasets = len(dataset_names)
    
    # Load entry name library if not provided
    if entry_name_library is None:
        entry_name_library = load_entry_name_mapping_library(library_file)
        manual_mapping = load_manual_mapping(manual_mapping_file, verbose=False)
        entry_name_library = {**entry_name_library, **manual_mapping}
    
    protein_sets = {}
    for dataset_name, data in all_data.items():
        raw_proteins = set(data['Protein'].unique())
        # Normalize to UniProt accessions for comparison
        protein_sets[dataset_name] = normalize_protein_set_for_comparison(raw_proteins, entry_name_library=entry_name_library)
    
    pairwise_similarities = {}
    all_similarities = []
    
    total_pairs = n_datasets * (n_datasets - 1) // 2
    processed = 0
    
    for i, dataset_i in enumerate(dataset_names):
        for j, dataset_j in enumerate(dataset_names):
            if i >= j:
                continue
            
            shared_frac = compute_shared_fraction(protein_sets[dataset_i], protein_sets[dataset_j])
            pairwise_similarities[(dataset_i, dataset_j)] = shared_frac
            all_similarities.append(shared_frac)
            
            processed += 1
            if processed % 50 == 0 or processed == total_pairs:
                print(f"  Processed {processed}/{total_pairs} pairs...")
    
    print(f"  Computed {len(all_similarities)} pairwise similarities")
    print(f"  Mean similarity: {np.mean(all_similarities):.3f}")
    print(f"  Median similarity: {np.median(all_similarities):.3f}")
    print(f"  Min similarity: {np.min(all_similarities):.3f}")
    print(f"  Max similarity: {np.max(all_similarities):.3f}")
    
    return pairwise_similarities, all_similarities

def compute_per_dataset_connectivity(pairwise_similarities, dataset_names):
    """Compute mean connectivity score for each dataset."""
    print("\nComputing per-dataset connectivity scores...")
    
    connectivity_scores = {}
    
    for dataset_i in dataset_names:
        similarities_for_i = []
        for (ds_i, ds_j), similarity in pairwise_similarities.items():
            if ds_i == dataset_i:
                similarities_for_i.append(similarity)
            elif ds_j == dataset_i:
                similarities_for_i.append(similarity)
        
        if len(similarities_for_i) > 0:
            mean_sim = np.mean(similarities_for_i)
            median_sim = np.median(similarities_for_i)
            connectivity_scores[dataset_i] = {
                'mean_similarity': mean_sim,
                'median_similarity': median_sim,
                'n_connections': len(similarities_for_i)
            }
        else:
            connectivity_scores[dataset_i] = {
                'mean_similarity': 0.0,
                'median_similarity': 0.0,
                'n_connections': 0
            }
    
    mean_scores = [scores['mean_similarity'] for scores in connectivity_scores.values()]
    print(f"  Mean connectivity across datasets: {np.mean(mean_scores):.3f}")
    print(f"  Median connectivity across datasets: {np.median(mean_scores):.3f}")
    print(f"  Min connectivity: {np.min(mean_scores):.3f}")
    print(f"  Max connectivity: {np.max(mean_scores):.3f}")
    
    return connectivity_scores

def create_connectivity_plot(pairwise_similarities, all_similarities, connectivity_scores, output_dir):
    """Create connectivity analysis plot and save data."""
    print("\n" + "=" * 80)
    print("Creating connectivity analysis plot...")
    print("=" * 80)
    
    # Create figure with single panel
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create histogram (no KDE)
    n, bins, patches = ax.hist(all_similarities, bins=50, alpha=0.6, color='steelblue', 
                               edgecolor='black', linewidth=0.5, density=False)
    
    # Set Y-axis to discrete counts with max + 2
    max_count = int(np.max(n))
    ax.set_ylim(0, max_count + 2)
    # Set Y-axis ticks to integers
    ax.set_yticks(range(0, max_count + 3, max(1, (max_count + 2) // 10)))
    
    ax.set_xlabel('Shared Proteome Fraction', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Dataset Pairs', fontsize=12, fontweight='bold')
    ax.set_title('Pairwise Similarity Distribution', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim(0, 1)
    
    # Add statistics text
    mean_sim = np.mean(all_similarities)
    median_sim = np.median(all_similarities)
    ax.text(0.02, 0.98, f'Mean: {mean_sim:.3f}\nMedian: {median_sim:.3f}\nn = {len(all_similarities)} pairs',
             transform=ax.transAxes, fontsize=9,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / "connectivity_analysis.png"
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved: {output_file.name}")
    
    # Save data
    print("\nSaving data files...")
    
    # Save pairwise similarities
    pairwise_df = pd.DataFrame([
        {'Dataset_1': ds_i, 'Dataset_2': ds_j, 'Shared_Fraction': sim}
        for (ds_i, ds_j), sim in pairwise_similarities.items()
    ])
    pairwise_df = pairwise_df.sort_values('Shared_Fraction', ascending=False)
    pairwise_df.to_csv(output_dir / "pairwise_similarities.csv", index=False)
    print(f"  Saved: pairwise_similarities.csv")
    
    # Save dataset connectivity scores (sorted by least connected first)
    dataset_scores = []
    for dataset, scores in connectivity_scores.items():
        dataset_scores.append({
            'Dataset': dataset,
            'Mean_Connectivity': scores['mean_similarity'],
            'Median_Connectivity': scores['median_similarity'],
            'N_Connections': scores['n_connections']
        })
    
    # Sort by mean connectivity (ascending = least connected first)
    dataset_scores_df = pd.DataFrame(dataset_scores)
    dataset_scores_df = dataset_scores_df.sort_values('Mean_Connectivity', ascending=True)
    dataset_scores_df.to_csv(output_dir / "dataset_connectivity_scores.csv", index=False)
    print(f"  Saved: dataset_connectivity_scores.csv")
    
    # Print summary of least connected datasets
    print(f"\n  Least connected datasets (bottom 5):")
    for idx, row in dataset_scores_df.head(5).iterrows():
        print(f"    {row['Dataset']}: {row['Mean_Connectivity']:.3f}")

def create_pairwise_similarity_matrix(plasma_serum_data, normalized_protein_sets_by_dataset, output_dir, entry_name_library=None):
    """Create pairwise similarity matrix plot with shared proteins and abundance correlation.
    
    Matrix shows:
    - Circle size: number of shared proteins
    - Circle color: abundance correlation between datasets
    - Only upper triangle (no diagonal, no lower half)
    """
    print("  Creating pairwise similarity matrix...")
    
    # Start with sorted dataset names, will re-sort by similarity later
    dataset_names_initial = sorted(list(plasma_serum_data.keys()))
    n_datasets = len(dataset_names_initial)
    
    if n_datasets < 2:
        print("    Warning: Need at least 2 datasets for pairwise comparison")
        return
    
    # Pre-compute normalized protein sets per sample (much faster than doing it in nested loops)
    # Priority: 1) B_'s metadata (pre-calculated), 2) E_'s cache, 3) Compute on-the-fly
    cache_file = cache_dir / "normalized_proteins_per_sample.json"
    normalized_proteins_per_sample = {}  # {dataset_name: {sample: set(normalized_proteins)}}
    
    print("    Loading normalized protein sets...")
    # FIRST: Try to load from B_'s metadata files (already normalized, pre-calculated)
    datasets_with_metadata = 0
    for dataset_name in plasma_serum_data.keys():
        metadata = load_metadata(dataset_name)
        if metadata and 'protein_sets_per_sample' in metadata:
            # Proteins in metadata are already normalized from B_
            normalized_proteins_per_sample[dataset_name] = {
                sample: set(proteins) for sample, proteins in metadata['protein_sets_per_sample'].items()
            }
            datasets_with_metadata += 1
    
    if datasets_with_metadata > 0:
        print(f"      Loaded {datasets_with_metadata} dataset(s) from B_ metadata (pre-calculated)")
    
    # SECOND: Try to load from E_'s cache for remaining datasets
    if cache_file.exists():
        try:
            print("      Loading from E_ cache for remaining datasets...")
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                # Convert lists back to sets
                for dataset_name, sample_dict in cached_data.items():
                    if dataset_name not in normalized_proteins_per_sample:
                        normalized_proteins_per_sample[dataset_name] = {
                            sample: set(protein_list) for sample, protein_list in sample_dict.items()
                        }
            additional_count = len([d for d in normalized_proteins_per_sample.keys() if d in cached_data and d not in [k for k in normalized_proteins_per_sample.keys() if load_metadata(k)]])
            if additional_count > 0:
                print(f"      Loaded {additional_count} additional dataset(s) from E_ cache")
        except Exception as e:
            print(f"      Warning: Could not load E_ cache: {e}")
    
    # THIRD: Compute missing datasets
    datasets_to_process = []
    for dataset_name, data in plasma_serum_data.items():
        if dataset_name not in normalized_proteins_per_sample:
            datasets_to_process.append((dataset_name, data))
    
    if datasets_to_process:
        print(f"      Computing normalized protein sets for {len(datasets_to_process)} dataset(s)...")
        for dataset_name, data in datasets_to_process:
            normalized_proteins_per_sample[dataset_name] = {}
            samples = data['Sample'].unique()
            print(f"        Processing {len(samples)} samples in {dataset_name}...")
            
            for sample in samples:
                sample_data = data[data['Sample'] == sample]
                sample_proteins = set(sample_data['Protein'].unique())
                # Note: Proteins from B_ cache are already normalized, but we normalize again
                # for cross-dataset consistency (in case some datasets are from old cache)
                normalized_sample_proteins = normalize_protein_set_for_comparison(sample_proteins, entry_name_library=entry_name_library)
                normalized_proteins_per_sample[dataset_name][sample] = normalized_sample_proteins
        
        # Save newly computed datasets to E_'s cache
        print("      Saving computed normalized protein sets to E_ cache...")
        try:
            # Load existing cache
            existing_cache = {}
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        existing_cache = json.load(f)
                except:
                    pass
            
            # Update with newly computed datasets
            for dataset_name, _ in datasets_to_process:
                if dataset_name in normalized_proteins_per_sample:
                    existing_cache[dataset_name] = {
                        sample: list(protein_set) for sample, protein_set in normalized_proteins_per_sample[dataset_name].items()
                    }
            
            # Save updated cache
            with open(cache_file, 'w') as f:
                json.dump(existing_cache, f, indent=2)
            print(f"      Cached data for {len(datasets_to_process)} dataset(s)")
        except Exception as e:
            print(f"      Warning: Could not save E_ cache: {e}")
    
    # Calculate protein abundance by dataset (feature-based) and map to normalized proteins
    print("    Calculating feature-based protein abundances...")
    protein_abundance_by_dataset = {}
    for dataset_name, data in plasma_serum_data.items():
        raw_abundance = _compute_feature_abundance_by_protein(data, protein_col="Protein")
        if not raw_abundance:
            protein_abundance_by_dataset[dataset_name] = {}
            continue
        _, norm_map = normalize_protein_set_for_comparison(
            set(raw_abundance.keys()), entry_name_library=entry_name_library, return_mapping=True
        )
        norm_abundance = {}
        for norm_p, originals in norm_map.items():
            best = 0.0
            for o in originals:
                best = max(best, float(raw_abundance.get(o, 0.0)))
            norm_abundance[norm_p] = best
        protein_abundance_by_dataset[dataset_name] = norm_abundance
    
    # Pre-compute raw protein sets per dataset for debugging
    # (Proteins as they come directly from plasma_serum_data / cache)
    print("    Debug: comparing raw vs normalized protein sets per dataset...")
    raw_proteins_by_dataset = {}
    
    # Also check what's directly in cache files for comparison
    print("    Debug: checking cache files directly for comparison...")
    from shared_utils import find_cache_file
    cache_proteins_by_dataset = {}
    for dataset_name in dataset_names_initial:
        cache_file = find_cache_file(dataset_name, cache_dir)
        if cache_file:
            try:
                cache_df = pd.read_parquet(cache_file)
                cache_proteins = set(cache_df['Protein'].unique())
                cache_proteins_by_dataset[dataset_name] = cache_proteins
            except:
                cache_proteins_by_dataset[dataset_name] = set()
        else:
            cache_proteins_by_dataset[dataset_name] = set()
    
    for dataset_name in dataset_names_initial:
        raw_set = set(plasma_serum_data[dataset_name]['Protein'].unique())
        norm_set = normalized_protein_sets_by_dataset[dataset_name]
        cache_set = cache_proteins_by_dataset.get(dataset_name, set())
        raw_proteins_by_dataset[dataset_name] = raw_set
        
        # Compare all three
        if len(raw_set) != len(cache_set):
            print(f"      {dataset_name}: WARNING - plasma_serum_data ({len(raw_set)}) != cache ({len(cache_set)})")
        if len(raw_set) != len(norm_set):
            lost = len(raw_set) - len(norm_set)
            print(f"      {dataset_name}: raw={len(raw_set)}, normalized={len(norm_set)}, delta={lost}")
        else:
            print(f"      {dataset_name}: raw={len(raw_set)}, normalized={len(norm_set)} (no change)")
    
    # Calculate pairwise metrics
    print("    Computing pairwise similarities and abundance correlations...")
    total_pairs = (n_datasets * (n_datasets - 1)) // 2  # Upper triangle only
    print(f"      Computing {total_pairs} pairwise comparisons...")
    
    shared_proteins_matrix = np.zeros((n_datasets, n_datasets))
    abundance_corr_matrix = np.zeros((n_datasets, n_datasets))
    
    pair_count = 0
    for i, dataset_i in enumerate(dataset_names_initial):
        for j, dataset_j in enumerate(dataset_names_initial):
            if i >= j:  # Skip lower triangle and diagonal
                continue
            
            pair_count += 1
            if pair_count % 50 == 0 or pair_count == total_pairs:
                print(f"        Progress: {pair_count}/{total_pairs} pairs ({pair_count*100//total_pairs}%)")
            
            proteins_i = normalized_protein_sets_by_dataset[dataset_i]
            proteins_j = normalized_protein_sets_by_dataset[dataset_j]
            
            # Shared proteins (normalized identifiers)
            shared = proteins_i.intersection(proteins_j)
            shared_proteins_matrix[i, j] = len(shared)
            
            # Debug specific problematic pairs
            debug_pairs = [
                ("PXD002854-serum", "PXD008441"),
                ("PXD002854-serum", "PXD023650"),
                ("PXD062484", "PXD002854-serum"),
            ]
            if (dataset_i, dataset_j) in debug_pairs or (dataset_j, dataset_i) in debug_pairs:
                print(f"        DEBUG {dataset_i} vs {dataset_j}:")
                print(f"          {dataset_i}: {len(proteins_i)} proteins")
                print(f"          {dataset_j}: {len(proteins_j)} proteins")
                print(f"          Shared: {len(shared)} proteins")
                if len(shared) == 0:
                    print(f"          WARNING: 0 shared proteins!")
                    # Show sample proteins from each
                    sample_i = sorted(list(proteins_i))[:5]
                    sample_j = sorted(list(proteins_j))[:5]
                    print(f"          Sample {dataset_i}: {sample_i}")
                    print(f"          Sample {dataset_j}: {sample_j}")
            
            # Debug: check if raw proteins show sharing but normalized do not
            raw_i = raw_proteins_by_dataset[dataset_i]
            raw_j = raw_proteins_by_dataset[dataset_j]
            raw_shared = raw_i.intersection(raw_j)
            
            # Also check cache directly
            cache_i = cache_proteins_by_dataset.get(dataset_i, set())
            cache_j = cache_proteins_by_dataset.get(dataset_j, set())
            cache_shared = cache_i.intersection(cache_j)
            
            if len(shared) == 0 and len(raw_shared) > 0:
                # This indicates normalization removed shared proteins between these datasets
                print(f"        WARNING: raw vs normalized mismatch for {dataset_i} vs {dataset_j}")
                print(f"                 raw_shared={len(raw_shared)}, normalized_shared=0")
                print(f"                 sample raw shared: {sorted(list(raw_shared))[:10]}")
            
            if len(shared) == 0 and len(cache_shared) > 0:
                # This indicates plasma_serum_data doesn't match cache files
                print(f"        WARNING: cache vs plasma_serum_data mismatch for {dataset_i} vs {dataset_j}")
                print(f"                 cache_shared={len(cache_shared)}, normalized_shared=0")
                print(f"                 sample cache shared: {sorted(list(cache_shared))[:10]}")
                print(f"                 cache_i={len(cache_i)}, cache_j={len(cache_j)}")
                print(f"                 raw_i={len(raw_i)}, raw_j={len(raw_j)}")
            
            # Abundance correlation for shared proteins
            ab_i = protein_abundance_by_dataset.get(dataset_i, {})
            ab_j = protein_abundance_by_dataset.get(dataset_j, {})
            
            # Build vectors for shared proteins only (more efficient)
            shared_ab_i = []
            shared_ab_j = []
            for protein in shared:
                if protein in ab_i and protein in ab_j:
                    shared_ab_i.append(ab_i[protein])
                    shared_ab_j.append(ab_j[protein])
            
            if len(shared_ab_i) > 1:
                correlation = np.corrcoef(shared_ab_i, shared_ab_j)[0, 1]
                if not np.isnan(correlation):
                    abundance_corr_matrix[i, j] = correlation
                else:
                    abundance_corr_matrix[i, j] = 0
            else:
                abundance_corr_matrix[i, j] = 0
    
    # Sort datasets by abundance correlation (average correlation) - most correlated to the top
    # Calculate average abundance correlation for each dataset
    dataset_avg_correlation = {}
    for i, dataset_i in enumerate(dataset_names_initial):
        correlations = []
        for j, dataset_j in enumerate(dataset_names_initial):
            if i != j:  # Exclude self
                if i < j:
                    correlations.append(abundance_corr_matrix[i, j])
                else:
                    correlations.append(abundance_corr_matrix[j, i])
        dataset_avg_correlation[dataset_i] = np.mean(correlations) if correlations else 0
    
    # Sort datasets by average abundance correlation (descending - most correlated first)
    dataset_names = sorted(dataset_names_initial, key=lambda x: dataset_avg_correlation[x], reverse=True)
    
    # Rebuild matrices with new order
    # Create mapping from old index to new index
    old_to_new = {old_name: new_idx for new_idx, old_name in enumerate(dataset_names)}
    
    # Rebuild matrices in new order
    shared_proteins_matrix_sorted = np.zeros((n_datasets, n_datasets))
    abundance_corr_matrix_sorted = np.zeros((n_datasets, n_datasets))
    
    for i_old, dataset_i in enumerate(dataset_names_initial):
        for j_old, dataset_j in enumerate(dataset_names_initial):
            if i_old < j_old:  # Upper triangle in original order
                i_new = old_to_new[dataset_i]
                j_new = old_to_new[dataset_j]
                # Ensure we store in upper triangle of sorted matrix (i_new < j_new)
                if i_new < j_new:
                    shared_proteins_matrix_sorted[i_new, j_new] = shared_proteins_matrix[i_old, j_old]
                    abundance_corr_matrix_sorted[i_new, j_new] = abundance_corr_matrix[i_old, j_old]
                else:
                    # If after sorting, i_new > j_new, store in upper triangle (swap indices)
                    shared_proteins_matrix_sorted[j_new, i_new] = shared_proteins_matrix[i_old, j_old]
                    abundance_corr_matrix_sorted[j_new, i_new] = abundance_corr_matrix[i_old, j_old]
    
    # Use sorted matrices
    shared_proteins_matrix = shared_proteins_matrix_sorted
    abundance_corr_matrix = abundance_corr_matrix_sorted
    
    # Create mask for lower triangle and diagonal (like corrplot)
    # Mask should be True for cells we want to hide (lower triangle + diagonal)
    mask = np.zeros_like(abundance_corr_matrix, dtype=bool)
    # Set lower triangle (including diagonal) to True (masked/hidden)
    for i in range(n_datasets):
        for j in range(n_datasets):
            if i >= j:  # Lower triangle including diagonal
                mask[i, j] = True
    
    # Create the plot with heatmap style (like corrplot)
    fig, ax = plt.subplots(figsize=(max(12, n_datasets * 0.6), max(12, n_datasets * 0.6)))
    
    # Create white background with grid lines manually
    # Draw white squares for upper triangle only
    for i in range(n_datasets):
        for j in range(n_datasets):
            if i < j:  # Upper triangle only
                rect = plt.Rectangle((j, i), 1, 1, facecolor='white', 
                                   edgecolor='gray', linewidth=0.5)
                ax.add_patch(rect)
    
    # Set axis limits and ticks
    ax.set_xlim(0, n_datasets)
    ax.set_ylim(0, n_datasets)
    ax.set_xticks(np.arange(0.5, n_datasets, 1))
    ax.set_yticks(np.arange(0.5, n_datasets, 1))
    ax.set_xticklabels(dataset_names)
    ax.set_yticklabels(dataset_names)
    ax.invert_yaxis()  # Invert y-axis so first dataset is at top
    
    # Add circles whose size represents shared protein counts (only in upper triangle)
    # Color circles by correlation value
    max_shared = np.max(shared_proteins_matrix) if np.max(shared_proteins_matrix) > 0 else 1
    
    # Prepare data for scatter plot (circles)
    circle_x = []
    circle_y = []
    circle_sizes = []
    circle_colors = []
    
    for i in range(n_datasets):
        for j in range(n_datasets):
            if i < j:  # Upper triangle only
                shared = shared_proteins_matrix[i, j]
                corr = abundance_corr_matrix[i, j]
                if shared > 0:  # Only plot if there are shared proteins
                    # Position in heatmap coordinates (center of cell)
                    circle_x.append(j + 0.5)
                    circle_y.append(i + 0.5)
                    # Scale circle size based on shared proteins
                    # Previous iteration: (shared / max_shared) * 400 + 50 (range 50-450)
                    # User wants 2-fold increase: range 100-900
                    # This keeps circles small and inside the rectangle cells
                    if max_shared > 0:
                        normalized_size = (shared / max_shared) * 800 + 100  # Scale between 100 and 900
                    else:
                        normalized_size = 100
                    circle_sizes.append(normalized_size)
                    # Color based on correlation: blue=low (0), red=high (1)
                    # Correlations are already in range 0 to 1, use directly with reversed colormap
                    color = plt.cm.RdBu_r(corr)  # RdBu_r: blue=0 (low), red=1 (high)
                    circle_colors.append(color)
    
    # Overlay circles on the heatmap, colored by correlation
    if len(circle_sizes) > 0:
        ax.scatter(circle_x, circle_y, s=circle_sizes, 
                  c=circle_colors, alpha=0.8, edgecolors='black', linewidths=1,
                  zorder=10)  # zorder ensures circles are on top
    
    # Add colorbar for correlation
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Abundance Correlation', fontsize=10, fontweight='bold')
    
    # Customize labels (twice as big as before)
    # Move x-axis labels to top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='left', fontsize=14)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=14)
    
    # Remove axis labels (corrplot style is minimal)
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Set title
    ax.set_title('Pairwise Similarity Matrix (Plasma/Serum)\nColor = Abundance correlation, Circle size = Shared proteins', 
                fontsize=11, fontweight='bold', pad=10)
    
    # Minimal margins (like corrplot)
    plt.tight_layout()
    
    # Save as PNG
    output_file_png = output_dir / "pairwise_similarity_matrix.png"
    plt.savefig(output_file_png, bbox_inches='tight', dpi=300, pad_inches=0.1)
    print(f"    Saved: {output_file_png.name}")
    
    # Save as PDF
    output_file_pdf = output_dir / "pairwise_similarity_matrix.pdf"
    plt.savefig(output_file_pdf, bbox_inches='tight', pad_inches=0.1)
    print(f"    Saved: {output_file_pdf.name}")
    
    plt.close()
    
    # Create second plot: Circle size = Percentage of smaller dataset that is shared
    print("    Creating pairwise similarity matrix (percentage-based circle sizes)...")
    fig, ax = plt.subplots(figsize=(max(12, n_datasets * 0.6), max(12, n_datasets * 0.6)))
    
    # Create white background with grid lines manually
    for i in range(n_datasets):
        for j in range(n_datasets):
            if i < j:  # Upper triangle only
                rect = plt.Rectangle((j, i), 1, 1, facecolor='white', 
                                   edgecolor='gray', linewidth=0.5)
                ax.add_patch(rect)
    
    # Set axis limits and ticks
    ax.set_xlim(0, n_datasets)
    ax.set_ylim(0, n_datasets)
    ax.set_xticks(np.arange(0.5, n_datasets, 1))
    ax.set_yticks(np.arange(0.5, n_datasets, 1))
    ax.set_xticklabels(dataset_names)
    ax.set_yticklabels(dataset_names)
    ax.invert_yaxis()  # Invert y-axis so first dataset is at top
    
    # Prepare data for scatter plot (circles) - percentage based
    circle_x_pct = []
    circle_y_pct = []
    circle_sizes_pct = []
    circle_colors_pct = []
    
    # Calculate percentage matrix
    percentage_matrix = np.zeros((n_datasets, n_datasets))
    for i in range(n_datasets):
        for j in range(n_datasets):
            if i < j:  # Upper triangle only
                shared = shared_proteins_matrix[i, j]
                size_i = len(normalized_protein_sets_by_dataset[dataset_names[i]])
                size_j = len(normalized_protein_sets_by_dataset[dataset_names[j]])
                min_size = min(size_i, size_j)
                if min_size > 0:
                    percentage = (shared / min_size) * 100  # Percentage of smaller dataset
                    percentage_matrix[i, j] = percentage
                else:
                    percentage_matrix[i, j] = 0
                
                corr = abundance_corr_matrix[i, j]
                if shared > 0:  # Only plot if there are shared proteins
                    # Position in heatmap coordinates (center of cell)
                    circle_x_pct.append(j + 0.5)
                    circle_y_pct.append(i + 0.5)
                    # Scale circle size based on percentage (0-100% maps to 100-900)
                    normalized_size_pct = 100 + (percentage / 100) * 800  # Scale between 100 and 900
                    circle_sizes_pct.append(normalized_size_pct)
                    # Color based on correlation
                    color = plt.cm.RdBu_r(corr)  # RdBu_r: blue=0 (low), red=1 (high)
                    circle_colors_pct.append(color)
    
    # Overlay circles on the heatmap, colored by correlation
    if len(circle_sizes_pct) > 0:
        ax.scatter(circle_x_pct, circle_y_pct, s=circle_sizes_pct, 
                  c=circle_colors_pct, alpha=0.8, edgecolors='black', linewidths=1,
                  zorder=10)  # zorder ensures circles are on top
    
    # Add colorbar for correlation
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Abundance Correlation', fontsize=10, fontweight='bold')
    
    # Customize labels (twice as big as before)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='left', fontsize=14)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=14)
    
    # Remove axis labels
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Set title
    ax.set_title('Pairwise Similarity Matrix (Plasma/Serum)\nColor = Abundance correlation, Circle size = % of smaller dataset shared', 
                fontsize=11, fontweight='bold', pad=10)
    
    # Minimal margins
    plt.tight_layout()
    
    # Save as PNG
    output_file_png_pct = output_dir / "pairwise_similarity_matrix_percentage.png"
    plt.savefig(output_file_png_pct, bbox_inches='tight', dpi=300, pad_inches=0.1)
    print(f"    Saved: {output_file_png_pct.name}")
    
    # Save as PDF
    output_file_pdf_pct = output_dir / "pairwise_similarity_matrix_percentage.pdf"
    plt.savefig(output_file_pdf_pct, bbox_inches='tight', pad_inches=0.1)
    print(f"    Saved: {output_file_pdf_pct.name}")
    
    plt.close()
    
    # Save data
    pairwise_data = []
    for i, dataset_i in enumerate(dataset_names):
        for j, dataset_j in enumerate(dataset_names):
            if i < j:
                shared = int(shared_proteins_matrix[i, j])
                size_i = len(normalized_protein_sets_by_dataset[dataset_i])
                size_j = len(normalized_protein_sets_by_dataset[dataset_j])
                min_size = min(size_i, size_j)
                percentage = (shared / min_size * 100) if min_size > 0 else 0
                
                pairwise_data.append({
                    'Dataset_1': dataset_i,
                    'Dataset_2': dataset_j,
                    'Shared_Proteins': shared,
                    'Percentage_of_Smaller_Dataset': round(percentage, 2),
                    'Abundance_Correlation': round(abundance_corr_matrix[i, j], 3)
                })
    
    pairwise_df = pd.DataFrame(pairwise_data)
    pairwise_df = pairwise_df.sort_values('Shared_Proteins', ascending=False)
    pairwise_df.to_csv(output_dir / "pairwise_similarity_matrix_data.csv", index=False)
    print(f"    Saved: pairwise_similarity_matrix_data.csv")

def create_accumulation_and_network_plots(all_data, plots_dir, entry_name_library=None):
    """Create proteome accumulation curves and network plots for plasma/serum."""
    print("=" * 50)
    print("CREATING PROTEOME ACCUMULATION CURVES (Plasma/Serum)")
    print("=" * 50)
    
    # Same plasma/serum definition as B / main E (blood, plasma, serum → Blood Plasma/Serum)
    plasma_serum_data = {}
    for dataset_name, data in all_data.items():
        if is_plasma_serum(dataset_name, data):
            plasma_serum_data[dataset_name] = data
    allowed_02 = get_blood_plasma_serum_datasets_from_02()
    if allowed_02 is not None:
        before = len(plasma_serum_data)
        plasma_serum_data = {k: v for k, v in plasma_serum_data.items() if k in allowed_02}
        if len(plasma_serum_data) < before:
            print(f"  Restricted to {len(plasma_serum_data)} Blood Plasma/Serum datasets from 02 (excluded {before - len(plasma_serum_data)} not in 02)")
    
    if len(plasma_serum_data) == 0:
        print("  Warning: No plasma/serum datasets found for accumulation curves")
        return
    
    print(f"  Found {len(plasma_serum_data)} plasma/serum datasets")
    
    plasma_serum_tissue_safe = "Blood_Plasma_Serum"
    plasma_serum_plot_dir = plots_dir / plasma_serum_tissue_safe
    plasma_serum_plot_dir.mkdir(exist_ok=True)
    
    # Normalize proteins for consistent counting (normalize union once, then map per dataset)
    print("  Normalizing protein identifiers to UniProt accessions for consistent counting...")
    all_raw_plasma = set()
    for data in plasma_serum_data.values():
        all_raw_plasma.update(data['Protein'].unique())
    all_plasma_serum_proteins_total, plasma_norm_mapping = normalize_protein_set_for_comparison(
        all_raw_plasma, entry_name_library=entry_name_library, return_mapping=True)
    orig_to_norm_plasma = {}
    for norm_p, originals in plasma_norm_mapping.items():
        for o in originals:
            orig_to_norm_plasma[o] = norm_p
    normalized_protein_sets_by_dataset = {}
    for dataset_name, data in plasma_serum_data.items():
        raw_proteins = set(data['Protein'].unique())
        normalized_protein_sets_by_dataset[dataset_name] = {orig_to_norm_plasma.get(p, p) for p in raw_proteins}
    
    # Restrict to 02 Blood Plasma/Serum protein list so accumulation curve matches 02 (e.g. 5864)
    proteins_02_plasma = get_02_blood_plasma_serum_protein_set()
    if proteins_02_plasma is not None:
        before = len(all_plasma_serum_proteins_total)
        all_plasma_serum_proteins_total = all_plasma_serum_proteins_total & proteins_02_plasma
        for dn in normalized_protein_sets_by_dataset:
            normalized_protein_sets_by_dataset[dn] = normalized_protein_sets_by_dataset[dn] & proteins_02_plasma
        print(f"  Restricted to 02 Blood Plasma/Serum list: {len(all_plasma_serum_proteins_total)} proteins (from {before})")
    
    total_unique_proteins = len(all_plasma_serum_proteins_total)
    print(f"\n  Summary: {total_unique_proteins} unique proteins across all plasma/serum datasets (normalized)")
    
    # Save all normalized proteins to CSV
    print("\n  Saving all normalized proteins to CSV...")
    proteins_df = pd.DataFrame({
        'Protein': sorted(list(all_plasma_serum_proteins_total))
    })
    proteins_csv_file = plasma_serum_plot_dir / "all_plasma_serum_proteins_normalized.csv"
    
    try:
        proteins_df.to_csv(proteins_csv_file, index=False)
        print(f"  Saved: {plasma_serum_tissue_safe}/all_plasma_serum_proteins_normalized.csv ({len(proteins_df)} proteins)")
    except PermissionError:
        print(f"  ERROR: Cannot save {proteins_csv_file.name}")
        print(f"  The file is likely open in another program (e.g., Excel).")
        print(f"  Please close the file and re-run the script.")
        print(f"  Skipping CSV save, but continuing with analysis...")
    except Exception as e:
        print(f"  Warning: Could not save CSV file: {e}")
        print(f"  Continuing with analysis...")
    
    # Proteome Accumulation Curve
    print("\nCreating proteome accumulation curve (random shuffles, plasma/serum only)...")
    
    n_iterations = 100
    n_datasets_total = len(plasma_serum_data)
    
    print("  Calculating protein accumulation curves...")
    protein_accumulation_curves = []
    np.random.seed(42)
    for iteration in range(n_iterations):
        dataset_names_shuffled = np.random.permutation(list(plasma_serum_data.keys()))
        cumulative_proteins = set()
        curve = []
        for i, dataset_name in enumerate(dataset_names_shuffled):
            # Use normalized proteins for consistent counting
            cumulative_proteins.update(normalized_protein_sets_by_dataset[dataset_name])
            curve.append(len(cumulative_proteins))
        protein_accumulation_curves.append(curve)
        if (iteration + 1) % 20 == 0:
            print(f"    Completed {iteration + 1}/{n_iterations} iterations...")
    
    protein_accumulation_array = np.array(protein_accumulation_curves)
    mean_protein_curve = np.mean(protein_accumulation_array, axis=0)
    min_protein_curve = np.min(protein_accumulation_array, axis=0)
    max_protein_curve = np.max(protein_accumulation_array, axis=0)
    q25_protein_curve = np.percentile(protein_accumulation_array, 25, axis=0)
    q75_protein_curve = np.percentile(protein_accumulation_array, 75, axis=0)
    final_mean_proteins = mean_protein_curve[-1]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    x_vals = range(1, n_datasets_total + 1)
    
    ax.fill_between(x_vals, q25_protein_curve, q75_protein_curve, alpha=0.3, color='steelblue', label='IQR (25th-75th percentile)')
    ax.fill_between(x_vals, min_protein_curve, max_protein_curve, alpha=0.1, color='steelblue', label='Min-Max range')
    ax.plot(x_vals, mean_protein_curve, linewidth=3, color='darkblue', label='Mean', zorder=10)
    
    ax.set_xlabel('Number of Datasets Added', fontsize=12)
    ax.set_ylabel('Cumulative Unique Proteins', fontsize=12)
    n_02_expected = len(proteins_02_plasma) if proteins_02_plasma is not None else None
    subtitle_02 = ''
    if n_02_expected is not None and int(final_mean_proteins) == n_02_expected:
        subtitle_02 = f' (matches 02 Blood Plasma/Serum list: {n_02_expected})'
    elif n_02_expected is not None:
        subtitle_02 = f' (02 list size: {n_02_expected}; check dataset inclusion if these differ)'
    ax.set_title(f'Proteome Accumulation Curve (Plasma/Serum)\n'
                f'Final: {int(final_mean_proteins)} unique proteins{subtitle_02}\n'
                f'(Cumulative union as datasets are added in random order)', 
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)
    
    ax.annotate(f'Final: {int(final_mean_proteins)} proteins', 
               xy=(n_datasets_total, final_mean_proteins), 
               xytext=(n_datasets_total * 0.7, final_mean_proteins * 0.3),
               arrowprops=dict(arrowstyle='->', color='red', lw=2),
               fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    print(f"  Final cumulative unique proteins: {int(final_mean_proteins)}")
    print(f"  Total unique proteins across all datasets: {total_unique_proteins}")
    
    plt.tight_layout()
    plt.savefig(plasma_serum_plot_dir / "proteome_accumulation_curve_random.png", bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved: {plasma_serum_tissue_safe}/proteome_accumulation_curve_random.png")
    
    # Create pairwise similarity matrix plot
    print("\nCreating pairwise similarity matrix plot...")
    create_pairwise_similarity_matrix(plasma_serum_data, normalized_protein_sets_by_dataset, plasma_serum_plot_dir, entry_name_library=entry_name_library)

# ============================================
# MAIN ANALYSIS
# ============================================

def main():
    print("=" * 80)
    print("PLASMA AND CELL TYPES COMPARISON")
    print("=" * 80)
    print("Comparing protein presence between plasma/serum and cell type datasets\n")
    
    # Load entry name mapping library once (used throughout; same as B_ via shared_utils)
    print("\nLoading entry name mapping library...")
    entry_name_library = load_entry_name_mapping_library(library_file)
    manual_mapping = load_manual_mapping(manual_mapping_file, verbose=False)
    entry_name_library = {**entry_name_library, **manual_mapping}  # Merge manual mapping
    print(f"  Total entry name mappings available: {len(entry_name_library)}")
    
    # Load all datasets from cache
    all_data = load_all_datasets_from_cache()
    
    if len(all_data) == 0:
        print("Error: No datasets loaded from cache.")
        print("Please run B_data_preparation_and_filtering.py first to create cache.")
        return
    
    # Identify plasma/serum datasets
    n_total_cached = len(all_data)
    print("\nIdentifying plasma/serum datasets...")
    print(f"  (Total cached datasets loaded: {n_total_cached}; rest are erythrocyte, platelet, cell types, etc.)")
    plasma_serum_datasets = []
    plasma_serum_proteins_by_sample = {}
    plasma_sample_to_dataset = {}
    
    # Check for cached proteins_per_sample (only use if built with same filter as current run)
    # Cache file: cache/plasma_serum_proteins_by_sample.json
    # Cache is invalidated if parquet suffix (2-peptide filter) changes so comparison uses filtered counts.
    cache_file = cache_dir / "plasma_serum_proteins_by_sample.json"
    cached_proteins_by_sample = {}
    current_suffix = get_default_cache_parquet_suffix(cache_dir)
    if cache_file.exists():
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                raw_cache = json.load(f)
            meta = raw_cache.pop("_cache_meta", None)
            if meta is None or meta.get("parquet_suffix") != current_suffix:
                if meta is None:
                    print(f"  Cache has no filter version; recomputing proteins_per_sample (use filtered data).")
                else:
                    print(f"  Cache was built with different filter (then: {meta.get('parquet_suffix')}, now: {current_suffix}); recomputing.")
            else:
                cached_proteins_by_sample = {k: set(v) for k, v in raw_cache.items() if isinstance(v, list)}
                print(f"  Loaded cached proteins_per_sample for {len(cached_proteins_by_sample)} samples (filter: {current_suffix})")
        except Exception as e:
            print(f"  Warning: Could not load cache: {e}")
    
    datasets_to_compute = []
    for dataset_name, data in all_data.items():
        if is_plasma_serum(dataset_name, data):
            plasma_serum_datasets.append(dataset_name)
            # Check if we have cached data for all samples of this dataset
            samples = data['Sample'].unique()
            all_cached = all(f"{dataset_name}_{sample}" in cached_proteins_by_sample for sample in samples)
            
            if all_cached:
                # Use cached data
                for sample in samples:
                    combined_sample_name = f"{dataset_name}_{sample}"
                    plasma_serum_proteins_by_sample[combined_sample_name] = cached_proteins_by_sample[combined_sample_name]
                    plasma_sample_to_dataset[combined_sample_name] = dataset_name
            else:
                # Need to compute
                datasets_to_compute.append((dataset_name, data))
    
    n_plasma_serum_in_cache = len(plasma_serum_datasets)
    allowed_02 = get_blood_plasma_serum_datasets_from_02()
    if allowed_02 is not None:
        n_in_02 = len(allowed_02)
        if n_plasma_serum_in_cache < n_in_02:
            missing = allowed_02 - set(plasma_serum_datasets)
            print(f"  Note: 02 lists {n_in_02} Blood Plasma/Serum datasets; {n_plasma_serum_in_cache} are in cache and classified as plasma/serum.")
            if len(missing) <= 10:
                print(f"  Datasets in 02 but not in cache / not plasma: {sorted(missing)}")
            else:
                print(f"  {len(missing)} datasets in 02 have no matching cache or are not classified as plasma/serum.")
    print(f"  Plasma/serum datasets in cache: {n_plasma_serum_in_cache} (of {n_total_cached} total cached).")
    
    # Compute proteins per sample for datasets not in cache (using efficient groupby)
    if datasets_to_compute:
        print(f"  Computing proteins_per_sample for {len(datasets_to_compute)} plasma/serum dataset(s)...")
        for dataset_name, data in datasets_to_compute:
            # Use groupby which is much faster than filtering for each sample
            for sample, sample_data in data.groupby('Sample'):
                sample_proteins = set(sample_data['Protein'].unique())
                combined_sample_name = f"{dataset_name}_{sample}"
                plasma_serum_proteins_by_sample[combined_sample_name] = sample_proteins
                plasma_sample_to_dataset[combined_sample_name] = dataset_name
        
        # Save to cache for next time (include filter version so we only reuse when same 2-peptide filter)
        try:
            cache_data = {k: list(v) for k, v in plasma_serum_proteins_by_sample.items()}
            cache_data["_cache_meta"] = {"parquet_suffix": current_suffix}
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
            print(f"  Cached proteins_per_sample for {len(plasma_serum_proteins_by_sample)} samples (filter: {current_suffix})")
        except Exception as e:
            print(f"  Warning: Could not save cache: {e}")
    
    # Restrict to Blood Plasma/Serum datasets from 02 so counts match 02 (e.g. 5864)
    if allowed_02 is not None:
        before = len(plasma_serum_datasets)
        disallowed = set(plasma_serum_datasets) - allowed_02
        plasma_serum_datasets = [d for d in plasma_serum_datasets if d in allowed_02]
        for key in list(plasma_serum_proteins_by_sample):
            for d in sorted(disallowed, key=len, reverse=True):
                if key.startswith(d + "_"):
                    del plasma_serum_proteins_by_sample[key]
                    plasma_sample_to_dataset.pop(key, None)
                    break
        if before > len(plasma_serum_datasets):
            print(f"  Restricted to {len(plasma_serum_datasets)} Blood Plasma/Serum datasets from 02 (excluded {before - len(plasma_serum_datasets)} not in 02)")
    
    print(f"  Found {len(plasma_serum_datasets)} plasma/serum dataset(s)")
    print(f"  Total samples in plasma/serum: {len(plasma_serum_proteins_by_sample)}")
    
    # Identify cell type datasets
    print("\nIdentifying cell type datasets...")
    cell_type_datasets = {}
    excluded_cell_types = ['cd19-positive', 'cd19 positive', 'CD19-positive', 'CD19 positive']
    
    for dataset_name, data in all_data.items():
        if is_plasma_serum(dataset_name, data):
            continue
        
        # Check if this dataset has multiple cell types (multiple conditions)
        unique_conditions = data['Condition'].unique() if 'Condition' in data.columns else []
        
        # Process each condition separately
        cell_types_in_dataset = set()
        for condition in unique_conditions:
            if pd.isna(condition):
                continue
            
            # Create a temporary data subset for this condition
            condition_str = str(condition)
            if "|" in condition_str:
                condition_str = condition_str.split("|")[0].strip()
            
            cell_type = get_cell_type(dataset_name, pd.DataFrame({'Condition': [condition_str]}))
            cell_type_lower = cell_type.lower()
            
            # Skip if it's plasma/serum
            if 'plasma' in cell_type_lower or 'serum' in cell_type_lower:
                continue
            
            # Skip excluded types
            if 'terminally' in cell_type_lower and 'differentiated' in cell_type_lower:
                continue
            
            if cell_type_lower in excluded_cell_types or 'cd19' in cell_type_lower:
                continue
            
            cell_types_in_dataset.add(cell_type)
        
        # If no valid cell types found, try the old method (dataset-level identification)
        if len(cell_types_in_dataset) == 0:
            cell_type = get_cell_type(dataset_name, data)
            cell_type_lower = cell_type.lower()
            
            if 'terminally' in cell_type_lower and 'differentiated' in cell_type_lower:
                continue
            
            if cell_type_lower in excluded_cell_types or 'cd19' in cell_type_lower:
                print(f"  Excluding CD19-positive dataset: {dataset_name}")
                continue
            
            cell_types_in_dataset.add(cell_type)
        
        # Add dataset to each identified cell type
        for cell_type in cell_types_in_dataset:
            if cell_type not in cell_type_datasets:
                cell_type_datasets[cell_type] = []
            # Store as tuple: (dataset_name, condition) so we can filter later
            cell_type_datasets[cell_type].append((dataset_name, None))  # None means use all conditions for this cell type
    
    print(f"  Found {len(cell_type_datasets)} cell type(s): {list(cell_type_datasets.keys())}")
    for cell_type, datasets in cell_type_datasets.items():
        # Extract dataset names from tuples if needed
        dataset_names = [d[0] if isinstance(d, tuple) else d for d in datasets]
        print(f"    {cell_type}: {len(datasets)} dataset(s) - {dataset_names}")
    
    # Calculate intra-dataset analysis results (needed for some plots)
    print("\n" + "=" * 80)
    print("CALCULATING INTRA-DATASET ANALYSIS")
    print("=" * 80)
    intra_results = {}
    for dataset_name, data in all_data.items():
        intra_results[dataset_name] = intra_dataset_analysis(data, dataset_name)
    
    # Create all plots from proteomics_analysis.py
    print("\n" + "=" * 80)
    print("CREATING ALL PLOTS")
    print("=" * 80)
    
    # 1. Intra-dataset plots
    create_intra_dataset_plots(all_data, intra_results, plots_dir, tables_dir)
    
    # 2. Inter-dataset plots
    create_inter_dataset_plots(all_data, plots_dir, tables_dir)
    
    # 3. Tissue-grouped plots
    create_tissue_grouped_plots(all_data, intra_results, plots_dir, tables_dir, entry_name_library=entry_name_library)
    
    # 4. Accumulation curves and network plots
    create_accumulation_and_network_plots(all_data, plots_dir, entry_name_library=entry_name_library)
    
    # 5. Scatter plots and UpSet plots (already existing code below)
    if len(plasma_serum_proteins_by_sample) == 0 or len(cell_type_datasets) == 0:
        print("  Warning: No plasma/serum samples or cell types found, skipping scatter/upset plots...")
        return
    
    # Calculate feature-based protein abundance for plasma/serum.
    print("  Normalizing protein identifiers to UniProt accessions for feature-based abundance...")
    n_plasma_serum_samples = len(plasma_serum_proteins_by_sample)
    
    # Collect all unique raw proteins and normalize once (much faster than per-sample when many samples)
    all_raw_proteins = set()
    for raw_proteins in plasma_serum_proteins_by_sample.values():
        all_raw_proteins.update(raw_proteins)
    normalized_all, plasma_normalized_mapping = normalize_protein_set_for_comparison(
        all_raw_proteins, entry_name_library=entry_name_library, return_mapping=True)
    # Build original -> normalized for fast per-sample lookup
    original_to_normalized = {}
    for norm_protein, originals in plasma_normalized_mapping.items():
        for o in originals:
            original_to_normalized[o] = norm_protein
    # Remove duplicates from mapping lists (same as before)
    plasma_normalized_mapping = {k: list(set(v)) for k, v in plasma_normalized_mapping.items()}
    
    # Apply mapping to each sample (set lookup only, no repeated normalization)
    normalized_plasma_serum_proteins_by_sample = {}
    for sample_name, raw_proteins in plasma_serum_proteins_by_sample.items():
        normalized_plasma_serum_proteins_by_sample[sample_name] = {
            original_to_normalized.get(p, p) for p in raw_proteins
        }
    
    # Set of all normalized proteins (already have from single normalization)
    all_plasma_serum_proteins = normalized_all
    
    # Use 02's Blood Plasma/Serum protein list as canonical set so comparison plot matches 02 exactly (B_ writes this file)
    path_02_txt = (data_prep_output_dir / "02_blood_plasma_serum_proteins.txt").resolve()
    proteins_02_set = get_02_blood_plasma_serum_protein_set()
    if proteins_02_set is not None:
        n_02 = len(proteins_02_set)
        # Use 02 list as the canonical set (not intersection) so plot always shows 02 count
        all_plasma_serum_proteins = proteins_02_set
        in_both = len(normalized_all & proteins_02_set)
        print(f"  Using 02 protein list for comparison: {path_02_txt}")
        print(f"  Plasma/serum proteins for plot: {n_02} (from 02); {in_both} of these appear in current cache data.")
    else:
        print(f"  02 protein list not found at: {path_02_txt}")
        print(f"  Run B_data_preparation_and_filtering.py to create it; comparison plot will use cache union ({len(all_plasma_serum_proteins)} proteins, may not match 02).")
    
    # Calculate abundance using normalized proteins
    plasma_serum_protein_abundance = {}
    plasma_serum_protein_n_samples = {}
    
    # First, calculate abundance for original proteins (for normalized lookup)
    original_plasma_serum_protein_abundance = {}
    original_plasma_serum_protein_n_samples = {}
    plasma_serum_rows = []
    for dataset_name in plasma_serum_datasets:
        if dataset_name not in all_data:
            continue
        ds = all_data[dataset_name]
        if 'Condition' in ds.columns:
            mask = ds['Condition'].apply(lambda x: _is_plasma_serum_label(str(x)) if pd.notna(x) else False)
            ds = ds[mask]
        if len(ds) > 0 and 'Protein' in ds.columns:
            plasma_serum_rows.append(ds)
    if plasma_serum_rows:
        plasma_all_df = pd.concat(plasma_serum_rows, ignore_index=True)
        original_plasma_serum_protein_abundance = _compute_feature_abundance_by_protein(plasma_all_df, protein_col='Protein')
    for sample_name, raw_proteins in plasma_serum_proteins_by_sample.items():
        for orig_protein in raw_proteins:
            if orig_protein not in original_plasma_serum_protein_n_samples:
                original_plasma_serum_protein_n_samples[orig_protein] = 0
            original_plasma_serum_protein_n_samples[orig_protein] += 1
    
    # Then, map to normalized proteins
    for normalized_protein in all_plasma_serum_proteins:
        n_present = sum(1 for normalized_proteins in normalized_plasma_serum_proteins_by_sample.values() if normalized_protein in normalized_proteins)
        plasma_serum_protein_n_samples[normalized_protein] = n_present
        orig_list = plasma_normalized_mapping.get(normalized_protein, [normalized_protein])
        plasma_serum_protein_abundance[normalized_protein] = max(
            [float(original_plasma_serum_protein_abundance.get(p, 0.0)) for p in orig_list] or [0.0]
        )
    
    # Filter: proteins appearing in 5+ samples
    plasma_serum_proteins_filtered = {p for p, n in plasma_serum_protein_n_samples.items() if n > 4}
    
    n_plasma_proteins = len(all_plasma_serum_proteins)
    expected_02 = get_02_correct_proteins_after_plasma_serum()
    print(f"\n  Total proteins in plasma/serum (normalized): {n_plasma_proteins}")
    if expected_02 is not None:
        print(f"  Expected (02 Correct_Proteins_After for Blood Plasma/Serum): {expected_02}")
        if n_plasma_proteins > expected_02:
            print(f"  WARNING: Count ({n_plasma_proteins}) exceeds 02 ({expected_02}). The comparison plot may use unfiltered or stale cache.")
            print(f"  To match 02: re-run B_data_preparation_and_filtering.py, then delete cache/plasma_serum_proteins_by_sample.json and re-run E_.")
    print(f"  Proteins in 5+ samples: {len(plasma_serum_proteins_filtered)}")
    print(f"  Proteins removed by filter: {len(all_plasma_serum_proteins - plasma_serum_proteins_filtered)}")
    
    # Collect data for each cell type
    print("\nCollecting protein data for each cell type...")
    cell_type_data = {}
    
    # Pre-compute condition to cell type mappings for each dataset (for performance)
    dataset_condition_mappings = {}
    for dataset_name, data in all_data.items():
        if 'Condition' not in data.columns:
            continue
        unique_conditions = data['Condition'].dropna().unique()
        condition_to_celltype = {}
        for condition in unique_conditions:
            condition_str = str(condition)
            if "|" in condition_str:
                condition_str = condition_str.split("|")[0].strip()
            cell_type = get_cell_type(dataset_name, pd.DataFrame({'Condition': [condition_str]}))
            condition_to_celltype[condition] = cell_type
        dataset_condition_mappings[dataset_name] = condition_to_celltype
    
    for cell_type, dataset_list in cell_type_datasets.items():
        print(f"  Processing {cell_type}...")
        cell_type_proteins_by_sample = {}
        for dataset_entry in dataset_list:
            # Handle both old format (string) and new format (tuple)
            if isinstance(dataset_entry, tuple):
                dataset_name = dataset_entry[0]
            else:
                dataset_name = dataset_entry
            
            if dataset_name in all_data:
                data = all_data[dataset_name]
                
                # Use pre-computed mapping for fast filtering
                if dataset_name in dataset_condition_mappings:
                    condition_to_celltype = dataset_condition_mappings[dataset_name]
                    # Filter data to only include samples with this cell type
                    condition_mask = data['Condition'].map(condition_to_celltype) == cell_type
                    data_filtered = data[condition_mask]
                else:
                    # Fallback: filter by checking conditions (slower)
                    condition_mask = data['Condition'].apply(lambda x: 
                        get_cell_type(dataset_name, pd.DataFrame({'Condition': [str(x)]})) == cell_type 
                        if pd.notna(x) else False)
                    data_filtered = data[condition_mask]
                
                if len(data_filtered) == 0:
                    continue
                
                # One groupby instead of filtering once per sample (much faster for many samples)
                sample_to_proteins = data_filtered.groupby('Sample')['Protein'].apply(lambda x: set(x.unique())).to_dict()
                for sample, sample_proteins in sample_to_proteins.items():
                    combined_sample_name = f"{dataset_name}_{sample}"
                    cell_type_proteins_by_sample[combined_sample_name] = sample_proteins
        
        if len(cell_type_proteins_by_sample) == 0:
            print(f"  Warning: No valid samples for {cell_type}, skipping...")
            continue
        
        n_cell_type_samples = len(cell_type_proteins_by_sample)
        all_cell_type_proteins = set()
        protein_n_present = {}  # count samples containing each protein (one pass)
        for proteins in cell_type_proteins_by_sample.values():
            all_cell_type_proteins.update(proteins)
            for p in proteins:
                protein_n_present[p] = protein_n_present.get(p, 0) + 1
        cell_type_protein_pct = {
            protein: (protein_n_present.get(protein, 0) / n_cell_type_samples) * 100
            for protein in all_cell_type_proteins
        }
        cell_type_protein_abundance = _compute_feature_abundance_by_protein(data_filtered, protein_col='Protein')
        
        cell_type_data[cell_type] = {
            'proteins': all_cell_type_proteins,
            'protein_pct': cell_type_protein_pct,
            'protein_abundance': cell_type_protein_abundance,
            'n_samples': n_cell_type_samples
        }
    
    # Load entry name mapping library if not provided
    if entry_name_library is None:
        print("  Loading entry name mapping library...")
        entry_name_library = load_entry_name_mapping_library(library_file)
        manual_mapping = load_manual_mapping(manual_mapping_file, verbose=False)
        entry_name_library = {**entry_name_library, **manual_mapping}  # Merge manual mapping
    
    # Normalize cell type protein sets once; restrict to 02 list per tissue so counts match 02
    normalized_cell_type_data = {}
    cell_type_normalized_mappings = {}
    for cell_type, data_dict in cell_type_data.items():
        normalized_proteins, normalized_mapping = normalize_protein_set_for_comparison(
            data_dict['proteins'], entry_name_library=entry_name_library, return_mapping=True)
        proteins_02_ct = get_02_protein_set_for_tissue(cell_type)
        if proteins_02_ct is not None:
            before_ct = len(normalized_proteins)
            normalized_proteins = normalized_proteins & proteins_02_ct
            if before_ct != len(normalized_proteins):
                print(f"  Restricted {cell_type} to 02 list: {len(normalized_proteins)} proteins (from {before_ct})")
        normalized_abundance = {}
        for norm_p, originals in normalized_mapping.items():
            best = 0.0
            for o in originals:
                best = max(best, float(data_dict.get('protein_abundance', {}).get(o, 0.0)))
            normalized_abundance[norm_p] = best
        normalized_cell_type_data[cell_type] = {
            **data_dict,
            'proteins': normalized_proteins,
            'protein_abundance_normalized': normalized_abundance,
        }
        cell_type_normalized_mappings[cell_type] = normalized_mapping
    
    # Create volcano plots for all proteins
    # Note: all_plasma_serum_proteins is already normalized
    for version_name, plasma_serum_proteins_set in [("all_proteins", all_plasma_serum_proteins)]:
        print(f"\n{'='*60}")
        print(f"Processing: {version_name}")
        print(f"  Using normalized protein identifiers (UniProt accessions)")
        
        # Proteins are already normalized, so use them directly
        normalized_plasma_serum_proteins = plasma_serum_proteins_set
        
        print(f"  Plasma/Serum proteins in set: {len(normalized_plasma_serum_proteins)} (normalized)")
        print(f"{'='*60}")
        
        # Prepare scatter plot data
        scatter_data_dict = {}
        valid_cell_types = []
        
        # Identify proteins only in plasma + each specific cell type (not in other cell types)
        # This will be used for coloring in scatter plots
        proteins_only_plasma_celltype = {}
        for cell_type, data_dict in normalized_cell_type_data.items():
            if _is_plasma_serum_label(cell_type):
                continue
            # Proteins in plasma and this cell type (using normalized sets)
            plasma_and_this_celltype = normalized_plasma_serum_proteins.intersection(data_dict['proteins'])
            
            # Check if these proteins are in any OTHER cell types
            proteins_only_in_this_pair = set()
            for protein in plasma_and_this_celltype:
                # Check if protein is in any other cell type
                in_other_celltype = False
                for other_cell_type, other_data_dict in normalized_cell_type_data.items():
                    if other_cell_type != cell_type and protein in other_data_dict['proteins']:
                        in_other_celltype = True
                        break
                
                # If not in any other cell type, it's only in plasma + this cell type
                if not in_other_celltype:
                    proteins_only_in_this_pair.add(protein)
            
            proteins_only_plasma_celltype[cell_type] = proteins_only_in_this_pair
            if len(proteins_only_in_this_pair) > 0:
                print(f"  {cell_type}: {len(proteins_only_in_this_pair)} proteins only in plasma+{cell_type} (not in other cell types)")
        
        for cell_type, data_dict in normalized_cell_type_data.items():
            if _is_plasma_serum_label(cell_type):
                continue  # don't show plasma/serum as a separate "cell type" in comparison
            shared_proteins = normalized_plasma_serum_proteins.intersection(data_dict['proteins'])
            
            if len(shared_proteins) == 0:
                print(f"  Warning: No shared proteins between plasma/serum and {cell_type}, skipping...")
                continue
            
            print(f"  {cell_type}: {len(shared_proteins)} shared proteins")
            valid_cell_types.append(cell_type)
            
            scatter_data = []
            for normalized_protein in shared_proteins:
                # Use normalized protein percentages directly (already calculated)
                plasma_pct = plasma_serum_protein_abundance.get(normalized_protein, 0)
                plasma_n_samples = plasma_serum_protein_n_samples.get(normalized_protein, 0)
                
                cell_pct = data_dict.get('protein_abundance_normalized', {}).get(normalized_protein, 0)
                
                # Check if this protein is only in plasma + this cell type
                is_only_plasma_celltype = normalized_protein in proteins_only_plasma_celltype.get(cell_type, set())
                
                # Calculate log2 fold change for volcano plot (add 1 to avoid log(0))
                log2_fc = np.log2((cell_pct + 1) / (plasma_pct + 1))
                # Calculate mean percentage for y-axis
                mean_pct = (plasma_pct + cell_pct) / 2
                
                scatter_data.append({
                    'Protein': normalized_protein,
                    'Plasma_Serum_Pct': plasma_pct,
                    'Cell_Type_Pct': cell_pct,
                    'Log2_FC': log2_fc,
                    'Mean_Pct': mean_pct,
                    'Only_Plasma_CellType': is_only_plasma_celltype
                })
            
            scatter_data_dict[cell_type] = pd.DataFrame(scatter_data)
            print(f"    Scatter plot data points: {len(scatter_data)}")
        
        if len(valid_cell_types) == 0:
            print(f"  Warning: No valid cell types for {version_name}, skipping...")
            continue
        
        # Create multipanel scatter plot (original abundance comparison)
        print(f"\nCreating multipanel scatter plot ({version_name})...")
        n_cell_types = len(valid_cell_types)
        n_cols = min(3, n_cell_types)
        n_rows = (n_cell_types + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_cell_types == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, cell_type in enumerate(valid_cell_types):
            ax = axes[idx]
            df_scatter = scatter_data_dict[cell_type].copy()
            # Log-transform first to spread values close to zero, then min-max to [0,1].
            df_scatter["Plasma_Serum_Log"] = np.log1p(pd.to_numeric(df_scatter["Plasma_Serum_Pct"], errors="coerce").fillna(0.0))
            df_scatter["Cell_Type_Log"] = np.log1p(pd.to_numeric(df_scatter["Cell_Type_Pct"], errors="coerce").fillna(0.0))
            df_scatter["Plasma_Serum_Norm01"] = _minmax_01(df_scatter["Plasma_Serum_Log"])
            df_scatter["Cell_Type_Norm01"] = _minmax_01(df_scatter["Cell_Type_Log"])
            
            # Check if dataframe is empty or has issues
            if len(df_scatter) == 0:
                print(f"      Warning: {cell_type} has no scatter data points, skipping plot")
                ax.text(0.5, 0.5, f'No data for {cell_type}', 
                       transform=ax.transAxes, ha='center', va='center', fontsize=12)
                ax.set_title(cell_type, fontsize=11, fontweight='bold')
                continue
            
            # Check for constant values (which would cause a straight line)
            if df_scatter['Plasma_Serum_Pct'].nunique() == 1 or df_scatter['Cell_Type_Pct'].nunique() == 1:
                print(f"      Warning: {cell_type} has constant values - Plasma_Serum_Pct unique: {df_scatter['Plasma_Serum_Pct'].nunique()}, Cell_Type_Pct unique: {df_scatter['Cell_Type_Pct'].nunique()}")
            
            # Separate proteins into two groups: only plasma+celltype vs others
            only_plasma_celltype = df_scatter[df_scatter['Only_Plasma_CellType'] == True]
            other_proteins = df_scatter[df_scatter['Only_Plasma_CellType'] == False]
            
            # Calculate number of unique proteins before plotting
            n_unique_proteins = len(only_plasma_celltype)
            
            # Plot other proteins in default color
            if len(other_proteins) > 0:
                ax.scatter(other_proteins['Plasma_Serum_Norm01'], other_proteins['Cell_Type_Norm01'], 
                          alpha=0.6, s=30, edgecolors='black', linewidth=0.5, 
                          color='steelblue', label='Other proteins')
            
            # Plot proteins only in plasma+this cell type in orange
            if len(only_plasma_celltype) > 0:
                ax.scatter(only_plasma_celltype['Plasma_Serum_Norm01'], only_plasma_celltype['Cell_Type_Norm01'], 
                          alpha=0.8, s=40, edgecolors='black', linewidth=0.5, 
                          color='orange', label=f'Only plasma+{cell_type} (n={n_unique_proteins})')
                print(f"      {cell_type}: {len(only_plasma_celltype)} proteins only in plasma+{cell_type} (colored orange)")
            
            # Add diagonal line
            max_val = max(df_scatter['Plasma_Serum_Norm01'].max(), df_scatter['Cell_Type_Norm01'].max(), 1e-6)
            min_val = 0.0
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.5, label='y=x')
            
            # Calculate correlation
            correlation = df_scatter['Plasma_Serum_Norm01'].corr(df_scatter['Cell_Type_Norm01'])
            n_proteins = len(df_scatter)
            
            # Get number of samples for this cell type
            n_cell_type_samples = normalized_cell_type_data[cell_type]['n_samples']
            
            ax.text(0.05, 0.95, f'r = {correlation:.3f}\nn = {n_proteins} proteins',
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel('Plasma/Serum\n(feature abundance, min-max 0-1)', fontsize=10)
            ax.set_ylabel(f'{cell_type}\n(feature abundance, min-max 0-1)', fontsize=10)
            ax.set_title(f'{cell_type} (n={n_cell_type_samples} samples)', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1.02)
            ax.set_ylim(0, 1.02)
            # Add legend
            if len(only_plasma_celltype) > 0 or idx == 0:
                ax.legend(loc='lower right', fontsize=8)
        
        # Hide unused subplots
        for idx in range(len(valid_cell_types), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Protein Presence: Plasma/Serum vs Cell Types ({version_name.replace("_", " ").title()})',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])

        scatter_plot_path = plots_dir / "cell_contamination_all_proteins.png"
        plt.savefig(scatter_plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  Saved: {scatter_plot_path.name}")
        
        # Create multipanel volcano plot
        print(f"\nCreating multipanel volcano plot ({version_name})...")
        n_cell_types = len(valid_cell_types)
        n_cols = min(3, n_cell_types)
        n_rows = (n_cell_types + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_cell_types == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, cell_type in enumerate(valid_cell_types):
            ax = axes[idx]
            df_scatter = scatter_data_dict[cell_type].copy()
            # For volcano, apply signed-log on fold change and log on mean abundance before scaling.
            df_scatter["Log2_FC_Log"] = _signed_log1p(df_scatter["Log2_FC"])
            df_scatter["Mean_Pct_Log"] = np.log1p(pd.to_numeric(df_scatter["Mean_Pct"], errors="coerce").fillna(0.0))
            df_scatter["Log2_FC_Norm01"] = _minmax_01(df_scatter["Log2_FC_Log"])
            df_scatter["Mean_Pct_Norm01"] = _minmax_01(df_scatter["Mean_Pct_Log"])
            
            # Check if dataframe is empty or has issues
            if len(df_scatter) == 0:
                print(f"      Warning: {cell_type} has no scatter data points, skipping plot")
                ax.text(0.5, 0.5, f'No data for {cell_type}', 
                       transform=ax.transAxes, ha='center', va='center', fontsize=12)
                ax.set_title(cell_type, fontsize=11, fontweight='bold')
                continue
            
            # Separate proteins into two groups: only plasma+celltype vs others
            only_plasma_celltype = df_scatter[df_scatter['Only_Plasma_CellType'] == True]
            other_proteins = df_scatter[df_scatter['Only_Plasma_CellType'] == False]
            
            # Calculate number of unique proteins before plotting
            n_unique_proteins = len(only_plasma_celltype)
            
            # Plot other proteins in default color
            if len(other_proteins) > 0:
                ax.scatter(other_proteins['Log2_FC_Norm01'], other_proteins['Mean_Pct_Norm01'], 
                          alpha=0.6, s=30, edgecolors='black', linewidth=0.5, 
                          color='steelblue', label='Other proteins')
            
            # Plot proteins only in plasma+this cell type in orange
            if len(only_plasma_celltype) > 0:
                ax.scatter(only_plasma_celltype['Log2_FC_Norm01'], only_plasma_celltype['Mean_Pct_Norm01'], 
                          alpha=0.8, s=40, edgecolors='black', linewidth=0.5, 
                          color='orange', label=f'Only plasma+{cell_type} (n={n_unique_proteins})')
                print(f"      {cell_type}: {len(only_plasma_celltype)} proteins only in plasma+{cell_type} (colored orange)")
            
            # Add vertical line at log2(FC) = 0
            ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            
            # Get number of samples for this cell type
            n_cell_type_samples = normalized_cell_type_data[cell_type]['n_samples']
            n_proteins = len(df_scatter)
            
            ax.text(0.05, 0.95, f'n = {n_proteins} proteins',
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel('Log2 Fold Change\n(min-max 0-1)', fontsize=10)
            ax.set_ylabel('Mean Abundance\n(min-max 0-1)', fontsize=10)
            ax.set_title(f'{cell_type} (n={n_cell_type_samples} samples)', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1.02)
            ax.set_ylim(0, 1.02)
            # Add legend
            if len(only_plasma_celltype) > 0 or idx == 0:
                ax.legend(loc='upper right', fontsize=8)
        
        # Hide unused subplots
        for idx in range(len(valid_cell_types), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Volcano Plot: Plasma/Serum vs Cell Types ({version_name.replace("_", " ").title()})',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        volcano_plot_path = plots_dir / "cell_contamination_all_proteins_volcano.png"
        plt.savefig(volcano_plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  Saved: {volcano_plot_path.name}")
        
        # Create upset plots
        print(f"\nCreating upset plots ({version_name})...")
        
        # Protein-based upset plot (use normalized sets for consistency)
        if UPSET_AVAILABLE:
            protein_sets_for_upset = {CANONICAL_PLASMA_SERUM_LABEL: list(normalized_plasma_serum_proteins)}
            for cell_type, data_dict in normalized_cell_type_data.items():
                if _is_plasma_serum_label(cell_type):
                    continue  # avoid duplicate plasma/serum set (e.g. "plasma" vs "Plasma/Serum")
                protein_sets_for_upset[cell_type] = list(data_dict['proteins'])
            
            protein_sets_for_upset = {k: v for k, v in protein_sets_for_upset.items() if len(v) > 0}
            
            if len(protein_sets_for_upset) >= 2:
                try:
                    upset_data_protein = from_contents(protein_sets_for_upset)
                    fig = plt.figure(figsize=(16, 10))
                    upset = UpSet(upset_data_protein, subset_size='count', show_counts=True, 
                                 sort_by='cardinality', min_subset_size=100)
                    upset.plot(fig=fig)
                    plt.suptitle(f'Protein Intersections: {CANONICAL_PLASMA_SERUM_LABEL} vs Cell Types ({version_name.replace("_", " ").title()})',
                                fontsize=14, fontweight='bold', y=0.98)
                    plt.tight_layout(rect=[0, 0, 1, 0.97])

                    upset_plot_path = plots_dir / "plasma_and_cell_types_comparison_all_proteins.png"
                    plt.savefig(upset_plot_path, bbox_inches='tight', dpi=300)
                    plt.close()
                    print(f"    Saved: {upset_plot_path.name} (proteins)")
                except Exception as e:
                    print(f"    Error creating protein upset plot: {e}")
            
            # Gene-based upset plot removed - keeping only protein-based plots
        
        # Save scatter plot data
        all_scatter_data = []
        for cell_type in valid_cell_types:
            df = scatter_data_dict[cell_type].copy()
            df['Cell_Type'] = cell_type
            all_scatter_data.append(df)
        
        if all_scatter_data:
            combined_scatter_df = pd.concat(all_scatter_data, ignore_index=True)
            scatter_data_path = tables_dir / "cell_contamination_all_proteins_data.csv"
            combined_scatter_df.to_csv(scatter_data_path, index=False)
            print(f"  Saved: {scatter_data_path.name}")

        # Reduced deepdive outputs per cell type (merged from E_cell_contamination_deepdive.py)
        try:
            print("\nWriting contamination_deepdive per-cell-type outputs (lists, candidates, overlays)...")
            write_contamination_deepdive_outputs(
                entry_name_library=entry_name_library,
                normalized_plasma_serum_proteins_by_sample=normalized_plasma_serum_proteins_by_sample,
                plasma_sample_to_dataset=plasma_sample_to_dataset,
                normalized_plasma_serum_proteins=normalized_plasma_serum_proteins,
                scatter_data_dict=scatter_data_dict,
                proteins_only_plasma_celltype=proteins_only_plasma_celltype,
                valid_cell_types=valid_cell_types,
            )
            print(f"  Saved per-cell-type outputs in: {deepdive_dir}")
        except Exception as e:
            print(f"  Warning: Could not write contamination_deepdive outputs: {e}")
    
    # Create proteins per tissue/cell type CSV
    print("\n" + "=" * 80)
    print("CREATING PROTEINS PER TISSUE/CELL TYPE CSV")
    print("=" * 80)
    
    def group_to_tissue_for_csv(condition):
        """Group condition into tissue/cell type for CSV output (one canonical name per tissue)."""
        condition_lower = condition.lower().strip()
        if 'plasma' in condition_lower or 'serum' in condition_lower:
            return CANONICAL_PLASMA_SERUM_LABEL.replace('/', '_').replace(' ', '_')  # Blood_Plasma_Serum
        elif 'erythrocyte' in condition_lower or 'red blood cell' in condition_lower:
            return 'Erythrocyte'
        elif 'platelet' in condition_lower:
            return 'Platelet'
        elif 'cd4' in condition_lower or 'cd8' in condition_lower or 't cell' in condition_lower:
            return 'T_Cell'
        elif 'b cell' in condition_lower or 'cd19' in condition_lower:
            return 'B_Cell'
        elif 'nk cell' in condition_lower or 'natural killer' in condition_lower:
            return 'NK_Cell'
        elif 'dendritic cell' in condition_lower or 'dc' in condition_lower:
            return 'Dendritic_Cell'
        elif 'monocyte' in condition_lower:
            return 'Monocyte'
        else:
            return condition.replace('/', '_').replace('\\', '_').replace(' ', '_').replace(':', '_').replace('(', '').replace(')', '')
    
    tissue_proteins_dict = {}
    for dataset_name, data in all_data.items():
        condition = str(data['Condition'].iloc[0])
        if "|" in condition:
            condition = condition.split("|")[0].strip()
        
        tissue = group_to_tissue_for_csv(condition)
        tissue_lower = tissue.lower()
        
        if 'terminally' in tissue_lower and 'differentiated' in tissue_lower:
            continue
        if 'cd19' in tissue_lower:
            continue
        
        unique_proteins = set(data['Protein'].unique())
        if tissue not in tissue_proteins_dict:
            tissue_proteins_dict[tissue] = set()
        tissue_proteins_dict[tissue].update(unique_proteins)
    
    # Restrict each tissue to 02 protein list when available
    for tissue in list(tissue_proteins_dict.keys()):
        proteins_02_t = get_02_protein_set_for_tissue(tissue)
        if proteins_02_t is not None:
            tissue_proteins_dict[tissue] = tissue_proteins_dict[tissue] & proteins_02_t
    
    print(f"Found {len(tissue_proteins_dict)} tissue/cell types:")
    for tissue, proteins in tissue_proteins_dict.items():
        print(f"  {tissue}: {len(proteins)} unique proteins")
    
    # Create CSV
    max_proteins = max(len(proteins) for proteins in tissue_proteins_dict.values()) if tissue_proteins_dict else 0
    proteins_per_tissue_data = {}
    for tissue in sorted(tissue_proteins_dict.keys()):
        proteins = sorted(list(tissue_proteins_dict[tissue]))
        proteins_padded = proteins + [''] * (max_proteins - len(proteins))
        proteins_per_tissue_data[tissue] = proteins_padded
    
    proteins_per_tissue_df = pd.DataFrame(proteins_per_tissue_data)
    proteins_per_tissue_file = tables_dir / "proteins_per_tissue_cell_type.csv"
    proteins_per_tissue_df.to_csv(proteins_per_tissue_file, index=False)
    print(f"\n  Saved: {proteins_per_tissue_file.name}")
    
    # Summary CSV
    summary_data = []
    for tissue in sorted(tissue_proteins_dict.keys()):
        summary_data.append({
            'Tissue_Cell_Type': tissue,
            'Number_of_Unique_Proteins': len(tissue_proteins_dict[tissue])
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = tables_dir / "proteins_per_tissue_cell_type_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"  Saved: {summary_file.name}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Plots saved in: {plots_dir}")
    print(f"Tables saved in: {tables_dir}")

if __name__ == "__main__":
    main()
