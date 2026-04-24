"""
B_data_preparation_and_filtering.py

Comprehensive data preparation script that:
1. Normalizes DDA/DIA protein naming systems
2. Filters contaminants, decoys, non-human (keeps entrapments for analysis)
3. Analyzes entrapment percentages per dataset and tissue/cell type
4. Combines datasets by tissue/cell type
5. Applies peptide filtering
6. Removes entrapments and creates final cache files

This script replaces B_data_analysis_cache_and_filtering.py and provides
detailed reporting of all filtering steps.

Abundance definition (current):
- Early quality filter per (Sample, Protein):
  remove groups with <=1 unique peptide sequence.
  Multiple feature rows of the same peptide do NOT pass this filter.
- For retained rows, protein abundance is counted as the number of unique
  (PeptideSequence, PrecursorCharge, FragmentIon, ProductCharge, IsotopeLabelType)
  tuples for each Protein across the dataset split.
- The value is stored in the legacy-named column `PeptideCount`
  for backward compatibility with downstream scripts.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
import csv
import json
import re
import shutil
import warnings
from collections import Counter, defaultdict
from datetime import datetime
import time
warnings.filterwarnings('ignore')

# Add parent directory to path to import shared_utils
work_dir = Path(r"G:\My Drive\Ikasketak\Postdoc\Cambridge\Cursor\blood_proteome_analysis")
sys.path.insert(0, str(work_dir))

# Import shared utilities
from shared_utils import (
    is_decoy_entrap_protein,
    is_non_human_protein,
    extract_uniprot_id,
    convert_entry_name_to_accession,
    PEPTIDE_COLUMN_CANDIDATES,
    compute_protein_feature_counts,
    get_default_cache_parquet_suffix,
    group_condition_to_tissue,
    load_entry_name_mapping_library,
    load_manual_mapping,
)

# Set working directory
os.chdir(work_dir)

# Directories
msstats_dir = work_dir / "msstats"
cache_dir = work_dir / "cache"
output_dir = work_dir / "data_preparation_output"
checkpoint_dir = output_dir / "checkpoints"  # For intermediate results
mapping_library_dir = work_dir / "entry_name_mapping"
sdrf_dir = work_dir / "sdrf_files"

# Entry name mapping library files
library_file = mapping_library_dir / "entry_name_to_accession.json"
manual_mapping_file = mapping_library_dir / "manual_id_mapping.xlsx"

# Tissue/cell types of interest for this blood-focused analysis.
# Anything outside this set will be skipped early when splitting datasets by condition.
TISSUES_OF_INTEREST = {
    'Blood Plasma/Serum',
    'Erythrocyte',
    'Platelet',
    'T Cell',
    'B Cell',
    'Dendritic Cell',
    'Monocyte',
    'Macrophage',
    'NK Cell',
    'Granulocyte',
    'Neutrophil',
    'Eosinophil',
    'Basophil',
}

# ============================================
# CONFIGURATION: CHOOSE WHICH FILTERED DATA TO USE
# ============================================
# This setting determines which version of filtered data will be saved as the default cache
# Other scripts (C_, D_, E_, F_, G_) will use this version unless they specify otherwise
# Options:
#   - 'unfiltered': No peptide filter (all proteins, even with 1 peptide)
#   - '2_peptides': Filter out proteins with < 2 peptides (default, recommended)
#   - '3_peptides': Filter out proteins with < 3 peptides (stricter)
#   - 'custom': Use CUSTOM_MIN_PEPTIDES value below
DEFAULT_PEPTIDE_FILTER = '2_peptides'  # Change this to choose default filter level
CUSTOM_MIN_PEPTIDES = 2  # Only used if DEFAULT_PEPTIDE_FILTER = 'custom'

# Whether to save multiple versions (unfiltered + filtered) for flexibility
SAVE_MULTIPLE_VERSIONS = True  # If True, saves both unfiltered and filtered versions
# If sample cardinality is too high, try alternative columns (Run/BioReplicate/Condition)
MAX_REASONABLE_SAMPLES = 5000

# Create output directories
cache_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# Configuration file path (for other scripts to read)
config_file = cache_dir / "cache_config.json"

# Check if Parquet is available
try:
    import pyarrow
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    print("ERROR: pyarrow not available. Install with 'pip install pyarrow' for Parquet caching.")
    sys.exit(1)

# ============================================
# HELPER FUNCTIONS
# ============================================

def _default_cache_parquet_path(dataset_name, cache_dir, min_peptides):
    """Default Step 5 cache path (same naming as shared_utils / E_ load)."""
    cache_dir = Path(cache_dir)
    if min_peptides == 2:
        return cache_dir / f"{dataset_name}_processed.parquet"
    if min_peptides == 0:
        return cache_dir / f"{dataset_name}_processed_unfiltered.parquet"
    return cache_dir / f"{dataset_name}_processed_min{min_peptides}pep.parquet"


def union_correct_proteins_from_parquet_cache(dataset_names, cache_dir, min_peptides):
    """
    Union of Protein column from on-disk caches (after peptide filter + ENTRAP row removal).
    Source of truth for counts that must match E_ when E_ reads *_processed.parquet.
    Returns (protein_set, missing_dataset_names_no_file, list_of_(dataset, error)).
    """
    union = set()
    missing = []
    errors = []
    cache_dir = Path(cache_dir)
    for ds in dataset_names:
        path = _default_cache_parquet_path(ds, cache_dir, min_peptides)
        if not path.is_file():
            missing.append(str(ds))
            continue
        try:
            part = pd.read_parquet(path, columns=["Protein"])
            union.update(part["Protein"].dropna().astype(str).str.strip().unique())
        except Exception as e:
            errors.append((str(ds), str(e)))
    return union, missing, errors


def is_contaminant_protein(protein_name):
    """Check if a protein is a contaminant."""
    if pd.isna(protein_name):
        return False
    protein_str = str(protein_name).strip().upper()
    
    # Check for CONTAM pattern
    if 'CONTAM' in protein_str:
        return True
    
    # Check for common contaminant patterns
    contaminant_patterns = [
        'KERATIN', 'TRYPSIN', 'ALBUMIN_BOVINE', 'LYSOZYME', 
        'CON__', 'CONTAMINANT'
    ]
    return any(pattern in protein_str for pattern in contaminant_patterns)

def classify_protein(protein_name):
    """Classify a protein into categories."""
    if pd.isna(protein_name):
        return 'other'
    
    protein_str = str(protein_name).strip()
    
    # Check for ENTRAP
    if 'ENTRAP' in protein_str.upper():
        return 'ENTRAP'
    
    # Check for DECOY (but not ENTRAP)
    if is_decoy_entrap_protein(protein_str):
        if 'DECOY' in protein_str.upper() or 'REV__' in protein_str.upper():
            return 'decoy'
        if 'ENTRAP' in protein_str.upper():
            return 'ENTRAP'
        return 'decoy'
    
    # Check for contaminants
    if is_contaminant_protein(protein_str):
        return 'contaminant'
    
    # Check for non-human
    if is_non_human_protein(protein_str):
        return 'non_human'
    
    # If none of the above, it's a correct protein
    return 'correct'

def normalize_protein_name(protein_name, entry_name_library):
    """Normalize a single protein name to common format (UniProt accession when possible)."""
    if pd.isna(protein_name):
        return None
    
    protein_str = str(protein_name).strip()
    
    # Check if it's DDA format (contains '|')
    if '|' in protein_str:
        # DDA format: sp|ACCESSION|ENTRY_NAME;sp|ACCESSION2|ENTRY_NAME2
        # Split on ';', take first entry
        first_entry = protein_str.split(';')[0].strip()
        # Extract UniProt accession (between first and second '|')
        uniprot_id = extract_uniprot_id(first_entry)
        if uniprot_id:
            return uniprot_id
        else:
            # Fallback: keep original if extraction fails
            return protein_name
    else:
        # DIA format or direct UniProt accession
        # First, try to extract as direct UniProt accession
        uniprot_id = extract_uniprot_id(protein_str)
        if uniprot_id:
            return uniprot_id
        else:
            # Try to convert entry name to accession (DIA format: ENTRY_NAME_HUMAN)
            # Split on ';', take first entry name
            first_entry = protein_str.split(';')[0].strip()
            
            # Try direct lookup first
            accession = convert_entry_name_to_accession(first_entry, entry_name_library)
            if accession:
                return accession
            
            # If not found and ends with _HUMAN, try without _HUMAN suffix
            if first_entry.endswith('_HUMAN'):
                entry_without_suffix = first_entry[:-6]  # Remove '_HUMAN'
                accession = convert_entry_name_to_accession(entry_without_suffix, entry_name_library)
                if accession:
                    return accession
                # Also try with _HUMAN suffix added back (in case mapping uses full name)
                entry_with_suffix = entry_without_suffix + '_HUMAN'
                accession = convert_entry_name_to_accession(entry_with_suffix, entry_name_library)
                if accession:
                    return accession
            
            # If we can't convert to accession, return the first entry (may still have _HUMAN)
            return first_entry

def get_acquisition_method(dataset_name):
    """Get acquisition method (DDA/DIA) for a dataset.
    
    Priority order:
    1. Check dataset name for explicit DIA/DDA indicators (highest priority)
    2. Check methodology CSV file
    3. Check SDRF file
    4. Return Unknown if not found
    """
    # FIRST: Check dataset name for explicit DIA/DDA indicators (highest priority)
    # This handles cases like "PXD038669-DIA-blood_serum" or "PXD037340-DIA"
    dataset_name_lower = dataset_name.lower()
    
    # Normalize separators to check patterns
    normalized_name = dataset_name_lower.replace('_', '-')
    
    # Check for DIA indicators (check before DDA since DIA is more specific)
    # Look for patterns like: -DIA, -DIA-, _DIA, _DIA_, or DIA as a separate part
    if 'dia' in normalized_name:
        # Split by common separators and check if 'dia' is a standalone part
        parts = re.split(r'[-_]', normalized_name)
        if 'dia' in parts:
            return 'DIA'
        # Also check for explicit patterns
        if re.search(r'[-_]dia[-_]|[-_]dia$|^dia[-_]', normalized_name):
            return 'DIA'
    
    # Check for DDA indicators
    if 'dda' in normalized_name:
        parts = re.split(r'[-_]', normalized_name)
        if 'dda' in parts:
            return 'DDA'
        # Also check for explicit patterns
        if re.search(r'[-_]dda[-_]|[-_]dda$|^dda[-_]', normalized_name):
            return 'DDA'
    
    # TMT / iTRAQ in name (typically DDA) – e.g. MSV000079033-Blood-Plasma-TMT10
    if 'tmt' in normalized_name or 'itraq' in normalized_name:
        return 'DDA'
    
    # SECOND: Try to load from methodology CSV
    methodology_csv = work_dir / "methodology_analysis" / "methodology_dataset_summary.csv"
    if methodology_csv.exists():
        try:
            methodology_df = pd.read_csv(methodology_csv)
            if 'Dataset' in methodology_df.columns and 'Acquisition' in methodology_df.columns:
                # Try exact match first
                dataset_match = methodology_df[methodology_df['Dataset'] == dataset_name]
                if len(dataset_match) > 0:
                    acquisition = dataset_match['Acquisition'].iloc[0]
                    if pd.notna(acquisition) and acquisition in ['DDA', 'DIA']:
                        return acquisition
                
                # Try partial match (in case methodology CSV has base name)
                # Extract base name for matching
                base_name = dataset_name
                for suffix in ['-plasma', '-serum', '-erythrocyte', '-blood_serum', '-blood_plasma']:
                    if base_name.endswith(suffix):
                        base_name = base_name[:-len(suffix)]
                        break
                
                # Try matching base name
                if base_name != dataset_name:
                    base_match = methodology_df[methodology_df['Dataset'] == base_name]
                    if len(base_match) > 0:
                        acquisition = base_match['Acquisition'].iloc[0]
                        if pd.notna(acquisition) and acquisition in ['DDA', 'DIA']:
                            return acquisition
        except Exception as e:
            pass
    
    # THIRD: Try to parse SDRF file
    # Extract base name for SDRF lookup (but preserve DIA/DDA if present)
    base_name = dataset_name
    # Remove tissue/cell type suffixes but keep DIA/DDA
    for suffix in ['-LFQ', '-plasma', '-serum', '-erythrocyte', '-blood_serum', '-blood_plasma']:
        if base_name.endswith(suffix):
            base_name = base_name[:-len(suffix)]
            break
    
    sdrf_file = sdrf_dir / f"{base_name}.sdrf.tsv"
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
            pass
    
    # If nothing found, return Unknown
    return 'Unknown'

def load_msstats_file(dataset_name):
    """Load an msstats CSV file.
    Prefer standard CSV parsing using the file header. If parsing fails due to malformed rows
    (commonly: unquoted commas inside the Condition field), fall back to a conservative repair
    that merges the extra columns back into Condition based on the header index.

    IMPORTANT: This function should not rewrite msstats files on disk. We only normalize values
    in-memory to keep downstream grouping consistent.
    """
    filename = f"{dataset_name}.sdrf_openms_design_msstats_in.csv"
    filepath = msstats_dir / filename
    
    if not filepath.exists():
        return None
    
    try:
        print(f"  Reading {filename}...", flush=True)
        # 0) Quick structural check: if any early rows have the wrong field count,
        # go straight to repair mode (pandas may succeed but still misalign columns
        # depending on engine/version).
        needs_repair = False
        header = None
        expected_cols = None
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, quotechar='"')
            header = [h.strip() for h in next(reader)]
            expected_cols = len(header)
            # Scan a small prefix for inconsistent row lengths
            for _ in range(500):
                try:
                    row = next(reader)
                except StopIteration:
                    break
                if len(row) != expected_cols:
                    needs_repair = True
                    break

        data = None
        if not needs_repair:
            # 1) First try: parse with the real header (do NOT assume canonical order / count).
            try:
                data = pd.read_csv(filepath, low_memory=False)
            except Exception:
                data = None

        # 2) Repair mode: conservative reconstruction when rows have too many fields
        # (commonly: unquoted commas inside the Condition field).
        if data is None:
            rows_out = []
            n_fixed = 0
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, quotechar='"')
                header = [h.strip() for h in next(reader)]
                expected_cols = len(header)

                # Find Condition index from header (case-insensitive)
                cond_idx = None
                header_l = [h.strip().lower() for h in header]
                for i, h in enumerate(header_l):
                    if h == 'condition':
                        cond_idx = i
                        break

                def _is_num(x) -> bool:
                    try:
                        float(str(x).strip())
                        return True
                    except Exception:
                        return False

                # Find indexes used in row realignment checks
                biorep_idx = None
                for i, h in enumerate(header_l):
                    if h == 'bioreplicate':
                        biorep_idx = i
                        break

                n_shifted_rows = 0
                for row in reader:
                    if len(row) == expected_cols:
                        rows_out.append(row)
                        continue
                    if len(row) < expected_cols:
                        # Skip malformed short rows
                        continue
                    if cond_idx is None:
                        # Without Condition index, we cannot safely repair; skip.
                        continue

                    # Strategy: merge as many fields as needed into Condition so that the
                    # remainder matches the header length exactly.
                    extra = len(row) - expected_cols
                    merged_condition = ",".join([r.strip() for r in row[cond_idx:cond_idx + extra + 1]])
                    recon = row[:cond_idx] + [merged_condition] + row[cond_idx + extra + 1:]
                    if len(recon) != expected_cols:
                        continue

                    # Additional safeguard requested by user:
                    # if BioReplicate is non-numeric after reconstruction, remove that value
                    # and shift all right-side fields one position left until BioReplicate
                    # becomes numeric or we run out of safe shifts.
                    if biorep_idx is not None and biorep_idx < len(recon):
                        shifted_this_row = False
                        max_shifts = max(0, len(recon) - biorep_idx - 1)
                        shift_count = 0
                        while shift_count < max_shifts and not _is_num(recon[biorep_idx]):
                            # Drop current BioReplicate cell and shift right part left
                            recon = recon[:biorep_idx] + recon[biorep_idx + 1:] + ['']
                            shift_count += 1
                            shifted_this_row = True
                        if shifted_this_row:
                            n_shifted_rows += 1

                    rows_out.append(recon)
                    n_fixed += 1

            if not rows_out:
                return None

            data = pd.DataFrame(rows_out, columns=header)
            if n_fixed > 0:
                print(f"  Fixed malformed rows in {filename}: {n_fixed} row(s) (re-merged Condition field)", flush=True)
            if n_shifted_rows > 0:
                print(f"  Shift-corrected BioReplicate alignment in {filename}: {n_shifted_rows} row(s)", flush=True)

        # ------------------------------------------------------------------
        # Sanity checks / repair for corrupted MSStats exports
        # Some files (observed in MSV*) can have columns shifted such that:
        #   - ProteinName is numeric-like
        #   - PeptideSequence contains 'sp|P...|...'
        # If we detect this, we warn loudly and attempt a minimal repair:
        # swap ProteinName and PeptideSequence so downstream normalization/ENTRAP works.
        # NOTE: this cannot recover true peptide sequences if they were overwritten upstream.
        # ------------------------------------------------------------------
        def _protein_like(series: pd.Series) -> float:
            s = series.dropna().astype(str)
            if len(s) == 0:
                return 0.0
            sample = s.head(2000)
            return float(sample.str.contains(r"\b(sp|tr)\|[A-Z0-9]{6,10}\|", regex=True).mean())

        def _numeric_like(series: pd.Series) -> float:
            s = series.dropna().astype(str)
            if len(s) == 0:
                return 0.0
            sample = s.head(2000)
            return float(sample.str.match(r"^[+-]?\d+(\.\d+)?(e[+-]?\d+)?$", case=False).mean())

        if 'ProteinName' in data.columns and 'PeptideSequence' in data.columns:
            prot_numeric = _numeric_like(data['ProteinName'])
            pep_protein_like = _protein_like(data['PeptideSequence'])
            prot_protein_like = _protein_like(data['ProteinName'])
            if prot_numeric > 0.5 and pep_protein_like > 0.2 and prot_protein_like < 0.2:
                print(f"  WARNING: {filename} looks corrupted (ProteinName numeric, PeptideSequence contains protein IDs).", flush=True)
                print("           Attempting repair: swapping ProteinName and PeptideSequence.", flush=True)
                data[['ProteinName', 'PeptideSequence']] = data[['PeptideSequence', 'ProteinName']]
                # Recompute after swap stats for logging
                prot_numeric2 = _numeric_like(data['ProteinName'])
                prot_protein_like2 = _protein_like(data['ProteinName'])
                print(f"           After swap: ProteinName protein-like={prot_protein_like2:.2f}, numeric-like={prot_numeric2:.2f}", flush=True)
                print("           If this file was overwritten previously, restore from *.backup if available.", flush=True)
        
        # Convert numeric columns.
        # IMPORTANT: keep `Run` as string. In many msstats exports it's a sample ID
        # like "1_1_3", "BBP_1", etc. Converting it to numeric would turn valid
        # IDs into NaN and break downstream sample selection.
        for col in ['PrecursorCharge', 'Intensity']:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Normalize Condition values for consistent splitting/grouping across runs (IN-MEMORY ONLY).
        # Do not modify msstats files on disk.
        if 'Condition' in data.columns:
            data['Condition'] = (
                data['Condition']
                .astype(str)
                .str.strip()
                .str.lower()
                .str.replace(r'\s+', ' ', regex=True)
            )

        return data
    except Exception as e:
        print(f"  Error reading {filename}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ============================================
# MAIN PROCESSING FUNCTIONS
# ============================================

def split_dataset_by_condition(data, dataset_name, condition_col):
    """Split a dataset into multiple sub-datasets if it contains multiple conditions."""
    if data is None or condition_col is None or condition_col not in data.columns:
        return {dataset_name: data}
    
    condition_series = data[condition_col].astype(str)
    unique_conditions = set()
    excluded_conditions = set()
    
    for val in condition_series:
        val_str = str(val).strip()
        if val_str and val_str.lower() not in ["", "nan", "none", "na"] and not val_str.strip().isdigit():
            if "|" in val_str:
                condition = val_str.split("|")[0].strip()
            else:
                condition = val_str.strip()
            if condition:
                tissue_group = group_condition_to_tissue(condition)
                if tissue_group in TISSUES_OF_INTEREST:
                    unique_conditions.add(condition)
                else:
                    excluded_conditions.add(condition)
    
    if len(unique_conditions) <= 1:
        if excluded_conditions:
            print(
                f"  Note: {dataset_name} has {len(excluded_conditions)} excluded condition(s) "
                f"(not in tissues of interest). Example(s): {', '.join(sorted(list(excluded_conditions))[:5])}",
                flush=True
            )
        return {dataset_name: data}
    
    print(f"  Splitting {dataset_name} into {len(unique_conditions)} sub-datasets by condition: {sorted(unique_conditions)}")
    if excluded_conditions:
        print(
            f"    Excluding {len(excluded_conditions)} non-target condition(s): {sorted(list(excluded_conditions))[:10]}"
            f"{' ...' if len(excluded_conditions) > 10 else ''}",
            flush=True
        )
    
    split_datasets = {}
    for condition in unique_conditions:
        condition_mask = (condition_series == condition) | \
                        condition_series.str.startswith(f"{condition}|", na=False)
        condition_data = data[condition_mask].copy()
        
        if len(condition_data) > 0:
            condition_data[condition_col] = condition
            safe_condition = re.sub(r'\s+', ' ', condition.strip().lower())
            safe_condition = safe_condition.replace(' ', '_')
            sub_dataset_name = f"{dataset_name}-{safe_condition}"
            split_datasets[sub_dataset_name] = condition_data
            print(f"    Created {sub_dataset_name}: {len(condition_data)} rows")
    
    return split_datasets

def process_dataset(dataset_name, entry_name_library):
    """Process a single dataset: normalize, filter, and classify proteins.
    Returns a list of results (one per condition if dataset is split)."""
    print(f"\nProcessing {dataset_name}...", flush=True)
    
    # Load data
    data = load_msstats_file(dataset_name)
    if data is None:
        print(f"  Error: Could not load data for {dataset_name}")
        return []
    if len(data) == 0:
        print(f"  Warning: {dataset_name} has no data rows")
        return []
    
    # Find columns
    protein_col = None
    peptide_col = None
    condition_col = None
    biorep_col = None
    sample_col = None
    disease_col = None
    
    for col in data.columns:
        col_lower = col.lower()
        if col_lower == 'proteinname' or (protein_col is None and 'protein' in col_lower):
            protein_col = col
        if col_lower == 'peptidesequence' or (peptide_col is None and 'peptide' in col_lower and 'sequence' in col_lower):
            peptide_col = col
        if col_lower == 'condition':
            condition_col = col
        if col_lower == 'bioreplicate' or col_lower == 'biorep':
            biorep_col = col
        if col_lower == 'reference' or col_lower == 'sample':
            sample_col = col
        if col_lower == 'disease':
            disease_col = col
    
    if sample_col is None:
        for col in data.columns:
            if col.lower() == 'run':
                sample_col = col
                break
    
    if protein_col is None:
        print(f"  Error: Could not find ProteinName column")
        return []
    
    # Note: Column misalignment detection and fixing is now handled in load_msstats_file()
    # No need to check again here - if the file was malformed, it should have been fixed during loading
    
    # Split dataset by condition if needed
    split_datasets = split_dataset_by_condition(data, dataset_name, condition_col)
    
    results = []
    for sub_dataset_name, sub_data in split_datasets.items():
        # Get condition for this sub-dataset
        if condition_col and condition_col in sub_data.columns:
            condition = str(sub_data[condition_col].iloc[0]) if len(sub_data) > 0 else "unknown"
            if "|" in condition:
                condition = condition.split("|")[0].strip()
        else:
            condition = "unknown"
    
        # Get unique proteins and their classifications
        unique_proteins = sub_data[protein_col].dropna().unique()
        
        # Classify proteins
        protein_classifications = {prot: classify_protein(prot) for prot in unique_proteins}
        
        # Count classifications
        classification_counts = Counter(protein_classifications.values())
        
        # Filter: remove decoys, contaminants, non-human (keep correct and ENTRAP)
        filtered_proteins = []
        for prot in unique_proteins:
            cls = protein_classifications[prot]
            if cls in ['correct', 'ENTRAP']:
                filtered_proteins.append(prot)
        
        # Normalize protein names
        n_proteins = len(filtered_proteins)
        print(f"  Normalizing protein names (DDA/DIA -> common format) for {sub_dataset_name} ({n_proteins} proteins)...", flush=True)
        protein_normalization_map = {}  # original -> normalized
        normalized_to_entrap = {}  # normalized -> is_entrap (bool)
        unnormalized_human_proteins = set()  # Track proteins that still have _HUMAN
        
        for i, orig_prot in enumerate(filtered_proteins):
            if n_proteins > 2000 and (i + 1) % 2000 == 0:
                print(f"    Normalized {i + 1}/{n_proteins} proteins...", flush=True)
            normalized = normalize_protein_name(orig_prot, entry_name_library)
            if normalized:
                protein_normalization_map[orig_prot] = normalized
                # Track if this normalized protein came from an ENTRAP original
                is_entrap = protein_classifications[orig_prot] == 'ENTRAP'
                if normalized not in normalized_to_entrap:
                    normalized_to_entrap[normalized] = is_entrap
                else:
                    # If any original was ENTRAP, mark as ENTRAP
                    normalized_to_entrap[normalized] = normalized_to_entrap[normalized] or is_entrap
                
                # Track proteins that still have _HUMAN suffix (not normalized)
                if '_HUMAN' in normalized:
                    unnormalized_human_proteins.add(normalized)
        
        # Report unnormalized _HUMAN proteins
        if unnormalized_human_proteins:
            print(f"    Warning: {len(unnormalized_human_proteins)} proteins still have _HUMAN suffix (not in mapping)")
            if len(unnormalized_human_proteins) <= 10:
                print(f"      Examples: {', '.join(sorted(list(unnormalized_human_proteins))[:10])}")
            else:
                print(f"      Examples: {', '.join(sorted(list(unnormalized_human_proteins))[:10])} ... and {len(unnormalized_human_proteins) - 10} more")
            print(f"      Consider adding these to manual_id_mapping.xlsx")
        
        # Count unique normalized proteins
        unique_normalized = len(set(protein_normalization_map.values()))
        proteins_before_unique_peptide_filter = set(protein_normalization_map.values())
        
        # Create processed dataframe
        processed_data = sub_data[sub_data[protein_col].isin(filtered_proteins)].copy()
        processed_data['NormalizedProtein'] = processed_data[protein_col].map(protein_normalization_map)
        processed_data = processed_data[processed_data['NormalizedProtein'].notna()].copy()
        
        # Rename protein column
        processed_data = processed_data.rename(columns={protein_col: 'OriginalProtein', 'NormalizedProtein': 'Protein'})
        
        # Ensure required columns exist
        if condition_col and condition_col in processed_data.columns:
            processed_data['Condition'] = processed_data[condition_col]
        else:
            processed_data['Condition'] = condition
        
        if biorep_col and biorep_col in processed_data.columns:
            processed_data['BioReplicate'] = processed_data[biorep_col]
        else:
            processed_data['BioReplicate'] = processed_data.index
        
        if sample_col and sample_col in processed_data.columns:
            processed_data['Sample'] = processed_data[sample_col]
        else:
            processed_data['Sample'] = processed_data.index

        # Print initial sample cardinality for monitoring in Step 1
        n_rows = len(processed_data)
        n_samples_current = processed_data['Sample'].nunique()
        print(f"    Initial Sample unique count: {n_samples_current} (rows={n_rows})", flush=True)

        # If sample count is unrealistically high, switch to a coarser identifier.
        # Requested behavior: if too many samples, use msstats `Run` (when available).
        sample_switched_due_to_cardinality = False
        if n_samples_current > MAX_REASONABLE_SAMPLES:
            if 'Run' in processed_data.columns:
                processed_data['Sample'] = processed_data['Run']
                n_samples_current = processed_data['Sample'].nunique()
                sample_switched_due_to_cardinality = True
                print(
                    f"    Sample count > {MAX_REASONABLE_SAMPLES}; forcing Sample=Run "
                    f"({n_samples_current} unique)",
                    flush=True
                )
            else:
                print(
                    f"    Sample count > {MAX_REASONABLE_SAMPLES} but no 'Run' column found; keeping current Sample.",
                    flush=True
                )

        # For TMT/iTRAQ, Reference (or Run) is often unique per row -> 1 protein per "sample".
        # Use BioReplicate or Condition as sample when they give a sensible grouping.
        n_rows = len(processed_data)
        n_samples_current = processed_data['Sample'].nunique()
        mean_proteins_per_sample = n_rows / n_samples_current if n_samples_current > 0 else 0
        use_fallback = (
            n_samples_current >= n_rows * 0.5 or
            (mean_proteins_per_sample < 2 and n_samples_current > 1)
        )
        if use_fallback and not sample_switched_due_to_cardinality:
            for fallback_col, fallback_name in [(biorep_col, 'BioReplicate'), (condition_col, 'Condition')]:
                if fallback_col and fallback_col in processed_data.columns:
                    n_fallback = processed_data[fallback_col].nunique()
                    if n_fallback < n_samples_current and n_fallback >= 1:
                        processed_data['Sample'] = processed_data[fallback_col]
                        print(f"    Using {fallback_name} as Sample (Reference/Run had {n_samples_current} values, {fallback_name} has {n_fallback})", flush=True)
                        break
        
        # Ensure no row has invalid Sample (NaN or empty) so downstream (e.g. E_) doesn't drop all rows
        sample_invalid = processed_data['Sample'].isna() | (processed_data['Sample'].astype(str).str.strip() == '')
        if sample_invalid.any():
            for fallback_col, fallback_name in [(biorep_col, 'BioReplicate'), (condition_col, 'Condition')]:
                if fallback_col and fallback_col in processed_data.columns:
                    processed_data.loc[sample_invalid, 'Sample'] = processed_data.loc[sample_invalid, fallback_col]
                    still_invalid = processed_data['Sample'].isna() | (processed_data['Sample'].astype(str).str.strip() == '')
                    if not still_invalid.any():
                        print(f"    Filled invalid Sample with {fallback_name} ({sample_invalid.sum()} rows)", flush=True)
                        break
                    sample_invalid = still_invalid
        
        # Early quality filter:
        # Remove (Sample, Protein) groups supported by <=1 unique peptide sequence.
        # NOTE: if one peptide has many feature rows, it is still removed.
        # This is intentional and keeps weak single-peptide identifications out
        # before abundance is computed and cached.
        peptide_col_in_processed = None
        if peptide_col and peptide_col in processed_data.columns:
            peptide_col_in_processed = peptide_col
        else:
            lower_map = {str(c).lower(): c for c in processed_data.columns}
            for cand in PEPTIDE_COLUMN_CANDIDATES:
                if cand.lower() in lower_map:
                    peptide_col_in_processed = lower_map[cand.lower()]
                    break
            if peptide_col_in_processed is None:
                for c in processed_data.columns:
                    cl = str(c).lower()
                    if "peptide" in cl and "sequence" in cl:
                        peptide_col_in_processed = c
                        break

        if peptide_col_in_processed:
            before_rows = len(processed_data)
            pep_tmp = processed_data[["Sample", "Protein", peptide_col_in_processed]].copy()
            pep_tmp[peptide_col_in_processed] = pep_tmp[peptide_col_in_processed].astype(str).str.strip()
            pep_tmp = pep_tmp[pep_tmp[peptide_col_in_processed].ne("")]
            pep_tmp = pep_tmp[pep_tmp[peptide_col_in_processed].str.lower().ne("nan")]
            pep_counts = (
                pep_tmp.groupby(["Sample", "Protein"])[peptide_col_in_processed]
                .nunique()
                .reset_index(name="n_unique_peptides")
            )
            keep_pairs = pep_counts[pep_counts["n_unique_peptides"] > 1][["Sample", "Protein"]]
            processed_data = processed_data.merge(keep_pairs, on=["Sample", "Protein"], how="inner")
            removed_rows = before_rows - len(processed_data)
            if removed_rows > 0:
                print(
                    f"    Removed {removed_rows} rows where protein had <=1 unique peptide per sample",
                    flush=True,
                )
        else:
            print(
                "    Warning: peptide sequence column not found; skipping per-sample unique-peptide filter.",
                flush=True,
            )

        if disease_col and disease_col in processed_data.columns:
            processed_data['Disease'] = processed_data[disease_col]
        
        # Abundance metric:
        # Count unique (PeptideSequence + feature tuple) per Protein across this dataset split.
        # Feature tuple = (PrecursorCharge, FragmentIon, ProductCharge, IsotopeLabelType).
        # Stored in `PeptideCount` column name for downstream compatibility.
        protein_feature_counts = compute_protein_feature_counts(
            processed_data,
            protein_col='Protein',
            peptide_col=peptide_col_in_processed,
        )
        if protein_feature_counts:
            processed_data['PeptideCount'] = processed_data['Protein'].map(protein_feature_counts).fillna(1).astype(int)
        else:
            processed_data['PeptideCount'] = 1
        
        # Select and reorder columns
        final_columns = ['Protein', 'Condition', 'BioReplicate', 'Sample', 'PeptideCount']
        if 'Disease' in processed_data.columns:
            final_columns.append('Disease')
        processed_data = processed_data[final_columns].copy()
        
        # Remove duplicates (keep unique protein-sample combinations)
        processed_data = processed_data.drop_duplicates(subset=['Protein', 'Sample'])
        processed_data['Dataset'] = sub_dataset_name
        
        # Calculate entrapment percentage
        entrap_count = classification_counts.get('ENTRAP', 0)
        correct_count = classification_counts.get('correct', 0)
        total_before_filter = len(unique_proteins)
        total_after_filter = len(filtered_proteins)
        entrap_pct = (entrap_count / total_after_filter * 100) if total_after_filter > 0 else 0.0
        
        print(f"  Total proteins: {total_before_filter}")
        print(f"  After filtering (removed decoys/contaminants/non-human): {total_after_filter}")
        print(f"  Correct: {correct_count}, ENTRAP: {entrap_count} ({entrap_pct:.2f}%)")
        print(f"  Unique normalized proteins: {unique_normalized}")

        # Sanity warning: flag likely msstats corruption (column shift) without false positives.
        # We warn when BOTH ENTRAP and contaminants are absent AND ProteinName looks wrong (numeric-like),
        # which is the typical signature of misaligned MSStats exports (observed in MSV files).
        contaminant_count = classification_counts.get('contaminant', 0)
        if entrap_count == 0 and contaminant_count == 0 and total_after_filter > 0:
            sample_vals = sub_data[protein_col].dropna().astype(str).head(5).tolist()
            looks_numeric = sum(1 for v in sample_vals if v.replace('.', '', 1).isdigit()) >= 3
            if looks_numeric:
                print(
                    f"  WARNING: {sub_dataset_name} has ENTRAP=0 and Contaminant=0, and ProteinName looks numeric -> "
                    f"possible column shift/corruption. Sample ProteinName values: {sample_vals}",
                    flush=True
                )
        
        result = {
            'dataset': sub_dataset_name,
            'condition': condition,
            'total_proteins_before': total_before_filter,
            'decoy': classification_counts.get('decoy', 0),
            'contaminant': classification_counts.get('contaminant', 0),
            'non_human': classification_counts.get('non_human', 0),
            'other': classification_counts.get('other', 0),
            'correct': correct_count,
            'ENTRAP': entrap_count,
            'total_proteins_after_filter': total_after_filter,
            'entrapment_pct': entrap_pct,
            'unique_normalized_proteins': unique_normalized,
            # Protein universe before the early per-sample unique-peptide support filter.
            # Used for true "before filter" tissue summaries in 02.
            'proteins_before_unique_peptide_filter': sorted(proteins_before_unique_peptide_filter),
            'normalized_to_entrap': normalized_to_entrap,  # Track which normalized proteins are ENTRAP
            'processed_data': processed_data
        }
        
        results.append(result)
    
    return results


def _build_combined_tissue_entry(tissue, results):
    """Build one combined_by_tissue entry from a list of result dicts (same logic as combine_datasets_by_tissue for one tissue)."""
    if not results:
        return None
    all_proteins = []
    all_classifications = []
    all_peptide_counts = {}
    all_processed_data = []
    for result in results:
        processed_data = result['processed_data']
        unique_proteins = processed_data['Protein'].unique()
        normalized_to_entrap = result.get('normalized_to_entrap', {})
        for prot in unique_proteins:
            prot_data = processed_data[processed_data['Protein'] == prot]
            max_peptides = prot_data['PeptideCount'].max() if len(prot_data) > 0 else 1
            is_entrap = normalized_to_entrap.get(prot, False)
            all_proteins.append(prot)
            all_classifications.append('ENTRAP' if is_entrap else 'correct')
            if prot not in all_peptide_counts:
                all_peptide_counts[prot] = max_peptides
            else:
                all_peptide_counts[prot] = max(all_peptide_counts[prot], max_peptides)
        all_processed_data.append(processed_data)
    protein_to_class = {}
    priority = {'correct': 0, 'ENTRAP': 1}
    for prot, cls in zip(all_proteins, all_classifications):
        if prot not in protein_to_class:
            protein_to_class[prot] = cls
        else:
            if priority.get(cls, 99) < priority.get(protein_to_class[prot], 99):
                protein_to_class[prot] = cls
    unique_proteins = list(protein_to_class.keys())
    final_classifications = [protein_to_class[prot] for prot in unique_proteins]
    entrap_count = sum(1 for c in final_classifications if c == 'ENTRAP')
    correct_count = sum(1 for c in final_classifications if c == 'correct')
    total_unique = len(unique_proteins)
    entrap_pct = (entrap_count / total_unique * 100) if total_unique > 0 else 0.0
    combined_data = pd.concat(all_processed_data, ignore_index=True)
    combined_data = combined_data.drop_duplicates(subset=['Protein', 'Sample'])
    for prot in unique_proteins:
        mask = combined_data['Protein'] == prot
        if mask.any():
            combined_data.loc[mask, 'PeptideCount'] = all_peptide_counts.get(prot, 1)
    unique_datasets = sorted(list(set(r['dataset'] for r in results)))
    return {
        'tissue': tissue,
        'datasets': unique_datasets,
        'total_unique_proteins': total_unique,
        'correct_proteins': correct_count,
        'ENTRAP_proteins': entrap_count,
        'entrapment_pct': entrap_pct,
        'protein_peptide_counts': all_peptide_counts,
        'combined_data': combined_data
    }


def _compute_tissue_stats_from_results(results):
    """Compute total_unique_proteins, correct_proteins, ENTRAP_proteins, entrapment_pct from a list of result dicts (same logic as combine_datasets_by_tissue)."""
    if not results:
        return 0, 0, 0, 0.0
    all_proteins = []
    all_classifications = []
    for result in results:
        processed_data = result['processed_data']
        unique_proteins = processed_data['Protein'].unique()
        normalized_to_entrap = result.get('normalized_to_entrap', {})
        for prot in unique_proteins:
            is_entrap = normalized_to_entrap.get(prot, False)
            all_proteins.append(prot)
            all_classifications.append('ENTRAP' if is_entrap else 'correct')
    protein_to_class = {}
    priority = {'correct': 0, 'ENTRAP': 1}
    for prot, cls in zip(all_proteins, all_classifications):
        if prot not in protein_to_class:
            protein_to_class[prot] = cls
        else:
            if priority.get(cls, 99) < priority.get(protein_to_class[prot], 99):
                protein_to_class[prot] = cls
    unique_proteins = list(protein_to_class.keys())
    final_classifications = [protein_to_class[prot] for prot in unique_proteins]
    entrap_count = sum(1 for c in final_classifications if c == 'ENTRAP')
    correct_count = sum(1 for c in final_classifications if c == 'correct')
    total_unique = len(unique_proteins)
    entrap_pct = (entrap_count / total_unique * 100) if total_unique > 0 else 0.0
    return total_unique, correct_count, entrap_count, entrap_pct


def combine_datasets_by_tissue(all_results, entry_name_library):
    """Combine datasets by tissue/cell type and remove duplicates."""
    print("\n" + "=" * 80)
    print("COMBINING DATASETS BY TISSUE/CELL TYPE")
    print("=" * 80)
    
    # Group datasets by tissue/cell type
    tissue_groups = defaultdict(list)
    for result in all_results:
        if result is None:
            continue
        condition = result['condition']
        tissue = group_condition_to_tissue(condition)
        tissue_groups[tissue].append(result)
    
    print(f"\nGrouped into {len(tissue_groups)} tissue/cell types:")
    for tissue, results in tissue_groups.items():
        print(f"  {tissue}: {len(results)} dataset(s)")
    
    # Combine datasets for each tissue
    combined_by_tissue = {}
    
    for tissue, results in tissue_groups.items():
        print(f"\nProcessing {tissue}...")
        
        # Combine all proteins from datasets in this tissue
        all_proteins = []
        all_classifications = []
        all_peptide_counts = {}
        all_processed_data = []
        
        for result in results:
            dataset_name = result['dataset']
            processed_data = result['processed_data']
            
            # Get unique proteins and their classifications
            unique_proteins = processed_data['Protein'].unique()
            
            # Get ENTRAP mapping for this dataset
            normalized_to_entrap = result.get('normalized_to_entrap', {})
            
            # Classify proteins (should be correct or ENTRAP)
            for prot in unique_proteins:
                prot_data = processed_data[processed_data['Protein'] == prot]
                # Get max peptide count for this protein in this dataset
                max_peptides = prot_data['PeptideCount'].max() if len(prot_data) > 0 else 1
                
                # Determine classification using normalized_to_entrap mapping
                is_entrap = normalized_to_entrap.get(prot, False)
                classification = 'ENTRAP' if is_entrap else 'correct'
                
                all_proteins.append(prot)
                all_classifications.append(classification)
                
                # Track max peptide count
                if prot not in all_peptide_counts:
                    all_peptide_counts[prot] = max_peptides
                else:
                    all_peptide_counts[prot] = max(all_peptide_counts[prot], max_peptides)
            
            all_processed_data.append(processed_data)
        
        # Remove duplicates - prioritize correct over ENTRAP
        protein_to_class = {}
        priority = {'correct': 0, 'ENTRAP': 1}
        
        for prot, cls in zip(all_proteins, all_classifications):
            if prot not in protein_to_class:
                protein_to_class[prot] = cls
            else:
                current_priority = priority.get(protein_to_class[prot], 99)
                new_priority = priority.get(cls, 99)
                if new_priority < current_priority:
                    protein_to_class[prot] = cls
        
        # Get unique proteins
        unique_proteins = list(protein_to_class.keys())
        final_classifications = [protein_to_class[prot] for prot in unique_proteins]
        
        # Count entrapments
        entrap_count = sum(1 for cls in final_classifications if cls == 'ENTRAP')
        correct_count = sum(1 for cls in final_classifications if cls == 'correct')
        total_unique = len(unique_proteins)
        entrap_pct = (entrap_count / total_unique * 100) if total_unique > 0 else 0.0
        
        print(f"  Total unique proteins (after duplicate removal): {total_unique}")
        print(f"  Correct: {correct_count}, ENTRAP: {entrap_count} ({entrap_pct:.2f}%)")
        
        # Combine processed data
        combined_data = pd.concat(all_processed_data, ignore_index=True)
        
        # Remove duplicates based on Protein+Sample combination
        combined_data = combined_data.drop_duplicates(subset=['Protein', 'Sample'])
        
        # Update peptide counts to max across all datasets
        for prot in unique_proteins:
            mask = combined_data['Protein'] == prot
            if mask.any():
                max_peptides = all_peptide_counts.get(prot, 1)
                combined_data.loc[mask, 'PeptideCount'] = max_peptides
        
        # Get unique datasets (remove duplicates)
        unique_datasets = sorted(list(set([r['dataset'] for r in results])))
        
        combined_by_tissue[tissue] = {
            'tissue': tissue,
            'datasets': unique_datasets,  # Use unique datasets
            'total_unique_proteins': total_unique,
            'correct_proteins': correct_count,
            'ENTRAP_proteins': entrap_count,
            'entrapment_pct': entrap_pct,
            'protein_peptide_counts': all_peptide_counts,
            'combined_data': combined_data
        }
    
    return combined_by_tissue

def apply_peptide_filter(combined_data, min_peptides=2):
    """Apply peptide filter to combined data."""
    if min_peptides <= 0:
        return combined_data.copy()
    
    filtered = combined_data[combined_data['PeptideCount'] >= min_peptides].copy()
    return filtered

def get_min_peptides_from_config():
    """Get minimum peptides value from configuration."""
    if DEFAULT_PEPTIDE_FILTER == 'unfiltered':
        return 0
    elif DEFAULT_PEPTIDE_FILTER == '2_peptides':
        return 2
    elif DEFAULT_PEPTIDE_FILTER == '3_peptides':
        return 3
    elif DEFAULT_PEPTIDE_FILTER == 'custom':
        return CUSTOM_MIN_PEPTIDES
    else:
        # Default to 2 if invalid config
        return 2

def get_cache_file_suffix(min_peptides):
    """Get suffix for cache file based on peptide filter level."""
    if min_peptides == 0:
        return "_processed_unfiltered.parquet"
    elif min_peptides == 2:
        return "_processed.parquet"  # Default, no special suffix
    else:
        return f"_processed_min{min_peptides}pep.parquet"


def cache_file_exists_for_dataset(dataset_name):
    """Return True if the default cache parquet file for this dataset exists.
    Used so that if the user deletes cache files, we re-process those datasets
    instead of skipping them based on checkpoint."""
    suffix = get_cache_file_suffix(get_min_peptides_from_config())
    return (cache_dir / f"{dataset_name}{suffix}").exists()


def result_belongs_to_current_datasets(result_name, current_dataset_set):
    """True if this result (e.g. PXD004352-plasma) belongs to the current dataset list.
    current_dataset_set has base names from msstats filenames (e.g. PXD004352); results
    can be base or sub-names (e.g. PXD004352-plasma). Names are compared stripped to avoid whitespace mismatch."""
    r = (result_name or "").strip()
    if r in current_dataset_set:
        return True
    for base in current_dataset_set:
        if r == base or r.startswith(base + '-'):
            return True
    return False


def base_dataset_fully_cached(base_name, all_results):
    """Return True if this base dataset (from all_datasets) is fully cached.
    For split datasets, all_results has sub-names (e.g. PXD062484-plasma); the base
    (PXD062484) is in all_datasets but never in result['dataset']. So we consider
    the base fully cached iff every result that belongs to it has its cache file.
    Names compared stripped for consistency with checkpoint matching."""
    base = (base_name or "").strip()
    sub_results = [r for r in all_results if (r.get('dataset') or "").strip() == base or (r.get('dataset') or "").strip().startswith(base + '-')]
    if len(sub_results) == 0:
        return False
    return all(cache_file_exists_for_dataset(r['dataset']) for r in sub_results)


def base_dataset_in_checkpoint(base_name, all_results):
    """Return True if this base dataset is in the checkpoint AND at least one
    default cache parquet exists on disk for its results.

    This lets us resume without re-running Step 1 for datasets that are already cached,
    while still allowing Step 1 to re-run when you delete cache outputs (e.g. to pick up
    changes in Sample assignment logic)."""
    base = (base_name or "").strip()
    sub_results = [
        r for r in all_results
        if ((r.get('dataset') or "").strip() == base or (r.get('dataset') or "").strip().startswith(base + '-'))
    ]
    if len(sub_results) == 0:
        return False
    return any(cache_file_exists_for_dataset(r['dataset']) for r in sub_results)

def save_cache_config():
    """Save configuration to JSON file for other scripts to read."""
    config = {
        'default_peptide_filter': DEFAULT_PEPTIDE_FILTER,
        'min_peptides': get_min_peptides_from_config(),
        'save_multiple_versions': SAVE_MULTIPLE_VERSIONS,
        'description': {
            'unfiltered': 'No peptide filter - all proteins included',
            '2_peptides': 'Filter out proteins with < 2 peptides (recommended)',
            '3_peptides': 'Filter out proteins with < 3 peptides (stricter)',
            'custom': f'Custom filter: min {CUSTOM_MIN_PEPTIDES} peptides'
        }
    }
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        print(f"\nSaved cache configuration: {config_file}")
        print(f"  Default filter: {DEFAULT_PEPTIDE_FILTER} (min {config['min_peptides']} peptides)")
        print(f"  Save multiple versions: {SAVE_MULTIPLE_VERSIONS}")
    except Exception as e:
        print(f"  Warning: Could not save cache configuration: {e}")

def save_checkpoint(checkpoint_name, data):
    """Save checkpoint data to disk."""
    checkpoint_file = checkpoint_dir / f"{checkpoint_name}.pkl"
    try:
        import pickle
        
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"  Checkpoint saved: {checkpoint_name}")
    except Exception as e:
        print(f"  Warning: Could not save checkpoint {checkpoint_name}: {e}")


def load_checkpoint(checkpoint_name):
    """Load checkpoint data from disk. Can be slow on network/Google Drive for large step1 checkpoint."""
    checkpoint_file = checkpoint_dir / f"{checkpoint_name}.pkl"
    if checkpoint_file.exists():
        try:
            import pickle
            with open(checkpoint_file, 'rb') as f:
                data = pickle.load(f)
            print(f"  Checkpoint loaded: {checkpoint_name}", flush=True)
            return data
        except Exception as e:
            print(f"  Warning: Could not load checkpoint {checkpoint_name}: {e}")
    return None


def check_and_write_cache_normalization_status():
    """Check if all cache parquet files have normalized Protein column (UniProt accessions).
    Writes cache_dir/normalization_status.json so E_ can skip re-normalization when status is 'ok'.
    """
    status_file = cache_dir / "normalization_status.json"
    cache_files = list(cache_dir.glob("*_processed*.parquet"))
    if not cache_files:
        out = {"status": "no_cache", "timestamp": datetime.now().isoformat(), "checked_files": 0,
               "reason": "No cache parquet files found."}
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Cache normalization status: no cache files (wrote {status_file.name})")
        return
    max_sample = 15000  # per file, to avoid loading huge columns
    needs_normalization = False
    reason = None
    for cf in cache_files:
        try:
            df = pd.read_parquet(cf, columns=["Protein"])
            proteins = df["Protein"].dropna().astype(str).str.strip().unique()
            if len(proteins) > max_sample:
                proteins = np.random.choice(proteins, size=max_sample, replace=False)
            for p in proteins:
                if "|" in p or (p.endswith("_HUMAN") and len(p) > 6):
                    needs_normalization = True
                    reason = f"Un-normalized IDs found (e.g. '|' or '_HUMAN') in cache (e.g. {cf.name})."
                    break
            if needs_normalization:
                break
        except Exception as e:
            needs_normalization = True
            reason = f"Could not check {cf.name}: {e}"
            break
    out = {
        "status": "ok" if not needs_normalization else "needs_normalization",
        "timestamp": datetime.now().isoformat(),
        "checked_files": len(cache_files),
    }
    if reason:
        out["reason"] = reason
    with open(status_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Cache normalization status: {out['status']} (checked {len(cache_files)} files, wrote {status_file.name})")


def main():
    """Main processing function."""
    print("=" * 80)
    print("DATA PREPARATION AND FILTERING")
    print("=" * 80)
    print("This script prepares data for downstream analysis by:")
    print("  1. Normalizing DDA/DIA protein naming systems")
    print("  2. Filtering contaminants, decoys, non-human (keeps entrapments)")
    print("  3. Analyzing entrapment percentages")
    print("  4. Combining datasets by tissue/cell type")
    print("  5. Applying peptide filters")
    print("  6. Removing entrapments and creating final cache files")
    print("=" * 80)
    print("\nNote: Checkpoints are saved after each major step.")
    print("      If the script stops, it will resume from the last checkpoint.")
    print("      If you update manual_id_mapping.xlsx, proteins with _HUMAN will be re-normalized automatically.\n")
    
    # Load entry name mapping library
    print("\nLoading entry name mapping library...")
    entry_name_library = load_entry_name_mapping_library(library_file)
    manual_mapping = load_manual_mapping(manual_mapping_file)
    print(f"  Loaded {len(entry_name_library)} from JSON library")
    print(f"  Loaded {len(manual_mapping)} from manual mapping file")
    # Manual mapping takes precedence (overwrites JSON library entries)
    entry_name_library = {**entry_name_library, **manual_mapping}
    print(f"  Total entry name mappings: {len(entry_name_library)}")
    
    # Auto-detect datasets (strip whitespace so names match checkpoint exactly)
    print("\nAuto-detecting datasets...")
    msstats_files = list(msstats_dir.glob("*.sdrf_openms_design_msstats_in.csv"))
    all_datasets = sorted([f.stem.replace('.sdrf_openms_design_msstats_in', '').strip() for f in msstats_files])
    
    if len(all_datasets) == 0:
        print("Error: No msstats files found")
        return
    
    print(f"Found {len(all_datasets)} datasets")
    
    # Process each dataset
    print("\n" + "=" * 80)
    print("STEP 1: PROCESSING INDIVIDUAL DATASETS")
    print("=" * 80, flush=True)
    
    # Try to load checkpoint (can be slow on network/Google Drive if file is large)
    print("  Loading Step 1 checkpoint (may take a minute on network drives)...", flush=True)
    t_ckpt0 = time.perf_counter()
    checkpoint_step1 = load_checkpoint("step1_processed_datasets")
    t_ckpt1 = time.perf_counter()
    print(f"  Step 1 checkpoint load time: {t_ckpt1 - t_ckpt0:.1f}s", flush=True)
    
    if checkpoint_step1 is not None:
        print("  Resuming from checkpoint...")
        all_results_raw = checkpoint_step1.get('all_results', [])
        dataset_summary_raw = checkpoint_step1.get('dataset_summary', [])
        # Keep only entries for datasets that exist in current msstats (avoid carrying over stale data from old runs)
        # Use result_belongs_to_current_datasets so sub-dataset names (e.g. PXD004352-plasma) are kept when base (PXD004352) is in all_datasets
        current_dataset_set = set(all_datasets)
        all_results = [r for r in all_results_raw if result_belongs_to_current_datasets(r['dataset'], current_dataset_set)]
        dataset_summary = [d for d in dataset_summary_raw if result_belongs_to_current_datasets(d['Dataset'], current_dataset_set)]
        n_dropped = len(all_results_raw) - len(all_results)
        if n_dropped > 0:
            print(f"  Note: Checkpoint had {len(all_results_raw)} results; {n_dropped} were dropped (names don't match current msstats list). Kept {len(all_results)}.")
        else:
            print(f"  Checkpoint has {len(all_results)} results (all match current dataset list).")
        # Don't drop results that lack a parquet: Step 5 writes parquets, so after a Step 5 crash many are in checkpoint but have no file yet. Keep them so Step 5 can run.
        # remaining = base names that still need Step 1 (not in checkpoint and not fully cached on disk)
        remaining_datasets = [ds for ds in all_datasets if not (
            base_dataset_fully_cached(ds, all_results) or base_dataset_in_checkpoint(ds, all_results)
        )]
        n_fully_cached = len(all_datasets) - len(remaining_datasets)
        n_with_parquet = sum(1 for ds in all_datasets if base_dataset_fully_cached(ds, all_results))
        print(f"  Found {n_fully_cached} base datasets already processed (of {len(all_datasets)} current); {n_with_parquet} have cache parquets.")
        
        if remaining_datasets:
            print(f"  Processing {len(remaining_datasets)} remaining datasets...")
            if len(remaining_datasets) <= 5:
                print(f"  Remaining: {remaining_datasets}")
            else:
                print(f"  Remaining (first 5): {remaining_datasets[:5]} ...")
            # If user just ran Step 1 and names should match, checkpoint may be from another path/sync or truncated
            if len(all_results) < len(all_datasets):
                print(f"  Note: Checkpoint has {len(all_results)} results for {len(all_datasets)} current datasets. If you just ran Step 1, check that the same checkpoint file is being loaded (e.g. sync on network/Google Drive, or same working directory).")
        else:
            print("  All datasets already processed!")
    else:
        all_results = []
        dataset_summary = []
        remaining_datasets = all_datasets
        print(f"  No checkpoint: processing all {len(remaining_datasets)} datasets.", flush=True)
    
    total_remaining = len(remaining_datasets)
    for i, dataset_name in enumerate(remaining_datasets, start=1):
        print(f"  STEP 1 progress: {i}/{total_remaining} -> {dataset_name}", flush=True)
        results = process_dataset(dataset_name, entry_name_library)
        if results:
            all_results.extend(results)
            
            # Get acquisition method (check the actual dataset name, don't strip DIA/DDA)
            # The function will check the name first, then fall back to other methods
            # Use the first result's dataset name (or original dataset_name if results is empty)
            if results and len(results) > 0:
                acquisition = get_acquisition_method(results[0]['dataset'])
            else:
                acquisition = get_acquisition_method(dataset_name)
            
            # Add summary for each result (sub-dataset)
            for result in results:
                dataset_summary.append({
                    'Dataset': result['dataset'],
                    'Acquisition': acquisition,
                    'Condition': result['condition'],
                    'Total_Proteins_Before_Filter': result['total_proteins_before'],
                    'Decoy_Removed': result['decoy'],
                    'Contaminant_Removed': result['contaminant'],
                    'Non_Human_Removed': result['non_human'],
                    'Other_Removed': result['other'],
                    'Correct_Proteins': result['correct'],
                    'ENTRAP_Proteins': result['ENTRAP'],
                    'Total_Proteins_After_Filter': result['total_proteins_after_filter'],
                    'Entrapment_Percentage': round(result['entrapment_pct'], 2),
                    'Unique_Normalized_Proteins': result['unique_normalized_proteins']
                })
        
        # Save checkpoint after each dataset (in case of crash)
        # Note: save_checkpoint will automatically add mapping file hash
        save_checkpoint("step1_processed_datasets", {
            'all_results': all_results,
            'dataset_summary': dataset_summary
        })
    
    # Build dataset summary (before filter); combined with after-filter data written in step 5
    summary_df = pd.DataFrame(dataset_summary)
    
    # Group by acquisition method (get unique base datasets)
    base_datasets = {}
    for _, row in summary_df.iterrows():
        base_name = row['Dataset']
        for suffix in ['-plasma', '-serum', '-erythrocyte', '-DDA', '-DIA']:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                break
        if base_name not in base_datasets:
            base_datasets[base_name] = row['Acquisition']
    
    dda_datasets = [ds for ds, acq in base_datasets.items() if acq == 'DDA']
    dia_datasets = [ds for ds, acq in base_datasets.items() if acq == 'DIA']
    unknown_datasets = [ds for ds, acq in base_datasets.items() if acq == 'Unknown']
    
    print(f"\nAcquisition methods (base datasets):")
    print(f"  DDA: {len(dda_datasets)} datasets - {', '.join(dda_datasets)}")
    print(f"  DIA: {len(dia_datasets)} datasets - {', '.join(dia_datasets)}")
    if unknown_datasets:
        print(f"  Unknown: {len(unknown_datasets)} datasets - {', '.join(unknown_datasets)}")
    
    # Group by tissue/cell type
    print("\n" + "=" * 80)
    print("STEP 2: GROUPING DATASETS BY TISSUE/CELL TYPE")
    print("=" * 80)
    
    tissue_grouping = defaultdict(set)  # Use set to avoid duplicates
    for result in all_results:
        if result is None:
            continue
        condition = result['condition']
        tissue = group_condition_to_tissue(condition)
        tissue_grouping[tissue].add(result['dataset'])  # Use add() for set
    
    # Convert sets to sorted lists (full list of tissues/datasets; 02 will be written after step 3 with stats)
    tissue_grouping_df = pd.DataFrame([
        {'Tissue_CellType': tissue, 'Datasets': ', '.join(sorted(datasets)), 'Count': len(datasets)}
        for tissue, datasets in tissue_grouping.items()
    ])
    tissue_grouping_df = tissue_grouping_df.drop_duplicates(subset=['Tissue_CellType'], keep='first')
    print(f"  Total unique tissues: {len(tissue_grouping_df)}")
    
    # Combine datasets by tissue
    print("\n" + "=" * 80)
    print("STEP 3: COMBINING DATASETS BY TISSUE/CELL TYPE")
    print("=" * 80)
    
    # Try to load checkpoint
    checkpoint_step3 = load_checkpoint("step3_combined_by_tissue")
    
    if checkpoint_step3 is not None:
        print("  Resuming from checkpoint...")
        combined_by_tissue = checkpoint_step3
        print(f"  Found combined data for {len(combined_by_tissue)} tissues")
    else:
        combined_by_tissue = combine_datasets_by_tissue(all_results, entry_name_library)
        # Save checkpoint
        save_checkpoint("step3_combined_by_tissue", combined_by_tissue)
    
    # Backfill: ensure every tissue in tissue_grouping is in combined_by_tissue (so 04/05 include all cell types/datasets)
    for _, row in tissue_grouping_df.iterrows():
        tissue = row['Tissue_CellType']
        if tissue in combined_by_tissue:
            continue
        dataset_set = set(d.strip() for d in row['Datasets'].split(','))
        tissue_results = [r for r in all_results if r is not None and r.get('dataset') in dataset_set]
        if not tissue_results:
            continue
        entry = _build_combined_tissue_entry(tissue, tissue_results)
        if entry is not None:
            combined_by_tissue[tissue] = entry
            print(f"  Backfilled combined data for {tissue} ({len(tissue_results)} dataset(s))")
    
    # Build before-filter summary (merged with after-filter into single 02 file below)
    rows_02 = []
    for _, row in tissue_grouping_df.iterrows():
        tissue = row['Tissue_CellType']
        r = {
            'Tissue_CellType': tissue,
            'Datasets': row['Datasets'],
            'Count': row['Count'],
        }
        if tissue in combined_by_tissue:
            data = combined_by_tissue[tissue]
            r['Total_Unique_Proteins'] = data['total_unique_proteins']
            r['Correct_Proteins'] = data['correct_proteins']
            r['ENTRAP_Proteins'] = data['ENTRAP_proteins']
            r['Entrapment_Percentage'] = round(data['entrapment_pct'], 2)
        else:
            # Fallback: compute stats from all_results (e.g. when step3 checkpoint is stale and missing this tissue)
            dataset_set = set(d.strip() for d in row['Datasets'].split(','))
            tissue_results = [res for res in all_results if res is not None and res.get('dataset') in dataset_set]
            total_u, correct, entrap, pct = _compute_tissue_stats_from_results(tissue_results)
            r['Total_Unique_Proteins'] = total_u
            r['Correct_Proteins'] = correct
            r['ENTRAP_Proteins'] = entrap
            r['Entrapment_Percentage'] = round(pct, 2)
        rows_02.append(r)
    tissue_grouping_df = pd.DataFrame(rows_02)  # Keep for report and for combined 02 file (written after step 4)
    
    # Apply peptide filter
    default_min_peptides = get_min_peptides_from_config()
    print("\n" + "=" * 80)
    print(f"STEP 4: APPLYING PEPTIDE FILTER (min {default_min_peptides} peptides)")
    print("=" * 80)
    print(f"Configuration: DEFAULT_PEPTIDE_FILTER = '{DEFAULT_PEPTIDE_FILTER}'")
    print(f"  This means cache files will use min {default_min_peptides} peptides filter")
    if SAVE_MULTIPLE_VERSIONS:
        print(f"  Multiple versions will be saved (unfiltered + filtered)")
    print("=" * 80)
    
    # Source of truth for which datasets belong to each tissue (same as 02)
    tissue_to_datasets_and_count = {}
    for _, row in tissue_grouping_df.iterrows():
        t = row['Tissue_CellType']
        dataset_str = row['Datasets']
        dataset_list = sorted([d.strip() for d in str(dataset_str).split(',') if d.strip()])
        tissue_to_datasets_and_count[t] = (dataset_list, int(row['Count']))

    # ----------------------------
    # Step 4 redesign (incremental):
    # Cache per-dataset protein sets (unfiltered + peptide-filtered) and ENTRAP sets.
    # Tissue summaries become fast unions over these sets, so adding a dataset only computes that dataset.
    # ----------------------------
    STEP4_DATASET_CACHE_VERSION = 2
    step4_dataset_cache = load_checkpoint("step4_dataset_cache")
    dataset_cache = {}
    cache_sig = None
    if isinstance(step4_dataset_cache, dict):
        cache_sig = step4_dataset_cache.get("signature")
        dataset_cache = step4_dataset_cache.get("datasets", {}) if isinstance(step4_dataset_cache.get("datasets"), dict) else {}

    current_sig = {
        "version": STEP4_DATASET_CACHE_VERSION,
        "min_peptides": default_min_peptides,
    }

    if cache_sig != current_sig:
        # Invalidate dataset cache if filter or logic changes
        if dataset_cache:
            print("  Step 4 dataset cache signature changed; recomputing per-dataset sets as needed.")
        dataset_cache = {}

    # Compute per-dataset cached sets only for datasets needed by current tissue grouping
    needed_datasets = set()
    for tissue, (ds_list, _) in tissue_to_datasets_and_count.items():
        for ds in ds_list:
            needed_datasets.add(ds)

    # Build lookup from dataset name -> step1 result
    result_by_dataset = {r.get("dataset"): r for r in all_results if r is not None and r.get("dataset") is not None}

    missing = sorted([ds for ds in needed_datasets if ds not in dataset_cache])
    if missing:
        print(f"\n  Step 4 sub-step A: building per-dataset cached sets ({len(missing)} new dataset(s))")
        for i, ds in enumerate(missing, 1):
            print(f"    [ {i} / {len(missing)} ] {ds} ...", flush=True)
            r = result_by_dataset.get(ds)
            if r is None:
                print("      Warning: dataset not found in Step 1 results; skipping.")
                continue
            proc = r.get("processed_data")
            if proc is None or len(proc) == 0 or "Protein" not in proc.columns:
                print("      Warning: processed_data missing/empty; skipping.")
                continue

            proteins_unfiltered = set(proc["Protein"].dropna().astype(str).unique())
            proteins_before_support_filter = set(
                r.get("proteins_before_unique_peptide_filter", proteins_unfiltered)
            )
            entrap_unfiltered = set()
            n2e = r.get("normalized_to_entrap")
            if isinstance(n2e, dict):
                entrap_unfiltered = {p for p, is_e in n2e.items() if is_e}

            filtered_df = apply_peptide_filter(proc, min_peptides=default_min_peptides)
            proteins_filtered = set(filtered_df["Protein"].dropna().astype(str).unique()) if len(filtered_df) > 0 else set()
            entrap_filtered = entrap_unfiltered & proteins_filtered

            dataset_cache[ds] = {
                # True pre-filter protein universe (before early unique-peptide support filtering).
                "proteins_unfiltered": sorted(proteins_before_support_filter),
                "entrap_unfiltered": sorted(entrap_unfiltered),
                "proteins_filtered": sorted(proteins_filtered),
                "entrap_filtered": sorted(entrap_filtered),
            }
            print(f"      unfiltered={len(proteins_before_support_filter)} (ENTRAP={len(entrap_unfiltered)}), "
                  f"filtered={len(proteins_filtered)} (ENTRAP={len(entrap_filtered)})", flush=True)

        save_checkpoint("step4_dataset_cache", {"signature": current_sig, "datasets": dataset_cache})
        print("  Step 4 sub-step A complete: per-dataset cache saved.", flush=True)
    else:
        print("\n  Step 4 sub-step A: per-dataset cache is up to date (no new datasets).", flush=True)

    # Build per-tissue before/after summaries from cached per-dataset sets
    print("\n  Step 4 sub-step B: building tissue unions from cached per-dataset sets...", flush=True)
    # Global ENTRAP IDs — same rule as Step 5 when writing parquet (remove row if Protein in this set).
    # Per-dataset normalized_to_entrap alone can miss proteins: e.g. flagged ENTRAP in dataset A but observed
    # only in dataset B's n2e dict without that key → old 02 union counted them as "correct" while all caches dropped them.
    all_entrap_proteins_global = set()
    for result in all_results:
        if result is None:
            continue
        n2e = result.get("normalized_to_entrap")
        if isinstance(n2e, dict):
            for prot, is_e in n2e.items():
                if is_e:
                    all_entrap_proteins_global.add(prot)
    print(
        f"  Global ENTRAP protein IDs (aligned with Step 5 cache removal): {len(all_entrap_proteins_global)}",
        flush=True,
    )

    before_summary = []
    filtered_summary = []
    filtered_by_tissue = {}

    for tissue, (ds_list, _) in tissue_to_datasets_and_count.items():
        print(f"    Tissue: {tissue} ({len(ds_list)} dataset(s))", flush=True)
        proteins_before_union = set()
        entrap_before_union = set()
        proteins_after_union = set()

        for ds in ds_list:
            entry = dataset_cache.get(ds)
            if entry is None:
                continue
            proteins_before_union.update(entry.get("proteins_unfiltered", []))
            entrap_before_union.update(entry.get("entrap_unfiltered", []))
            proteins_after_union.update(entry.get("proteins_filtered", []))

        entrap_before_union = entrap_before_union & proteins_before_union
        # After peptide filter: ENTRAP = any ID in the tissue union that is globally flagged (matches Step 5)
        entrap_after_union = proteins_after_union & all_entrap_proteins_global
        correct_before_union = proteins_before_union - entrap_before_union
        correct_after_union = proteins_after_union - entrap_after_union

        before_entrap_pct = (len(entrap_before_union) / len(proteins_before_union) * 100) if proteins_before_union else 0.0
        after_entrap_pct = (len(entrap_after_union) / len(proteins_after_union) * 100) if proteins_after_union else 0.0

        datasets_str = ", ".join(sorted(ds_list))
        before_summary.append({
            "Tissue_CellType": tissue,
            "Datasets": datasets_str,
            "Count": len(ds_list),
            "Total_Unique_Proteins": len(proteins_before_union),
            "Correct_Proteins": len(correct_before_union),
            "ENTRAP_Proteins": len(entrap_before_union),
            "Entrapment_Percentage": round(before_entrap_pct, 2),
        })

        filtered_summary.append({
            "Tissue_CellType": tissue,
            "Datasets": datasets_str,
            "Count": len(ds_list),
            "Total_Proteins_After_Peptide_Filter": len(proteins_after_union),
            "Correct_Proteins": len(correct_after_union),
            "ENTRAP_Proteins": len(entrap_after_union),
            "Entrapment_Percentage": round(after_entrap_pct, 2),
        })

        filtered_by_tissue[tissue] = {
            "tissue": tissue,
            "datasets": sorted(ds_list),
            "total_proteins": len(proteins_after_union),
            "correct_proteins": len(correct_after_union),
            "ENTRAP_proteins": len(entrap_after_union),
            "entrapment_pct": after_entrap_pct,
            # No giant filtered_data dataframe here; Step 5 will work per-dataset (and is checkpointed).
            "filtered_data": None,
            "correct_protein_ids": sorted(correct_after_union),
        }
    
    # Remove any duplicate tissue entries (shouldn't happen, but just in case)
    filtered_summary_df = pd.DataFrame(filtered_summary)
    filtered_summary_df = filtered_summary_df.drop_duplicates(subset=['Tissue_CellType'], keep='first')
    # Build combined 02: before-filter columns + 3 empty columns + after-filter columns (replaces old 02 and 03)
    before_cols = ['Tissue_CellType', 'Datasets', 'Count', 'Total_Unique_Proteins', 'Correct_Proteins', 'ENTRAP_Proteins', 'Entrapment_Percentage']
    after_cols = ['Total_Proteins_After_Peptide_Filter', 'Correct_Proteins', 'ENTRAP_Proteins', 'Entrapment_Percentage']
    after_rename = {'Correct_Proteins': 'Correct_Proteins_After', 'ENTRAP_Proteins': 'ENTRAP_Proteins_After', 'Entrapment_Percentage': 'Entrapment_Percentage_After'}
    df_after = filtered_summary_df[['Tissue_CellType'] + after_cols].rename(columns=after_rename)
    # Always build 02 'before' from per-dataset cached unions (keeps 02 consistent even when Step 3 is resumed from an old checkpoint).
    before_df = pd.DataFrame(before_summary)
    combined_02 = before_df.merge(df_after, on='Tissue_CellType', how='left')
    combined_02.insert(7, '', '')
    combined_02.insert(8, ' ', '')
    combined_02.insert(9, '  ', '')
    combined_file = output_dir / "02_tissue_summary_before_and_after_filtering.csv"
    combined_02.to_csv(combined_file, index=False)
    print(f"\nSaved tissue summary (before + after filtering): {combined_file}")
    print(f"  Total unique tissues: {len(combined_02)}")
    
    # Export 02 correct protein list for every tissue (so E_ and other scripts use same filtered counts)
    def _tissue_to_safe(tissue_name):
        return str(tissue_name).replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '').replace(':', '_')
    
    for tissue_name, data_t in filtered_by_tissue.items():
        if 'correct_protein_ids' in data_t:
            proteins_02 = data_t['correct_protein_ids']
            safe = _tissue_to_safe(tissue_name)
            out_txt = output_dir / f"02_proteins_{safe}.txt"
            out_txt = out_txt.resolve()
            with open(out_txt, 'w', encoding='utf-8') as f:
                f.write("\n".join(proteins_02))
            print(f"  Saved 02 protein list: {out_txt.name} ({tissue_name}, {len(proteins_02)} proteins)")
            if tissue_name == 'Blood Plasma/Serum':
                legacy_txt = output_dir / "02_blood_plasma_serum_proteins.txt"
                with open(legacy_txt, 'w', encoding='utf-8') as f:
                    f.write("\n".join(proteins_02))
                print(f"  Saved (legacy) Blood Plasma/Serum list for E_: {legacy_txt.name}")
    
    # Save lightweight Step 4 tissue checkpoint (derived from per-dataset cache; mostly for convenience/debug)
    save_checkpoint("step4_filtered_by_tissue", {
        "filtered_by_tissue": filtered_by_tissue,
        "signature": current_sig,
    })
    
    # Remove entrapments and create final cache files
    print("\n" + "=" * 80)
    print("STEP 5: REMOVING ENTRAPMENTS AND CREATING FINAL CACHE FILES")
    print("=" * 80)
    
    # Try to load checkpoint for processed cache files
    checkpoint_step5 = load_checkpoint("step5_cache_summary")
    if checkpoint_step5 is not None:
        # Only treat as processed if cache file still exists (re-process if user deleted cache)
        final_cache_summary = [e for e in checkpoint_step5 if cache_file_exists_for_dataset(e['Dataset'])]
        datasets_processed = set([entry['Dataset'] for entry in final_cache_summary])
        dropped = len(checkpoint_step5) - len(final_cache_summary)
        if dropped:
            print(f"  Found {len(datasets_processed)} previously processed cache files ({dropped} missing from disk will be re-processed)")
        else:
            print(f"  Found {len(datasets_processed)} previously processed cache files")
    else:
        final_cache_summary = []
        datasets_processed = set()
    
    # Build ENTRAP protein set from all results
    all_entrap_proteins = set()
    for result in all_results:
        if 'normalized_to_entrap' in result:
            for prot, is_entrap in result['normalized_to_entrap'].items():
                if is_entrap:
                    all_entrap_proteins.add(prot)
    
    print(f"  Identified {len(all_entrap_proteins)} unique ENTRAP proteins to remove")
    
    total_step5 = len(all_results)
    step5_index = 0
    for result in all_results:
        dataset_name = result['dataset']
        if dataset_name in datasets_processed:
            continue
        
        # Skip if already cached (so restarting the run doesn't re-process)
        default_min_peptides = get_min_peptides_from_config()
        if default_min_peptides == 2:
            default_cache_file = cache_dir / f"{dataset_name}_processed.parquet"
        elif default_min_peptides == 0:
            default_cache_file = cache_dir / f"{dataset_name}_processed_unfiltered.parquet"
        else:
            default_cache_file = cache_dir / f"{dataset_name}_processed_min{default_min_peptides}pep.parquet"
        if default_cache_file.exists():
            datasets_processed.add(dataset_name)
            step5_index += 1
            print(f"\n[ {step5_index} / {total_step5} ] {dataset_name}...", flush=True)
            print(f"  Already cached, skipping.", flush=True)
            # Add to summary so final CSV and checkpoint stay consistent
            dataset_tissue_skip = group_condition_to_tissue(result.get('condition', 'unknown'))
            total_skip = 0
            metadata_file_skip = cache_dir / f"{dataset_name}_metadata.json"
            if metadata_file_skip.exists():
                try:
                    with open(metadata_file_skip, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                    total_skip = int(meta.get('total_proteins', 0))
                    dataset_tissue_skip = meta.get('tissue_celltype', dataset_tissue_skip)
                except Exception:
                    pass
            final_cache_summary.append({
                'Dataset': dataset_name,
                'Tissue_CellType': dataset_tissue_skip,
                'Total_Proteins_Final': total_skip,
                'Total_Proteins_After_Peptide_Filter': None,
                'Correct_Proteins_After': None,
                'ENTRAP_Proteins_After': None,
                'Entrapment_Percentage_After': None
            })
            save_checkpoint("step5_cache_summary", final_cache_summary)
            continue
        
        datasets_processed.add(dataset_name)
        step5_index += 1
        print(f"\n[ {step5_index} / {total_step5} ] Processing {dataset_name}...", flush=True)
        
        # Get the filtered data for this dataset (after peptide filter)
        # Find which tissue this dataset belongs to
        dataset_tissue = None
        
        # First try to find by checking datasets list
        for tissue, data in filtered_by_tissue.items():
            if dataset_name in data.get('datasets', []):
                dataset_tissue = tissue
                break
        
        # If not found, determine tissue from condition (case-insensitive; e.g. Neutrophil == neutrophil)
        if dataset_tissue is None:
            condition = result.get('condition', 'unknown')
            dataset_tissue = group_condition_to_tissue(condition)
        
        if dataset_tissue is None:
            print(f"  Warning: Could not determine tissue for {dataset_name}, skipping")
            continue
        
        # Get filtered data: prefer tissue pool (only if Step 4 stored a DataFrame); otherwise use original result + peptide filter
        dataset_data = None
        if dataset_tissue in filtered_by_tissue:
            tissue_filtered_data = filtered_by_tissue[dataset_tissue].get('filtered_data')
            if tissue_filtered_data is not None and hasattr(tissue_filtered_data, 'columns') and 'Dataset' in tissue_filtered_data.columns:
                subset = tissue_filtered_data[tissue_filtered_data['Dataset'] == dataset_name]
                if len(subset) > 0:
                    dataset_data = subset.copy()
        
        if dataset_data is None or len(dataset_data) == 0:
            print(f"  Warning: Dataset not found in tissue data, processing from original result...")
            # Get original processed data
            original_data = result.get('processed_data')
            if original_data is None or len(original_data) == 0:
                print(f"  Warning: No original data found for {dataset_name}, skipping")
                continue
            
            # Apply peptide filter (use configured default)
            dataset_data = apply_peptide_filter(original_data, min_peptides=default_min_peptides)
            
            if len(dataset_data) == 0:
                print(f"  Warning: No data found for {dataset_name} after peptide filter")
                continue
        
        # After peptide filter (before ENTRAP removal): total, correct, ENTRAP, % for 01 combined file
        unique_after_filter = dataset_data['Protein'].unique()
        total_after_peptide_filter = len(unique_after_filter)
        entrap_after_count = sum(1 for p in unique_after_filter if p in all_entrap_proteins)
        correct_after_count = total_after_peptide_filter - entrap_after_count
        entrap_pct_after = (entrap_after_count / total_after_peptide_filter * 100) if total_after_peptide_filter > 0 else 0.0
        
        # Remove entrapments
        final_data = dataset_data[~dataset_data['Protein'].isin(all_entrap_proteins)].copy()
        
        # Remove OriginalProtein and Dataset columns (not needed in cache)
        columns_to_drop = []
        if 'OriginalProtein' in final_data.columns:
            columns_to_drop.append('OriginalProtein')
        if 'Dataset' in final_data.columns:
            columns_to_drop.append('Dataset')
        if columns_to_drop:
            final_data = final_data.drop(columns_to_drop, axis=1)
        
        total_final = final_data['Protein'].nunique()
        print(f"  After removing ENTRAP: {total_final} proteins", flush=True)
        
        # Save cache file(s) based on configuration
        # Determine default cache file name
        if default_min_peptides == 2:
            default_cache_file = cache_dir / f"{dataset_name}_processed.parquet"
        elif default_min_peptides == 0:
            default_cache_file = cache_dir / f"{dataset_name}_processed_unfiltered.parquet"
        else:
            default_cache_file = cache_dir / f"{dataset_name}_processed_min{default_min_peptides}pep.parquet"
        
        # Save default version (the one other scripts will use)
        final_data.to_parquet(default_cache_file, compression='snappy', index=False)
        print(f"  Saved default cache: {default_cache_file.name} (min {default_min_peptides} peptides)")
        
        # If SAVE_MULTIPLE_VERSIONS, also save unfiltered version
        if SAVE_MULTIPLE_VERSIONS and default_min_peptides > 0:
            # Get unfiltered data (before peptide filter)
            unfiltered_data = result.get('processed_data')
            if unfiltered_data is not None and len(unfiltered_data) > 0:
                # Remove ENTRAP from unfiltered data
                unfiltered_final = unfiltered_data[~unfiltered_data['Protein'].isin(all_entrap_proteins)].copy()
                
                # Remove columns
                columns_to_drop_unf = []
                if 'OriginalProtein' in unfiltered_final.columns:
                    columns_to_drop_unf.append('OriginalProtein')
                if 'Dataset' in unfiltered_final.columns:
                    columns_to_drop_unf.append('Dataset')
                if columns_to_drop_unf:
                    unfiltered_final = unfiltered_final.drop(columns_to_drop_unf, axis=1)
                
                unfiltered_cache_file = cache_dir / f"{dataset_name}_processed_unfiltered.parquet"
                unfiltered_final.to_parquet(unfiltered_cache_file, compression='snappy', index=False)
                print(f"  Saved unfiltered cache: {unfiltered_cache_file.name} ({unfiltered_final['Protein'].nunique()} proteins)", flush=True)
        
        # Create and save metadata for this dataset (proteins per sample, etc.)
        n_samples = final_data['Sample'].nunique()
        print(f"  Building metadata ({n_samples} samples)...", flush=True)
        metadata = {
            'dataset': dataset_name,
            'tissue_celltype': dataset_tissue,
            'total_proteins': int(total_final),
            'total_samples': int(final_data['Sample'].nunique()),
            'proteins_per_sample': {},  # {sample: count}
            'protein_sets_per_sample': {}  # {sample: [list of proteins]}
        }
        
        # Calculate proteins per sample
        for sample in final_data['Sample'].unique():
            sample_data = final_data[final_data['Sample'] == sample]
            protein_count = int(sample_data['Protein'].nunique())
            protein_list = sorted(sample_data['Protein'].unique().tolist())
            metadata['proteins_per_sample'][str(sample)] = protein_count
            metadata['protein_sets_per_sample'][str(sample)] = protein_list
        
        # Save metadata as JSON (compact to speed up write for large datasets)
        metadata_file = cache_dir / f"{dataset_name}_metadata.json"
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=None, separators=(',', ':'))
            print(f"  Saved metadata: {metadata_file.name}", flush=True)
        except Exception as e:
            print(f"  Warning: Could not save metadata for {dataset_name}: {e}", flush=True)
        
        final_cache_summary.append({
            'Dataset': dataset_name,
            'Tissue_CellType': dataset_tissue,
            'Total_Proteins_Final': total_final,
            'Total_Proteins_After_Peptide_Filter': total_after_peptide_filter,
            'Correct_Proteins_After': correct_after_count,
            'ENTRAP_Proteins_After': entrap_after_count,
            'Entrapment_Percentage_After': round(entrap_pct_after, 2)
        })
        
        # Save checkpoint after each dataset (in case of crash)
        save_checkpoint("step5_cache_summary", final_cache_summary)
        print(f"  Checkpoint saved.", flush=True)
    
    # Group final summary by tissue (use set to avoid duplicates)
    final_cache_by_tissue = defaultdict(set)
    for entry in final_cache_summary:
        final_cache_by_tissue[entry['Tissue_CellType']].add(entry['Dataset'])
    
    # Remove any duplicate dataset entries per tissue (shouldn't happen, but just in case)
    final_cache_summary_df = pd.DataFrame(final_cache_summary)
    final_cache_summary_df = final_cache_summary_df.drop_duplicates(subset=['Dataset'], keep='first')
    # Ensure after-filter columns exist (e.g. if checkpoint had old format)
    for col in ['Total_Proteins_After_Peptide_Filter', 'Correct_Proteins_After', 'ENTRAP_Proteins_After', 'Entrapment_Percentage_After']:
        if col not in final_cache_summary_df.columns:
            final_cache_summary_df[col] = None
    
    # Backfill empty after-filter stats (for "already cached" or old checkpoint entries)
    default_min_peptides = get_min_peptides_from_config()
    missing = final_cache_summary_df['Total_Proteins_After_Peptide_Filter'].isna()
    if missing.any():
        n_from_tissue = 0
        n_from_results = 0
        for idx in final_cache_summary_df.index[missing]:
            row = final_cache_summary_df.loc[idx]
            dataset_name = row['Dataset']
            tissue = row['Tissue_CellType']
            filled = False
            # Try 1: from filtered_by_tissue (works when step 4 ran with this dataset)
            if tissue in filtered_by_tissue:
                data = filtered_by_tissue[tissue]
                filtered_data = data.get('filtered_data')
                if filtered_data is not None and 'Dataset' in filtered_data.columns:
                    subset = filtered_data[filtered_data['Dataset'] == dataset_name]
                    if len(subset) > 0:
                        unique_prots = subset['Protein'].unique()
                        total_af = len(unique_prots)
                        entrap_af = sum(1 for p in unique_prots if p in all_entrap_proteins)
                        correct_af = total_af - entrap_af
                        pct_af = (entrap_af / total_af * 100) if total_af > 0 else 0.0
                        final_cache_summary_df.at[idx, 'Total_Proteins_After_Peptide_Filter'] = total_af
                        final_cache_summary_df.at[idx, 'Correct_Proteins_After'] = correct_af
                        final_cache_summary_df.at[idx, 'ENTRAP_Proteins_After'] = entrap_af
                        final_cache_summary_df.at[idx, 'Entrapment_Percentage_After'] = round(pct_af, 2)
                        n_from_tissue += 1
                        filled = True
            # Try 2: from all_results (for recently added datasets not in step 4 checkpoint's filtered_data)
            if not filled:
                for result in all_results:
                    if result is None or result.get('dataset') != dataset_name:
                        continue
                    processed_data = result.get('processed_data')
                    if processed_data is None or len(processed_data) == 0:
                        continue
                    dataset_data = apply_peptide_filter(processed_data, min_peptides=default_min_peptides)
                    if len(dataset_data) == 0:
                        continue
                    unique_prots = dataset_data['Protein'].unique()
                    total_af = len(unique_prots)
                    entrap_af = sum(1 for p in unique_prots if p in all_entrap_proteins)
                    correct_af = total_af - entrap_af
                    pct_af = (entrap_af / total_af * 100) if total_af > 0 else 0.0
                    final_cache_summary_df.at[idx, 'Total_Proteins_After_Peptide_Filter'] = total_af
                    final_cache_summary_df.at[idx, 'Correct_Proteins_After'] = correct_af
                    final_cache_summary_df.at[idx, 'ENTRAP_Proteins_After'] = entrap_af
                    final_cache_summary_df.at[idx, 'Entrapment_Percentage_After'] = round(pct_af, 2)
                    n_from_results += 1
                    break
        if n_from_tissue > 0 or n_from_results > 0:
            print(f"  Backfilled after-filter stats: {n_from_tissue} from filtered_by_tissue, {n_from_results} from all_results")
    
    # Align 02 "Correct_Proteins_After" and 02_proteins_*.txt with what is actually on disk in Step 5 parquets.
    # Step 4 unions come from in-memory processed_data; Step 5 can differ (skipped datasets, already-cached paths, etc.),
    # which inflated 02 vs E_ (~8774 vs ~8460). E_ reads *_processed.parquet — use the same union here.
    if PARQUET_AVAILABLE and filtered_by_tissue:
        pep_min_reconcile = get_min_peptides_from_config()
        print("\n" + "=" * 80, flush=True)
        print("RECONCILING 02 WITH PARQUET CACHE (same proteins as E_ sees)", flush=True)
        print("=" * 80, flush=True)
        for tissue_name, tdata in filtered_by_tissue.items():
            ds_list = tdata.get("datasets") or []
            if not ds_list:
                continue
            disk_union, missing_ds, read_errors = union_correct_proteins_from_parquet_cache(
                ds_list, cache_dir, pep_min_reconcile
            )
            n_disk = len(disk_union)
            n_prev = int(tdata.get("correct_proteins", 0))
            if missing_ds:
                print(
                    f"  {tissue_name}: {len(missing_ds)} dataset(s) missing default parquet — "
                    f"not in union (e.g. {missing_ds[:3]}{'...' if len(missing_ds) > 3 else ''})",
                    flush=True,
                )
            for ds, msg in read_errors[:5]:
                print(f"    Warning: parquet read failed for {ds}: {msg}", flush=True)
            if n_disk != n_prev:
                print(
                    f"  {tissue_name}: Correct_Proteins_After {n_prev} → {n_disk} (parquet union)",
                    flush=True,
                )
            tdata["correct_protein_ids"] = sorted(disk_union)
            tdata["correct_proteins"] = n_disk

        if "Correct_Proteins_After" in combined_02.columns:
            tmap = {
                t: int(filtered_by_tissue[t]["correct_proteins"])
                for t in filtered_by_tissue
                if "correct_proteins" in filtered_by_tissue[t]
            }
            for idx in combined_02.index:
                tt = combined_02.at[idx, "Tissue_CellType"]
                if tt in tmap:
                    combined_02.at[idx, "Correct_Proteins_After"] = tmap[tt]
        try:
            combined_02.to_csv(combined_file, index=False)
            print(f"  Re-wrote {combined_file.name} (Correct_Proteins_After from parquets).", flush=True)
        except Exception as e:
            print(f"  Warning: could not re-write 02 CSV: {e}", flush=True)

        def _tissue_to_safe_reconcile(tn):
            return (
                str(tn).replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "").replace(":", "_")
            )

        for tissue_name, data_t in filtered_by_tissue.items():
            ids = data_t.get("correct_protein_ids")
            if ids is None:
                continue
            safe = _tissue_to_safe_reconcile(tissue_name)
            out_txt = (output_dir / f"02_proteins_{safe}.txt").resolve()
            try:
                with open(out_txt, "w", encoding="utf-8") as f:
                    f.write("\n".join(ids))
                print(f"  Re-wrote {out_txt.name} ({tissue_name}, {len(ids)} proteins)", flush=True)
                if tissue_name == "Blood Plasma/Serum":
                    legacy_txt = output_dir / "02_blood_plasma_serum_proteins.txt"
                    with open(legacy_txt, "w", encoding="utf-8") as f:
                        f.write("\n".join(ids))
                    print(f"  Re-wrote {legacy_txt.name}", flush=True)
            except Exception as e:
                print(f"  Warning: could not write {out_txt.name}: {e}", flush=True)
    
    # Build combined 01: before-filter (summary_df) + Tissue_CellType + 3 empty columns + after-filter (final_cache_summary_df), same structure as 02
    # Before: drop Total_Proteins_After_Filter (duplicate of Unique_Normalized_Proteins); order: ... Unique_Normalized_Proteins, ENTRAP_Proteins, Entrapment_Percentage, Correct_Proteins
    before_01_order = [
        'Dataset', 'Acquisition', 'Condition', 'Tissue_CellType', 'Total_Proteins_Before_Filter',
        'Decoy_Removed', 'Contaminant_Removed', 'Non_Human_Removed', 'Other_Removed',
        'Unique_Normalized_Proteins', 'ENTRAP_Proteins', 'Entrapment_Percentage', 'Correct_Proteins'
    ]
    summary_with_tissue = summary_df.merge(
        final_cache_summary_df[['Dataset', 'Tissue_CellType']], on='Dataset', how='left'
    )
    summary_before = summary_with_tissue[[c for c in before_01_order if c in summary_with_tissue.columns]].copy()
    df_after_01 = final_cache_summary_df[[
        'Dataset', 'Total_Proteins_After_Peptide_Filter',
        'Correct_Proteins_After', 'ENTRAP_Proteins_After', 'Entrapment_Percentage_After'
    ]].copy()
    combined_01 = summary_before.merge(df_after_01, on='Dataset', how='left')
    n_before = len(before_01_order)
    combined_01.insert(n_before, '', '')
    combined_01.insert(n_before + 1, ' ', '')
    combined_01.insert(n_before + 2, '  ', '')
    combined_01_file = output_dir / "01_dataset_summary_before_and_after_filtering.csv"
    combined_01.to_csv(combined_01_file, index=False)
    print(f"\nSaved dataset summary (before + after filtering): {combined_01_file}")
    print(f"  Total datasets: {len(combined_01)}")
    
    # Write plasma/serum proteins-by-sample cache for E_ (so E_ skips recomputing from parquet)
    plasma_serum_datasets = final_cache_summary_df[
        final_cache_summary_df['Tissue_CellType'] == 'Blood Plasma/Serum'
    ]['Dataset'].tolist()
    if plasma_serum_datasets:
        plasma_by_sample = {}
        for dataset_name in plasma_serum_datasets:
            meta_path = cache_dir / f"{dataset_name}_metadata.json"
            if not meta_path.exists():
                continue
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
            except Exception:
                continue
            for sample, protein_list in meta.get('protein_sets_per_sample', {}).items():
                plasma_by_sample[f"{dataset_name}_{sample}"] = protein_list
        if plasma_by_sample:
            plasma_cache_path = cache_dir / "plasma_serum_proteins_by_sample.json"
            suffix = get_default_cache_parquet_suffix(cache_dir)
            out_data = {k: v for k, v in plasma_by_sample.items()}
            out_data["_cache_meta"] = {"parquet_suffix": suffix, "source": "B"}
            try:
                with open(plasma_cache_path, 'w', encoding='utf-8') as f:
                    json.dump(out_data, f, indent=2)
                print(f"  Wrote {plasma_cache_path.name} for E_ ({len(plasma_by_sample)} samples, {len(plasma_serum_datasets)} plasma/serum datasets)")
            except Exception as e:
                print(f"  Warning: Could not write plasma/serum cache: {e}")
    
    # Create proteins per tissue file with abundance values (from final cache files)
    print("\n" + "=" * 80)
    print("CREATING PROTEINS PER TISSUE FILE WITH ABUNDANCE")
    print("=" * 80)
    
    proteins_per_tissue_data = []
    
    # Load final cache files and group by tissue
    if PARQUET_AVAILABLE:
        # Get default min peptides for cache file naming
        default_min_peptides = get_min_peptides_from_config()
        
        # Load dataset summary to get tissue information
        summary_file = output_dir / "01_dataset_summary_before_and_after_filtering.csv"
        if summary_file.exists():
            try:
                prep_summary = pd.read_csv(summary_file)
                
                # Group datasets by tissue (use set to avoid duplicates)
                tissue_to_datasets = defaultdict(set)
                for _, row in prep_summary.iterrows():
                    dataset_name = str(row['Dataset']).strip()
                    condition = str(row.get('Condition', '')).strip()
                    tissue = group_condition_to_tissue(condition)
                    tissue_to_datasets[tissue].add(dataset_name)
                
                # Load cache files for each tissue (convert set to list)
                for tissue, dataset_set in tissue_to_datasets.items():
                    dataset_list = sorted(list(dataset_set))  # Convert set to sorted list
                    tissue_proteins = {}  # {protein: max_abundance}
                    
                    for dataset_name in dataset_list:
                        # Find cache file
                        cache_file = None
                        if default_min_peptides == 2:
                            cache_file = cache_dir / f"{dataset_name}_processed.parquet"
                        elif default_min_peptides == 0:
                            cache_file = cache_dir / f"{dataset_name}_processed_unfiltered.parquet"
                        else:
                            cache_file = cache_dir / f"{dataset_name}_processed_min{default_min_peptides}pep.parquet"
                        
                        # Try alternative naming if not found
                        if not cache_file.exists():
                            cache_file = cache_dir / f"{dataset_name}_processed.parquet"
                        
                        if cache_file.exists():
                            try:
                                df = pd.read_parquet(cache_file)
                                if len(df) > 0 and 'Protein' in df.columns and 'PeptideCount' in df.columns:
                                    # Get max abundance per protein for this dataset
                                    protein_abundance = df.groupby('Protein')['PeptideCount'].max()
                                    for protein, abundance in protein_abundance.items():
                                        # Keep maximum abundance across all datasets in this tissue
                                        if protein not in tissue_proteins or abundance > tissue_proteins[protein]:
                                            tissue_proteins[protein] = abundance
                            except Exception as e:
                                print(f"  Warning: Error loading {dataset_name}: {e}")
                    
                    # Create dataframe for this tissue
                    if tissue_proteins:
                        tissue_df = pd.DataFrame([
                            {'Protein': protein, 'Abundance': abundance, 'Tissue_CellType': tissue}
                            for protein, abundance in tissue_proteins.items()
                        ])
                        proteins_per_tissue_data.append(tissue_df)
                        print(f"  {tissue}: {len(tissue_proteins)} proteins")
                
            except Exception as e:
                print(f"  Warning: Error creating proteins per tissue file: {e}")
    
    if proteins_per_tissue_data:
        # Combine all tissues
        all_proteins_per_tissue = pd.concat(proteins_per_tissue_data, ignore_index=True)
        
        # Sort by tissue and then by abundance (descending)
        all_proteins_per_tissue = all_proteins_per_tissue.sort_values(
            ['Tissue_CellType', 'Abundance'], 
            ascending=[True, False]
        )
        
        # Save to CSV
        proteins_per_tissue_file = output_dir / "05_proteins_per_tissue_with_abundance.csv"
        all_proteins_per_tissue.to_csv(proteins_per_tissue_file, index=False)
        print(f"\n  Saved proteins per tissue: {proteins_per_tissue_file}")
        print(f"  Total unique proteins across all tissues: {all_proteins_per_tissue['Protein'].nunique()}")
        
        # Print summary by tissue
        tissue_counts = all_proteins_per_tissue.groupby('Tissue_CellType')['Protein'].nunique()
        print(f"\n  Proteins per tissue:")
        for tissue, count in tissue_counts.sort_values(ascending=False).items():
            print(f"    {tissue}: {count} proteins")
    else:
        print("  Warning: No protein data found to create tissue file")
    
    # Create comprehensive text report
    print("\n" + "=" * 80)
    print("CREATING COMPREHENSIVE TEXT REPORT")
    print("=" * 80)
    
    report_file = output_dir / "data_preparation_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("DATA PREPARATION AND FILTERING REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total datasets processed: {len(all_datasets)}\n\n")
        
        f.write("DATASET ACQUISITION METHODS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"DDA datasets ({len(dda_datasets)}):\n")
        for ds in dda_datasets:
            f.write(f"  - {ds}\n")
        f.write(f"\nDIA datasets ({len(dia_datasets)}):\n")
        for ds in dia_datasets:
            f.write(f"  - {ds}\n")
        if unknown_datasets:
            f.write(f"\nUnknown acquisition ({len(unknown_datasets)}):\n")
            for ds in unknown_datasets:
                f.write(f"  - {ds}\n")
        f.write("\n")
        
        f.write("PER-DATASET FILTERING RESULTS:\n")
        f.write("-" * 80 + "\n")
        for _, row in summary_df.iterrows():
            f.write(f"\n{row['Dataset']} ({row['Acquisition']}):\n")
            f.write(f"  Condition: {row['Condition']}\n")
            f.write(f"  Total proteins (before filter): {row['Total_Proteins_Before_Filter']}\n")
            f.write(f"  Removed - Decoy: {row['Decoy_Removed']}, Contaminant: {row['Contaminant_Removed']}, ")
            f.write(f"Non-human: {row['Non_Human_Removed']}, Other: {row['Other_Removed']}\n")
            f.write(f"  After filter: {row['Total_Proteins_After_Filter']} proteins ")
            f.write(f"(Correct: {row['Correct_Proteins']}, ENTRAP: {row['ENTRAP_Proteins']})\n")
            f.write(f"  Entrapment %: {row['Entrapment_Percentage']:.2f}%\n")
            f.write(f"  Unique normalized proteins: {row['Unique_Normalized_Proteins']}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("TISSUE/CELL TYPE GROUPING (with combined stats):\n")
        f.write("-" * 80 + "\n")
        for _, row in tissue_grouping_df.iterrows():
            f.write(f"\n{row['Tissue_CellType']} ({row['Count']} dataset(s)):\n")
            f.write(f"  Datasets: {row['Datasets']}\n")
            if pd.notna(row.get('Total_Unique_Proteins')):
                f.write(f"  Total unique proteins: {int(row['Total_Unique_Proteins'])}\n")
                f.write(f"  Correct: {int(row['Correct_Proteins'])}, ENTRAP: {int(row['ENTRAP_Proteins'])}\n")
                f.write(f"  Entrapment %: {row['Entrapment_Percentage']:.2f}%\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"AFTER PEPTIDE FILTER (min {default_min_peptides} peptides):\n")
        f.write("-" * 80 + "\n")
        for _, row in filtered_summary_df.iterrows():
            f.write(f"\n{row['Tissue_CellType']}:\n")
            f.write(f"  Total proteins: {row['Total_Proteins_After_Peptide_Filter']}\n")
            f.write(f"  Correct: {row['Correct_Proteins']}, ENTRAP: {row['ENTRAP_Proteins']}\n")
            f.write(f"  Entrapment %: {row['Entrapment_Percentage']:.2f}%\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("FINAL CACHE FILES (ENTRAP removed, ready for downstream analysis):\n")
        f.write("-" * 80 + "\n")
        f.write(f"Configuration: DEFAULT_PEPTIDE_FILTER = '{DEFAULT_PEPTIDE_FILTER}' (min {default_min_peptides} peptides)\n")
        if SAVE_MULTIPLE_VERSIONS:
            f.write("Multiple versions saved:\n")
            f.write("  - _processed_unfiltered.parquet: No peptide filter\n")
            if default_min_peptides == 2:
                f.write("  - _processed.parquet: Min 2 peptides (default, used by other scripts)\n")
            else:
                f.write(f"  - _processed_min{default_min_peptides}pep.parquet: Min {default_min_peptides} peptides (default)\n")
        else:
            f.write(f"Single version saved: min {default_min_peptides} peptides (default)\n")
        f.write("\nNote: Cache files contain normalized proteins with PeptideCount as abundance metric.\n")
        f.write("      ENTRAP proteins have been removed. Files are ready for use by other scripts.\n")
        f.write(f"      Other scripts will use the default version (min {default_min_peptides} peptides) unless configured otherwise.\n\n")
        for tissue, datasets_set in final_cache_by_tissue.items():
            # Convert set to sorted list for display
            datasets = sorted(list(datasets_set))
            tissue_data = final_cache_summary_df[final_cache_summary_df['Tissue_CellType'] == tissue]
            total_proteins = tissue_data['Total_Proteins_Final'].sum()
            f.write(f"\n{tissue}:\n")
            f.write(f"  Total proteins (final): {total_proteins}\n")
            f.write(f"  Datasets ({len(datasets)}): {', '.join(datasets)}\n")
            for _, row in tissue_data.iterrows():
                f.write(f"    {row['Dataset']}: {row['Total_Proteins_Final']} proteins\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("REPORT COMPLETE\n")
        f.write("=" * 80 + "\n")
    
    print(f"Saved comprehensive report: {report_file}")
    
    # Re-normalize proteins with _HUMAN suffix using updated mappings
    print("\n" + "=" * 80)
    print("STEP 6: RE-NORMALIZING PROTEINS WITH _HUMAN SUFFIX")
    print("=" * 80)
    print("Checking cache files for proteins with _HUMAN suffix and re-normalizing if mappings exist...")
    
    if PARQUET_AVAILABLE:
        # Reload mapping to get latest updates
        print("  Reloading mapping files to get latest updates...")
        updated_entry_name_library = load_entry_name_mapping_library(library_file)
        updated_manual_mapping = load_manual_mapping(manual_mapping_file)
        print(f"  JSON library: {len(updated_entry_name_library)} entries")
        print(f"  Manual mapping: {len(updated_manual_mapping)} entries")
        # Manual mapping takes precedence (overwrites JSON library entries)
        updated_entry_name_library = {**updated_entry_name_library, **updated_manual_mapping}
        print(f"  Total combined mappings: {len(updated_entry_name_library)}")
        
        # Find all cache files
        cache_files = list(cache_dir.glob("*_processed*.parquet"))
        total_renormalized = 0
        total_files_updated = 0
        remaining_human_proteins = set()  # Track proteins that still have _HUMAN after normalization attempt
        normalization_attempts = {}  # Track what was tried for debugging
        
        for cache_file in cache_files:
            try:
                df = pd.read_parquet(cache_file)
                if len(df) == 0 or 'Protein' not in df.columns:
                    continue
                
                # Find proteins with _HUMAN suffix
                human_proteins = df[df['Protein'].astype(str).str.contains('_HUMAN', na=False)]
                
                if len(human_proteins) == 0:
                    continue
                
                # Check which ones can now be normalized
                renormalization_map = {}  # old_name -> new_name
                for old_protein in human_proteins['Protein'].unique():
                    old_protein_str = str(old_protein).strip()
                    
                    # Try direct lookup first (for debugging)
                    direct_lookup = None
                    if old_protein_str in updated_entry_name_library:
                        direct_lookup = updated_entry_name_library[old_protein_str]
                    elif old_protein_str.upper() in updated_entry_name_library:
                        direct_lookup = updated_entry_name_library[old_protein_str.upper()]
                    elif old_protein_str.endswith('_HUMAN') and old_protein_str[:-6] in updated_entry_name_library:
                        direct_lookup = updated_entry_name_library[old_protein_str[:-6]]
                    
                    # Try to normalize using the function
                    new_protein = normalize_protein_name(old_protein_str, updated_entry_name_library)
                    
                    # Debug: track what was tried
                    if old_protein_str not in normalization_attempts:
                        normalization_attempts[old_protein_str] = {
                            'original': old_protein_str,
                            'normalized': new_protein,
                            'direct_lookup': direct_lookup,
                            'in_mapping': (old_protein_str in updated_entry_name_library or 
                                        old_protein_str.upper() in updated_entry_name_library or
                                        (old_protein_str.endswith('_HUMAN') and old_protein_str[:-6] in updated_entry_name_library))
                        }
                    
                    # Use direct lookup if normalize_protein_name didn't work but we found it
                    if not new_protein or new_protein == old_protein_str or '_HUMAN' in new_protein:
                        if direct_lookup:
                            new_protein = direct_lookup
                    
                    if new_protein and new_protein != old_protein_str and '_HUMAN' not in new_protein:
                        renormalization_map[old_protein] = new_protein
                    else:
                        # Still has _HUMAN or couldn't be normalized - add to remaining list
                        remaining_human_proteins.add(old_protein_str)
                
                # If we found proteins to re-normalize, update the dataframe
                if renormalization_map:
                    print(f"  {cache_file.name}: Re-normalizing {len(renormalization_map)} proteins")
                    df['Protein'] = df['Protein'].replace(renormalization_map)
                    
                    # Save updated cache file
                    df.to_parquet(cache_file, compression='snappy', index=False)
                    total_renormalized += len(renormalization_map)
                    total_files_updated += 1
                    
                    # Check for any remaining _HUMAN proteins after re-normalization
                    remaining_after = df[df['Protein'].astype(str).str.contains('_HUMAN', na=False)]
                    if len(remaining_after) > 0:
                        for prot in remaining_after['Protein'].unique():
                            remaining_human_proteins.add(str(prot))
                    
                    # Also update metadata file if it exists
                    metadata_file = cache_dir / f"{cache_file.stem.replace('_processed', '')}_metadata.json"
                    if metadata_file.exists():
                        try:
                            with open(metadata_file, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)
                            
                            # Update protein sets in metadata
                            for sample_key, protein_list in metadata.get('protein_sets_per_sample', {}).items():
                                updated_list = [renormalization_map.get(p, p) for p in protein_list]
                                metadata['protein_sets_per_sample'][sample_key] = updated_list
                            
                            # Update total proteins count
                            metadata['total_proteins'] = int(df['Protein'].nunique())
                            
                            with open(metadata_file, 'w', encoding='utf-8') as f:
                                json.dump(metadata, f, indent=2)
                        except Exception as e:
                            print(f"    Warning: Could not update metadata file: {e}")
                else:
                    # No proteins could be re-normalized, but they still have _HUMAN
                    # (already added to remaining_human_proteins above)
                    pass
            
            except Exception as e:
                print(f"  Warning: Error processing {cache_file.name}: {e}")
        
        if total_renormalized > 0:
            print(f"\n  Re-normalized {total_renormalized} proteins across {total_files_updated} cache files")
        else:
            print(f"  No proteins with _HUMAN suffix found that can be re-normalized")
            print(f"  (Either all are already normalized, or no mappings exist for them)")
        
        # Save remaining _HUMAN proteins to text file for manual mapping
        if remaining_human_proteins:
            remaining_proteins_file = output_dir / "remaining_human_proteins_to_map.txt"
            try:
                with open(remaining_proteins_file, 'w', encoding='utf-8') as f:
                    f.write("=" * 80 + "\n")
                    f.write("REMAINING PROTEINS WITH _HUMAN SUFFIX\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(f"Total: {len(remaining_human_proteins)} unique proteins\n\n")
                    f.write("These proteins still have _HUMAN suffix and need to be added to manual_id_mapping.xlsx\n")
                    f.write("Format: ENTRY_NAME_HUMAN -> UniProt_Accession\n\n")
                    f.write("IMPORTANT: Make sure the entry name column matches exactly (including _HUMAN suffix)\n")
                    f.write("           The script will also check without _HUMAN suffix automatically.\n\n")
                    f.write("-" * 80 + "\n")
                    f.write("PROTEIN LIST (one per line):\n")
                    f.write("-" * 80 + "\n")
                    for protein in sorted(remaining_human_proteins):
                        # Check if it's in mapping (for debugging)
                        in_mapping = (protein in updated_entry_name_library or 
                                    protein.upper() in updated_entry_name_library or
                                    (protein.endswith('_HUMAN') and protein[:-6] in updated_entry_name_library))
                        status = " [IN MAPPING BUT NOT NORMALIZED - CHECK FORMAT]" if in_mapping else ""
                        f.write(f"{protein}{status}\n")
                    
                    # Add debugging info
                    f.write("\n" + "=" * 80 + "\n")
                    f.write("DEBUGGING INFO:\n")
                    f.write("=" * 80 + "\n")
                    f.write(f"Total mappings loaded: {len(updated_entry_name_library)}\n")
                    f.write(f"Manual mapping entries: {len(updated_manual_mapping)}\n")
                    f.write("\nSample of mappings (first 10):\n")
                    for i, (key, val) in enumerate(list(updated_manual_mapping.items())[:10]):
                        f.write(f"  {key} -> {val}\n")
                    
                    f.write("\n" + "=" * 80 + "\n")
                    f.write("INSTRUCTIONS:\n")
                    f.write("=" * 80 + "\n")
                    f.write("1. Open manual_id_mapping.xlsx\n")
                    f.write("2. Add these proteins with their UniProt accessions\n")
                    f.write("3. Make sure the entry name column has the exact format (e.g., 'M3K1_HUMAN')\n")
                    f.write("4. The accession column should have the UniProt ID (e.g., 'Q9Y6R4')\n")
                    f.write("5. Use UniProt website (https://www.uniprot.org/) to find accessions if needed\n")
                    f.write("6. Save the Excel file and re-run B_data_preparation_and_filtering.py\n")
                    f.write("=" * 80 + "\n")
                
                print(f"\n  Saved list of remaining _HUMAN proteins: {remaining_proteins_file.name}")
                print(f"  Total: {len(remaining_human_proteins)} proteins need to be added to manual_id_mapping.xlsx")
                
                # Print some examples of what's in the mapping vs what's needed
                example_needed = list(remaining_human_proteins)[:5]
                print(f"\n  Example proteins that need mapping:")
                for prot in example_needed:
                    # Check various formats
                    found = False
                    if prot in updated_entry_name_library:
                        print(f"    {prot} -> Found in mapping as '{prot}' -> {updated_entry_name_library[prot]}")
                        found = True
                    elif prot.upper() in updated_entry_name_library:
                        print(f"    {prot} -> Found in mapping as '{prot.upper()}' -> {updated_entry_name_library[prot.upper()]}")
                        found = True
                    elif prot.endswith('_HUMAN') and prot[:-6] in updated_entry_name_library:
                        print(f"    {prot} -> Found in mapping as '{prot[:-6]}' -> {updated_entry_name_library[prot[:-6]]}")
                        found = True
                    if not found:
                        print(f"    {prot} -> NOT FOUND in mapping")
            except Exception as e:
                print(f"  Warning: Could not save remaining proteins list: {e}")
        else:
            print(f"\n  No remaining _HUMAN proteins found - all have been normalized!")
    
    # Save configuration file for other scripts
    save_cache_config()
    
    # Check cache Protein column is normalized; E_ can skip re-normalization when status is 'ok'
    check_and_write_cache_normalization_status()
    
    print("\n" + "=" * 80)
    print("DATA PREPARATION COMPLETE")
    print("=" * 80)
    print(f"All outputs saved to: {output_dir}")
    print(f"Cache files saved to: {cache_dir}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"Configuration saved to: {config_file}")
    print("\nCache Configuration:")
    print(f"  Default filter: {DEFAULT_PEPTIDE_FILTER} (min {get_min_peptides_from_config()} peptides)")
    print(f"  Save multiple versions: {SAVE_MULTIPLE_VERSIONS}")
    if SAVE_MULTIPLE_VERSIONS:
        print(f"  Available cache versions:")
        print(f"    - _processed_unfiltered.parquet (no peptide filter)")
        if default_min_peptides == 2:
            print(f"    - _processed.parquet (min 2 peptides - DEFAULT)")
        else:
            print(f"    - _processed_min{default_min_peptides}pep.parquet (min {default_min_peptides} peptides - DEFAULT)")
    print("\nNote: Checkpoints allow the script to resume if interrupted.")
    print("      To restart from scratch, delete files in the checkpoints directory.")
    print("      To change filter level, modify DEFAULT_PEPTIDE_FILTER at top of script and re-run.")
    print("\nOutput files:")
    print(f"  1. 01_dataset_summary_before_and_after_filtering.csv - Per-dataset summary before and after peptide filter")
    print(f"  2. 02_tissue_summary_before_and_after_filtering.csv - Tissue summary before and after peptide filter")
    print(f"  3. 05_proteins_per_tissue_with_abundance.csv - Proteins per tissue with abundance")
    print(f"  4. data_preparation_report.txt - Comprehensive text report")
    print("=" * 80)

if __name__ == "__main__":
    main()
