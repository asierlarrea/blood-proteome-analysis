"""
Database Comparison Script (PROTEIN-LEVEL)
Compares quantm (Plasma/Serum) database with external databases (PeptideAtlas, HPA, PAXDB, GPMDB)

This script:
1. Loads proteins from protein-level cache (B_data_preparation_and_filtering.py)
2. Converts quantm proteins to genes at the end for comparison with external databases
3. Ensures protein counts match other scripts by using the same cache/data source

Creates plots in protein_level_data/database_comparison/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import sys
import pickle
import csv
import re
import warnings
from scipy import stats
from scipy.stats import norm
warnings.filterwarnings('ignore')

# Optional: UpSet plot for set comparisons
try:
    from upsetplot import UpSet, from_contents
    UPSET_AVAILABLE = True
except ImportError:
    UPSET_AVAILABLE = False
    print("Note: upsetplot not available. Install with 'pip install upsetplot' for UpSet plots.")

# Set working directory and add to path for imports
work_dir = Path(r"G:\My Drive\Ikasketak\Postdoc\Cambridge\Cursor\blood_proteome_analysis")
sys.path.insert(0, str(work_dir))
os.chdir(work_dir)

# Directories
msstats_dir = work_dir / "msstats"  # Shared raw data
external_data_dir = work_dir / "external_databases"
cache_dir = work_dir / "cache"
data_prep_output_dir = work_dir / "data_preparation_output"
mapping_library_dir = work_dir / "entry_name_mapping"
output_base_dir = work_dir / "database_comparison"
output_base_dir.mkdir(parents=True, exist_ok=True)

# Entry name mapping library files
library_file = mapping_library_dir / "entry_name_to_accession.json"
manual_mapping_file = mapping_library_dir / "manual_id_mapping.xlsx"

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Parameters
HEALTHY_KEYWORDS = ['normal', 'healthy', 'control', 'not available']  # Keywords for healthy samples

def is_healthy(disease_value):
    """Check if a disease value represents healthy/normal condition."""
    if pd.isna(disease_value) or disease_value == '':
        return False
    disease_str = str(disease_value).lower().strip()
    return any(keyword in disease_str for keyword in HEALTHY_KEYWORDS)

# ============================================
# PROTEIN FILTERING AND NORMALIZATION FUNCTIONS
# ============================================

# Import from shared_utils instead of defining here
from shared_utils import (
    extract_uniprot_id,
    load_protein_mapping_cache,
    convert_protein_to_gene,
    filter_and_normalize_proteins,
    normalize_protein_to_gene,
    is_decoy_entrap_protein,
    convert_entry_name_to_accession,
    compute_protein_feature_counts,
    group_condition_to_tissue,
)
import json

# Note: All protein filtering and normalization functions are now imported from shared_utils above


def is_blood_plasma_serum_condition(condition_value) -> bool:
    """Same tissue assignment as B_/E_ (blood / plasma / serum → Blood Plasma/Serum)."""
    if condition_value is None or (isinstance(condition_value, float) and pd.isna(condition_value)):
        return False
    return group_condition_to_tissue(str(condition_value).strip()) == "Blood Plasma/Serum"


# ============================================
# ENTRY NAME MAPPING LIBRARY FUNCTIONS
# ============================================

def load_entry_name_mapping_library():
    """Load the entry name to accession mapping library from JSON file."""
    if library_file.exists():
        try:
            with open(library_file, 'r') as f:
                library = json.load(f)
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
        entry_name_library: Dictionary mapping entry_name -> accession
        return_mapping: If True, also return a mapping from normalized -> list of original proteins
    
    Returns:
        Set of normalized protein identifiers (UniProt accessions when possible)
        If return_mapping=True, also returns dict mapping normalized -> list of original proteins
    """
    normalized_set = set()
    normalized_to_original = {} if return_mapping else None
    
    for protein_id in protein_set:
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
                normalized_set.add(first_entry)
                if return_mapping:
                    if first_entry not in normalized_to_original:
                        normalized_to_original[first_entry] = []
                    normalized_to_original[first_entry].append(protein_id)
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
# REVERSE MAPPING (Gene -> Protein)
# ============================================

def create_gene_to_protein_mapping(mapping_cache):
    """Create reverse mapping from gene symbol to protein identifiers.
    
    Args:
        mapping_cache: Dictionary mapping protein_id -> gene_symbol
    
    Returns:
        Dictionary mapping gene_symbol -> set of protein_ids
    """
    gene_to_proteins = {}
    for protein_id, gene_symbol in mapping_cache.items():
        if gene_symbol and str(gene_symbol).strip():
            gene = str(gene_symbol).strip().upper()
            if gene not in gene_to_proteins:
                gene_to_proteins[gene] = set()
            gene_to_proteins[gene].add(protein_id)
    return gene_to_proteins

def convert_gene_to_proteins(gene_symbol, gene_to_protein_mapping):
    """Convert gene symbol to set of protein identifiers.
    
    Args:
        gene_symbol: Gene symbol to convert
        gene_to_protein_mapping: Dictionary mapping gene -> set of proteins
    
    Returns:
        Set of protein identifiers, or empty set if not found
    """
    if not gene_symbol:
        return set()
    gene_upper = str(gene_symbol).strip().upper()
    return gene_to_protein_mapping.get(gene_upper, set())

# ============================================
# CACHE FILE HELPERS
# ============================================

def find_cache_file(dataset_name):
    """Find cache file for a dataset (default B_ filter only: 2 peptide + ENTRAP removed)."""
    from shared_utils import get_default_cache_parquet_suffix
    default_suffix = get_default_cache_parquet_suffix(cache_dir)
    cache_file = cache_dir / f"{dataset_name}{default_suffix}"
    if cache_file.exists():
        return cache_file
    for suffix in ['-plasma', '-serum', '-erythrocyte', '-DDA', '-DIA', '-blood_serum', '-blood_plasma']:
        cache_file = cache_dir / f"{dataset_name}{suffix}{default_suffix}"
        if cache_file.exists():
            return cache_file
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
            pass
    return None

# ============================================
# ALTERNATIVE NORMALIZATION (Mapping Cache Priority)
# ============================================

def normalize_protein_to_gene_cache_priority(protein_name, mapping_cache=None):
    """Alternative normalization that prioritizes mapping cache over direct extraction.
    
    This is the opposite priority of the current normalize_protein_to_gene:
    - FIRST: Try mapping cache (gives official gene symbols like LGALS1)
    - SECOND: Extract gene name from identifier (gives LEG1)
    
    This helps test if external databases use official gene symbols vs identifier-based names.
    """
    if pd.isna(protein_name) or not protein_name:
        return None

    protein_str = str(protein_name).strip()
    
    # Handle DECOY/ENTRAP proteins
    if is_decoy_entrap_protein(protein_str):
        return None
    
    # Check for ENSP IDs
    if re.match(r'^ENSP\d{11}$', protein_str, re.IGNORECASE):
        if mapping_cache:
            if protein_str in mapping_cache:
                gene = mapping_cache[protein_str]
                if gene and str(gene).strip():
                    return str(gene).strip().upper()
            for key, value in mapping_cache.items():
                if str(key).upper() == protein_str.upper():
                    gene = value
                    if gene and str(gene).strip():
                        return str(gene).strip().upper()
        return None
    
    # Check for UniProt IDs
    if re.match(r'^[A-Z]\d{5,9}$', protein_str, re.IGNORECASE):
        if mapping_cache:
            if protein_str in mapping_cache:
                gene = mapping_cache[protein_str]
                if gene and str(gene).strip():
                    return str(gene).strip().upper()
            for key, value in mapping_cache.items():
                if str(key).upper() == protein_str.upper():
                    gene = value
                    if gene and str(gene).strip():
                        return str(gene).strip().upper()
    
    # Format 1: sp|ACCESSION|GENE_HUMAN or tr|ACCESSION|GENE_HUMAN (DDA format)
    if '|' in protein_str:
        # Handle multiple proteins separated by semicolon
        if ';' in protein_str:
            first_protein = protein_str.split(';')[0].strip()
            if '|' in first_protein:
                parts = first_protein.split('|')
                # FIRST: Try mapping cache (prioritize official gene symbols)
                if len(parts) >= 2 and mapping_cache:
                    uniprot_id = parts[1].strip()
                    if uniprot_id in mapping_cache:
                        gene = mapping_cache[uniprot_id]
                        if gene and str(gene).strip():
                            return str(gene).strip().upper()
                
                # SECOND: Extract from identifier as fallback
                if len(parts) >= 3:
                    gene_part = parts[-1]
                    if '_HUMAN' in gene_part:
                        gene = gene_part.split('_HUMAN')[0].upper()
                        if len(gene) >= 2 and len(gene) <= 20 and gene.replace('_', '').replace('-', '').isalnum():
                            return gene
        
        # Single protein entry
        parts = protein_str.split('|')
        # FIRST: Try mapping cache
        if len(parts) >= 2 and mapping_cache:
            uniprot_id = parts[1].strip()
            if uniprot_id in mapping_cache:
                gene = mapping_cache[uniprot_id]
                if gene and str(gene).strip():
                    return str(gene).strip().upper()
        
        # SECOND: Extract from identifier
        if len(parts) >= 3:
            gene_part = parts[-1]
            if ';' in gene_part:
                gene_part = gene_part.split(';')[0].strip()
            if '_HUMAN' in gene_part:
                gene = gene_part.split('_HUMAN')[0].upper()
                if len(gene) >= 2 and len(gene) <= 20 and gene.replace('_', '').replace('-', '').isalnum():
                    return gene
            else:
                gene = gene_part.upper()
                if len(gene) >= 2 and len(gene) <= 20 and gene.replace('_', '').replace('-', '').isalnum():
                    return gene
    
    # Format 2: GENE_HUMAN or GENE1_HUMAN;GENE2_HUMAN (DIA format)
    if '_HUMAN' in protein_str:
        # For DIA format, try mapping cache first if we can extract UniProt ID
        # But DIA format doesn't have UniProt ID, so just extract gene name
        gene_part = protein_str.split('_HUMAN')[0]
        if ';' in gene_part:
            gene_part = gene_part.split(';')[0].strip()
        gene = gene_part.strip().upper()
        if len(gene) >= 2 and len(gene) <= 20 and gene.replace('_', '').replace('-', '').isalnum():
            return gene
    
    # Fallback: return uppercase if it looks like a gene symbol
    if len(protein_str) >= 2 and len(protein_str) <= 20 and protein_str.replace('_', '').replace('-', '').isalnum():
            return protein_str.upper()
    
    return None

# ============================================
# LOAD MSSTATS DATA
# ============================================

def load_dataset(dataset_name, data_dir=None):
    """Load and process a dataset from CSV."""
    if data_dir is None:
        data_dir = msstats_dir
    else:
        data_dir = Path(data_dir)
    
    filename = f"{dataset_name}.sdrf_openms_design_msstats_in.csv"
    filepath = data_dir / filename
    filepath = filepath.resolve()
    
    if not filepath.exists():
        return None
    
    try:
        # Try pandas first
        data = pd.read_csv(str(filepath), low_memory=False, on_bad_lines='skip')
    except Exception as e:
        print(f"  Warning: Error reading {filename}: {e}")
        return None
    
    # Skip datasets with no data rows (header only, e.g. PXD007975)
    if len(data) == 0:
        print(f"  Skipping {dataset_name}: CSV has no data rows (header only).")
        return None
    
    # Extract relevant columns
    col_names = data.columns.tolist()
    
    # Find columns by name
    protein_col = None
    condition_col = None
    sample_col = None
    disease_col = None
    
    for col in col_names:
        col_lower = col.lower()
        if col_lower == 'proteinname' or (protein_col is None and 'protein' in col_lower):
            protein_col = col
        if col_lower == 'condition':
            condition_col = col
        if col_lower == 'reference' or col_lower == 'sample':
            sample_col = col
        if col_lower == 'disease':
            disease_col = col
    
    if protein_col is None:
        protein_col = col_names[0] if len(col_names) > 0 else None
    if condition_col is None:
        condition_col = col_names[6] if len(col_names) > 6 else None
    if sample_col is None:
        sample_col = col_names[-1] if len(col_names) > 0 else None
    
    if protein_col is None or condition_col is None:
        return None
    
    # Ensure single Series (CSV may have duplicate column names)
    def _to_series(x):
        if x is None:
            return None
        if isinstance(x, pd.DataFrame):
            x = x.iloc[:, 0]
        return x
    protein_series = _to_series(data[protein_col])
    condition_series = _to_series(data[condition_col])
    sample_series = _to_series(data[sample_col]) if sample_col else None
    n_rows = len(protein_series)
    # Build from 1D arrays so column names are always Protein/Condition/Sample (no nested DataFrame columns)
    processed = pd.DataFrame({
        'Protein': np.asarray(protein_series).ravel(),
        'Condition': np.asarray(condition_series).ravel(),
        'Sample': np.asarray(sample_series).ravel() if sample_series is not None else [None] * n_rows
    })
    
    # Add disease column if available
    if disease_col:
        d = data[disease_col]
        processed['Disease'] = d.iloc[:, 0] if isinstance(d, pd.DataFrame) else d
    else:
        processed['Disease'] = None
    
    # Filter out Decoy/Entrap proteins and normalize to gene names
    if processed.shape[1] == 0:
        print(f"  Warning: {dataset_name} has no columns after extraction, skipping.")
        return None
    if 'Protein' not in processed.columns:
        print(f"  Warning: {dataset_name} missing Protein column (columns: {list(processed.columns)}), skipping.")
        return None
    print(f"  Filtering Decoy/Entrap proteins and normalizing to gene names for {dataset_name}...")
    mapping_cache = load_protein_mapping_cache()
    processed = filter_and_normalize_proteins(processed, protein_column='Protein', inplace=False, mapping_cache=mapping_cache)
    
    # Additional filtering for empty/NaN values
    protein_str = processed['Protein'].astype(str)
    valid_mask = (
        processed['Protein'].notna() & 
        (protein_str.str.strip() != '') & 
        (protein_str.str.lower() != 'nan') &
        (protein_str.str.lower() != 'none') &
        (protein_str.str.lower() != 'na') &
        (~protein_str.str.contains('CONTAM', case=False, na=False))
    )
    processed = processed[valid_mask].copy()
    
    # Remove duplicates
    processed = processed.drop_duplicates()
    processed['Dataset'] = dataset_name
    
    return processed

def load_all_msstats_data():
    """Load all msstats plasma/serum datasets."""
    print("Loading msstats plasma/serum datasets...")
    
    csv_files = list(msstats_dir.glob("*.sdrf_openms_design_msstats_in.csv"))
    datasets = []
    for f in csv_files:
        name = f.name.replace(".sdrf_openms_design_msstats_in.csv", "")
        datasets.append(name)
    
    all_data = {}
    for dataset in datasets:
        data = load_dataset(dataset, msstats_dir)
        if data is not None and len(data) > 0:
            condition = str(data["Condition"].iloc[0]) if "Condition" in data.columns else ""
            if is_blood_plasma_serum_condition(condition):
                all_data[dataset] = data
    
    print(f"  Loaded {len(all_data)} plasma/serum dataset(s)")
    return all_data

def load_msstats_plasma_serum_from_data(all_data, mapping_cache):
    """Extract plasma and serum proteins from loaded msstats data and convert to gene symbols."""
    all_genes = set()
    unmapped_proteins = set()
    
    for dataset_name, data in all_data.items():
        proteins = data['Protein'].unique()
        for protein in proteins:
            if protein and str(protein).lower() not in ["", "nan", "none", "na"]:
                # Convert protein ID to gene symbol
                gene = convert_protein_to_gene(str(protein), mapping_cache)
                if gene:
                    all_genes.add(gene)
                else:
                    unmapped_proteins.add(str(protein))
    
    print(f"  Found {len(all_genes)} unique genes from msstats plasma/serum datasets")
    if unmapped_proteins and len(unmapped_proteins) <= 10:
        print(f"    Sample unmapped: {list(unmapped_proteins)[:10]}")
    
    return all_genes

def load_msstats_plasma_serum_dual_format(all_data, mapping_cache):
    """Load plasma/serum proteins using BOTH normalization strategies for comparison.
    
    Uses the SAME datasets and SAME raw protein names as the original analysis.
    The key difference: applies both normalization strategies to the SAME set of raw proteins.
    
    Strategy 1: Extract gene name directly from identifier (current approach) - uses normalize_protein_to_gene
    Strategy 2: Prioritize mapping cache to get official gene symbols - uses normalize_protein_to_gene_cache_priority
    
    Note: This should give similar counts to the original if using the same filtering.
    The original uses convert_protein_to_gene on already-normalized data, which is why counts differ.
    
    Returns:
        Tuple of (genes_strategy1, genes_strategy2)
    """
    print("\nLoading msstats plasma/serum with DUAL normalization strategies...")
    print("  Strategy 1: Use convert_protein_to_gene (matches original analysis)")
    print("  Strategy 2: Prioritize mapping cache (normalize_protein_to_gene_cache_priority)")
    print(f"  Processing {len(all_data)} plasma/serum dataset(s)...")
    print("  Note: Original analysis uses convert_protein_to_gene on already-normalized data.")
    print("        This analysis applies both strategies to raw protein identifiers from CSV.")
    
    all_genes_strategy1 = set()
    all_genes_strategy2 = set()
    all_raw_proteins = set()  # Track unique raw proteins processed
    
    # Use the same datasets as all_data (already filtered to plasma/serum)
    for dataset_name in all_data.keys():
        # Load raw protein names from CSV (not from cache) to get original identifiers
        csv_file = msstats_dir / f"{dataset_name}.sdrf_openms_design_msstats_in.csv"
        if not csv_file.exists():
            print(f"  Warning: CSV file not found for {dataset_name}")
            continue
        
        try:
            raw_data = pd.read_csv(csv_file, low_memory=False, on_bad_lines='skip')
            
            # Find columns (same logic as load_dataset)
            protein_col = None
            condition_col = None
            for col in raw_data.columns:
                col_lower = col.lower()
                if col_lower == 'proteinname' or (protein_col is None and 'protein' in col_lower):
                    protein_col = col
                if col_lower == 'condition':
                    condition_col = col
            
            if protein_col is None:
                protein_col = raw_data.columns[0] if len(raw_data.columns) > 0 else None
            if condition_col is None:
                condition_col = raw_data.columns[6] if len(raw_data.columns) > 6 else None
            
            if protein_col is None:
                continue
            
            # Filter to Blood Plasma/Serum only (same rule as B_/E_)
            if condition_col:
                raw_data = raw_data[
                    raw_data[condition_col].apply(lambda c: is_blood_plasma_serum_condition(c))
                ]
            
            if len(raw_data) == 0:
                continue
            
            # Get unique raw protein names (before any normalization)
            raw_proteins = raw_data[protein_col].dropna().unique()
            
            for protein in raw_proteins:
                protein_str = str(protein).strip()
                if protein_str and protein_str.lower() not in ["", "nan", "none", "na"]:
                    # Skip decoy/entrap (same filtering as filter_and_normalize_proteins)
                    if is_decoy_entrap_protein(protein_str):
                        continue
                    
                    all_raw_proteins.add(protein_str)  # Track for debugging
                    
                    # Strategy 1: Use convert_protein_to_gene (matches original analysis)
                    # This primarily uses mapping cache, similar to what original does
                    gene1 = convert_protein_to_gene(protein_str, mapping_cache)
                    if gene1:
                        all_genes_strategy1.add(gene1)
                    
                    # Strategy 2: Cache priority (mapping cache first, then extract)
                    # This is similar to Strategy 1 but with explicit cache priority
                    gene2 = normalize_protein_to_gene_cache_priority(protein_str, mapping_cache)
                    if gene2:
                        all_genes_strategy2.add(gene2)
        except Exception as e:
            print(f"  Warning: Error loading {dataset_name}: {e}")
            continue
    
    print(f"  Processed {len(all_raw_proteins)} unique raw protein identifiers")
    print(f"  Strategy 1 (convert_protein_to_gene): {len(all_genes_strategy1)} unique genes")
    print(f"  Strategy 2 (cache priority normalization): {len(all_genes_strategy2)} unique genes")
    
    return all_genes_strategy1, all_genes_strategy2

def compare_normalization_strategies(genes_strategy1, genes_strategy2, external_databases, output_dir):
    """Compare the two normalization strategies and create comparison plots."""
    print("\n" + "=" * 60)
    print("COMPARING NORMALIZATION STRATEGIES")
    print("=" * 60)
    
    # Calculate overlaps
    only_strategy1 = genes_strategy1 - genes_strategy2
    only_strategy2 = genes_strategy2 - genes_strategy1
    common = genes_strategy1.intersection(genes_strategy2)
    
    print(f"\nStrategy 1 (convert_protein_to_gene): {len(genes_strategy1)} genes")
    print(f"Strategy 2 (cache priority normalization): {len(genes_strategy2)} genes")
    print(f"Common genes: {len(common)}")
    print(f"Only in Strategy 1: {len(only_strategy1)}")
    print(f"Only in Strategy 2: {len(only_strategy2)}")
    
    # Compare with external databases
    print("\nComparing with external databases:")
    for db_name, db_genes in external_databases.items():
        overlap1 = len(genes_strategy1.intersection(db_genes))
        overlap2 = len(genes_strategy2.intersection(db_genes))
        only_quantm1 = len(genes_strategy1 - db_genes)
        only_quantm2 = len(genes_strategy2 - db_genes)
        
        print(f"\n  {db_name}:")
        print(f"    Strategy 1 overlap: {overlap1} ({overlap1/len(genes_strategy1)*100:.1f}% of quantm)")
        print(f"    Strategy 2 overlap: {overlap2} ({overlap2/len(genes_strategy2)*100:.1f}% of quantm)")
        print(f"    Strategy 1 only in quantm: {only_quantm1}")
        print(f"    Strategy 2 only in quantm: {only_quantm2}")
        print(f"    Difference: {abs(overlap2 - overlap1)} ({'+' if overlap2 > overlap1 else ''}{overlap2 - overlap1})")
    
    # Create UpSet plots for both strategies
    if UPSET_AVAILABLE:
        print("\nCreating UpSet plots for both strategies...")
        
        # Strategy 1 (Gene Conversion)
        gene_lists_strategy1 = {
            'quantm': genes_strategy1
        }
        gene_lists_strategy1.update(external_databases)
        create_upset_plot(
            gene_lists_strategy1,
            output_dir,
            filename="proteome_comparison_gene_conversion.png",
            title="Proteome Comparison: quantm vs External Databases (Gene-based)"
        )
        
        # Strategy 2
        gene_lists_strategy2 = {
            'quantm (Strategy 2)': genes_strategy2
        }
        gene_lists_strategy2.update(external_databases)
        create_upset_plot(
            gene_lists_strategy2,
            output_dir,
            filename="proteome_comparison_upset_strategy2_cache.png",
            title="Proteome Comparison: quantm (Strategy 2: Cache Priority Normalization) vs External Databases"
        )
    
    # Save comparison summary
    comparison_summary = {
        'Metric': [
            'Total genes (Strategy 1)',
            'Total genes (Strategy 2)',
            'Common genes',
            'Only in Strategy 1',
            'Only in Strategy 2'
        ],
        'Count': [
            len(genes_strategy1),
            len(genes_strategy2),
            len(common),
            len(only_strategy1),
            len(only_strategy2)
        ]
    }
    
    # Add database overlaps
    for db_name, db_genes in external_databases.items():
        comparison_summary['Metric'].extend([
            f'{db_name} overlap (Strategy 1)',
            f'{db_name} overlap (Strategy 2)',
            f'{db_name} only in quantm (Strategy 1)',
            f'{db_name} only in quantm (Strategy 2)'
        ])
        comparison_summary['Count'].extend([
            len(genes_strategy1.intersection(db_genes)),
            len(genes_strategy2.intersection(db_genes)),
            len(genes_strategy1 - db_genes),
            len(genes_strategy2 - db_genes)
        ])
    
    comparison_df = pd.DataFrame(comparison_summary)
    comparison_df.to_csv(output_dir / "normalization_strategy_comparison.csv", index=False)
    print(f"\n  Saved: normalization_strategy_comparison.csv")
    
    # Save gene lists
    pd.DataFrame({'gene': sorted(only_strategy1)}).to_csv(
        output_dir / "genes_only_strategy1.csv", index=False)
    pd.DataFrame({'gene': sorted(only_strategy2)}).to_csv(
        output_dir / "genes_only_strategy2.csv", index=False)
    pd.DataFrame({'gene': sorted(common)}).to_csv(
        output_dir / "genes_common_both_strategies.csv", index=False)
    
    print(f"  Saved: genes_only_strategy1.csv ({len(only_strategy1)} genes)")
    print(f"  Saved: genes_only_strategy2.csv ({len(only_strategy2)} genes)")
    print(f"  Saved: genes_common_both_strategies.csv ({len(common)} genes)")

def load_msstats_abundance_data(all_data, mapping_cache, min_samples=0):
    """Load msstats plasma/serum abundance data.
    Note: Data from protein-level cache contains PROTEINS (not genes).
    This function converts proteins to genes for abundance analysis.
    
    Args:
        all_data: Dictionary of dataset_name -> DataFrame with 'Protein' (proteins) and 'Sample' columns
        mapping_cache: Protein to gene mapping cache
        min_samples: Minimum number of samples a protein must appear in (default: 0 = all proteins)
    """
    msstats_abundance_data = []
    protein_sample_sets = {}  # Track unique samples per gene across all datasets
    protein_dataset_sets = {}  # Track unique datasets per gene across all datasets
    gene_feature_values = {}  # Track feature richness proxy per gene
    
    # First pass: collect sample support per gene (used for filtering and PA-like normalization)
    for dataset_name, data in all_data.items():
        protein_str = data['Protein'].astype(str)
        valid_mask = (
            data['Protein'].notna() &
            (~protein_str.apply(is_decoy_entrap_protein))
        )
        data_filtered = data[valid_mask].copy()
        if 'Sample' not in data_filtered.columns:
            continue
        for protein in data_filtered['Protein'].unique():
            if protein and str(protein).lower() not in ["", "nan", "none", "na"]:
                gene = convert_protein_to_gene(str(protein), mapping_cache)
                if not gene:
                    gene = str(protein).strip().upper()
                if gene and not is_decoy_entrap_protein(gene):
                    protein_samples = set(data_filtered[data_filtered['Protein'] == protein]['Sample'].astype(str).unique())
                    if gene not in protein_sample_sets:
                        protein_sample_sets[gene] = set()
                    protein_sample_sets[gene].update(protein_samples)
                    if gene not in protein_dataset_sets:
                        protein_dataset_sets[gene] = set()
                    protein_dataset_sets[gene].add(str(dataset_name))
    
    protein_sample_counts = {gene: len(samples) for gene, samples in protein_sample_sets.items()}
    
    # Second pass: collect abundance data
    for dataset_name, data in all_data.items():
        # Data is already filtered and normalized from cache/CSV loading, no need to re-filter
        data_filtered = data.copy()
        
        # Use feature-count abundance when available from B_ cache.
        # Fallback computes feature-count directly from the dataset.
        if 'PeptideCount' in data_filtered.columns:
            protein_abundance = data_filtered.groupby('Protein')['PeptideCount'].max().to_dict()
        else:
            protein_abundance = compute_protein_feature_counts(data_filtered, protein_col='Protein')
        
        # Convert proteins to genes for abundance analysis
        for protein, abundance_value in protein_abundance.items():
            if protein and str(protein).lower() not in ["", "nan", "none", "na"]:
                # Convert protein to gene using mapping cache
                gene = convert_protein_to_gene(str(protein), mapping_cache)
                if not gene:
                    # If mapping fails, skip this protein
                    continue
                
                if gene and not is_decoy_entrap_protein(gene):
                    if gene not in gene_feature_values:
                        gene_feature_values[gene] = []
                    gene_feature_values[gene].append(float(abundance_value))
                    msstats_abundance_data.append({
                        'gene': gene,
                        'abundance': float(abundance_value),
                        'source': 'quantm (Plasma/Serum)'
                    })
    
    if len(msstats_abundance_data) > 0:
        msstats_df = pd.DataFrame(msstats_abundance_data)
        # Aggregate by gene (take sum abundance across datasets)
        msstats_aggregated = msstats_df.groupby(['gene', 'source'], as_index=False).agg({
            'abundance': 'sum'
        })
        
        # Keep original quantm abundance for all existing plots.
        # Also compute a PA-like normalized abundance (extra column) for optional comparisons.
        # Detectability proxy: gene feature richness (median feature-abundance value across
        # observations of proteins that map to that gene).
        msstats_aggregated['n_samples'] = msstats_aggregated['gene'].map(protein_sample_counts).fillna(0).astype(float)
        msstats_aggregated['n_datasets_plasma'] = (
            msstats_aggregated['gene'].map(lambda g: len(protein_dataset_sets.get(g, set()))).fillna(0).astype(float)
        )
        msstats_aggregated['abundance_raw'] = msstats_aggregated['abundance'].astype(float)
        feature_proxy = {}
        for g, vals in gene_feature_values.items():
            vv = [float(v) for v in vals if pd.notna(v) and float(v) > 0]
            feature_proxy[g] = float(np.median(vv)) if vv else 0.0
        msstats_aggregated['feature_detectability_proxy'] = (
            msstats_aggregated['gene'].map(feature_proxy).fillna(0.0).astype(float)
        )
        msstats_aggregated['abundance_pa_like_proxy'] = (
            msstats_aggregated['abundance_raw'] /
            msstats_aggregated['feature_detectability_proxy'].replace(0, np.nan)
        )
        msstats_aggregated['abundance_pa_like_proxy'] = msstats_aggregated['abundance_pa_like_proxy'].fillna(0.0)
        total_proxy = float(msstats_aggregated['abundance_pa_like_proxy'].sum())
        if total_proxy > 0:
            msstats_aggregated['abundance_norm_pa_like'] = (
                msstats_aggregated['abundance_pa_like_proxy'] / total_proxy
            ) * 100000.0
        else:
            msstats_aggregated['abundance_norm_pa_like'] = 0.0
        
        # Filter by minimum sample count if specified
        if min_samples > 0:
            # Filter to proteins appearing in min_samples+ samples
            msstats_aggregated = msstats_aggregated[msstats_aggregated['n_samples'] > min_samples].copy()
            print(f"  Loaded {len(msstats_aggregated)} unique genes from quantm (Plasma/Serum) (filtered to {min_samples+1}+ samples)")
        else:
            print(f"  Loaded {len(msstats_aggregated)} unique genes from quantm (Plasma/Serum)")
        print("  quantm PA-like normalization: abundance_raw / feature_detectability_proxy, scaled per 100K")
        return msstats_aggregated
    return pd.DataFrame()

# ============================================
# LOAD EXTERNAL DATABASES
# ============================================

def load_gpmdb_with_filtering(df, source_name, gene_column=None, mapping_cache=None, entry_name_library=None):
    """Load GPMDB data with filtering and deduplication.
    
    Returns:
        Dictionary with keys 'genes' (set) and 'proteins' (set of UniProt accessions from Entry column)
    """
    print(f"  Processing {source_name} with filtering and deduplication...")
    print(f"  Input rows: {len(df)}")
    
    # First, check if "Entry" column exists (user-prepared UniProt IDs)
    entry_col = None
    for col in df.columns:
        if col.lower() == 'entry':
            entry_col = col
            break
    
    if entry_col:
        # Use Entry column directly (already UniProt IDs)
        proteins = set(df[entry_col].astype(str).dropna().unique())
        proteins = {p.strip() for p in proteins if p.strip() and p.lower() not in ["", "nan", "none", "na"]}
        
        # Filter out decoy/entrap proteins
        proteins = {p for p in proteins if not is_decoy_entrap_protein(p)}
        
        # Convert to genes
        genes = set()
        for protein in proteins:
            gene = convert_protein_to_gene(protein, mapping_cache)
            if gene:
                genes.add(gene)
        
        print(f"  Loaded {len(genes)} genes, {len(proteins)} proteins (from Entry column) from {source_name}")
        return {'genes': genes, 'proteins': proteins}
    
    # Fallback to original filtering approach if Entry column not found
    # Find columns
    accession_col = None
    description_col = None
    gene_col = None
    total_col = None
    
    for col in ['accession', 'Accession', 'protein_accession', 'ProteinAccession', 'ensp', 'ENSP']:
        if col in df.columns:
            accession_col = col
            break
    
    for col in ['description', 'Description', 'desc', 'Desc']:
        if col in df.columns:
            description_col = col
            break
    
    if gene_column and gene_column in df.columns:
        gene_col = gene_column
    else:
        for col in ['gene', 'Gene', 'gene_name', 'GeneName', 'gene_symbol', 'GeneSymbol']:
            if col in df.columns:
                gene_col = col
                break
    
    for col in ['total', 'Total', 'abundance', 'Abundance', 'value', 'Value', 'count', 'Count']:
        if col in df.columns:
            total_col = col
            break
    
    if total_col is None:
        print(f"  Warning: Total/abundance column not found for {source_name}")
        return {'genes': set(), 'proteins': set()}
    
    if accession_col is None:
        print(f"  Warning: Accession column not found for {source_name}")
        return {'genes': set(), 'proteins': set()}
    
    # Map identifiers to genes
    df['gene_from_accession'] = None
    df['gene_from_desc'] = None
    df['gene'] = None
    
    if accession_col:
        df['gene_from_accession'] = df[accession_col].astype(str).apply(
            lambda x: convert_protein_to_gene(x, mapping_cache) if x and str(x).strip() and str(x).lower() not in ["", "nan", "none", "na"] else None
        )
    
    if description_col:
        pattern = r'[A-Z0-9]+(?=,| |$)'
        df['gene_from_desc'] = df[description_col].astype(str).apply(
            lambda x: re.search(pattern, x).group(0) if x and re.search(pattern, x) else None
        )
    
    if gene_col:
        df['gene'] = df[gene_col].astype(str).apply(
            lambda x: x.strip().upper() if x and str(x).strip() and str(x).lower() not in ["", "nan", "none", "na"] else None
        )
    
    # Combine gene mappings
    mask = df['gene'].isna() | (df['gene'] == '')
    if accession_col:
        df.loc[mask, 'gene'] = df.loc[mask, 'gene_from_accession']
    
    mask = df['gene'].isna() | (df['gene'] == '')
    if description_col:
        df.loc[mask, 'gene'] = df.loc[mask, 'gene_from_desc']
    
    # Filter invalid entries (matching R script: !is.na(gene), gene != "", !is.na(total), total > 0)
    df_clean = df[
        (df['gene'].notna()) & 
        (df['gene'] != '') & 
        (df[total_col].notna()) & 
        (df[total_col] > 0) &
        (df[accession_col].notna()) &
        (df[accession_col].astype(str).str.strip() != '')
    ].copy()
    
    print(f"  Valid genes after mapping: {len(df_clean)}")
    
    if len(df_clean) == 0:
        return {'genes': set(), 'proteins': set()}
    
    # Deduplicate genes using median, keeping original accessions (matching R script)
    agg_dict = {total_col: 'median'}
    if accession_col:
        agg_dict[accession_col] = 'first'
    if description_col:
        agg_dict[description_col] = 'first'
    
    df_dedup = df_clean.groupby('gene', as_index=False).agg(agg_dict)
    print(f"  Final unique genes after deduplication: {len(df_dedup)}")
    
    # Validate gene symbols (matching R script's is_gene_symbol function)
    # Gene symbols should be 2-10 characters, start with a letter, contain only letters, numbers, and hyphens
    def is_valid_gene_symbol(gene_str):
        if not gene_str or len(gene_str) < 2 or len(gene_str) > 10:
            return False
        # Check pattern: ^[A-Z][A-Z0-9-]{1,9}$ (matching R script)
        return bool(re.match(r'^[A-Z][A-Z0-9-]{1,9}$', gene_str))
    
    # Filter to only valid gene symbols (matching R script behavior)
    valid_genes = []
    for gene in df_dedup['gene'].dropna().unique():
        gene_str = str(gene).strip().upper()
        if gene_str and is_valid_gene_symbol(gene_str):
            valid_genes.append(gene_str)
    
    genes = set(valid_genes)
    print(f"  Valid gene symbols after validation: {len(genes)}")
    
    # Filter df_dedup to only include valid genes
    df_dedup = df_dedup[df_dedup['gene'].isin(genes)].copy()
    
    # For GPMDB, after deduplication by gene, we have one entry per gene
    # The R script uses unique genes as the final "protein" count
    # So we use the deduplicated accessions (one per gene) as proteins
    # This matches the R script behavior where ~2200 unique genes = ~2200 "proteins"
    proteins_raw = set()
    if accession_col:
        # Get the first accession for each gene (from deduplicated dataframe)
        # This ensures one protein per gene, matching R script behavior
        for _, row in df_dedup.iterrows():
            acc = row[accession_col]
            if pd.notna(acc):
                acc_str = str(acc).strip()
                if acc_str and acc_str.lower() not in ["", "nan", "none", "na"]:
                    # Filter out decoy/entrap proteins
                    if not is_decoy_entrap_protein(acc_str):
                        proteins_raw.add(acc_str)
    
    # Normalize proteins to UniProt accessions
    if entry_name_library is None:
        entry_name_library = load_entry_name_mapping_library()
        manual_mapping = load_manual_mapping()
        entry_name_library = {**entry_name_library, **manual_mapping}
    
    proteins_normalized = normalize_protein_set_for_comparison(proteins_raw, entry_name_library=entry_name_library)
    
    print(f"  Loaded {len(genes)} unique genes, {len(proteins_raw)} raw proteins -> {len(proteins_normalized)} normalized proteins from {source_name}")
    return {'genes': genes, 'proteins': proteins_normalized}

def load_external_database(filepath, source_name, gene_column=None, mapping_cache=None, entry_name_library=None):
    """Load protein/gene list from external database file and convert to both gene symbols and proteins.
    
    Uses the R script approach: extracts original identifiers, converts them to genes using 
    convert_protein_to_gene (strategy1_convert), and keeps original identifiers as proteins.
    
    Returns:
        Dictionary with keys 'genes' (set) and 'proteins' (set of original identifiers)
    """
    if not filepath.exists():
        print(f"  Warning: {source_name} file not found: {filepath}")
        return {'genes': set(), 'proteins': set()}
    
    if mapping_cache is None:
        mapping_cache = load_protein_mapping_cache()
    
    try:
        df = pd.read_csv(filepath, on_bad_lines='skip', engine='python')
        
        # Special handling for databases that use "Entry" column directly (GPMDB, PAXDB, HPA)
        # These databases have UniProt IDs already prepared by the user
        use_entry_column = (source_name.startswith("GPMDB") or 
                           source_name.startswith("PAXDB") or 
                           source_name.startswith("HPA"))
        
        if use_entry_column:
            # Look for "Entry" column (case-insensitive)
            entry_col = None
            for col in df.columns:
                if col.lower() == 'entry':
                    entry_col = col
                    break
            
            if entry_col is None:
                print(f"  Warning: 'Entry' column not found in {source_name}, falling back to standard approach")
                use_entry_column = False
            else:
                # Extract proteins directly from Entry column (already UniProt IDs)
                proteins = set(df[entry_col].astype(str).dropna().unique())
                proteins = {p.strip() for p in proteins if p.strip() and p.lower() not in ["", "nan", "none", "na"]}
                
                # Filter out decoy/entrap proteins
                proteins = {p for p in proteins if not is_decoy_entrap_protein(p)}
                
                # Convert to genes for comparison
                genes = set()
                for protein in proteins:
                    gene = convert_protein_to_gene(protein, mapping_cache)
                    if gene:
                        genes.add(gene)
                
                print(f"  Loaded {len(genes)} genes, {len(proteins)} proteins (from Entry column) from {source_name}")
                return {'genes': genes, 'proteins': proteins}
        
        # Special handling for GPMDB (if Entry column not found, use filtering approach)
        if source_name.startswith("GPMDB"):
            result = load_gpmdb_with_filtering(df, source_name, gene_column, mapping_cache, entry_name_library)
            # GPMDB returns both genes and normalized protein accessions
            return result
        
        # For PeptideAtlas and other databases: extract ALL proteins directly, no gene conversion required
        # Find identifier column (prioritize protein/accession columns)
        identifier_col = None
        
        # First, try to find protein/accession columns (these are the original identifiers)
        for col in ['biosequence_accession', 'protein', 'Protein', 'protein_name', 'ProteinName', 
                   'accession', 'Accession', 'protein_accession', 'ProteinAccession',
                   'ensp', 'ENSP', 'uniprot', 'UniProt']:
            if col in df.columns:
                identifier_col = col
                break
        
        # If no protein column found, try gene column (for PeptideAtlas might have gene names)
        if identifier_col is None:
            if gene_column and gene_column in df.columns:
                identifier_col = gene_column
            else:
                for col in ['gene', 'Gene', 'gene_name', 'GeneName', 'gene_symbol', 'GeneSymbol', 'biosequence_gene_name']:
                    if col in df.columns:
                        identifier_col = col
                        break
        
        # Fallback to first column
        if identifier_col is None:
            identifier_col = df.columns[0]
            print(f"  Warning: Using first column '{identifier_col}' for {source_name}")
        
        # Extract ALL unique identifiers directly (no gene conversion filtering)
        identifiers = set(df[identifier_col].astype(str).dropna().unique())
        identifiers = {id.strip() for id in identifiers if id.strip() and id.lower() not in ["", "nan", "none", "na"]}
        
        # Filter out decoy/entrap proteins
        proteins_raw = {id for id in identifiers if not is_decoy_entrap_protein(id)}
        
        # Preprocess PAXDB identifiers (if not using Entry column)
        if source_name.startswith("PAXDB"):
            proteins_raw = {re.sub(r"^9606\.", "", id) for id in proteins_raw}
        
        # Normalize proteins to UniProt accessions (for PeptideAtlas and consistency)
        if entry_name_library is None:
            entry_name_library = load_entry_name_mapping_library()
            manual_mapping = load_manual_mapping()
            entry_name_library = {**entry_name_library, **manual_mapping}
        
        proteins_normalized = normalize_protein_set_for_comparison(proteins_raw, entry_name_library=entry_name_library)
        
        # Convert to genes for gene-based comparisons (optional, not required for protein list)
        genes = set()
        for protein in proteins_normalized:
            gene = convert_protein_to_gene(protein, mapping_cache)
            if gene:
                genes.add(gene)
        
        print(f"  Loaded {len(genes)} genes, {len(proteins_raw)} raw proteins -> {len(proteins_normalized)} normalized proteins from {source_name}")
        return {'genes': genes, 'proteins': proteins_normalized}
    
    except Exception as e:
        print(f"  Error loading {source_name}: {e}")
        return {'genes': set(), 'proteins': set()}

def load_all_external_databases(mapping_cache=None, entry_name_library=None):
    """Load all external databases from individual CSV files and return both genes and proteins.
    
    Args:
        mapping_cache: Protein to gene mapping cache
        entry_name_library: Entry name to accession mapping library
    
    Returns:
        Dictionary with keys 'genes' and 'proteins', each containing a dict of database_name -> set
    """
    print("\nLoading external databases from individual CSV files...")
    
    if mapping_cache is None:
        mapping_cache = load_protein_mapping_cache()
    
    if entry_name_library is None:
        entry_name_library = load_entry_name_mapping_library()
        manual_mapping = load_manual_mapping()
        entry_name_library = {**entry_name_library, **manual_mapping}
    
    databases_genes = {}
    databases_proteins = {}
    
    # PeptideAtlas
    peptideatlas_file = external_data_dir / "peptideatlas.csv"
    if peptideatlas_file.exists():
        result = load_external_database(peptideatlas_file, "PeptideAtlas", gene_column='biosequence_gene_name', 
                                       mapping_cache=mapping_cache, entry_name_library=entry_name_library)
        databases_genes['PeptideAtlas'] = result['genes']
        databases_proteins['PeptideAtlas'] = result['proteins']
    
    # HPA MS
    hpa_ms_file = external_data_dir / "hpa_ms.csv"
    if hpa_ms_file.exists():
        result = load_external_database(hpa_ms_file, "HPA MS", gene_column='gene', 
                                       mapping_cache=mapping_cache, entry_name_library=entry_name_library)
        databases_genes['HPA MS'] = result['genes']
        databases_proteins['HPA MS'] = result['proteins']
    
    # PAXDB - Plasma only (no serum file in the folder)
    paxdb_plasma_file = external_data_dir / "paxdb_plasma.csv"
    if paxdb_plasma_file.exists():
        result = load_external_database(paxdb_plasma_file, "PAXDB", gene_column='gene', 
                                       mapping_cache=mapping_cache, entry_name_library=entry_name_library)
        databases_genes['PAXDB'] = result['genes']
        databases_proteins['PAXDB'] = result['proteins']
    
    # GPMDB
    gpmdb_file = external_data_dir / "gpmdb.csv"
    if gpmdb_file.exists():
        result = load_external_database(gpmdb_file, "GPMDB Plasma", gene_column='gene', 
                                       mapping_cache=mapping_cache, entry_name_library=entry_name_library)
        databases_genes['GPMDB Plasma'] = result['genes']
        databases_proteins['GPMDB Plasma'] = result['proteins']
    
    # HPA PEA
    hpa_pea_file = external_data_dir / "hpa_pea.csv"
    if hpa_pea_file.exists():
        result = load_external_database(hpa_pea_file, "HPA PEA", gene_column='gene', 
                                       mapping_cache=mapping_cache, entry_name_library=entry_name_library)
        databases_genes['HPA PEA'] = result['genes']
        databases_proteins['HPA PEA'] = result['proteins']
    
    # HPA Immunoassay - Plasma only
    hpa_immuno_plasma_file = external_data_dir / "hpa_immunoassay_plasma.csv"
    if hpa_immuno_plasma_file.exists():
        result = load_external_database(hpa_immuno_plasma_file, "HPA Immunoassay", gene_column='gene', 
                                       mapping_cache=mapping_cache, entry_name_library=entry_name_library)
        databases_genes['HPA Immunoassay'] = result['genes']
        databases_proteins['HPA Immunoassay'] = result['proteins']
    
    return {'genes': databases_genes, 'proteins': databases_proteins}

def load_external_database_with_abundance(filepath, source_name, gene_column=None, abundance_column=None, mapping_cache=None):
    """Load external database with abundance data and convert to gene symbols.
    
    For PeptideAtlas: Uses the same logic as 6_database_comparison.py (looks for 'abundance' column or auto-detects)
    For all other databases: Uses 'Concentration' column for abundance.
    """
    if not filepath.exists():
        return pd.DataFrame()
    
    if mapping_cache is None:
        mapping_cache = load_protein_mapping_cache()
    
    try:
        df = pd.read_csv(filepath, on_bad_lines='skip', engine='python')
        
        # Find gene column
        gene_col = None
        if gene_column and gene_column in df.columns:
            gene_col = gene_column
        else:
            # Try common gene column names
            for col in ['gene', 'Gene', 'gene_name', 'GeneName', 'gene_symbol', 'GeneSymbol', 'biosequence_gene_name']:
                if col in df.columns:
                    gene_col = col
                    break
            
            # If not found, try protein/accession columns
            if gene_col is None:
                for col in ['Entry', 'entry', 'protein', 'Protein', 'protein_name', 'ProteinName', 
                           'accession', 'Accession', 'protein_accession', 'ProteinAccession',
                           'biosequence_accession', 'ensp', 'ENSP', 'uniprot', 'UniProt']:
                    if col in df.columns:
                        gene_col = col
                        break
        
        if gene_col is None:
            gene_col = df.columns[0]
        
        # Find abundance column
        abundance_col = None
        
        # For PeptideAtlas: use same logic as 6_database_comparison.py (look for 'abundance' or auto-detect)
        if source_name == "PeptideAtlas":
            if abundance_column and abundance_column in df.columns:
                abundance_col = abundance_column
            else:
                # Try 'abundance' first
                for col in ['abundance', 'Abundance']:
                    if col in df.columns:
                        abundance_col = col
                        break
                # If not found, try other common names
                if abundance_col is None:
                    for col in ['value', 'Value', 'count', 'Count', 
                               'total', 'Total', 'intensity', 'Intensity', 'expression', 'Expression',
                               'norm_PSMs_per_100K']:
                        if col in df.columns:
                            abundance_col = col
                            break
                # If still not found, find first numeric column
                if abundance_col is None:
                    for col in df.columns:
                        if df[col].dtype in [np.int64, np.float64]:
                            abundance_col = col
                            break
        else:
            # For all other databases: use 'Concentration' column
            if 'Concentration' in df.columns:
                abundance_col = 'Concentration'
            elif 'concentration' in df.columns:
                abundance_col = 'concentration'
            else:
                print(f"  Warning: 'Concentration' column not found for {source_name}, trying to auto-detect...")
                # Fallback: try to find numeric columns
                for col in df.columns:
                    if df[col].dtype in [np.int64, np.float64] and col.lower() not in ['entry', 'from']:
                        abundance_col = col
                        break
        
        if abundance_col is None:
            print(f"  Warning: No abundance column found for {source_name}")
            return pd.DataFrame()
        
        # Extract identifiers and abundances
        result_data = []
        for _, row in df.iterrows():
            identifier = str(row[gene_col]).strip() if pd.notna(row[gene_col]) else None
            abundance = row[abundance_col] if pd.notna(row[abundance_col]) else None
            
            if not identifier or identifier.lower() in ["", "nan", "none", "na"]:
                continue
            if abundance is None or (isinstance(abundance, (int, float)) and (pd.isna(abundance) or abundance <= 0)):
                continue
            
            # Preprocess PAXDB identifiers
            if source_name.startswith("PAXDB"):
                identifier = re.sub(r"^9606\.", "", identifier)
            
            # Convert to gene symbol
            gene = convert_protein_to_gene(identifier, mapping_cache)
            if gene:
                result_data.append({
                    'gene': gene,
                    'abundance': float(abundance),
                    'source': source_name
                })
        
        result_df = pd.DataFrame(result_data)
        if len(result_df) > 0:
            print(f"  Loaded {len(result_df)} entries from {source_name} ({result_df['gene'].nunique()} unique genes)")
        return result_df
    
    except Exception as e:
        print(f"  Error loading {source_name}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def apply_quantile_normalization(data):
    """Apply quantile-to-normal normalization to abundance data."""
    data = data[data['abundance'] > 0].copy()
    data['log_abundance'] = np.log10(data['abundance'] + 1)
    
    normalized_data = []
    for source in data['source'].unique():
        source_data = data[data['source'] == source].copy()
        source_data['rank_quantile'] = source_data['log_abundance'].rank() / (len(source_data) + 1)
        source_data['z_score'] = norm.ppf(source_data['rank_quantile'])
        normalized_data.append(source_data)
    
    return pd.concat(normalized_data, ignore_index=True)

# ============================================
# PLOTTING FUNCTIONS
# ============================================

def create_upset_plot(protein_lists, output_dir, filename="proteome_comparison_upset.png", title="Proteome Comparison: quantm (Plasma/Serum) vs External Databases", min_subset_size=175):
    """Create UpSet plot from protein lists.
    
    Args:
        protein_lists: Dictionary mapping database names to sets of proteins
        output_dir: Output directory for plots
        filename: Output filename (default: "proteome_comparison_upset.png")
        title: Plot title (default: "Proteome Comparison: quantm (Plasma/Serum) vs External Databases")
        min_subset_size: Minimum number of proteins in intersection to show (default: 175)
    """
    print("\nCreating UpSet plot...")
    
    protein_lists = {k: v for k, v in protein_lists.items() if len(v) > 0}
    
    if len(protein_lists) == 0:
        print("  Error: No data to plot")
        return
    
    if len(protein_lists) < 2:
        print(f"  Warning: Only {len(protein_lists)} database(s) with data. UpSet plot requires at least 2 databases.")
        return
    
    if not UPSET_AVAILABLE:
        print("  Warning: upsetplot not available, skipping upset plot...")
        return
    
    try:
        protein_lists_for_upset = {k: list(v) for k, v in protein_lists.items()}
        upset_data = from_contents(protein_lists_for_upset)
        
        fig = plt.figure(figsize=(16, 10))
        upset = UpSet(upset_data, subset_size='count', show_counts=True, 
                      sort_by='cardinality', min_subset_size=min_subset_size)
        upset.plot(fig=fig)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(output_dir / filename, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  Saved: {filename} (showing only intersections with >= {min_subset_size} proteins)")
    
    except Exception as e:
        print(f"  Error creating UpSet plot: {e}")
        import traceback
        traceback.print_exc()

def create_abundance_plots(normalized_data_sum, output_dir, panel_b_sources, db_colors, suffix=""):
    """Create all abundance plots (Panel B plots).
    
    Args:
        normalized_data_sum: DataFrame with normalized abundance data
        output_dir: Output directory for plots
        panel_b_sources: List of source names to include
        db_colors: Dictionary mapping source names to colors
        suffix: Suffix to add to output file names (e.g., "_filtered_5plus_samples")
    """
    if len(normalized_data_sum) == 0:
        print("  Error: No normalized data for abundance plots")
        return
    
    # Get PeptideAtlas data and create ordering
    peptideatlas_data_sum = normalized_data_sum[normalized_data_sum['source'] == 'PeptideAtlas'].copy()
    
    if len(peptideatlas_data_sum) == 0:
        print("  Warning: No PeptideAtlas data for ordering")
        return
    
    # Sort by z_score and assign order
    peptideatlas_data_sum = peptideatlas_data_sum.sort_values('z_score')
    peptideatlas_data_sum['order'] = range(1, len(peptideatlas_data_sum) + 1)
    
    # Create mapping: gene -> order
    gene_to_order_sum = dict(zip(peptideatlas_data_sum['gene'], peptideatlas_data_sum['order']))
    
    # Join with other databases
    dot_plot_data_sum = normalized_data_sum[normalized_data_sum['source'].isin(panel_b_sources)].copy()
    dot_plot_data_sum['order'] = dot_plot_data_sum['gene'].map(gene_to_order_sum)
    dot_plot_data_sum = dot_plot_data_sum[dot_plot_data_sum['order'].notna()]
    
    # Group by gene and source, take median z_score
    dot_plot_data_sum = dot_plot_data_sum.groupby(['gene', 'source'], as_index=False).agg({
        'z_score': 'median',
        'order': 'first'
    })
    
    # PLOT 1: Panel B with all databases (summed)
    print("\nCreating Panel B plot with all databases...")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    plot_order = [s for s in panel_b_sources if s != 'quantm (Plasma/Serum)']
    plot_order.append('quantm (Plasma/Serum)')
    
    for source in plot_order:
        source_data = dot_plot_data_sum[dot_plot_data_sum['source'] == source]
        if len(source_data) > 0:
            ax.scatter(source_data['order'], source_data['z_score'],
                     label=source, alpha=0.7, s=8, color=db_colors.get(source, 'gray'))
    
    ax.set_xlabel('Protein Rank (by PeptideAtlas z-score)', fontsize=12)
    ax.set_ylabel('Quantile-normalized Values (z-score)', fontsize=12)
    ax.set_title('(B) Protein abundance/detection frequency correlation with PeptideAtlas\n(quantm with summed abundances)',
                fontsize=14, fontweight='bold')
    ax.legend(title='Data Source', loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks([])
    ax.set_xticklabels([])
    
    plt.tight_layout()
    filename = f"cross_database_abundance_summed{suffix}.png"
    plt.savefig(output_dir / filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved: {filename}")
    
    # Save data
    filename = f"data_summed{suffix}.csv"
    dot_plot_data_sum.to_csv(output_dir / filename, index=False)
    print(f"  Saved: {filename}")
    
    # Calculate similarity scores
    # Use normalized_data_sum directly (before filtering to PeptideAtlas genes only)
    # This ensures we have all the data for comparison
    print("\nCalculating similarity scores with PeptideAtlas...")
    print(f"  Available sources in normalized_data_sum: {sorted(normalized_data_sum['source'].unique())}")
    
    pa_data = normalized_data_sum[normalized_data_sum['source'] == 'PeptideAtlas'].copy()
    
    if len(pa_data) == 0:
        print("  Warning: No PeptideAtlas data found in normalized_data_sum")
        print(f"  Available sources: {normalized_data_sum['source'].unique()}")
        similarity_scores = []
    else:
        # Group PeptideAtlas data by gene (take median z_score if multiple entries)
        pa_grouped = pa_data.groupby('gene', as_index=False).agg({
            'z_score': 'median',
            'abundance': 'first',
            'log_abundance': 'first'
        })
        pa_dict = dict(zip(pa_grouped['gene'], pa_grouped['z_score']))
        print(f"  PeptideAtlas has {len(pa_dict)} unique genes")
        
        similarity_scores = []
        for source in panel_b_sources:
            if source == 'PeptideAtlas':
                continue
            
            source_data = normalized_data_sum[normalized_data_sum['source'] == source].copy()
            if len(source_data) == 0:
                print(f"  Warning: No data found for {source} in normalized_data_sum")
                # Check if it exists with a different name
                available_sources = normalized_data_sum['source'].unique()
                matching = [s for s in available_sources if source.lower() in str(s).lower() or str(s).lower() in source.lower()]
                if matching:
                    print(f"    Found similar source names: {matching}")
                continue
            
            # Group source data by gene (take median z_score if multiple entries)
            source_grouped = source_data.groupby('gene', as_index=False).agg({
                'z_score': 'median',
                'abundance': 'first',
                'log_abundance': 'first'
            })
            source_genes = set(source_grouped['gene'].unique())
            print(f"  {source} has {len(source_genes)} unique genes")
            
            common_genes = set(pa_dict.keys()).intersection(source_genes)
            print(f"  Common genes between PeptideAtlas and {source}: {len(common_genes)}")
            
            if len(common_genes) > 0:
                pa_scores = [pa_dict[g] for g in common_genes]
                source_dict = dict(zip(source_grouped['gene'], source_grouped['z_score']))
                source_scores = [source_dict[g] for g in common_genes]
                correlation, _ = stats.pearsonr(pa_scores, source_scores)
                rmse = np.sqrt(np.mean((np.array(pa_scores) - np.array(source_scores))**2))
                
                similarity_scores.append({
                    'Database': source,
                    'Correlation': correlation,
                    'RMSE': rmse,
                    'Common_Genes': len(common_genes)
                })
            else:
                print(f"    Warning: No common genes found between PeptideAtlas and {source}")
                # Show sample genes from each to help debug
                if len(pa_dict) > 0 and len(source_genes) > 0:
                    sample_pa = list(pa_dict.keys())[:5]
                    sample_source = list(source_genes)[:5]
                    print(f"      Sample PeptideAtlas genes: {sample_pa}")
                    print(f"      Sample {source} genes: {sample_source}")
    
    if len(similarity_scores) > 0:
        similarity_df = pd.DataFrame(similarity_scores)
        similarity_df = similarity_df.sort_values('Correlation', ascending=False)
        filename = f"similarity_scores{suffix}.csv"
        similarity_df.to_csv(output_dir / filename, index=False)
        print("\nSimilarity scores with PeptideAtlas:")
        print(similarity_df.to_string(index=False))
        print(f"  Saved: {filename}")
    else:
        print("\nWarning: No similarity scores calculated - no common genes found between PeptideAtlas and other databases")
        print("  This might be due to:")
        print("    - Gene name normalization differences")
        print("    - Missing PeptideAtlas data")
        print("    - No overlapping genes after filtering")
        # Create empty DataFrame with expected columns
        similarity_df = pd.DataFrame(columns=['Database', 'Correlation', 'RMSE', 'Common_Genes'])
        filename = f"similarity_scores{suffix}.csv"
        similarity_df.to_csv(output_dir / filename, index=False)
        print(f"  Saved: {filename} (empty)")
    
    # PLOT 2: Panel B with only PeptideAtlas and quantm
    print("\nCreating Panel B plot with only PeptideAtlas and quantm...")
    panel_b_sources_simple = ['PeptideAtlas', 'quantm (Plasma/Serum)']
    normalized_data_simple_sum = normalized_data_sum[normalized_data_sum['source'].isin(panel_b_sources_simple)]
    
    if len(normalized_data_simple_sum) > 0:
        peptideatlas_data_simple_sum = normalized_data_simple_sum[normalized_data_simple_sum['source'] == 'PeptideAtlas'].copy()
        
        if len(peptideatlas_data_simple_sum) > 0:
            peptideatlas_data_simple_sum = peptideatlas_data_simple_sum.sort_values('z_score')
            peptideatlas_data_simple_sum['order'] = range(1, len(peptideatlas_data_simple_sum) + 1)
            
            gene_to_order_simple_sum = dict(zip(peptideatlas_data_simple_sum['gene'], peptideatlas_data_simple_sum['order']))
            
            dot_plot_data_simple_sum = normalized_data_simple_sum[normalized_data_simple_sum['source'].isin(panel_b_sources_simple)].copy()
            dot_plot_data_simple_sum['order'] = dot_plot_data_simple_sum['gene'].map(gene_to_order_simple_sum)
            dot_plot_data_simple_sum = dot_plot_data_simple_sum[dot_plot_data_simple_sum['order'].notna()]
            
            dot_plot_data_simple_sum = dot_plot_data_simple_sum.groupby(['gene', 'source'], as_index=False).agg({
                'z_score': 'median',
                'order': 'first'
            })
            
            fig, ax = plt.subplots(figsize=(14, 8))
            
            for source in panel_b_sources_simple:
                source_data = dot_plot_data_simple_sum[dot_plot_data_simple_sum['source'] == source]
                if len(source_data) > 0:
                    ax.scatter(source_data['order'], source_data['z_score'],
                             label=source, alpha=0.7, s=8, color=db_colors.get(source, 'gray'))
            
            ax.set_xlabel('Protein Rank (by PeptideAtlas z-score)', fontsize=12)
            ax.set_ylabel('Quantile-normalized Values (z-score)', fontsize=12)
            ax.set_title('(B) Protein abundance correlation: PeptideAtlas vs quantm (Plasma/Serum)\n(quantm with summed abundances)',
                        fontsize=14, fontweight='bold')
            ax.legend(title='Data Source', loc='best', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_xticks([])
            ax.set_xticklabels([])
            
            plt.tight_layout()
            filename = f"peptideatlas_msstats_summed{suffix}.png"
            plt.savefig(output_dir / filename, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"  Saved: {filename}")
            
            filename = f"peptideatlas_msstats_data_summed{suffix}.csv"
            dot_plot_data_simple_sum.to_csv(output_dir / filename, index=False)
            print(f"  Saved: {filename}")
    
    # PLOT 3: msstats only, ordered by abundance
    print("\nCreating quantm-only plot ordered by abundance...")
    msstats_only = normalized_data_sum[normalized_data_sum['source'] == 'quantm (Plasma/Serum)'].copy()
    
    if len(msstats_only) > 0:
        msstats_only = msstats_only.sort_values('z_score')
        msstats_only['order'] = range(1, len(msstats_only) + 1)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        ax.scatter(msstats_only['order'], msstats_only['z_score'],
                 alpha=0.7, s=8, color='#9b59b6', label='quantm (Plasma/Serum)')
        
        ax.set_xlabel('Protein Rank (by abundance, low to high)', fontsize=12)
        ax.set_ylabel('Quantile-normalized Values (z-score)', fontsize=12)
        ax.set_title('quantm (Plasma/Serum) Protein Abundance Distribution\n(Ordered from low to high abundance)',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks([])
        ax.set_xticklabels([])
        
        plt.tight_layout()
        filename = f"msstats_abundance_distribution{suffix}.png"
        plt.savefig(output_dir / filename, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  Saved: {filename}")
        
        filename = f"msstats_abundance_distribution_data{suffix}.csv"
        msstats_only[['gene', 'order', 'z_score', 'abundance', 'log_abundance']].to_csv(
            output_dir / filename, index=False)
        print(f"  Saved: {filename}")
    
    # PLOT 4: msstats as reference, PeptideAtlas as cloud
    print("\nCreating plot with quantm as reference and PeptideAtlas as cloud...")
    msstats_ref = normalized_data_sum[normalized_data_sum['source'] == 'quantm (Plasma/Serum)'].copy()
    pa_cloud = normalized_data_sum[normalized_data_sum['source'] == 'PeptideAtlas'].copy()
    
    if len(msstats_ref) > 0 and len(pa_cloud) > 0:
        msstats_ref = msstats_ref.sort_values('z_score')
        msstats_ref['order'] = range(1, len(msstats_ref) + 1)
        
        gene_to_order_msstats = dict(zip(msstats_ref['gene'], msstats_ref['order']))
        
        pa_cloud['order'] = pa_cloud['gene'].map(gene_to_order_msstats)
        pa_cloud = pa_cloud[pa_cloud['order'].notna()]
        
        msstats_ref_grouped = msstats_ref.groupby(['gene'], as_index=False).agg({
            'z_score': 'median',
            'order': 'first'
        })
        pa_cloud_grouped = pa_cloud.groupby(['gene'], as_index=False).agg({
            'z_score': 'median',
            'order': 'first'
        })
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        ax.scatter(pa_cloud_grouped['order'], pa_cloud_grouped['z_score'],
                 alpha=0.7, s=8, color='#3498db', label='PeptideAtlas')
        
        msstats_ref_grouped = msstats_ref_grouped.sort_values('order')
        ax.plot(msstats_ref_grouped['order'], msstats_ref_grouped['z_score'],
               color='#1a1a1a', linewidth=2, label='quantm (Plasma/Serum)', alpha=0.9)
        
        ax.set_xlabel('Protein Rank (by quantm z-score)', fontsize=12)
        ax.set_ylabel('Quantile-normalized Values (z-score)', fontsize=12)
        ax.set_title('Protein abundance correlation: quantm (Plasma/Serum) vs PeptideAtlas\n(quantm as reference, PeptideAtlas as cloud)',
                    fontsize=14, fontweight='bold')
        ax.legend(title='Data Source', loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks([])
        ax.set_xticklabels([])
        
        plt.tight_layout()
        plt.savefig(output_dir / "msstats_reference_peptideatlas_cloud.png", bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  Saved: msstats_reference_peptideatlas_cloud.png")
        
        combined_data = pd.merge(
            msstats_ref_grouped[['gene', 'order', 'z_score']].rename(columns={'z_score': 'msstats_z_score'}),
            pa_cloud_grouped[['gene', 'order', 'z_score']].rename(columns={'z_score': 'peptideatlas_z_score'}),
            on=['gene', 'order'], how='outer'
        )
        combined_data.to_csv(output_dir / "msstats_reference_data.csv", index=False)
        print(f"  Saved: msstats_reference_data.csv")
    
    # PLOT 5: Density dot plot - quantm vs PeptideAtlas (using z-scores for consistency with similarity_scores)
    print("\nCreating density dot plot: quantm vs PeptideAtlas (z-scores)...")
    quantm_data = normalized_data_sum[normalized_data_sum['source'] == 'quantm (Plasma/Serum)'].copy()
    pa_data = normalized_data_sum[normalized_data_sum['source'] == 'PeptideAtlas'].copy()
    
    if len(quantm_data) > 0 and len(pa_data) > 0:
        # Use z-scores for consistency with similarity_scores.csv
        merged_data = quantm_data[['gene', 'z_score', 'abundance', 'log_abundance']].merge(
            pa_data[['gene', 'z_score', 'abundance', 'log_abundance']], 
            on='gene', 
            suffixes=('_quantm', '_peptideatlas'),
            how='inner'
        )
        
        if len(merged_data) > 0:
            # Define filter conditions for 3 separate plots
            filter_conditions = [
                ("All proteins", None, ""),  # No filter
                ("Abundance ≥ 3", merged_data['abundance_quantm'] >= 3, "_filtered_abundance_ge_3"),  # Remove abundance = 2
                ("Abundance ≥ 4", merged_data['abundance_quantm'] >= 4, "_filtered_abundance_ge_4"),  # Remove abundance = 2 or 3
            ]
            
            for filter_name, filter_mask, file_suffix in filter_conditions:
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Apply filter if specified
                if filter_mask is not None:
                    filtered_data = merged_data[filter_mask].copy()
                else:
                    filtered_data = merged_data.copy()
                
                if len(filtered_data) > 0:
                    # Use z-scores
                    x = filtered_data['z_score_peptideatlas']
                    y = filtered_data['z_score_quantm']
                    
                    valid_mask = np.isfinite(x) & np.isfinite(y)
                    x = x[valid_mask]
                    y = y[valid_mask]
                    
                    if len(x) > 0 and len(y) > 0:
                        # Shift z-scores so they start at 0 (subtract minimum value)
                        min_x = np.min(x)
                        min_y = np.min(y)
                        min_overall = min(min_x, min_y)
                        x_shifted = x - min_overall
                        y_shifted = y - min_overall
                        
                        hb = ax.hexbin(x_shifted, y_shifted, gridsize=50, cmap='YlOrRd', mincnt=1)
                        cb = plt.colorbar(hb, ax=ax)
                        cb.set_label('Number of Proteins', fontsize=12)
                        
                        ax.scatter(x_shifted, y_shifted, alpha=0.3, s=10, c='black', edgecolors='none')
                        
                        correlation = np.corrcoef(x_shifted, y_shifted)[0, 1]
                        n_proteins = len(x)
                        
                        # Add text with correlation and protein count
                        ax.text(0.05, 0.95, f'Pearson r = {correlation:.3f}\nn = {n_proteins} proteins', 
                               transform=ax.transAxes, fontsize=12,
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                        
                        ax.set_xlabel('PeptideAtlas z-score (shifted, min=0)', fontsize=12)
                        ax.set_ylabel('quantm (Plasma/Serum) z-score (shifted, min=0)', fontsize=12)
                        ax.set_title(f'Density Plot: quantm vs PeptideAtlas\n{filter_name}',
                                   fontsize=14, fontweight='bold')
                        ax.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        filename = f"density_plot_quantm_vs_peptideatlas{file_suffix}{suffix}.png"
                        plt.savefig(output_dir / filename, bbox_inches='tight', dpi=300)
                        plt.close()
                        print(f"  Saved: {filename}")
                    else:
                        ax.text(0.5, 0.5, 'No valid data points', 
                               transform=ax.transAxes, ha='center', va='center', fontsize=12)
                        ax.set_title(f'Density Plot: quantm vs PeptideAtlas\n{filter_name}',
                                   fontsize=14, fontweight='bold')
                        plt.tight_layout()
                        filename = f"density_plot_quantm_vs_peptideatlas{file_suffix}{suffix}.png"
                        plt.savefig(output_dir / filename, bbox_inches='tight', dpi=300)
                        plt.close()
                        print(f"  Saved: {filename} (no data)")
                else:
                    ax.text(0.5, 0.5, 'No data after filtering', 
                           transform=ax.transAxes, ha='center', va='center', fontsize=12)
                    ax.set_title(f'Density Plot: quantm vs PeptideAtlas\n{filter_name}',
                               fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    filename = f"density_plot_quantm_vs_peptideatlas{file_suffix}{suffix}.png"
                    plt.savefig(output_dir / filename, bbox_inches='tight', dpi=300)
                    plt.close()
                    print(f"  Saved: {filename} (no data)")
            
            filename = f"density_plot_quantm_vs_peptideatlas_data{suffix}.csv"
            merged_data.to_csv(output_dir / filename, index=False)
            print(f"  Saved: {filename}")

            # Additional density plot variant:
            # keep original workflow unchanged, and add PA-like normalized quantm for comparison.
            if 'abundance_norm_pa_like' in quantm_data.columns:
                print("\nCreating additional density dot plot with PA-like normalized quantm abundance...")
                quantm_norm = quantm_data[['gene', 'abundance_norm_pa_like']].dropna().copy()
                quantm_norm = quantm_norm[quantm_norm['abundance_norm_pa_like'] > 0]
                if len(quantm_norm) > 0:
                    quantm_norm['log_abundance'] = np.log10(quantm_norm['abundance_norm_pa_like'] + 1)
                    quantm_norm['rank_quantile'] = quantm_norm['log_abundance'].rank() / (len(quantm_norm) + 1)
                    quantm_norm['z_score_quantm_pa_like'] = norm.ppf(quantm_norm['rank_quantile'])
                    merged_data_pa_like = quantm_norm[['gene', 'z_score_quantm_pa_like', 'abundance_norm_pa_like']].merge(
                        pa_data[['gene', 'z_score', 'abundance', 'log_abundance']],
                        on='gene',
                        how='inner'
                    ).rename(columns={
                        'z_score': 'z_score_peptideatlas',
                        'abundance': 'abundance_peptideatlas',
                        'log_abundance': 'log_abundance_peptideatlas',
                    })
                    if len(merged_data_pa_like) > 0:
                        fig, ax = plt.subplots(figsize=(10, 8))
                        x = merged_data_pa_like['z_score_peptideatlas'].to_numpy(float)
                        y = merged_data_pa_like['z_score_quantm_pa_like'].to_numpy(float)
                        valid_mask = np.isfinite(x) & np.isfinite(y)
                        x = x[valid_mask]
                        y = y[valid_mask]
                        if len(x) > 0 and len(y) > 0:
                            min_overall = min(np.min(x), np.min(y))
                            x_shifted = x - min_overall
                            y_shifted = y - min_overall
                            hb = ax.hexbin(x_shifted, y_shifted, gridsize=50, cmap='YlOrRd', mincnt=1)
                            cb = plt.colorbar(hb, ax=ax)
                            cb.set_label('Number of Proteins', fontsize=12)
                            ax.scatter(x_shifted, y_shifted, alpha=0.3, s=10, c='black', edgecolors='none')
                            correlation = np.corrcoef(x_shifted, y_shifted)[0, 1]
                            n_proteins = len(x)
                            ax.text(
                                0.05, 0.95,
                                f'Pearson r = {correlation:.3f}\\nn = {n_proteins} proteins',
                                transform=ax.transAxes, fontsize=12, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                            )
                            ax.set_xlabel('PeptideAtlas z-score (shifted, min=0)', fontsize=12)
                            ax.set_ylabel('quantm z-score (PA-like normalized; shifted, min=0)', fontsize=12)
                            ax.set_title('Density Plot: quantm (PA-like normalization) vs PeptideAtlas', fontsize=14, fontweight='bold')
                            ax.grid(True, alpha=0.3)
                            plt.tight_layout()
                            filename_pa = f"density_plot_quantm_vs_peptideatlas_pa_like{suffix}.png"
                            plt.savefig(output_dir / filename_pa, bbox_inches='tight', dpi=300)
                            plt.close()
                            print(f"  Saved: {filename_pa}")
                        filename_pa_csv = f"density_plot_quantm_vs_peptideatlas_pa_like_data{suffix}.csv"
                        merged_data_pa_like.to_csv(output_dir / filename_pa_csv, index=False)
                        print(f"  Saved: {filename_pa_csv}")

                        # Additional check plot: remove two densest horizontal Y bands
                        # (top two most frequent z_score_quantm_pa_like values, rounded for tie grouping).
                        y_rounded = merged_data_pa_like['z_score_quantm_pa_like'].round(6)
                        top_bands = y_rounded.value_counts().head(2).index.tolist()
                        if len(top_bands) > 0:
                            filtered_no_bands = merged_data_pa_like[
                                ~y_rounded.isin(top_bands)
                            ].copy()
                            if len(filtered_no_bands) > 0:
                                fig2, ax2 = plt.subplots(figsize=(10, 8))
                                x2 = filtered_no_bands['z_score_peptideatlas'].to_numpy(float)
                                y2 = filtered_no_bands['z_score_quantm_pa_like'].to_numpy(float)
                                valid_mask2 = np.isfinite(x2) & np.isfinite(y2)
                                x2 = x2[valid_mask2]
                                y2 = y2[valid_mask2]
                                if len(x2) > 0 and len(y2) > 0:
                                    min_overall2 = min(np.min(x2), np.min(y2))
                                    x2_shifted = x2 - min_overall2
                                    y2_shifted = y2 - min_overall2
                                    hb2 = ax2.hexbin(x2_shifted, y2_shifted, gridsize=50, cmap='YlOrRd', mincnt=1)
                                    cb2 = plt.colorbar(hb2, ax=ax2)
                                    cb2.set_label('Number of Proteins', fontsize=12)
                                    ax2.scatter(x2_shifted, y2_shifted, alpha=0.3, s=10, c='black', edgecolors='none')
                                    correlation2 = np.corrcoef(x2_shifted, y2_shifted)[0, 1]
                                    n_proteins2 = len(x2)
                                    ax2.text(
                                        0.05, 0.95,
                                        f'Pearson r = {correlation2:.3f}\\nn = {n_proteins2} proteins\\nremoved bands: {top_bands}',
                                        transform=ax2.transAxes, fontsize=11, verticalalignment='top',
                                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85)
                                    )
                                    ax2.set_xlabel('PeptideAtlas z-score (shifted, min=0)', fontsize=12)
                                    ax2.set_ylabel('quantm z-score (PA-like; shifted, min=0)', fontsize=12)
                                    ax2.set_title('Density Plot: quantm (PA-like) vs PeptideAtlas\\nExcluding two densest horizontal lines', fontsize=14, fontweight='bold')
                                    ax2.grid(True, alpha=0.3)
                                    plt.tight_layout()
                                    filename_no_bands = f"density_plot_quantm_vs_peptideatlas_pa_like_without_2_dense_lines{suffix}.png"
                                    plt.savefig(output_dir / filename_no_bands, bbox_inches='tight', dpi=300)
                                    plt.close()
                                    print(f"  Saved: {filename_no_bands}")
                                filename_no_bands_csv = f"density_plot_quantm_vs_peptideatlas_pa_like_without_2_dense_lines_data{suffix}.csv"
                                filtered_no_bands.to_csv(output_dir / filename_no_bands_csv, index=False)
                                print(f"  Saved: {filename_no_bands_csv}")

                        # Additional variants by plasma-dataset support
                        ds_support = quantm_data[['gene', 'n_datasets_plasma']].drop_duplicates(subset=['gene'])
                        merged_with_support = merged_data_pa_like.merge(ds_support, on='gene', how='left')
                        support_variants = [
                            ("no_single_dataset", merged_with_support['n_datasets_plasma'] >= 2),
                            ("no_two_or_less_datasets", merged_with_support['n_datasets_plasma'] >= 3),
                        ]
                        for vname, vm in support_variants:
                            vdf = merged_with_support[vm].copy()
                            if len(vdf) == 0:
                                continue
                            xv = vdf['z_score_peptideatlas'].to_numpy(float)
                            yv = vdf['z_score_quantm_pa_like'].to_numpy(float)
                            valid_v = np.isfinite(xv) & np.isfinite(yv)
                            xv = xv[valid_v]
                            yv = yv[valid_v]
                            if len(xv) == 0:
                                continue
                            min_v = min(np.min(xv), np.min(yv))
                            xv_shift = xv - min_v
                            yv_shift = yv - min_v
                            figv, axv = plt.subplots(figsize=(10, 8))
                            hbv = axv.hexbin(xv_shift, yv_shift, gridsize=50, cmap='YlOrRd', mincnt=1)
                            cbv = plt.colorbar(hbv, ax=axv)
                            cbv.set_label('Number of Proteins', fontsize=12)
                            axv.scatter(xv_shift, yv_shift, alpha=0.3, s=10, c='black', edgecolors='none')
                            rv = np.corrcoef(xv_shift, yv_shift)[0, 1]
                            axv.text(
                                0.05, 0.95,
                                f'Pearson r = {rv:.3f}\\nn = {len(xv)} proteins',
                                transform=axv.transAxes, fontsize=12, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85)
                            )
                            axv.set_xlabel('PeptideAtlas z-score (shifted, min=0)', fontsize=12)
                            axv.set_ylabel('quantm z-score (PA-like; shifted, min=0)', fontsize=12)
                            axv.set_title(
                                f"Density Plot: quantm (PA-like) vs PeptideAtlas\n{vname.replace('_', ' ')}",
                                fontsize=14, fontweight='bold'
                            )
                            axv.grid(True, alpha=0.3)
                            plt.tight_layout()
                            fpng = f"density_plot_quantm_vs_peptideatlas_pa_like_{vname}{suffix}.png"
                            plt.savefig(output_dir / fpng, bbox_inches='tight', dpi=300)
                            plt.close()
                            print(f"  Saved: {fpng}")
                            fcsv = f"density_plot_quantm_vs_peptideatlas_pa_like_{vname}_data{suffix}.csv"
                            vdf.to_csv(output_dir / fcsv, index=False)
                            print(f"  Saved: {fcsv}")
            
            # Extract proteins at Y ≈ -2 (lowest abundance proteins in quantms)
            # Filter for proteins with z_score_quantm around -2 (horizontal line in plot)
            y_threshold_low = -2.5
            y_threshold_high = -1.5
            low_abundance_proteins = merged_data[
                (merged_data['z_score_quantm'] >= y_threshold_low) & 
                (merged_data['z_score_quantm'] <= y_threshold_high)
            ].copy()
            
            if len(low_abundance_proteins) > 0:
                # Sort by quantms abundance (lowest first)
                low_abundance_proteins = low_abundance_proteins.sort_values('abundance_quantm', ascending=True)
                
                # Select relevant columns
                output_columns = ['gene', 'abundance_quantm', 'log_abundance_quantm', 'z_score_quantm', 
                                 'abundance_peptideatlas', 'log_abundance_peptideatlas', 'z_score_peptideatlas']
                low_abundance_output = low_abundance_proteins[output_columns].copy()
                
                # Rename columns for clarity
                low_abundance_output.columns = ['Gene', 'Abundance_quantms', 'Log_Abundance_quantms', 
                                                'Z_Score_quantms', 'Abundance_PeptideAtlas', 
                                                'Log_Abundance_PeptideAtlas', 'Z_Score_PeptideAtlas']
                
                # Save to CSV
                low_abundance_filename = f"lowest_abundance_proteins_y_around_minus2{suffix}.csv"
                low_abundance_output.to_csv(output_dir / low_abundance_filename, index=False)
                print(f"  Saved: {low_abundance_filename} ({len(low_abundance_output)} proteins with z_score_quantm between {y_threshold_low} and {y_threshold_high})")
            else:
                print(f"  No proteins found with z_score_quantm between {y_threshold_low} and {y_threshold_high}")
    
    # PLOT 6: Scatter plot with PeptideAtlas as reference line and other databases as points
    print("\nCreating scatter plot with PeptideAtlas as reference and other databases on top...")
    pa_ref = normalized_data_sum[normalized_data_sum['source'] == 'PeptideAtlas'].copy()
    
    if len(pa_ref) > 0:
        # Order by PeptideAtlas z-score
        pa_ref = pa_ref.sort_values('z_score')
        pa_ref['order'] = range(1, len(pa_ref) + 1)
        gene_to_order_pa = dict(zip(pa_ref['gene'], pa_ref['order']))
        
        # Get other databases (excluding PeptideAtlas and quantm)
        other_sources = [s for s in panel_b_sources if s not in ['PeptideAtlas', 'quantm (Plasma/Serum)']]
        
        if len(other_sources) > 0:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Plot PeptideAtlas as reference line
            pa_ref_grouped = pa_ref.groupby('gene', as_index=False).agg({
                'z_score': 'median',
                'order': 'first'
            })
            pa_ref_grouped = pa_ref_grouped.sort_values('order')
            ax.plot(pa_ref_grouped['order'], pa_ref_grouped['z_score'],
                   color='#1a1a1a', linewidth=2.5, label='PeptideAtlas (reference)', alpha=0.9, zorder=10)
            
            # Plot other databases as scatter points
            for source in other_sources:
                source_data = normalized_data_sum[normalized_data_sum['source'] == source].copy()
                if len(source_data) > 0:
                    source_grouped = source_data.groupby('gene', as_index=False).agg({
                        'z_score': 'median'
                    })
                    source_grouped['order'] = source_grouped['gene'].map(gene_to_order_pa)
                    source_grouped = source_grouped[source_grouped['order'].notna()]
                    
                    if len(source_grouped) > 0:
                        ax.scatter(source_grouped['order'], source_grouped['z_score'],
                                 label=source, alpha=0.7, s=8, color=db_colors.get(source, 'gray'), zorder=5)
            
            # Also plot quantm
            quantm_data = normalized_data_sum[normalized_data_sum['source'] == 'quantm (Plasma/Serum)'].copy()
            if len(quantm_data) > 0:
                quantm_grouped = quantm_data.groupby('gene', as_index=False).agg({
                    'z_score': 'median'
                })
                quantm_grouped['order'] = quantm_grouped['gene'].map(gene_to_order_pa)
                quantm_grouped = quantm_grouped[quantm_grouped['order'].notna()]
                
                if len(quantm_grouped) > 0:
                    ax.scatter(quantm_grouped['order'], quantm_grouped['z_score'],
                             label='quantm (Plasma/Serum)', alpha=0.7, s=8, 
                             color=db_colors.get('quantm (Plasma/Serum)', '#9b59b6'), zorder=5)
            
            ax.set_xlabel('Protein Rank (by PeptideAtlas z-score)', fontsize=12)
            ax.set_ylabel('Quantile-normalized Values (z-score)', fontsize=12)
            ax.set_title('Protein Abundance: PeptideAtlas as Reference\n(PeptideAtlas as line, other databases as points)',
                        fontsize=14, fontweight='bold')
            ax.legend(title='Data Source', loc='best', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_xticks([])
            ax.set_xticklabels([])
            
            plt.tight_layout()
            filename = f"peptideatlas_reference_other_databases{suffix}.png"
            plt.savefig(output_dir / filename, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"  Saved: {filename}")

# ============================================
# MAIN ANALYSIS FUNCTION
# ============================================

def run_comparison():
    """Run database comparison analysis."""
    print("=" * 50)
    print("DATABASE COMPARISON")
    print("=" * 50)
    output_dir = output_base_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load mapping cache
    print("\nLoading protein-to-gene mapping cache...")
    mapping_cache = load_protein_mapping_cache()
    
    # Try to load from Parquet cache first (much faster)
    print("\nAttempting to load datasets from Parquet cache...")
    all_data = {}
    cache_hits = 0
    cache_misses = 0
    missing_datasets = []  # names of datasets to load from CSV (no cache or load failed)
    
    # Try to load plasma/serum dataset list from B_'s output for better identification
    plasma_serum_datasets_from_prep = set()
    prep_summary_file = data_prep_output_dir / "01_dataset_summary_before_and_after_filtering.csv"
    if prep_summary_file.exists():
        try:
            prep_summary = pd.read_csv(prep_summary_file)
            if "Dataset" in prep_summary.columns:
                if "Tissue_CellType" in prep_summary.columns:
                    for _, row in prep_summary.iterrows():
                        if str(row.get("Tissue_CellType", "")).strip() == "Blood Plasma/Serum":
                            plasma_serum_datasets_from_prep.add(str(row["Dataset"]).strip())
                elif "Condition" in prep_summary.columns:
                    for _, row in prep_summary.iterrows():
                        if is_blood_plasma_serum_condition(row["Condition"]):
                            plasma_serum_datasets_from_prep.add(str(row["Dataset"]).strip())
            if len(plasma_serum_datasets_from_prep) > 0:
                print(
                    f"  Found {len(plasma_serum_datasets_from_prep)} Blood Plasma/Serum dataset(s) "
                    f"from 01_dataset_summary (aligned with B_/E_)"
                )
        except Exception as e:
            print(f"  Warning: Could not load dataset list from data preparation: {e}")
    
    # Check if we can use Parquet cache
    try:
        import pyarrow.parquet as pq
        PARQUET_AVAILABLE = True
    except ImportError:
        PARQUET_AVAILABLE = False
        print("  Note: pyarrow not available, will load from CSV")
    
    if PARQUET_AVAILABLE:
        # First, try to load datasets identified from B_'s output
        if len(plasma_serum_datasets_from_prep) > 0:
            for dataset_name in plasma_serum_datasets_from_prep:
                cache_file = find_cache_file(dataset_name)
                if cache_file and cache_file.exists():
                    try:
                        processed = pd.read_parquet(cache_file)
                        condition = (
                            str(processed["Condition"].iloc[0]) if len(processed) > 0 and "Condition" in processed.columns else ""
                        )
                        # Safety check: even prep-listed datasets must pass plasma/serum condition.
                        if len(processed) > 0 and is_blood_plasma_serum_condition(condition):
                            all_data[dataset_name] = processed
                            cache_hits += 1
                        else:
                            cache_misses += 1
                            missing_datasets.append(dataset_name)
                    except Exception as e:
                        print(f"  Warning: Error loading {dataset_name} from cache: {e}")
                        cache_misses += 1
                        missing_datasets.append(dataset_name)
                else:
                    cache_misses += 1
                    missing_datasets.append(dataset_name)
        
        # Also check CSV files for any datasets not in prep output
        csv_files = list(msstats_dir.glob("*.sdrf_openms_design_msstats_in.csv"))
        for f in csv_files:
            dataset_name = f.name.replace(".sdrf_openms_design_msstats_in.csv", "")
            
            # Skip if already loaded
            if dataset_name in all_data:
                continue
            
            cache_file = find_cache_file(dataset_name)
            
            if cache_file and cache_file.exists():
                try:
                    # Load from Parquet cache (protein-level cache contains proteins, not genes)
                    processed = pd.read_parquet(cache_file)
                    
                    # Cache already contains filtered proteins (not normalized to genes), no need to re-process
                    condition = (
                        str(processed["Condition"].iloc[0]) if len(processed) > 0 and "Condition" in processed.columns else ""
                    )
                    if is_blood_plasma_serum_condition(condition) and len(processed) > 0:
                        all_data[dataset_name] = processed
                        cache_hits += 1
                except Exception as e:
                    print(f"  Warning: Error loading {dataset_name} from cache: {e}")
                    cache_misses += 1
                    missing_datasets.append(dataset_name)
            else:
                cache_misses += 1
                missing_datasets.append(dataset_name)
        
        if cache_hits > 0:
            print(f"  Cache: {cache_hits} dataset(s) loaded from cache")
        if cache_misses > 0:
            print(f"  Cache: {cache_misses} dataset(s) not in cache, will load from CSV")
    
    # Load from CSV only the datasets that missed cache (keep cached data for the rest)
    if cache_misses > 0 or len(all_data) == 0:
        if cache_misses > 0:
            print(f"  Loading {len(missing_datasets)} missing dataset(s) from CSV...")
            for dataset_name in sorted(set(missing_datasets)):
                data = load_dataset(dataset_name, msstats_dir)
                if data is not None and len(data) > 0:
                    condition = str(data["Condition"].iloc[0]) if "Condition" in data.columns else ""
                    if is_blood_plasma_serum_condition(condition):
                        all_data[dataset_name] = data
        if len(all_data) == 0:
            print("  No data loaded from cache, loading all datasets from CSV...")
            all_data_csv = load_all_msstats_data()
            all_data.update(all_data_csv)
    
    if len(all_data) == 0:
        print("  Error: No msstats data loaded after trying cache and CSV")
        return
    
    if len(all_data) == 0:
        print("  Error: No msstats data loaded")
        return
    
    # Load entry name mapping library for protein normalization
    print("\nLoading entry name mapping library...")
    entry_name_library = load_entry_name_mapping_library()
    manual_mapping = load_manual_mapping()
    entry_name_library = {**entry_name_library, **manual_mapping}  # Merge manual mapping
    print(f"  Total entry name mappings available: {len(entry_name_library)}")
    
    # Load msstats proteins from cache (they are already proteins, not genes)
    # Get unique protein identifiers directly from cached data
    msstats_proteins_raw = set()
    for dataset_name, data in all_data.items():
        # Data from protein-level cache contains proteins (not genes)
        # Get unique protein identifiers directly
        proteins = data['Protein'].dropna().unique()
        for protein in proteins:
            protein_str = str(protein).strip()
            if protein_str and protein_str.lower() not in ["", "nan", "none", "na"]:
                # Proteins are already filtered (no decoy/entrap) in cache
                msstats_proteins_raw.add(protein_str)
    
    print(f"  Found {len(msstats_proteins_raw)} unique proteins from msstats plasma/serum datasets (from protein-level cache)")
    
    # Normalize quantms proteins to UniProt accessions
    # Note: Proteins from B_data_preparation cache are already normalized to UniProt accessions
    # We normalize again to handle any old cache files and ensure consistency
    print("  Normalizing quantms proteins to UniProt accessions...")
    print("    Note: Proteins from B_data_preparation cache are pre-normalized, but re-normalizing for consistency")
    msstats_proteins_normalized = normalize_protein_set_for_comparison(msstats_proteins_raw, entry_name_library=entry_name_library)
    print(f"  Normalized to {len(msstats_proteins_normalized)} unique UniProt accessions")
    
    # Match B_ parquet reconciliation / E_: restrict to canonical Blood Plasma/Serum list if present
    proteins_02_path = data_prep_output_dir / "02_blood_plasma_serum_proteins.txt"
    if proteins_02_path.exists():
        try:
            with open(proteins_02_path, "r", encoding="utf-8") as f:
                canon_plasma = {line.strip() for line in f if line.strip()}
            n_before_02 = len(msstats_proteins_normalized)
            msstats_proteins_normalized = msstats_proteins_normalized & canon_plasma
            print(
                f"  Intersected quantm set with B_ 02 Blood Plasma/Serum list: "
                f"{len(msstats_proteins_normalized)} proteins (was {n_before_02} before intersect)"
            )
        except Exception as e:
            print(f"  Warning: could not apply 02 Blood Plasma/Serum protein list: {e}")
    
    # Load external databases from individual CSV files
    print("\nLoading external databases from individual CSV files...")
    external_databases_result = load_all_external_databases(mapping_cache=mapping_cache, entry_name_library=entry_name_library)
    external_databases_proteins = external_databases_result['proteins']
    external_databases_genes = external_databases_result['genes']
    
    # Combine protein lists for main analysis (using normalized proteins)
    protein_lists = {
        'quantm (Plasma/Serum)': msstats_proteins_normalized
    }
    protein_lists.update(external_databases_proteins)
    
    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY (PROTEIN-BASED ANALYSIS)")
    print("=" * 50)
    for db_name, proteins in protein_lists.items():
        print(f"{db_name:30s}: {len(proteins):6d} proteins")
    
    # Create UpSet plot using proteins (only plot, no CSV files)
    create_upset_plot(protein_lists, output_dir, filename="proteome_comparison_upset_proteins.png",
                     title="Proteome Comparison: quantm (Plasma/Serum) vs External Databases (Protein-based)",
                     min_subset_size=175)
    
    # Load external databases with abundance
    print("\n" + "=" * 50)
    print("PROTEIN ABUNDANCE ANALYSIS")
    print("=" * 50)
    print("\nLoading external databases with abundance data...")
    external_data_abundance = []
    
    # PeptideAtlas
    peptideatlas_file = external_data_dir / "peptideatlas.csv"
    if peptideatlas_file.exists():
        pa_data = load_external_database_with_abundance(peptideatlas_file, "PeptideAtlas", 
                                                       gene_column='biosequence_gene_name', abundance_column=None,
                                                       mapping_cache=mapping_cache)
        if len(pa_data) > 0:
            external_data_abundance.append(pa_data)
    
    # HPA MS
    hpa_ms_file = external_data_dir / "hpa_ms.csv"
    if hpa_ms_file.exists():
        hpa_data = load_external_database_with_abundance(hpa_ms_file, "HPA MS", 
                                                      gene_column='gene', abundance_column='Concentration',
                                                      mapping_cache=mapping_cache)
        if len(hpa_data) > 0:
            external_data_abundance.append(hpa_data)
    
    # PAXDB (uses 'Unnamed: 2' column for abundance, not 'Concentration')
    paxdb_plasma_file = external_data_dir / "paxdb_plasma.csv"
    if paxdb_plasma_file.exists():
        paxdb_data = load_external_database_with_abundance(paxdb_plasma_file, "PAXDB",
                                                            gene_column='Entry', abundance_column='Unnamed: 2',
                                                            mapping_cache=mapping_cache)
        if len(paxdb_data) > 0:
            external_data_abundance.append(paxdb_data)
    
    # GPMDB
    gpmdb_file = external_data_dir / "gpmdb.csv"
    if gpmdb_file.exists():
        gpmdb_data = load_external_database_with_abundance(gpmdb_file, "GPMDB Plasma",
                                                       gene_column='gene', abundance_column='Concentration',
                                                       mapping_cache=mapping_cache)
        if len(gpmdb_data) > 0:
            external_data_abundance.append(gpmdb_data)
    
    # HPA PEA
    hpa_pea_file = external_data_dir / "hpa_pea.csv"
    if hpa_pea_file.exists():
        hpa_pea_data = load_external_database_with_abundance(hpa_pea_file, "HPA PEA",
                                                       gene_column='gene', abundance_column='Concentration',
                                                       mapping_cache=mapping_cache)
        if len(hpa_pea_data) > 0:
            external_data_abundance.append(hpa_pea_data)
    
    # HPA Immunoassay
    hpa_immuno_file = external_data_dir / "hpa_immunoassay_plasma.csv"
    if hpa_immuno_file.exists():
        hpa_immuno_data = load_external_database_with_abundance(hpa_immuno_file, "HPA Immunoassay",
                                                       gene_column='gene', abundance_column='Concentration',
                                                       mapping_cache=mapping_cache)
        if len(hpa_immuno_data) > 0:
            external_data_abundance.append(hpa_immuno_data)
    
    # Load msstats abundance data (all proteins)
    print("\nLoading msstats plasma/serum abundance data (all proteins)...")
    msstats_abundance_all = load_msstats_abundance_data(all_data, mapping_cache, min_samples=0)
    
    # Define colors
    db_colors = {
        'PeptideAtlas': '#1a1a1a',
        'HPA MS': '#3498db',
        'PAXDB': '#e74c3c',
        'GPMDB Plasma': '#2ecc71',
        'quantm (Plasma/Serum)': '#9b59b6'
    }
    
    panel_b_sources = ['PeptideAtlas', 'HPA MS', 'PAXDB', 'GPMDB Plasma', 'quantm (Plasma/Serum)']
    
    # ===== ANALYSIS 1: All proteins =====
    print("\n" + "=" * 50)
    print("ANALYSIS 1: All proteins")
    print("=" * 50)
    
    external_data_abundance_all = []
    for data in external_data_abundance:
        if data['source'].iloc[0] != 'quantm (Plasma/Serum)':
            external_data_abundance_all.append(data)
    if len(msstats_abundance_all) > 0:
        external_data_abundance_all.append(msstats_abundance_all)
    
    if len(external_data_abundance_all) == 0:
        print("  Error: No abundance data loaded")
        return
    
    # Combine all external data
    all_external_df = pd.concat(external_data_abundance_all, ignore_index=True)
    
    # Apply quantile normalization
    print("\nApplying quantile normalization...")
    normalized_data_sum = apply_quantile_normalization(all_external_df)
    
    # Filter to only databases we want for Panel B
    normalized_data_sum = normalized_data_sum[normalized_data_sum['source'].isin(panel_b_sources)]
    
    # Create all abundance plots (all proteins)
    create_abundance_plots(normalized_data_sum, output_dir, panel_b_sources, db_colors, suffix="")
    
    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50)
    print(f"Results saved in: {output_dir}")

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    # Run database comparison analysis
    run_comparison()




