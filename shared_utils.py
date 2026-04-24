"""
Shared utilities for proteomics analysis scripts.
Contains common functions for protein filtering, normalization, and data loading.
"""

import json
import pandas as pd
import numpy as np
import re
from pathlib import Path

FEATURE_COUNT_COLUMNS = [
    "PrecursorCharge",
    "FragmentIon",
    "ProductCharge",
    "IsotopeLabelType",
]
PEPTIDE_COLUMN_CANDIDATES = [
    "PeptideSequence",
    "Peptide",
    "Sequence",
]

# Cache for case-insensitive mapping lookup (avoids O(n) scan per protein)
_mapping_cache_upper = {}


def _pick_matching_column(df, candidates):
    if df is None:
        return None
    cols = list(df.columns)
    lower_map = {str(c).lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    for c in cols:
        cl = str(c).lower()
        if "peptide" in cl and "sequence" in cl:
            return c
    return None


def compute_protein_feature_counts(df, protein_col="Protein", feature_cols=None, peptide_col=None):
    """Compute unique feature-combination counts per protein.

    A feature is the tuple:
    (PrecursorCharge, FragmentIon, ProductCharge, IsotopeLabelType)

    If any of those columns is missing, falls back to row counts per protein.
    """
    if df is None or len(df) == 0 or protein_col not in df.columns:
        return {}

    cols = feature_cols if feature_cols is not None else FEATURE_COUNT_COLUMNS
    peptide_col = peptide_col or _pick_matching_column(df, PEPTIDE_COLUMN_CANDIDATES)
    available_cols = [c for c in cols if c in df.columns]
    dedup_fields = [protein_col]
    if peptide_col and peptide_col in df.columns:
        dedup_fields.append(peptide_col)
    dedup_fields.extend(available_cols)
    base = df[dedup_fields].copy()
    base[protein_col] = base[protein_col].astype(str)
    base = base[base[protein_col].str.strip().ne("")]
    base = base[base[protein_col].str.lower().ne("nan")]

    if len(base) == 0:
        return {}

    if peptide_col and peptide_col in base.columns:
        base[peptide_col] = base[peptide_col].astype(str).str.strip()
        base = base[base[peptide_col].ne("")]
        base = base[base[peptide_col].str.lower().ne("nan")]

    if len(available_cols) < len(cols):
        return base.groupby(protein_col).size().astype(int).to_dict()

    for c in available_cols:
        base[c] = base[c].fillna("__NA__").astype(str).str.strip()

    dedup = base.drop_duplicates(subset=dedup_fields)
    return dedup.groupby(protein_col).size().astype(int).to_dict()

def _get_mapping_cache_upper(mapping_cache):
    """Build or return upper-case key lookup for mapping_cache (one-time per cache)."""
    if not mapping_cache:
        return {}
    cid = id(mapping_cache)
    if cid not in _mapping_cache_upper:
        _mapping_cache_upper[cid] = {str(k).upper(): v for k, v in mapping_cache.items()}
    return _mapping_cache_upper[cid]

# ============================================
# PROTEIN FILTERING AND NORMALIZATION FUNCTIONS
# ============================================

def extract_uniprot_id(protein_name):
    """Extract UniProt ID from protein name formats.
    
    DDA format: 'sp|P02768|ALB_HUMAN;sp|P01615|KVD28_HUMAN'
    - Split on ';', keep first entry
    - Extract UniProt accession (string between first and second '|')
    
    DIA format: 'ACTB_HUMAN;ACTG_HUMAN' or 'P02768'
    - For entry names, use convert_entry_name_to_accession()
    - For direct accession, return as-is
    
    Args:
        protein_name: Protein identifier string
    
    Returns:
        UniProt accession (e.g., 'P02768') or None
    """
    if not protein_name or pd.isna(protein_name):
        return None
    
    protein_str = str(protein_name).strip()
    
    # Handle protein groups (split on ';' and take first)
    if ';' in protein_str:
        protein_str = protein_str.split(';')[0].strip()
    
    # DDA format: sp|P02768|ALB_HUMAN or tr|Q6GZX4|Q6GZX4_HUMAN
    if '|' in protein_str:
        parts = protein_str.split('|')
        if len(parts) >= 2:
            # Extract the UniProt ID (second part, between first and second |)
            uniprot_id = parts[1].strip()
            # Remove isoform suffix if present (e.g., P02768-1 -> P02768)
            uniprot_id = uniprot_id.split('-')[0]
            return uniprot_id
    
    # Already a UniProt ID format? (e.g., P02768)
    if protein_str and len(protein_str) >= 6 and protein_str[0].isalpha() and protein_str[1].isdigit():
        # Remove isoform suffix if present
        uniprot_id = protein_str.split('-')[0]
        return uniprot_id
    
    return None

def load_protein_mapping_cache(cache_file=None):
    """Load protein to gene mapping cache."""
    if cache_file is None:
        cache_file = Path(r"C:\Users\HP zBook 15v\Documents\ebi-work\blood-review-data\data\cache\protein_to_gene_mappings.csv")
    
    if not cache_file.exists():
        print(f"  Warning: Mapping cache not found at {cache_file}")
        return {}
    
    try:
        df = pd.read_csv(cache_file)
        # Create mapping dictionary (protein_id -> gene_symbol)
        mapping = {}
        for _, row in df.iterrows():
            if pd.notna(row['protein_id']) and pd.notna(row['gene_symbol']):
                if row['mapping_status'] == 'success':
                    mapping[str(row['protein_id']).strip()] = str(row['gene_symbol']).strip().upper()
        print(f"  Loaded {len(mapping)} protein-to-gene mappings from cache")
        return mapping
    except Exception as e:
        print(f"  Warning: Error loading mapping cache: {e}")
        return {}

def build_entry_name_to_accession_mapping(mapping_cache):
    """Build a mapping from UniProt entry names (e.g., 'ACTB_HUMAN') to UniProt accessions (e.g., 'P60709').
    
    This is done by parsing DDA-format entries in the mapping cache which have the format:
    'sp|ACCESSION|ENTRY_NAME' or 'tr|ACCESSION|ENTRY_NAME'
    
    Args:
        mapping_cache: Dictionary mapping protein_id -> gene_symbol
    
    Returns:
        Dictionary mapping entry_name -> accession
    """
    entry_name_to_accession = {}
    
    for protein_id in mapping_cache.keys():
        # Check if it's a DDA format: sp|ACCESSION|ENTRY_NAME
        if '|' in str(protein_id):
            parts = str(protein_id).split('|')
            if len(parts) >= 3:
                accession = parts[1].strip()
                entry_name = parts[2].strip()
                # Handle protein groups - split on ';' and process each
                if ';' in entry_name:
                    entry_name = entry_name.split(';')[0].strip()
                # Remove isoform suffix from accession if present
                accession = accession.split('-')[0]
                # Store mapping (entry_name -> accession)
                if entry_name and accession:
                    entry_name_to_accession[entry_name] = accession
    
    return entry_name_to_accession

def convert_entry_name_to_accession(entry_name, entry_name_mapping):
    """Convert UniProt entry name (e.g., 'ACTB_HUMAN') to UniProt accession (e.g., 'P60709').
    
    Args:
        entry_name: UniProt entry name (e.g., 'ACTB_HUMAN')
        entry_name_mapping: Dictionary mapping entry_name -> accession
    
    Returns:
        UniProt accession or None if not found
    """
    if not entry_name or pd.isna(entry_name):
        return None
    
    entry_name_str = str(entry_name).strip()
    
    # Handle protein groups (split on ';' and take first)
    if ';' in entry_name_str:
        entry_name_str = entry_name_str.split(';')[0].strip()
    
    # Try multiple lookup strategies
    # 1. Exact match (case-sensitive)
    if entry_name_str in entry_name_mapping:
        return entry_name_mapping[entry_name_str]
    
    # 2. Case-insensitive match (uppercase)
    entry_upper = entry_name_str.upper()
    if entry_upper in entry_name_mapping:
        return entry_name_mapping[entry_upper]
    
    # 3. Case-insensitive match (lowercase)
    entry_lower = entry_name_str.lower()
    if entry_lower in entry_name_mapping:
        return entry_name_mapping[entry_lower]
    
    # 4. Try without _HUMAN suffix if present
    if entry_name_str.endswith('_HUMAN'):
        entry_without = entry_name_str[:-6]  # Remove '_HUMAN'
        if entry_without in entry_name_mapping:
            return entry_name_mapping[entry_without]
        if entry_without.upper() in entry_name_mapping:
            return entry_name_mapping[entry_without.upper()]
        if entry_without.lower() in entry_name_mapping:
            return entry_name_mapping[entry_without.lower()]
    
    # 5. Try with _HUMAN suffix if not present
    if not entry_name_str.endswith('_HUMAN'):
        entry_with = entry_name_str + '_HUMAN'
        if entry_with in entry_name_mapping:
            return entry_name_mapping[entry_with]
        if entry_with.upper() in entry_name_mapping:
            return entry_name_mapping[entry_with.upper()]
        if entry_with.lower() in entry_name_mapping:
            return entry_name_mapping[entry_with.lower()]
    
    return None

def convert_protein_to_gene(protein_id, mapping_cache):
    """Convert protein ID to gene symbol using cache."""
    # Try direct lookup
    if protein_id in mapping_cache:
        return mapping_cache[protein_id]
    
    # Try extracting UniProt ID first
    uniprot_id = extract_uniprot_id(protein_id)
    if uniprot_id and uniprot_id in mapping_cache:
        return mapping_cache[uniprot_id]
    
    # If it looks like a gene symbol already, return it
    if protein_id and len(protein_id) <= 15 and protein_id.replace('_', '').replace('-', '').isalnum():
        # Check if it's not a UniProt pattern
        if not (protein_id[0].isalpha() and len(protein_id) >= 6 and protein_id[1].isdigit()):
            return protein_id.upper()
    
    return None

def is_decoy_entrap_protein(protein_name):
    """Check if a protein is a Decoy or Entrap protein."""
    if pd.isna(protein_name):
        return True
    protein_str = str(protein_name).strip().upper()
    # Check for various decoy/entrap patterns
    decoy_patterns = ['DECOY', 'ENTRAP', 'REV__', 'CON__']
    return any(pattern in protein_str for pattern in decoy_patterns)

def is_non_human_protein(protein_name):
    """Check if a protein is from a non-human organism (e.g., BOVINE, MOUSE, etc.)."""
    if pd.isna(protein_name):
        return False
    protein_str = str(protein_name).strip().upper()
    # Check for non-human organism suffixes
    non_human_patterns = ['_BOVINE', '_MOUSE', '_RAT', '_PIG', '_CHICKEN', '_YEAST', '_ECOLI']
    return any(pattern in protein_str for pattern in non_human_patterns)

def normalize_protein_to_gene(protein_name, mapping_cache=None):
    """Normalize protein name to gene name.
    
    Handles different formats:
    - DDA format (sp|P98160|PGBM_HUMAN): Extract gene name after last |, remove _HUMAN suffix → PGBM
    - DIA format (K1C10_HUMAN): Extract gene name before _HUMAN → K1C10
    - ENSP IDs (ENSP00000299370): Try to map using mapping_cache
    - UniProt IDs (P02768): Try to map using mapping_cache
    - Other formats: Try to extract gene name
    
    Args:
        protein_name: Protein identifier to normalize
        mapping_cache: Optional dictionary mapping protein IDs to gene symbols
    
    Returns:
        Normalized gene name or None if extraction/conversion fails.
    """
    if pd.isna(protein_name) or not protein_name:
        return None
    
    protein_str = str(protein_name).strip()
    
    # Handle DECOY/ENTRAP proteins - return None (will be filtered)
    if is_decoy_entrap_protein(protein_str):
        return None
    
    # Check for ENSP IDs (Ensembl protein IDs) - format: ENSP followed by 11 digits
    if re.match(r'^ENSP\d{11}$', protein_str, re.IGNORECASE):
        if mapping_cache:
            if protein_str in mapping_cache:
                gene = mapping_cache[protein_str]
                if gene and str(gene).strip():
                    return str(gene).strip().upper()
            gene = _get_mapping_cache_upper(mapping_cache).get(protein_str.upper())
            if gene and str(gene).strip():
                return str(gene).strip().upper()
        return None
    
    # Check for UniProt IDs (format: starts with letter, then digits, 6-10 chars)
    # Examples: P02768, Q6GZX4
    if re.match(r'^[A-Z]\d{5,9}$', protein_str, re.IGNORECASE):
        if mapping_cache:
            if protein_str in mapping_cache:
                gene = mapping_cache[protein_str]
                if gene and str(gene).strip():
                    return str(gene).strip().upper()
            gene = _get_mapping_cache_upper(mapping_cache).get(protein_str.upper())
            if gene and str(gene).strip():
                return str(gene).strip().upper()
    
    # Format 1: sp|ACCESSION|GENE_HUMAN or tr|ACCESSION|GENE_HUMAN (DDA format)
    # PRIORITY: Extract gene name from identifier first (for consistency with DIA format)
    if '|' in protein_str:
        # Handle multiple proteins separated by semicolon (DDA format with multiple entries)
        if ';' in protein_str:
            # Split by semicolon first, then process each protein separately
            # Take the first protein entry
            first_protein = protein_str.split(';')[0].strip()
            if '|' in first_protein:
                parts = first_protein.split('|')
                if len(parts) >= 3:
                    # FIRST: Extract gene name from identifier
                    gene_part = parts[-1]
                    if '_HUMAN' in gene_part:
                        gene = gene_part.split('_HUMAN')[0].upper()
                        if len(gene) >= 2 and len(gene) <= 20 and gene.replace('_', '').replace('-', '').isalnum():
                            return gene
                    
                    # SECOND: If extraction didn't work, try mapping cache as fallback
                    if len(parts) >= 2 and mapping_cache:
                        uniprot_id = parts[1].strip()
                        if uniprot_id in mapping_cache:
                            gene = mapping_cache[uniprot_id]
                            if gene and str(gene).strip():
                                return str(gene).strip().upper()
        
        # Single protein entry (or first entry already processed above)
        parts = protein_str.split('|')
        if len(parts) >= 3:
            # FIRST: Extract gene name from identifier (prioritize consistency with DIA format)
            gene_part = parts[-1]
            # Handle case where gene_part might contain semicolon (multiple proteins)
            if ';' in gene_part:
                gene_part = gene_part.split(';')[0].strip()
            # Remove _HUMAN suffix for matching
            if '_HUMAN' in gene_part:
                gene = gene_part.split('_HUMAN')[0].upper()
                # Validate it looks like a gene symbol
                if len(gene) >= 2 and len(gene) <= 20 and gene.replace('_', '').replace('-', '').isalnum():
                    return gene
            else:
                gene = gene_part.upper()
                # Validate it looks like a gene symbol
                if len(gene) >= 2 and len(gene) <= 20 and gene.replace('_', '').replace('-', '').isalnum():
                    return gene
            
            # SECOND: If extraction from identifier didn't work, try mapping cache as fallback
            if len(parts) >= 2 and mapping_cache:
                uniprot_id = parts[1].strip()
                if uniprot_id in mapping_cache:
                    gene = mapping_cache[uniprot_id]
                    if gene and str(gene).strip():
                        return str(gene).strip().upper()
        elif len(parts) == 2:
            # sp|ACCESSION format - try mapping the accession first
            uniprot_id = parts[1].strip()
            if mapping_cache and uniprot_id in mapping_cache:
                gene = mapping_cache[uniprot_id]
                if gene and str(gene).strip():
                    return str(gene).strip().upper()
            # Fallback: return accession (might be a valid identifier)
            return uniprot_id.upper()
    
    # Format 2: GENE_HUMAN or GENE1_HUMAN;GENE2_HUMAN (DIA format)
    if '_HUMAN' in protein_str:
        # Handle multiple genes separated by semicolon
        if ';' in protein_str:
            # Take first gene
            first_gene = protein_str.split(';')[0]
            if '_HUMAN' in first_gene:
                gene = first_gene.split('_HUMAN')[0].upper()
                # Validate it looks like a gene symbol
                if len(gene) >= 2 and len(gene) <= 20 and gene.replace('_', '').replace('-', '').isalnum():
                    return gene
        else:
            gene = protein_str.split('_HUMAN')[0].upper()
            # Validate it looks like a gene symbol
            if len(gene) >= 2 and len(gene) <= 20 and gene.replace('_', '').replace('-', '').isalnum():
                return gene
    
    # Format 3: Check for date-like patterns (Excel formatting issues)
    if re.match(r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{2}$', protein_str, re.IGNORECASE):
        return None
    
    # Format 4: Check for numeric codes like "1433B", "1433E"
    if re.match(r'^\d+[A-Z]$', protein_str, re.IGNORECASE):
        # Try mapping cache first
        if mapping_cache and protein_str in mapping_cache:
            gene = mapping_cache[protein_str]
            if gene and str(gene).strip():
                return str(gene).strip().upper()
        # If no mapping, check if it's a known pattern
        if len(protein_str) >= 3 and len(protein_str) <= 10:
            return protein_str.upper()
    
    # Format 5: Just gene name or other format
    # Try to extract before common separators
    for sep in ['|', '-', '_', ';']:
        if sep in protein_str:
            parts = protein_str.split(sep)
            # Prefer parts that look like gene names (short, uppercase)
            for part in parts:
                part_clean = part.strip().upper()
                # More lenient validation for gene symbols
                if len(part_clean) >= 2 and len(part_clean) <= 20:
                    # Check if it's alphanumeric (allow some special chars for isoforms)
                    if part_clean.replace('_', '').replace('-', '').isalnum():
                        # Don't return if it looks like a date
                        if not re.match(r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{2}$', part_clean, re.IGNORECASE):
                            return part_clean
    
    # Fallback: return uppercase version (but check it's not a decoy or date)
    result = protein_str.upper().strip()
    if is_decoy_entrap_protein(result):
        return None
    # Check if it's a date pattern
    if re.match(r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{2}$', result, re.IGNORECASE):
        return None
    # Validate length
    if len(result) >= 2 and len(result) <= 50:
        return result
    
    return None

def filter_and_normalize_proteins(df, protein_column='Protein', inplace=False, mapping_cache=None):
    """Filter out Decoy/Entrap proteins and normalize protein names to gene names.
    
    Args:
        df: DataFrame with protein data
        protein_column: Name of column containing protein identifiers
        inplace: If True, modify df in place; if False, return new DataFrame
        mapping_cache: Optional dictionary mapping protein IDs to gene symbols
    
    Returns:
        DataFrame with filtered and normalized proteins
    """
    if not inplace:
        df = df.copy()
    
    # Guard: avoid iloc on empty columns
    if len(df.columns) == 0:
        return df
    
    # Resolve protein column (handles duplicate CSV headers or different column names)
    if protein_column not in df.columns:
        for alt in ['Protein', 'ProteinName', 'protein', 'proteinname']:
            if alt in df.columns:
                protein_column = alt
                break
        else:
            if len(df.columns) > 0:
                protein_column = df.columns[0]
            else:
                raise KeyError("No protein column found; dataframe has no columns")
    
    # Use column by position so we always have a valid reference (handles duplicate column names)
    loc = df.columns.get_loc(protein_column)
    if isinstance(loc, np.ndarray):
        protein_col_index = int(loc.flat[0]) if loc.size > 0 else 0
    elif isinstance(loc, slice):
        protein_col_index = int(getattr(loc, 'start', 0))
    else:
        protein_col_index = int(loc)
    protein_col_index = max(0, min(protein_col_index, len(df.columns) - 1))
    # Final guard: never use out-of-bounds index (e.g. empty frame edge case)
    if protein_col_index >= len(df.columns):
        protein_col_index = 0
    
    # Step 1: Filter out Decoy/Entrap proteins
    n_before = len(df)
    protein_str = df.iloc[:, protein_col_index].astype(str)
    decoy_mask = protein_str.apply(is_decoy_entrap_protein)
    df = df[~decoy_mask].copy()
    n_after_decoy = len(df)
    n_filtered_decoy = n_before - n_after_decoy
    
    if n_filtered_decoy > 0:
        print(f"  Filtered out {n_filtered_decoy} Decoy/Entrap proteins ({n_before} -> {n_after_decoy})")
    
    # Step 2: Normalize protein names to gene names (once per unique protein for speed)
    unique_proteins = df.iloc[:, protein_col_index].dropna().astype(str).str.strip().unique()
    problematic_formats = {'ENSP_IDs': [], 'date_like': [], 'numeric_codes': [], 'unmapped': []}
    protein_to_gene = {}
    for protein_name in unique_proteins:
        if not protein_name or str(protein_name).lower() in ['', 'nan', 'none', 'na']:
            continue
        gene = normalize_protein_to_gene(protein_name, mapping_cache=mapping_cache)
        if gene is not None:
            protein_to_gene[protein_name] = gene
        else:
            original = str(protein_name).strip()
            if re.match(r'^ENSP\d{11}$', original, re.IGNORECASE):
                problematic_formats['ENSP_IDs'].append(original)
            elif re.match(r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{2}$', original, re.IGNORECASE):
                problematic_formats['date_like'].append(original)
            elif re.match(r'^\d+[A-Z]$', original, re.IGNORECASE):
                problematic_formats['numeric_codes'].append(original)
            else:
                problematic_formats['unmapped'].append(original)
    df.iloc[:, protein_col_index] = df.iloc[:, protein_col_index].astype(str).str.strip().map(protein_to_gene)
    
    # Step 3: Remove rows where normalization returned None (invalid proteins)
    n_before_norm = len(df)
    df = df[df.iloc[:, protein_col_index].notna()].copy()
    n_after_norm = len(df)
    n_filtered_norm = n_before_norm - n_after_norm
    
    if n_filtered_norm > 0:
        print(f"  Filtered out {n_filtered_norm} proteins that could not be normalized ({n_before_norm} -> {n_after_norm})")
        
        # Print diagnostics for problematic formats
        if problematic_formats['ENSP_IDs']:
            unique_ensp = list(set(problematic_formats['ENSP_IDs']))[:10]
            print(f"    ENSP IDs (Ensembl protein IDs) that couldn't be mapped: {len(set(problematic_formats['ENSP_IDs']))} unique")
            print(f"      Examples: {unique_ensp[:5]}")
            if mapping_cache:
                print(f"      Note: Mapping cache available ({len(mapping_cache)} entries), but these IDs not found")
            else:
                print(f"      Note: No mapping cache available - install/update mapping cache to convert these")
        
        if problematic_formats['date_like']:
            unique_dates = list(set(problematic_formats['date_like']))[:10]
            print(f"    Date-like strings (likely Excel formatting issues): {len(set(problematic_formats['date_like']))} unique")
            print(f"      Examples: {unique_dates[:5]}")
            print(f"      Note: These are likely data corruption and should be filtered")
        
        if problematic_formats['numeric_codes']:
            unique_codes = list(set(problematic_formats['numeric_codes']))[:10]
            print(f"    Numeric codes (e.g., 1433B, 1433E): {len(set(problematic_formats['numeric_codes']))} unique")
            print(f"      Examples: {unique_codes[:5]}")
            print(f"      Note: These might be valid identifiers but need mapping to gene symbols")
        
        if problematic_formats['unmapped']:
            unique_unmapped = list(set(problematic_formats['unmapped']))[:10]
            print(f"    Other unmapped formats: {len(set(problematic_formats['unmapped']))} unique")
            print(f"      Examples: {unique_unmapped[:5]}")
    
    # Ensure output has 'Protein' column for callers (e.g. F_ load_dataset)
    actual_name = df.columns[protein_col_index]
    if actual_name != 'Protein':
        df = df.rename(columns={actual_name: 'Protein'})
    return df

def filter_proteins_only(df, protein_column='Protein', inplace=False, verbose=True):
    """Filter out Decoy/Entrap proteins but keep proteins as proteins (don't normalize to genes).
    
    This function is used for protein-level analysis where we want to preserve
    the original protein identifiers throughout the pipeline.
    
    Args:
        df: DataFrame with protein data
        protein_column: Name of column containing protein identifiers
        inplace: If True, modify df in place; if False, return new DataFrame
        verbose: If False, do not print row-count messages (e.g. batch tools / Streamlit)
    
    Returns:
        DataFrame with filtered proteins (still as proteins, not genes)
    """
    if not inplace:
        df = df.copy()
    
    # Step 1: Filter out Decoy/Entrap proteins
    n_before = len(df)
    protein_str = df[protein_column].astype(str)
    decoy_mask = protein_str.apply(is_decoy_entrap_protein)
    df = df[~decoy_mask].copy()
    n_after_decoy = len(df)
    n_filtered_decoy = n_before - n_after_decoy
    
    if verbose and n_filtered_decoy > 0:
        print(f"  Filtered out {n_filtered_decoy} Decoy/Entrap proteins ({n_before} -> {n_after_decoy})")
    
    # Step 2: Filter out non-human proteins (e.g., BOVINE)
    n_before_non_human = len(df)
    protein_str = df[protein_column].astype(str)  # Recalculate after decoy filtering
    non_human_mask = protein_str.apply(is_non_human_protein)
    df = df[~non_human_mask].copy()
    n_after_non_human = len(df)
    n_filtered_non_human = n_before_non_human - n_after_non_human
    
    if verbose and n_filtered_non_human > 0:
        print(f"  Filtered out {n_filtered_non_human} non-human proteins (e.g., BOVINE) ({n_before_non_human} -> {n_after_non_human})")
    
    # Step 3: Remove rows with invalid/empty protein identifiers
    n_before_clean = len(df)
    protein_str = df[protein_column].astype(str)  # Recalculate after non-human filtering
    valid_mask = (
        df[protein_column].notna() &
        (protein_str.str.strip() != '') &
        (protein_str.str.lower() != 'nan') &
        (protein_str.str.lower() != 'none') &
        (protein_str.str.lower() != 'na')
    )
    df = df[valid_mask].copy()
    n_after_clean = len(df)
    n_filtered_clean = n_before_clean - n_after_clean
    
    if verbose and n_filtered_clean > 0:
        print(f"  Filtered out {n_filtered_clean} invalid/empty protein identifiers ({n_before_clean} -> {n_after_clean})")
    
    return df

# ============================================
# CACHE CONFIGURATION FUNCTIONS
# ============================================

def load_cache_config(cache_dir=None):
    """Load cache configuration from JSON file.
    
    Args:
        cache_dir: Path to cache directory (default: work_dir / "cache")
    
    Returns:
        dict: Configuration dictionary with 'min_peptides', 'default_peptide_filter', etc.
              Returns None if config file doesn't exist
    """
    import json
    
    if cache_dir is None:
        # Try to infer from common location
        work_dir = Path(__file__).parent
        cache_dir = work_dir / "cache"
    else:
        cache_dir = Path(cache_dir)
    
    config_file = cache_dir / "cache_config.json"
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f"  Warning: Could not load cache config: {e}")
    return None

def find_cache_file(dataset_name, cache_dir=None, min_peptides=None):
    """Find cache file for a dataset, trying multiple naming patterns.
    
    Priority order:
    1. If min_peptides specified, try that specific version first
    2. Try to load config and use default from config
    3. Try default naming patterns
    
    Args:
        dataset_name: Name of the dataset (e.g., 'PXD002854' or 'PXD002854-plasma')
        cache_dir: Path to cache directory (default: work_dir / "cache")
        min_peptides: Specific peptide filter level to use (None = use config default)
    
    Returns:
        Path: Path to cache file, or None if not found
    """
    if cache_dir is None:
        work_dir = Path(__file__).parent
        cache_dir = work_dir / "cache"
    else:
        cache_dir = Path(cache_dir)
    
    # Determine which filter level to use
    if min_peptides is None:
        # Try to load config
        config = load_cache_config(cache_dir)
        if config:
            min_peptides = config.get('min_peptides', 2)
        else:
            # Default to 2 if no config
            min_peptides = 2
    
    # Build cache file name based on filter level
    if min_peptides == 2:
        # Default: no special suffix
        cache_file = cache_dir / f"{dataset_name}_processed.parquet"
    elif min_peptides == 0:
        cache_file = cache_dir / f"{dataset_name}_processed_unfiltered.parquet"
    else:
        cache_file = cache_dir / f"{dataset_name}_processed_min{min_peptides}pep.parquet"
    
    if cache_file.exists():
        return cache_file
    
    # If specific version not found, try other common patterns
    # Try exact match first
    cache_file = cache_dir / f"{dataset_name}_processed.parquet"
    if cache_file.exists():
        return cache_file
    
    # Try with common suffixes (for datasets split by condition)
    for suffix in ['-plasma', '-serum', '-erythrocyte', '-DDA', '-DIA', '-blood_serum', '-blood_plasma']:
        cache_file = cache_dir / f"{dataset_name}{suffix}_processed.parquet"
        if cache_file.exists():
            return cache_file
    
    # Try unfiltered version
    cache_file = cache_dir / f"{dataset_name}_processed_unfiltered.parquet"
    if cache_file.exists():
        return cache_file
    
    # Try to find any cache file that starts with the dataset name
    # BUT: Only if the dataset_name doesn't have a specific suffix (like -LFQ, -TMT)
    # This prevents matching PXD010899-LFQ with PXD010899-TMT
    base_name = dataset_name.split('-')[0]  # Get base PXD ID
    
    # If dataset_name has a suffix (like -LFQ, -TMT, -plasma), be more strict
    has_specific_suffix = '-' in dataset_name and len(dataset_name.split('-')) > 1
    if has_specific_suffix:
        # For datasets with suffixes, only try exact match or base name (no other suffixes)
        # Don't try to match with different suffixes
        return None
    else:
        # For base PXD IDs without suffixes, try to find any matching cache file
        cache_files = list(cache_dir.glob(f"{base_name}*_processed*.parquet"))
        if cache_files:
            # Prefer default version if multiple exist
            default_file = cache_dir / f"{base_name}_processed.parquet"
            if default_file in cache_files:
                return default_file
            return cache_files[0]  # Use first match
    
    return None

# ============================================
# CONDITION / TISSUE GROUPING AND MAPPING LOADERS (shared by B_ and E_)
# ============================================

def _normalize_condition_for_tissue(condition):
    """Normalize condition string for grouping: strip 'Condition=', take part before comma."""
    if condition is None or (isinstance(condition, float) and pd.isna(condition)):
        return ""
    s = str(condition).strip()
    if not s:
        return ""
    if "condition=" in s.lower():
        idx = s.lower().find("condition=")
        s = s[idx + len("condition=") :].strip()
    if "," in s:
        s = s.split(",")[0].strip()
    return re.sub(r"\s+", " ", s).lower().strip()


def group_condition_to_tissue(condition):
    """Group condition to tissue/cell type category (canonical for 02 and E_)."""
    condition_lower = _normalize_condition_for_tissue(condition)
    if not condition_lower:
        return 'Unknown'
    if 'plasma' in condition_lower or 'serum' in condition_lower:
        return 'Blood Plasma/Serum'
    elif 'erythrocyte' in condition_lower or 'red blood cell' in condition_lower:
        return 'Erythrocyte'
    elif 'platelet' in condition_lower:
        return 'Platelet'
    elif 'cd4' in condition_lower or 'cd8' in condition_lower or 't cell' in condition_lower or 'tcells' in condition_lower:
        return 'T Cell'
    elif 'b cell' in condition_lower or 'cd19' in condition_lower:
        return 'B Cell'
    elif 'dendritic' in condition_lower:
        return 'Dendritic Cell'
    elif 'monocyte' in condition_lower:
        return 'Monocyte'
    elif 'macrophage' in condition_lower:
        return 'Macrophage'
    elif 'natural killer' in condition_lower or 'nk' in condition_lower:
        return 'NK Cell'
    elif 'neutrophil' in condition_lower:
        return 'Neutrophil'
    elif 'eosinophil' in condition_lower:
        return 'Eosinophil'
    elif 'basophil' in condition_lower:
        return 'Basophil'
    elif 'granulocyte' in condition_lower:
        return 'Granulocyte'
    elif 'blood' in condition_lower:
        return 'Blood Plasma/Serum'
    else:
        return condition_lower


def load_entry_name_mapping_library(library_file):
    """Load the entry name to accession mapping library from JSON file.
    library_file: path to entry_name_to_accession.json (or similar).
    """
    path = Path(library_file) if not isinstance(library_file, Path) else library_file
    if not path.exists():
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"  Warning: Could not load entry name mapping library: {e}")
    return {}


def load_manual_mapping(manual_mapping_file, verbose=True):
    """Load manual entry name to accession mapping from Excel file.
    manual_mapping_file: path to manual_id_mapping.xlsx (or similar).
    verbose: if True, print progress and sample.
    """
    path = Path(manual_mapping_file) if not isinstance(manual_mapping_file, Path) else manual_mapping_file
    if not path.exists():
        if verbose:
            print(f"  Manual mapping file not found: {path}")
        return {}
    try:
        df = pd.read_excel(path)
        if verbose:
            print(f"  Reading manual mapping file: {len(df)} rows, columns: {list(df.columns)}")
        entry_col = None
        accession_col = None
        for col in df.columns:
            col_lower = str(col).lower()
            if entry_col is None:
                if ('entry' in col_lower and 'name' in col_lower) or col_lower in ('entry', 'from', 'entry_name'):
                    entry_col = col
            if accession_col is None:
                if 'accession' in col_lower or 'uniprot' in col_lower or col_lower in ('to', 'accession', 'uniprot_id'):
                    accession_col = col
        if entry_col and not accession_col and len(df.columns) >= 2:
            if df.columns[0] == entry_col:
                accession_col = df.columns[1]
            elif df.columns[1] == entry_col:
                accession_col = df.columns[0]
        if entry_col is None or accession_col is None:
            if verbose:
                print(f"  Warning: Could not identify columns. Available: {list(df.columns)}")
            return {}
        if verbose:
            print(f"  Using columns: '{entry_col}' -> '{accession_col}'")
        mapping = {}
        for _, row in df.iterrows():
            entry_name = str(row[entry_col]).strip() if pd.notna(row[entry_col]) else None
            accession = str(row[accession_col]).strip() if pd.notna(row[accession_col]) else None
            if entry_name and accession and entry_name != 'nan' and accession != 'nan':
                mapping[entry_name] = accession
                mapping[entry_name.upper()] = accession
                if entry_name.upper().endswith('_HUMAN'):
                    entry_without = entry_name[:-6]
                    mapping[entry_without] = accession
                    mapping[entry_without.upper()] = accession
                else:
                    entry_with_human = entry_name.rstrip() + '_HUMAN'
                    mapping[entry_with_human] = accession
                    mapping[entry_with_human.upper()] = accession
        if verbose:
            print(f"  Loaded {len(set(mapping.values()))} unique mappings from manual file")
        return mapping
    except Exception as e:
        print(f"  Warning: Could not load manual mapping: {e}")
    return {}


def get_cache_filter_info(cache_dir=None):
    """Get information about what filter level is currently configured.
    
    Args:
        cache_dir: Path to cache directory
    
    Returns:
        dict: Information about current filter configuration
    """
    config = load_cache_config(cache_dir)
    if config:
        return {
            'min_peptides': config.get('min_peptides', 2),
            'filter_type': config.get('default_peptide_filter', '2_peptides'),
            'save_multiple_versions': config.get('save_multiple_versions', False),
            'description': config.get('description', {})
        }
    else:
        # Default if no config
        return {
            'min_peptides': 2,
            'filter_type': '2_peptides',
            'save_multiple_versions': False,
            'description': {}
        }


def get_default_cache_parquet_suffix(cache_dir=None):
    """Return the filename suffix for the default B_ cache (2 peptide filter + ENTRAP removed).
    Other scripts (E_, D_, F_, G_) should load only these files so counts match 02_tissue_summary.
    
    Returns:
        str: e.g. '_processed.parquet', '_processed_unfiltered.parquet', or '_processed_minNpep.parquet'
    """
    info = get_cache_filter_info(cache_dir)
    min_pep = info.get('min_peptides', 2)
    if min_pep == 0:
        return "_processed_unfiltered.parquet"
    if min_pep == 2:
        return "_processed.parquet"
    return f"_processed_min{min_pep}pep.parquet"
