"""
Methodology Analysis Script
Analyzes the effect of acquisition method (DDA vs DIA), fractionation, and depletion
on plasma proteome coverage and protein detectability.

Downloads SDRF files for datasets with valid msstats files and creates a multipanel
figure comparing different methodology approaches.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import sys
import csv
import warnings
from scipy import stats
from collections import defaultdict
warnings.filterwarnings('ignore')

# Set working directory and add to path for imports
work_dir = Path(r"G:\My Drive\Ikasketak\Postdoc\Cambridge\Cursor\blood_proteome_analysis")
sys.path.insert(0, str(work_dir))
os.chdir(work_dir)

# Directories
msstats_dir = work_dir / "msstats"  # Shared raw data
sdrf_dir = work_dir / "sdrf_files"  # Shared SDRF files
sdrf_dir.mkdir(exist_ok=True)
cache_dir = work_dir / "cache"
data_prep_output_dir = work_dir / "data_preparation_output"
output_dir = work_dir / "methodology_analysis"
output_dir.mkdir(parents=True, exist_ok=True)

# Check if Parquet is available for caching
try:
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# ============================================
# PROTEIN FILTERING AND NORMALIZATION FUNCTIONS
# ============================================
# Import from shared_utils instead of defining here

from shared_utils import (
    filter_proteins_only,
    load_protein_mapping_cache
)

def get_all_datasets_with_msstats():
    """Get list of ALL datasets that have msstats files (no plasma/serum filter).
    Used for SDRF download so every dataset can get acquisition method (avoids grey points in C_).
    """
    msstats_files = list(msstats_dir.glob("*.sdrf_openms_design_msstats_in.csv"))
    datasets = {}
    for f in msstats_files:
        dataset_name = f.name.replace(".sdrf_openms_design_msstats_in.csv", "")
        if '-' in dataset_name:
            base_name = dataset_name.split('-')[0]
        else:
            base_name = dataset_name
        if base_name.startswith('PXD'):
            if base_name not in datasets:
                datasets[base_name] = []
            datasets[base_name].append(dataset_name)
    return datasets


def get_datasets_with_msstats():
    """Get list of datasets that have valid msstats files.
    Filters to only include plasma/serum datasets (excludes cell types).
    Used for methodology figures. For SDRF download we use get_all_datasets_with_msstats().
    """
    all_datasets = get_all_datasets_with_msstats()
    if not all_datasets:
        return all_datasets

    # Build plasma/serum set (exclude cell types); use default B_ cache only (2 peptide + ENTRAP removed)
    plasma_serum_datasets = set()
    if PARQUET_AVAILABLE:
        from shared_utils import get_default_cache_parquet_suffix
        default_suffix = get_default_cache_parquet_suffix(cache_dir)
        cache_files = list(cache_dir.glob(f"*{default_suffix}"))
        for cache_file in cache_files:
            try:
                df = pd.read_parquet(cache_file)
                if len(df) > 0 and 'Condition' in df.columns:
                    condition = str(df['Condition'].iloc[0]).lower()
                    if 'plasma' in condition or 'serum' in condition:
                        cell_type_keywords = ['erythrocyte', 'platelet', 'monocyte', 'neutrophil',
                                            'lymphocyte', 't cell', 'b cell', 'nk cell', 'cd4', 'cd8',
                                            'cd19', 'macrophage', 'dendritic', 'basophil', 'eosinophil']
                        is_cell_type = any(keyword in condition for keyword in cell_type_keywords)
                        if not is_cell_type:
                            dataset_name = cache_file.name.replace(default_suffix, "")
                            plasma_serum_datasets.add(dataset_name)
            except Exception:
                pass

    prep_summary_file = data_prep_output_dir / "01_dataset_summary_before_and_after_filtering.csv"
    if prep_summary_file.exists():
        try:
            prep_summary = pd.read_csv(prep_summary_file)
            if 'Dataset' in prep_summary.columns and 'Condition' in prep_summary.columns:
                for _, row in prep_summary.iterrows():
                    condition = str(row['Condition']).lower()
                    if 'plasma' in condition or 'serum' in condition:
                        cell_type_keywords = ['erythrocyte', 'platelet', 'monocyte', 'neutrophil',
                                            'lymphocyte', 't cell', 'b cell', 'nk cell', 'cd4', 'cd8',
                                            'cd19', 'macrophage', 'dendritic', 'basophil', 'eosinophil']
                        is_cell_type = any(keyword in condition for keyword in cell_type_keywords)
                        if not is_cell_type:
                            dataset_name = str(row['Dataset']).strip()
                            plasma_serum_datasets.add(dataset_name)
        except Exception:
            pass

    # Filter to plasma/serum only; if no filter info, keep all
    if len(plasma_serum_datasets) == 0:
        return all_datasets
    datasets = {}
    for base_name, variants in all_datasets.items():
        for dataset_name in variants:
            if dataset_name in plasma_serum_datasets:
                if base_name not in datasets:
                    datasets[base_name] = []
                datasets[base_name].append(dataset_name)
    return datasets

def parse_sdrf_file(sdrf_file_or_name):
    """Parse SDRF file and extract methodology information.
    
    Args:
        sdrf_file_or_name: Either a Path object to the SDRF file, or a string dataset name.
                          If a string, will try exact match first, then base PXD ID.
    
    Returns:
        Dictionary with methodology info, or None if file not found/parse failed.
    """
    # Handle both Path object and string dataset name
    if isinstance(sdrf_file_or_name, str):
        dataset_name = sdrf_file_or_name
        # First try exact match (e.g., PXD013231-DDA.sdrf.tsv)
        sdrf_file = sdrf_dir / f"{dataset_name}.sdrf.tsv"
        
        # If exact match doesn't exist, try base PXD ID (remove suffix like -DDA, -DIA, -plasma, etc.)
        if not sdrf_file.exists() and '-' in dataset_name:
            base_dataset = dataset_name.split('-')[0]
            sdrf_file = sdrf_dir / f"{base_dataset}.sdrf.tsv"
    elif isinstance(sdrf_file_or_name, Path):
        sdrf_file = sdrf_file_or_name
    else:
        # Try to convert to Path if possible
        sdrf_file = Path(sdrf_file_or_name)
    
    if not sdrf_file.exists():
        return None
    
    try:
        with open(sdrf_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            rows = list(reader)
        
        if len(rows) == 0:
            return None
        
        # Extract methodology information
        # Get unique values for each methodology column
        depletion_values = set()
        depletion_method_values = set()
        fractionation_values = set()
        acquisition_values = set()
        
        for row in rows:
            # Depletion
            depletion = row.get('characteristics[depletion]', '').strip().lower()
            if depletion:
                depletion_values.add(depletion)
            
            depletion_method = row.get('characteristics[depletion method]', '').strip()
            if depletion_method:
                depletion_method_values.add(depletion_method)
            
            # Fractionation
            fractionation = row.get('comment[fractionation method]', '').strip().lower()
            if fractionation:
                fractionation_values.add(fractionation)
            
            # Acquisition method
            acquisition = row.get('comment[proteomics data acquisition method]', '').strip()
            if acquisition:
                acquisition_values.add(acquisition)
        
        # Classify methodology
        # Depletion
        is_depleted = False
        if depletion_values:
            # Check if any value indicates depletion
            for val in depletion_values:
                if val and 'no depletion' not in val and 'not applicable' not in val:
                    is_depleted = True
                    break
        # If column is missing, default to not depleted
        
        # Fractionation
        is_fractionated = False
        if fractionation_values:
            for val in fractionation_values:
                val_lower = val.lower()
                if val and 'no fractionation' not in val_lower and 'not applicable' not in val_lower:
                    is_fractionated = True
                    break
        # If column is missing, default to not fractionated
        
        # Acquisition method
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
        
        return {
            'depletion': 'depleted' if is_depleted else 'non_depleted',
            'depletion_method': '; '.join(sorted(depletion_method_values)) if depletion_method_values else '',
            'fractionation': 'fractionated' if is_fractionated else 'unfractionated',
            'acquisition': acquisition_method,
            'n_rows': len(rows)
        }
    
    except Exception as e:
        print(f"  Error parsing {sdrf_file.name}: {e}")
        return None

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

def load_msstats_for_methodology(dataset_name):
    """Load msstats file and extract protein and sample information.
    Uses cached Parquet data if available for faster loading.
    """
    # Try to load from Parquet cache first
    if PARQUET_AVAILABLE:
        cache_file = find_cache_file(dataset_name)
        if cache_file and cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                
                # Cache already contains filtered proteins (protein-level cache), no need to re-process
                # Get unique proteins and samples
                unique_proteins = df['Protein'].nunique()
                unique_samples = df['Sample'].nunique()
                
                # Calculate abundance (fraction of samples where each protein is detected)
                protein_sample_counts = df.groupby('Protein')['Sample'].nunique().to_dict()
                total_samples = unique_samples
                
                return {
                    'n_proteins': unique_proteins,
                    'n_samples': unique_samples,
                    'protein_sample_counts': protein_sample_counts,
                    'total_samples': total_samples
                }
            except Exception as e:
                print(f"  Warning: Error loading from cache: {e}, falling back to CSV")
    
    # Fallback to CSV loading
    msstats_file = msstats_dir / f"{dataset_name}.sdrf_openms_design_msstats_in.csv"
    
    if not msstats_file.exists():
        return None
    
    try:
        # Read only necessary columns
        df = pd.read_csv(msstats_file, usecols=['ProteinName', 'Reference'], low_memory=False)
        
        # Rename ProteinName to Protein for consistency
        df = df.rename(columns={'ProteinName': 'Protein'})
        
        # Filter out Decoy/Entrap proteins (keep as proteins, don't normalize to genes)
        print(f"  Filtering Decoy/Entrap proteins (keeping as proteins) for {dataset_name}...")
        df = filter_proteins_only(df, protein_column='Protein', inplace=False)
        
        # Additional filtering for empty/NaN values
        protein_str = df['Protein'].astype(str)
        valid_mask = (
            df['Protein'].notna() & 
            (protein_str.str.strip() != '') & 
            (protein_str.str.lower() != 'nan') &
            (protein_str.str.lower() != 'none') &
            (protein_str.str.lower() != 'na') &
            (~protein_str.str.contains('CONTAM', case=False, na=False))
        )
        df = df[valid_mask].copy()
        
        # Get unique proteins and samples
        unique_proteins = df['Protein'].nunique()
        unique_samples = df['Reference'].nunique()
        
        # Calculate abundance (fraction of samples where each protein is detected)
        protein_sample_counts = df.groupby('Protein')['Reference'].nunique().to_dict()
        total_samples = unique_samples
        
        return {
            'n_proteins': unique_proteins,
            'n_samples': unique_samples,
            'protein_sample_counts': protein_sample_counts,
            'total_samples': total_samples
        }
    
    except Exception as e:
        print(f"  Error loading {msstats_file.name}: {e}")
        return None

def download_and_parse_sdrf_files():
    """Parse SDRF files for all datasets with msstats. SDRF files are downloaded by C_ script (first use)."""
    print("=" * 80)
    print("STEP 1: Parsing SDRF files")
    print("=" * 80)
    
    # Try to load acquisition methods from data preparation output first
    acquisition_from_prep = {}
    prep_summary_file = data_prep_output_dir / "01_dataset_summary_before_and_after_filtering.csv"
    if prep_summary_file.exists():
        try:
            prep_summary = pd.read_csv(prep_summary_file)
            if 'Dataset' in prep_summary.columns and 'Acquisition' in prep_summary.columns:
                for _, row in prep_summary.iterrows():
                    dataset_name = str(row['Dataset']).strip()
                    acquisition = str(row['Acquisition']).strip()
                    if acquisition in ['DDA', 'DIA']:
                        acquisition_from_prep[dataset_name] = acquisition
            print(f"  Loaded acquisition methods from data preparation output for {len(acquisition_from_prep)} datasets")
        except Exception as e:
            print(f"  Warning: Could not load acquisition methods from data preparation: {e}")
    
    # Use ALL datasets with msstats (SDRF files should already be in sdrf_files, downloaded by C_)
    datasets_dict = get_all_datasets_with_msstats()
    base_datasets = sorted(datasets_dict.keys())
    print(f"Found {len(base_datasets)} unique base datasets with msstats files")
    
    methodology_data = {}
    failed = 0
    already_parsed = set()
    
    for i, base_dataset in enumerate(base_datasets):
        print(f"\n[{i+1}/{len(base_datasets)}] Processing {base_dataset}...")
        
        if base_dataset in already_parsed:
            continue
        
        # Parse SDRF file for each variant (files in sdrf_files, downloaded by C_)
        parsed_any = False
        for variant_name in datasets_dict[base_dataset]:
            acquisition_method = None
            if variant_name in acquisition_from_prep:
                acquisition_method = acquisition_from_prep[variant_name]
            else:
                for suffix in ['-plasma', '-serum', '-erythrocyte', '-DDA', '-DIA', '-blood_serum', '-blood_plasma']:
                    dataset_name = f"{variant_name}{suffix}"
                    if dataset_name in acquisition_from_prep:
                        acquisition_method = acquisition_from_prep[dataset_name]
                        break
            
            methodology = parse_sdrf_file(variant_name)
            if methodology:
                if acquisition_method:
                    methodology['acquisition'] = acquisition_method
                methodology_data[variant_name] = methodology.copy()
                parsed_any = True
                if not already_parsed or base_dataset not in already_parsed:
                    print(f"  [OK] {variant_name}: {methodology['acquisition']}, {methodology['fractionation']}, {methodology['depletion']}")
            elif acquisition_method:
                methodology = {
                    'acquisition': acquisition_method,
                    'fractionation': 'unknown',
                    'depletion': 'unknown',
                    'depletion_method': '',
                    'n_rows': 0
                }
                methodology_data[variant_name] = methodology
                parsed_any = True
                print(f"  [OK] {variant_name}: {acquisition_method} (from data prep, SDRF parsing failed)")
        
        if parsed_any:
            already_parsed.add(base_dataset)
        else:
            print(f"  [WARN] Could not parse SDRF file for any variant of {base_dataset}")
            failed += 1
    
    print(f"\n[OK] Parsed SDRF for {len(already_parsed)} base datasets")
    print(f"  Could not parse: {failed}")
    print(f"  Successfully parsed methodology for {len(methodology_data)} dataset variants")
    
    return methodology_data

def load_protein_data_for_methodology(methodology_data):
    """Load protein data from msstats files for methodology analysis."""
    print("\n" + "=" * 80)
    print("STEP 2: Loading protein data from msstats files")
    print("=" * 80)
    
    dataset_protein_data = {}
    
    for i, (dataset, methodology) in enumerate(methodology_data.items()):
        print(f"[{i+1}/{len(methodology_data)}] Loading {dataset}...")
        
        protein_data = load_msstats_for_methodology(dataset)
        if protein_data:
            # Combine with methodology info
            # Check if methodology has all required fields
            if not methodology or 'acquisition' not in methodology:
                print(f"  [WARN] Dataset {dataset} missing methodology data!")
            dataset_protein_data[dataset] = {
                **methodology,
                **protein_data
            }
            print(f"  [OK] {protein_data['n_proteins']} proteins, {protein_data['n_samples']} samples")
        else:
            print(f"  [WARN] Could not load protein data")
    
    print(f"\n[OK] Loaded protein data for {len(dataset_protein_data)} datasets")
    return dataset_protein_data

def create_methodology_multipanel_figure(dataset_protein_data, output_dir):
    """Create three separate methodology analysis figures."""
    print("\n" + "=" * 80)
    print("STEP 3: Creating methodology figures (3 separate figures)")
    print("=" * 80)
    
    print("  Preparing dataset summary data...")
    # Prepare data
    df_list = []
    datasets_with_missing_info = []
    for dataset, data in dataset_protein_data.items():
        # Check if methodology fields exist and are valid
        acquisition = data.get('acquisition', None)
        fractionation = data.get('fractionation', None)
        depletion = data.get('depletion', None)
        
        # Track datasets with missing info
        if not acquisition or pd.isna(acquisition) or acquisition == '':
            datasets_with_missing_info.append((dataset, 'acquisition', acquisition))
        if not fractionation or pd.isna(fractionation) or fractionation == '':
            datasets_with_missing_info.append((dataset, 'fractionation', fractionation))
        if not depletion or pd.isna(depletion) or depletion == '':
            datasets_with_missing_info.append((dataset, 'depletion', depletion))
        
        df_list.append({
            'Dataset': dataset,
            'Acquisition': acquisition,
            'Fractionation': fractionation,
            'Depletion': depletion,
            'N_Proteins': data['n_proteins'],
            'N_Samples': data['n_samples']
        })
    
    # Report datasets with missing info
    if datasets_with_missing_info:
        print(f"  WARNING: Found {len(set(d[0] for d in datasets_with_missing_info))} datasets with missing methodology info:")
        for dataset, field, value in datasets_with_missing_info[:10]:  # Show first 10
            print(f"    {dataset}: missing {field} (value: {value})")
        if len(datasets_with_missing_info) > 10:
            print(f"    ... and {len(datasets_with_missing_info) - 10} more")
    
    df = pd.DataFrame(df_list)
    print(f"  [OK] Created summary dataframe with {len(df)} datasets")
    
    # Debug: Check for duplicates and missing values
    print(f"  Debug: Checking for issues in summary dataframe...")
    print(f"    Total rows: {len(df)}")
    print(f"    Unique datasets: {df['Dataset'].nunique()}")
    if len(df) != df['Dataset'].nunique():
        duplicates = df[df.duplicated(subset=['Dataset'], keep=False)]
        print(f"    WARNING: Found {len(duplicates)} duplicate dataset entries:")
        print(duplicates[['Dataset', 'Acquisition', 'Fractionation', 'Depletion']].to_string())
    
    # Check for missing values
    missing_acq = df['Acquisition'].isna().sum()
    missing_frac = df['Fractionation'].isna().sum()
    missing_dep = df['Depletion'].isna().sum()
    if missing_acq > 0 or missing_frac > 0 or missing_dep > 0:
        print(f"    WARNING: Missing values - Acquisition: {missing_acq}, Fractionation: {missing_frac}, Depletion: {missing_dep}")
        missing_rows = df[df['Acquisition'].isna() | df['Fractionation'].isna() | df['Depletion'].isna()]
        print(f"    Datasets with missing values:")
        print(missing_rows[['Dataset', 'Acquisition', 'Fractionation', 'Depletion']].to_string())
    
    # Check value counts before plotting
    print(f"    Acquisition value counts: {df['Acquisition'].value_counts().to_dict()}")
    print(f"    Fractionation value counts: {df['Fractionation'].value_counts().to_dict()}")
    print(f"    Depletion value counts: {df['Depletion'].value_counts().to_dict()}")
    
    print("  Calculating abundance data for panels G, H, I...")
    # Calculate abundance data (for panels G, H, I)
    abundance_data = []
    total_proteins = sum(len(data['protein_sample_counts']) for data in dataset_protein_data.values())
    processed_proteins = 0
    
    for i, (dataset, data) in enumerate(dataset_protein_data.items()):
        if (i + 1) % 5 == 0 or i == 0:
            print(f"    Processing dataset {i+1}/{len(dataset_protein_data)}: {dataset}...")
        for protein, sample_count in data['protein_sample_counts'].items():
            abundance = sample_count / data['total_samples']
            abundance_data.append({
                'Dataset': dataset,
                'Protein': protein,
                'Abundance': abundance,
                'Acquisition': data['acquisition'],
                'Fractionation': data['fractionation'],
                'Depletion': data['depletion']
            })
            processed_proteins += 1
    
    abundance_df = pd.DataFrame(abundance_data)
    print(f"  [OK] Created abundance dataframe with {len(abundance_df)} protein-dataset combinations")
    
    # Color scheme
    colors = {
        'DDA': '#3498db',
        'DIA': '#e74c3c',
        'fractionated': '#2ecc71',
        'unfractionated': '#f39c12',
        'depleted': '#9b59b6',
        'non_depleted': '#34495e'
    }
    
    print("  Creating Row 1: Dataset Composition (Panels A, B, C)...")
    # ============================================
    # ROW 1: Dataset Composition (Panels A, B, C)
    # ============================================
    
    # Filter to only include datasets with all methodology information
    # This ensures all panels count the same datasets
    df_complete = df.dropna(subset=['Acquisition', 'Fractionation', 'Depletion']).copy()
    print(f"  Debug: After filtering for complete methodology data: {len(df_complete)} datasets")
    if len(df) != len(df_complete):
        missing_info = df[~df.index.isin(df_complete.index)]
        print(f"    WARNING: {len(missing_info)} datasets excluded due to missing methodology info:")
        print(missing_info[['Dataset', 'Acquisition', 'Fractionation', 'Depletion']].to_string())
    
    # Use df_complete for all panels to ensure consistency
    df = df_complete
    
    # ============================================
    # FIGURE 1: Dataset Composition (Panels A, B, C)
    # ============================================
    print("\n  Creating Figure 1: Dataset Composition (Panels A, B, C)...")
    fig1 = plt.figure(figsize=(15, 5))
    gs1 = fig1.add_gridspec(1, 3, hspace=0.3, wspace=0.3)
    
    # Panel A: Acquisition
    print("    Creating Panel A: Acquisition method...")
    ax_a = fig1.add_subplot(gs1[0, 0])
    acquisition_counts = df['Acquisition'].value_counts()
    bars_a = ax_a.bar(acquisition_counts.index, acquisition_counts.values, 
                      color=[colors.get(x, 'gray') for x in acquisition_counts.index], alpha=0.7)
    ax_a.set_ylabel('Number of Datasets', fontsize=11, fontweight='bold')
    ax_a.set_title('(A) Acquisition Method', fontsize=12, fontweight='bold')
    ax_a.grid(True, alpha=0.3, axis='y')
    
    # Add sample count annotation
    for i, (method, count) in enumerate(acquisition_counts.items()):
        total_samples = df[df['Acquisition'] == method]['N_Samples'].sum()
        ax_a.text(i, count + count*0.05, f'n={int(count)}\n({int(total_samples)} samples)',
                 ha='center', va='bottom', fontsize=9)
    
    # Panel B: Fractionation
    print("    Creating Panel B: Fractionation...")
    ax_b = fig1.add_subplot(gs1[0, 1])
    fractionation_counts = df['Fractionation'].value_counts()
    bars_b = ax_b.bar(fractionation_counts.index, fractionation_counts.values,
                     color=[colors.get(x, 'gray') for x in fractionation_counts.index], alpha=0.7)
    ax_b.set_ylabel('Number of Datasets', fontsize=11, fontweight='bold')
    ax_b.set_title('(B) Fractionation', fontsize=12, fontweight='bold')
    ax_b.grid(True, alpha=0.3, axis='y')
    
    for i, (method, count) in enumerate(fractionation_counts.items()):
        total_samples = df[df['Fractionation'] == method]['N_Samples'].sum()
        ax_b.text(i, count + count*0.05, f'n={int(count)}\n({int(total_samples)} samples)',
                 ha='center', va='bottom', fontsize=9)
    
    # Panel C: Depletion
    print("    Creating Panel C: Depletion...")
    ax_c = fig1.add_subplot(gs1[0, 2])
    depletion_counts = df['Depletion'].value_counts()
    bars_c = ax_c.bar(depletion_counts.index, depletion_counts.values,
                     color=[colors.get(x, 'gray') for x in depletion_counts.index], alpha=0.7)
    ax_c.set_ylabel('Number of Datasets', fontsize=11, fontweight='bold')
    ax_c.set_title('(C) Depletion', fontsize=12, fontweight='bold')
    ax_c.grid(True, alpha=0.3, axis='y')
    
    for i, (method, count) in enumerate(depletion_counts.items()):
        total_samples = df[df['Depletion'] == method]['N_Samples'].sum()
        ax_c.text(i, count + count*0.05, f'n={int(count)}\n({int(total_samples)} samples)',
                 ha='center', va='bottom', fontsize=9)
    
    fig1.suptitle('Dataset Composition by Methodology', fontsize=14, fontweight='bold', y=1.02)
    plt.savefig(output_dir / "methodology_figure1_composition.png", bbox_inches='tight', dpi=300)
    plt.savefig(output_dir / "methodology_figure1_composition.pdf", bbox_inches='tight')
    plt.close()
    print("  [OK] Saved: methodology_figure1_composition.png and .pdf")
    
    # ============================================
    # FIGURE 2: Protein Coverage (Panels D, E, F1, F2)
    # ============================================
    print("\n  Creating Figure 2: Protein Coverage (Panels D, E, F1, F2)...")
    fig2 = plt.figure(figsize=(15, 10))
    gs2 = fig2.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Panel D: Protein count - DDA vs DIA
    print("    Creating Panel D: Protein count - DDA vs DIA...")
    ax_d = fig2.add_subplot(gs2[0, 0])
    dda_proteins = df[df['Acquisition'] == 'DDA']['N_Proteins'].values
    dia_proteins = df[df['Acquisition'] == 'DIA']['N_Proteins'].values
    
    # Count datasets
    n_dda = len(dda_proteins)
    n_dia = len(dia_proteins)
    
    if len(dda_proteins) > 0 and len(dia_proteins) > 0:
        positions = [1, 2]
        bp_d = ax_d.boxplot([dda_proteins, dia_proteins], positions=positions, 
                           patch_artist=True, widths=0.6, showmeans=True)
        bp_d['boxes'][0].set_facecolor(colors['DDA'])
        bp_d['boxes'][1].set_facecolor(colors['DIA'])
        bp_d['boxes'][0].set_alpha(0.7)
        bp_d['boxes'][1].set_alpha(0.7)
        
        # Statistical test
        if len(dda_proteins) > 1 and len(dia_proteins) > 1:
            stat, pval = stats.mannwhitneyu(dda_proteins, dia_proteins, alternative='two-sided')
            median_dda = np.median(dda_proteins)
            median_dia = np.median(dia_proteins)
            effect_size = (median_dia - median_dda) / np.median(np.concatenate([dda_proteins, dia_proteins]))
            
            ax_d.text(0.5, 0.95, f'p={pval:.3f}\nΔ={effect_size:.2f}', 
                     transform=ax_d.transAxes, fontsize=9,
                     verticalalignment='top', ha='center',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax_d.set_xticks(positions)
        ax_d.set_xticklabels(['DDA', 'DIA'])
        ax_d.set_ylabel('Number of Proteins per Dataset', fontsize=11, fontweight='bold')
        ax_d.set_title('(D) Protein Coverage: DDA vs DIA', fontsize=12, fontweight='bold')
        ax_d.grid(True, alpha=0.3, axis='y')
    
    # Add dataset count annotation (always show, even if one group is empty)
    ax_d.text(0.02, 0.98, f'n={n_dda} DDA, n={n_dia} DIA', 
             transform=ax_d.transAxes, fontsize=9,
             verticalalignment='top', ha='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Panel E: Protein count - Fractionation
    print("    Creating Panel E: Protein count - Fractionation...")
    ax_e = fig2.add_subplot(gs2[0, 1])
    frac_proteins = df[df['Fractionation'] == 'fractionated']['N_Proteins'].values
    unfrac_proteins = df[df['Fractionation'] == 'unfractionated']['N_Proteins'].values
    
    # Count datasets
    n_frac = len(frac_proteins)
    n_unfrac = len(unfrac_proteins)
    
    if len(frac_proteins) > 0 and len(unfrac_proteins) > 0:
        positions = [1, 2]
        bp_e = ax_e.boxplot([unfrac_proteins, frac_proteins], positions=positions,
                           patch_artist=True, widths=0.6, showmeans=True)
        bp_e['boxes'][0].set_facecolor(colors['unfractionated'])
        bp_e['boxes'][1].set_facecolor(colors['fractionated'])
        bp_e['boxes'][0].set_alpha(0.7)
        bp_e['boxes'][1].set_alpha(0.7)
        
        if len(unfrac_proteins) > 1 and len(frac_proteins) > 1:
            stat, pval = stats.mannwhitneyu(unfrac_proteins, frac_proteins, alternative='two-sided')
            median_unfrac = np.median(unfrac_proteins)
            median_frac = np.median(frac_proteins)
            effect_size = (median_frac - median_unfrac) / np.median(np.concatenate([unfrac_proteins, frac_proteins]))
            
            ax_e.text(0.5, 0.95, f'p={pval:.3f}\nΔ={effect_size:.2f}', 
                     transform=ax_e.transAxes, fontsize=9,
                     verticalalignment='top', ha='center',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax_e.set_xticks(positions)
        ax_e.set_xticklabels(['Unfractionated', 'Fractionated'])
        ax_e.set_ylabel('Number of Proteins per Dataset', fontsize=11, fontweight='bold')
        ax_e.set_title('(E) Protein Coverage: Fractionation', fontsize=12, fontweight='bold')
        ax_e.grid(True, alpha=0.3, axis='y')
    
    # Add dataset count annotation (always show, even if one group is empty)
    ax_e.text(0.02, 0.98, f'n={n_unfrac} Unfractionated, n={n_frac} Fractionated', 
             transform=ax_e.transAxes, fontsize=9,
             verticalalignment='top', ha='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Panel F1: Protein count - Depletion (DDA datasets only)
    print("    Creating Panel F1: Protein count - Depletion (DDA datasets)...")
    ax_f1 = fig2.add_subplot(gs2[1, 0])
    df_dda = df[df['Acquisition'] == 'DDA']
    dep_proteins_dda = df_dda[df_dda['Depletion'] == 'depleted']['N_Proteins'].values
    nondep_proteins_dda = df_dda[df_dda['Depletion'] == 'non_depleted']['N_Proteins'].values
    
    # Count datasets
    n_dep_dda = len(dep_proteins_dda)
    n_nondep_dda = len(nondep_proteins_dda)
    
    if len(dep_proteins_dda) > 0 and len(nondep_proteins_dda) > 0:
        positions = [1, 2]
        bp_f1 = ax_f1.boxplot([nondep_proteins_dda, dep_proteins_dda], positions=positions,
                           patch_artist=True, widths=0.6, showmeans=True)
        bp_f1['boxes'][0].set_facecolor(colors['non_depleted'])
        bp_f1['boxes'][1].set_facecolor(colors['depleted'])
        bp_f1['boxes'][0].set_alpha(0.7)
        bp_f1['boxes'][1].set_alpha(0.7)
        
        if len(nondep_proteins_dda) > 1 and len(dep_proteins_dda) > 1:
            stat, pval = stats.mannwhitneyu(nondep_proteins_dda, dep_proteins_dda, alternative='two-sided')
            median_nondep = np.median(nondep_proteins_dda)
            median_dep = np.median(dep_proteins_dda)
            effect_size = (median_dep - median_nondep) / np.median(np.concatenate([nondep_proteins_dda, dep_proteins_dda]))
            
            ax_f1.text(0.5, 0.95, f'p={pval:.3f}\nΔ={effect_size:.2f}', 
                      transform=ax_f1.transAxes, fontsize=9,
                     verticalalignment='top', ha='center',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax_f1.set_xticks(positions)
        ax_f1.set_xticklabels(['Non-depleted', 'Depleted'])
        ax_f1.set_ylabel('Number of Proteins per Dataset', fontsize=11, fontweight='bold')
        ax_f1.set_title('(F1) Protein Coverage: Depletion (DDA)', fontsize=12, fontweight='bold')
        ax_f1.grid(True, alpha=0.3, axis='y')
    
    # Add dataset count annotation (always show, even if one group is empty)
    ax_f1.text(0.02, 0.98, f'n={n_nondep_dda} Non-depleted, n={n_dep_dda} Depleted', 
               transform=ax_f1.transAxes, fontsize=9,
               verticalalignment='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Panel F2: Protein count - Depletion (DIA datasets only)
    print("    Creating Panel F2: Protein count - Depletion (DIA datasets)...")
    ax_f2 = fig2.add_subplot(gs2[1, 1])
    df_dia = df[df['Acquisition'] == 'DIA']
    dep_proteins_dia = df_dia[df_dia['Depletion'] == 'depleted']['N_Proteins'].values
    nondep_proteins_dia = df_dia[df_dia['Depletion'] == 'non_depleted']['N_Proteins'].values
    
    # Count datasets
    n_dep_dia = len(dep_proteins_dia)
    n_nondep_dia = len(nondep_proteins_dia)
    
    if len(dep_proteins_dia) > 0 and len(nondep_proteins_dia) > 0:
        positions = [1, 2]
        bp_f2 = ax_f2.boxplot([nondep_proteins_dia, dep_proteins_dia], positions=positions,
                              patch_artist=True, widths=0.6, showmeans=True)
        bp_f2['boxes'][0].set_facecolor(colors['non_depleted'])
        bp_f2['boxes'][1].set_facecolor(colors['depleted'])
        bp_f2['boxes'][0].set_alpha(0.7)
        bp_f2['boxes'][1].set_alpha(0.7)
        
        if len(nondep_proteins_dia) > 1 and len(dep_proteins_dia) > 1:
            stat, pval = stats.mannwhitneyu(nondep_proteins_dia, dep_proteins_dia, alternative='two-sided')
            median_nondep = np.median(nondep_proteins_dia)
            median_dep = np.median(dep_proteins_dia)
            effect_size = (median_dep - median_nondep) / np.median(np.concatenate([nondep_proteins_dia, dep_proteins_dia]))
            
            ax_f2.text(0.5, 0.95, f'p={pval:.3f}\nΔ={effect_size:.2f}', 
                      transform=ax_f2.transAxes, fontsize=9,
                      verticalalignment='top', ha='center',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax_f2.set_xticks(positions)
        ax_f2.set_xticklabels(['Non-depleted', 'Depleted'])
        ax_f2.set_ylabel('Number of Proteins per Dataset', fontsize=11, fontweight='bold')
        ax_f2.set_title('(F2) Protein Coverage: Depletion (DIA)', fontsize=12, fontweight='bold')
        ax_f2.grid(True, alpha=0.3, axis='y')
    
    # Add dataset count annotation (always show, even if one group is empty)
    ax_f2.text(0.02, 0.98, f'n={n_nondep_dia} Non-depleted, n={n_dep_dia} Depleted', 
               transform=ax_f2.transAxes, fontsize=9,
               verticalalignment='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig2.suptitle('Protein Coverage per Dataset by Methodology', fontsize=14, fontweight='bold', y=0.995)
    plt.savefig(output_dir / "methodology_figure2_coverage.png", bbox_inches='tight', dpi=300)
    plt.savefig(output_dir / "methodology_figure2_coverage.pdf", bbox_inches='tight')
    plt.close()
    print("  [OK] Saved: methodology_figure2_coverage.png and .pdf")
    
    # ============================================
    # FIGURE 3: Protein Abundance (Panels G, H)
    # ============================================
    print("\n  Creating Figure 3: Protein Abundance (Panels G, H)...")
    fig3 = plt.figure(figsize=(15, 6))
    gs3 = fig3.add_gridspec(1, 2, hspace=0.3, wspace=0.3)
    
    # Panel G: Cumulative protein accumulation curves by methodology
    print("    Creating Panel G: Cumulative protein accumulation curves...")
    ax_g = fig3.add_subplot(gs3[0, 0])
    
    # Load protein sets for each dataset
    print("      Loading protein sets for all datasets...")
    dataset_protein_sets = {}
    for dataset_name in dataset_protein_data.keys():
        protein_data = load_msstats_for_methodology(dataset_name)
        if protein_data:
            # Get unique proteins for this dataset
            if PARQUET_AVAILABLE:
                cache_file = find_cache_file(dataset_name)
                if cache_file and cache_file.exists():
                    try:
                        df_temp = pd.read_parquet(cache_file)
                        protein_str = df_temp['Protein'].astype(str)
                        valid_mask = (
                            df_temp['Protein'].notna() & 
                            (protein_str.str.strip() != '') & 
                            (protein_str.str.lower() != 'nan') &
                            (protein_str.str.lower() != 'none') &
                            (protein_str.str.lower() != 'na') &
                            (~protein_str.str.contains('CONTAM', case=False, na=False))
                        )
                        dataset_protein_sets[dataset_name] = set(df_temp[valid_mask]['Protein'].unique())
                    except Exception:
                        dataset_protein_sets[dataset_name] = set(protein_data['protein_sample_counts'].keys())
                else:
                    dataset_protein_sets[dataset_name] = set(protein_data['protein_sample_counts'].keys())
            else:
                dataset_protein_sets[dataset_name] = set(protein_data['protein_sample_counts'].keys())
    
    print(f"      Loaded protein sets for {len(dataset_protein_sets)} datasets")
    
    # Define groups for stratification
    groups = {
        'Acquisition': {
            'DDA': [d for d in dataset_protein_sets.keys() if dataset_protein_data[d]['acquisition'] == 'DDA'],
            'DIA': [d for d in dataset_protein_sets.keys() if dataset_protein_data[d]['acquisition'] == 'DIA']
        },
        'Fractionation': {
            'Fractionated': [d for d in dataset_protein_sets.keys() if dataset_protein_data[d]['fractionation'] == 'fractionated'],
            'Unfractionated': [d for d in dataset_protein_sets.keys() if dataset_protein_data[d]['fractionation'] == 'unfractionated']
        },
        'Depletion': {
            'Depleted': [d for d in dataset_protein_sets.keys() if dataset_protein_data[d]['depletion'] == 'depleted'],
            'Non-depleted': [d for d in dataset_protein_sets.keys() if dataset_protein_data[d]['depletion'] == 'non_depleted']
        }
    }
    
    # Function to calculate cumulative unique proteins with permutations
    def calculate_accumulation_curve(datasets_list, n_permutations=100):
        """Calculate mean ± SD cumulative unique proteins across permutations."""
        if len(datasets_list) == 0:
            return None
        
        max_datasets = len(datasets_list)
        all_curves = []
        
        for perm in range(n_permutations):
            # Randomly permute dataset order
            permuted_datasets = np.random.permutation(datasets_list).tolist()
            
            # Track cumulative unique proteins
            cumulative_proteins = set()
            curve = []
            
            for i, dataset in enumerate(permuted_datasets):
                if dataset in dataset_protein_sets:
                    cumulative_proteins.update(dataset_protein_sets[dataset])
                curve.append(len(cumulative_proteins))
            
            all_curves.append(curve)
        
        # Calculate mean and SD at each step
        all_curves_array = np.array(all_curves)
        mean_curve = np.mean(all_curves_array, axis=0)
        std_curve = np.std(all_curves_array, axis=0)
        
        return {
            'n_datasets': np.arange(1, max_datasets + 1),
            'mean': mean_curve,
            'std': std_curve,
            'mean_plus_std': mean_curve + std_curve,
            'mean_minus_std': mean_curve - std_curve
        }
    
    # Calculate curves for each group
    print("      Calculating accumulation curves with 100 permutations per group...")
    curves_data = {}
    
    for group_name, subgroups in groups.items():
        curves_data[group_name] = {}
        for subgroup_name, datasets_list in subgroups.items():
            if len(datasets_list) > 0:
                print(f"        {group_name} - {subgroup_name}: {len(datasets_list)} datasets")
                curve = calculate_accumulation_curve(datasets_list, n_permutations=100)
                if curve:
                    curves_data[group_name][subgroup_name] = curve
    
    # Plot all curves on the same axes
    plot_colors = {
        'DDA': colors['DDA'],
        'DIA': colors['DIA'],
        'Fractionated': colors['fractionated'],
        'Unfractionated': colors['unfractionated'],
        'Depleted': colors['depleted'],
        'Non-depleted': colors['non_depleted']
    }
    
    # Plot Acquisition curves
    if 'Acquisition' in curves_data:
        for method in ['DDA', 'DIA']:
            if method in curves_data['Acquisition']:
                curve = curves_data['Acquisition'][method]
                ax_g.plot(curve['n_datasets'], curve['mean'], 
                         color=plot_colors[method], label=f'{method}', linewidth=2)
                ax_g.fill_between(curve['n_datasets'], 
                                curve['mean_minus_std'], curve['mean_plus_std'],
                                color=plot_colors[method], alpha=0.2)
    
    # Plot Fractionation curves
    if 'Fractionation' in curves_data:
        for frac_type in ['Fractionated', 'Unfractionated']:
            if frac_type in curves_data['Fractionation']:
                curve = curves_data['Fractionation'][frac_type]
                ax_g.plot(curve['n_datasets'], curve['mean'], 
                         color=plot_colors[frac_type], label=f'{frac_type}', 
                         linewidth=2, linestyle='--')
                ax_g.fill_between(curve['n_datasets'], 
                                curve['mean_minus_std'], curve['mean_plus_std'],
                                color=plot_colors[frac_type], alpha=0.2)
    
    # Plot Depletion curves
    if 'Depletion' in curves_data:
        for dep_type in ['Depleted', 'Non-depleted']:
            if dep_type in curves_data['Depletion']:
                curve = curves_data['Depletion'][dep_type]
                ax_g.plot(curve['n_datasets'], curve['mean'], 
                         color=plot_colors[dep_type], label=f'{dep_type}', 
                         linewidth=2, linestyle=':')
                ax_g.fill_between(curve['n_datasets'], 
                                curve['mean_minus_std'], curve['mean_plus_std'],
                                color=plot_colors[dep_type], alpha=0.2)
    
    ax_g.set_xlabel('Number of Datasets', fontsize=11, fontweight='bold')
    ax_g.set_ylabel('Cumulative Unique Proteins', fontsize=11, fontweight='bold')
    ax_g.set_title('(G) Protein Accumulation by Methodology', fontsize=12, fontweight='bold')
    ax_g.legend(loc='lower right', fontsize=8, ncol=1)
    ax_g.grid(True, alpha=0.3)
    
    print("      [OK] Panel G complete")
    
    # Panel H: Abundance distribution - DDA vs DIA
    print("    Creating Panel H: Abundance distribution - DDA vs DIA...")
    ax_h = fig3.add_subplot(gs3[0, 1])
    dda_abundance = abundance_df[abundance_df['Acquisition'] == 'DDA']['Abundance'].values
    dia_abundance = abundance_df[abundance_df['Acquisition'] == 'DIA']['Abundance'].values
    
    if len(dda_abundance) > 0 and len(dia_abundance) > 0:
        parts_h = ax_h.violinplot([dda_abundance, dia_abundance], positions=[1, 2], 
                                  widths=0.6, showmeans=True, showmedians=True)
        for pc, color in zip(parts_h['bodies'], [colors['DDA'], colors['DIA']]):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        ax_h.set_xticks([1, 2])
        ax_h.set_xticklabels(['DDA', 'DIA'])
        ax_h.set_ylabel('Abundance\n(fraction of samples)', fontsize=11, fontweight='bold')
        ax_h.set_title('(H) Abundance Distribution: DDA vs DIA', fontsize=12, fontweight='bold')
        ax_h.set_ylim(0, 1.05)
        ax_h.grid(True, alpha=0.3, axis='y')
    
    fig3.suptitle('Protein Abundance by Methodology', fontsize=14, fontweight='bold', y=1.02)
    plt.savefig(output_dir / "methodology_figure3_abundance.png", bbox_inches='tight', dpi=300)
    plt.savefig(output_dir / "methodology_figure3_abundance.pdf", bbox_inches='tight')
    plt.close()
    print("  [OK] Saved: methodology_figure3_abundance.png and .pdf")
    print("  [OK] Step 3 complete!")
    
    # Save data tables
    print("\n  Saving data tables...")
    df.to_csv(output_dir / "methodology_dataset_summary.csv", index=False)
    abundance_df.to_csv(output_dir / "methodology_protein_abundance.csv", index=False)
    
    print(f"  [OK] Saved data tables")
    
    # Perform linear model analysis
    print("\nPerforming linear model analysis...")
    perform_linear_model_analysis(df, output_dir)
    
    # Perform additional stratified analyses
    print("\nPerforming stratified analyses...")
    perform_stratified_analyses(df, abundance_df, output_dir)

def perform_linear_model_analysis(df, output_dir):
    """Fit linear model: n_proteins ~ acquisition + fractionation + depletion + log(n_samples)."""
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import LabelEncoder
        
        # Prepare data
        model_df = df.copy()
        model_df['log_n_samples'] = np.log10(model_df['N_Samples'] + 1)
        
        # Encode categorical variables
        le_acq = LabelEncoder()
        le_frac = LabelEncoder()
        le_dep = LabelEncoder()
        
        model_df['acquisition_encoded'] = le_acq.fit_transform(model_df['Acquisition'])
        model_df['fractionation_encoded'] = le_frac.fit_transform(model_df['Fractionation'])
        model_df['depletion_encoded'] = le_dep.fit_transform(model_df['Depletion'])
        
        # Fit model
        X = model_df[['acquisition_encoded', 'fractionation_encoded', 'depletion_encoded', 'log_n_samples']]
        y = model_df['N_Proteins']
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Get coefficients
        coefficients = pd.DataFrame({
            'Feature': ['Acquisition (DDA=0, DIA=1)', 'Fractionation (Unfractionated=0, Fractionated=1)', 
                       'Depletion (Non-depleted=0, Depleted=1)', 'Log(N_Samples)'],
            'Coefficient': model.coef_,
            'Intercept': [model.intercept_] + [None] * (len(model.coef_) - 1)
        })
        
        # Calculate R-squared
        r_squared = model.score(X, y)
        coefficients['R_squared'] = [r_squared] + [None] * (len(model.coef_) - 1)
        
        coefficients.to_csv(output_dir / "methodology_linear_model.csv", index=False)
        print(f"  Saved: methodology_linear_model.csv")
        print(f"  R² = {r_squared:.3f}")
        print("\n  Model coefficients:")
        print(coefficients.to_string(index=False))
        
    except ImportError:
        print("  Warning: sklearn not available, skipping linear model analysis")
    except Exception as e:
        print(f"  Warning: Error in linear model analysis: {e}")

def perform_stratified_analyses(df, abundance_df, output_dir):
    """Perform stratified analyses to control for confounding."""
    stratified_results = []
    
    # 1. Compare DDA vs DIA only in unfractionated datasets
    unfractionated_df = df[df['Fractionation'] == 'unfractionated']
    if len(unfractionated_df[unfractionated_df['Acquisition'] == 'DDA']) > 0 and \
       len(unfractionated_df[unfractionated_df['Acquisition'] == 'DIA']) > 0:
        dda_unfrac = unfractionated_df[unfractionated_df['Acquisition'] == 'DDA']['N_Proteins'].values
        dia_unfrac = unfractionated_df[unfractionated_df['Acquisition'] == 'DIA']['N_Proteins'].values
        if len(dda_unfrac) > 1 and len(dia_unfrac) > 1:
            stat, pval = stats.mannwhitneyu(dda_unfrac, dia_unfrac, alternative='two-sided')
            stratified_results.append({
                'Analysis': 'DDA vs DIA (unfractionated only)',
                'DDA_n': len(dda_unfrac),
                'DDA_median': np.median(dda_unfrac),
                'DIA_n': len(dia_unfrac),
                'DIA_median': np.median(dia_unfrac),
                'p_value': pval
            })
    
    # 2. Compare fractionation only within DDA datasets
    dda_df = df[df['Acquisition'] == 'DDA']
    if len(dda_df[dda_df['Fractionation'] == 'fractionated']) > 0 and \
       len(dda_df[dda_df['Fractionation'] == 'unfractionated']) > 0:
        frac_dda = dda_df[dda_df['Fractionation'] == 'fractionated']['N_Proteins'].values
        unfrac_dda = dda_df[dda_df['Fractionation'] == 'unfractionated']['N_Proteins'].values
        if len(frac_dda) > 1 and len(unfrac_dda) > 1:
            stat, pval = stats.mannwhitneyu(unfrac_dda, frac_dda, alternative='two-sided')
            stratified_results.append({
                'Analysis': 'Fractionation (DDA only)',
                'Unfractionated_n': len(unfrac_dda),
                'Unfractionated_median': np.median(unfrac_dda),
                'Fractionated_n': len(frac_dda),
                'Fractionated_median': np.median(frac_dda),
                'p_value': pval
            })
    
    # 3. Compare depletion only within unfractionated datasets
    unfractionated_df = df[df['Fractionation'] == 'unfractionated']
    if len(unfractionated_df[unfractionated_df['Depletion'] == 'depleted']) > 0 and \
       len(unfractionated_df[unfractionated_df['Depletion'] == 'non_depleted']) > 0:
        dep_unfrac = unfractionated_df[unfractionated_df['Depletion'] == 'depleted']['N_Proteins'].values
        nondep_unfrac = unfractionated_df[unfractionated_df['Depletion'] == 'non_depleted']['N_Proteins'].values
        if len(dep_unfrac) > 1 and len(nondep_unfrac) > 1:
            stat, pval = stats.mannwhitneyu(nondep_unfrac, dep_unfrac, alternative='two-sided')
            stratified_results.append({
                'Analysis': 'Depletion (unfractionated only)',
                'Non_depleted_n': len(nondep_unfrac),
                'Non_depleted_median': np.median(nondep_unfrac),
                'Depleted_n': len(dep_unfrac),
                'Depleted_median': np.median(dep_unfrac),
                'p_value': pval
            })
    
    if len(stratified_results) > 0:
        stratified_df = pd.DataFrame(stratified_results)
        stratified_df.to_csv(output_dir / "methodology_stratified_analyses.csv", index=False)
        print(f"  Saved: methodology_stratified_analyses.csv")
        print("\n  Stratified analysis results:")
        print(stratified_df.to_string(index=False))
    else:
        print("  No stratified analyses possible (insufficient data)")

def main():
    """Main function to run methodology analysis."""
    print("=" * 80)
    print("METHODOLOGY ANALYSIS")
    print("=" * 80)
    print("Analyzing effect of acquisition method, fractionation, and depletion")
    print("on plasma proteome coverage and protein detectability\n")
    
    # Step 1: Download and parse SDRF files
    methodology_data = download_and_parse_sdrf_files()
    
    if len(methodology_data) == 0:
        print("\nError: No methodology data found. Exiting.")
        return
    
    # Step 2: Load protein data
    dataset_protein_data = load_protein_data_for_methodology(methodology_data)
    
    if len(dataset_protein_data) == 0:
        print("\nError: No protein data loaded. Exiting.")
        return
    
    # Restrict to plasma/serum only for methodology figures (exclude cell types)
    plasma_serum_dict = get_datasets_with_msstats()
    plasma_serum_names = set()
    for variants in plasma_serum_dict.values():
        plasma_serum_names.update(variants)
    dataset_protein_data = {k: v for k, v in dataset_protein_data.items() if k in plasma_serum_names}
    print(f"\n  Filtered to plasma/serum only: {len(dataset_protein_data)} datasets (cell types excluded)")
    if len(dataset_protein_data) == 0:
        print("\nError: No plasma/serum datasets after filtering. Exiting.")
        return
    
    # Step 3: Create multipanel figure
    create_methodology_multipanel_figure(dataset_protein_data, output_dir)
    
    print("\n" + "=" * 80)
    print("[OK] ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    main()
