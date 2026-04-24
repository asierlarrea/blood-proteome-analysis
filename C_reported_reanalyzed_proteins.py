"""
Reported vs Reanalyzed Protein Counts Analysis (PROTEIN-LEVEL)
Compares protein counts reported in articles vs reanalyzed with quantms
Uses protein-level cache to count proteins (not genes)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import sys
import warnings
import csv
warnings.filterwarnings('ignore')

# Add parent directory to path
work_dir = Path(r"G:\My Drive\Ikasketak\Postdoc\Cambridge\Cursor\blood_proteome_analysis")
sys.path.insert(0, str(work_dir))
os.chdir(work_dir)

# Import shared utilities
from shared_utils import (
    find_cache_file,
    get_cache_filter_info
)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# File paths
excel_file = Path(r"G:\My Drive\Ikasketak\Postdoc\Cambridge\Paper Plasma Proteomics\Dataset annotation\Datasets anotados 2025.xlsx")
output_dir = work_dir / "reported_vs_reanalyzed"
output_dir.mkdir(parents=True, exist_ok=True)
sdrf_dir = work_dir / "sdrf_files"  # Shared SDRF files directory

# Load protein counts from cache (PROTEINS, not genes)
print("=" * 60)
print("REPORTED VS REANALYZED PROTEIN COUNTS ANALYSIS (PROTEIN-LEVEL)")
print("=" * 60)

# Load data from Excel
print(f"\nLoading data from: {excel_file}")
df = pd.read_excel(excel_file, header=1)

print(f"Total rows loaded: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# Clean column names (remove extra spaces)
df.columns = df.columns.str.strip()

# Identify key columns
reported_col = "Proteins in article"
reanalyzed_col = "Proteins quantms"
pxd_col = "PXD"
technique_col = "Technique"
sample_col = "Sample"
tissue_col = "Tissue"

# Check if columns exist
required_cols = [reported_col, reanalyzed_col, pxd_col]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"ERROR: Missing required columns: {missing_cols}")
    print(f"Available columns: {df.columns.tolist()}")
    exit(1)

# Load actual protein counts from protein-level cache
print("\nLoading protein counts from protein-level cache...")
cache_dir = work_dir / "cache"
data_prep_output_dir = work_dir / "data_preparation_output"

try:
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    print("  Warning: pyarrow not available, cannot load from cache")

# Try to load acquisition methods from B_data_preparation output
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

# Count proteins from cache for each dataset
if PARQUET_AVAILABLE:
    # Get cache filter info
    cache_info = get_cache_filter_info(cache_dir)
    print(f"  Cache configuration: min {cache_info['min_peptides']} peptides ({cache_info['filter_type']})")
    
    cache_protein_counts = {}
    cache_file_mapping = {}  # Map PXD ID to actual cache file used
    
    # Get all available cache files to check for split datasets
    all_cache_files = list(cache_dir.glob("*_processed*.parquet"))
    cache_file_names = {f.stem.replace('_processed', '').replace('_unfiltered', '').replace('_min2pep', '').replace('_min3pep', ''): f for f in all_cache_files}
    
    for pxd_id in df[pxd_col].dropna().unique():
        pxd_str = str(pxd_id).strip()
        
        # Check if this PXD ID appears in Excel with a suffix (e.g., "PXD002854-plasma")
        # If it does, we should NOT combine - use exact match
        has_suffix_in_excel = '-' in pxd_str and pxd_str in df[pxd_col].values
        
        if has_suffix_in_excel:
            # Exact match required - don't combine
            cache_file = find_cache_file(pxd_str, cache_dir=cache_dir)
            if cache_file and cache_file.exists():
                try:
                    cached_data = pd.read_parquet(cache_file)
                    n_proteins = cached_data['Protein'].nunique()
                    cache_protein_counts[pxd_str] = n_proteins
                    cache_file_mapping[pxd_str] = cache_file.name
                    print(f"  {pxd_str}: {n_proteins} proteins (exact match)")
                except Exception as e:
                    print(f"  Warning: Error loading {pxd_str} from cache: {e}")
        else:
            # No suffix in Excel - check if dataset is split in cache files
            # Look for all cache files that start with this PXD ID
            base_pxd = pxd_str.split('-')[0]  # Get base PXD (e.g., "PXD004352" from "PXD004352")
            matching_cache_files = []
            
            # Try exact match first
            exact_match = find_cache_file(pxd_str, cache_dir=cache_dir)
            if exact_match and exact_match.exists():
                matching_cache_files.append(exact_match)
            
            # Also look for split versions (e.g., PXD004352-b_cell, PXD004352-platelet)
            for cache_name, cache_path in cache_file_names.items():
                # Check if cache file name starts with base PXD and has a suffix
                if cache_name.startswith(base_pxd + '-') and cache_name != pxd_str:
                    # This is a split version
                    if cache_path.exists():
                        matching_cache_files.append(cache_path)
            
            if matching_cache_files:
                # Combine all matching cache files
                all_proteins = set()
                files_used = []
                for cache_file in matching_cache_files:
                    try:
                        cached_data = pd.read_parquet(cache_file)
                        proteins = set(cached_data['Protein'].dropna().unique())
                        all_proteins.update(proteins)
                        files_used.append(cache_file.name)
                    except Exception as e:
                        print(f"  Warning: Error loading {cache_file.name}: {e}")
                
                if all_proteins:
                    n_proteins = len(all_proteins)
                    cache_protein_counts[pxd_str] = n_proteins
                    cache_file_mapping[pxd_str] = f"Combined: {', '.join(files_used)}"
                    print(f"  {pxd_str}: {n_proteins} unique proteins (combined from {len(files_used)} files: {', '.join([f.name for f in matching_cache_files])})")
            else:
                # Try standard lookup as fallback
                cache_file = find_cache_file(pxd_str, cache_dir=cache_dir)
                if cache_file and cache_file.exists():
                    try:
                        cached_data = pd.read_parquet(cache_file)
                        n_proteins = cached_data['Protein'].nunique()
                        cache_protein_counts[pxd_str] = n_proteins
                        cache_file_mapping[pxd_str] = cache_file.name
                        print(f"  {pxd_str}: {n_proteins} proteins")
                    except Exception as e:
                        print(f"  Warning: Error loading {pxd_str} from cache: {e}")
    
    # Update reanalyzed_proteins column with actual counts from cache
    def get_cache_count(pxd_val):
        if pd.isna(pxd_val):
            return np.nan
        pxd_str = str(pxd_val).strip()
        return cache_protein_counts.get(pxd_str, np.nan)
    
    df['reanalyzed_proteins_from_cache'] = df[pxd_col].apply(get_cache_count)
    print(f"  Loaded protein counts from cache for {len(cache_protein_counts)} datasets")
    
    # Use cache counts if available, otherwise use Excel values
    df['reanalyzed_proteins'] = df['reanalyzed_proteins_from_cache'].fillna(df[reanalyzed_col])
else:
    # Fallback to Excel values
    print("  Using protein counts from Excel (cache not available)")
    df['reanalyzed_proteins'] = df[reanalyzed_col]

# Create a new column with filtered counts (from cache, which have all filtering applied)
# These counts have contaminants, entrapments, decoys, and non-human proteins removed
print("\nCreating filtered protein counts column (from B_ script cache files)...")
print("  Note: Cache files contain proteins after removing contaminants, entrapments, decoys, and non-human proteins")

if 'reanalyzed_proteins_from_cache' in df.columns:
    df['reanalyzed_proteins_filtered'] = df['reanalyzed_proteins_from_cache'].copy()
    print(f"  Using cache counts for {df['reanalyzed_proteins_filtered'].notna().sum()} datasets")
else:
    # If cache not available, initialize with NaN (will be filled later)
    df['reanalyzed_proteins_filtered'] = np.nan
    print("  Warning: Cache not available, using Excel values as fallback")

# Fill with Excel values if cache not available (as fallback)
if df['reanalyzed_proteins_filtered'].isna().all():
    df['reanalyzed_proteins_filtered'] = df[reanalyzed_col]
    print("  Using Excel 'Proteins quantms' column as fallback")

# Clean and prepare data
print("\nCleaning data...")

# Convert protein counts to numeric, handling errors
def to_numeric_safe(val):
    """Convert value to numeric, handling text and errors."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    # Try to extract number from text
    val_str = str(val).strip()
    # Remove common text prefixes/suffixes
    val_str = val_str.replace(',', '').replace(' ', '')
    try:
        return float(val_str)
    except (ValueError, AttributeError):
        return np.nan

df['reported_proteins'] = df[reported_col].apply(to_numeric_safe)
df['reanalyzed_proteins'] = df['reanalyzed_proteins'].apply(to_numeric_safe)
df['reanalyzed_proteins_filtered'] = df['reanalyzed_proteins_filtered'].apply(to_numeric_safe)

# Calculate delta (reanalyzed - reported) - using original quantms column
df['delta_proteins'] = df['reanalyzed_proteins'] - df['reported_proteins']

# Calculate delta using filtered counts (from B_ script, contaminants/entrapments removed)
df['delta_proteins_filtered'] = df['reanalyzed_proteins_filtered'] - df['reported_proteins']

# Use only datasets that have msstats (cache) - the definitive ones
# So all plots show the same set of datasets (no grey/missing points)
has_cache = df['reanalyzed_proteins_from_cache'].notna() if 'reanalyzed_proteins_from_cache' in df.columns else pd.Series(False, index=df.index)

valid_mask_filtered = (
    df['reported_proteins'].notna() & 
    df['reanalyzed_proteins_filtered'].notna() &
    (df['reported_proteins'] > 0) &
    (df['reanalyzed_proteins_filtered'] > 0) &
    has_cache
)

df_clean_filtered = df[valid_mask_filtered].copy()

# For "Excel data" plots we use the same subset (only msstats) so both plot pairs show the same datasets
df_clean = df_clean_filtered.copy()
# In df_clean, reanalyzed_proteins is already from cache for these rows; we keep it for Y-axis in plots 1 & 3

print(f"Valid datasets (with msstats/cache - used for all plots): {len(df_clean_filtered)}")
print(f"Datasets with improvement (delta > 0): {(df_clean_filtered['delta_proteins_filtered'] > 0).sum()}")
print(f"Median delta: {df_clean_filtered['delta_proteins_filtered'].median():.1f}")

# Categorize by tissue type (plasma/serum vs cell types)
def categorize_tissue(tissue_val):
    """Categorize tissue into plasma/serum or cell types."""
    if pd.isna(tissue_val):
        return 'Unknown'
    tissue_str = str(tissue_val).lower().strip()
    if 'plasma' in tissue_str or 'serum' in tissue_str:
        return 'Plasma/Serum'
    else:
        return 'Cell Types'

if tissue_col in df_clean.columns:
    df_clean['tissue_category'] = df_clean[tissue_col].apply(categorize_tissue)
else:
    # If no tissue column, try to infer from other columns or set as unknown
    df_clean['tissue_category'] = 'Unknown'
    print("  Warning: No tissue column found, all datasets marked as 'Unknown'")

# Also add tissue_category to df_clean_filtered
if tissue_col in df_clean_filtered.columns:
    df_clean_filtered['tissue_category'] = df_clean_filtered[tissue_col].apply(categorize_tissue)
else:
    df_clean_filtered['tissue_category'] = 'Unknown'

print(f"\nTissue categories (original):")
print(df_clean['tissue_category'].value_counts())
print(f"\nTissue categories (filtered):")
print(df_clean_filtered['tissue_category'].value_counts())

# ============================================
# Determine DDA/DIA from SDRF files
# ============================================

def get_acquisition_method_from_sdrf(pxd_id):
    """Get acquisition method (DDA/DIA) from SDRF file.
    
    First tries exact match (e.g., PXD013231-DDA.sdrf.tsv),
    then falls back to base PXD ID (e.g., PXD013231.sdrf.tsv) if exact match doesn't exist.
    """
    if pd.isna(pxd_id):
        return None
    
    pxd_str = str(pxd_id).strip()
    
    # First try exact match (e.g., PXD013231-DDA.sdrf.tsv)
    sdrf_file = sdrf_dir / f"{pxd_str}.sdrf.tsv"
    
    # If exact match doesn't exist, try base PXD ID (remove suffix like -DDA, -DIA, -plasma, etc.)
    if not sdrf_file.exists():
        # Extract base PXD ID (everything before the first hyphen, if any)
        if '-' in pxd_str:
            base_pxd = pxd_str.split('-')[0]
            sdrf_file = sdrf_dir / f"{base_pxd}.sdrf.tsv"
    
    if not sdrf_file.exists():
        return None
    
    try:
        with open(sdrf_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            rows = list(reader)
        
        if len(rows) == 0:
            return None
        
        # Extract acquisition method
        acquisition_values = set()
        for row in rows:
            acquisition = row.get('comment[proteomics data acquisition method]', '').strip()
            if acquisition:
                acquisition_values.add(acquisition)
        
        # Classify acquisition method
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
        print(f"  Warning: Error parsing SDRF for {pxd_str}: {e}")
        return None

print("\nDetermining acquisition method (DDA/DIA)...")
acquisition_methods = {}

# First, try to get from data preparation output (most reliable)
for pxd_id in df_clean[pxd_col].unique():
    pxd_str = str(pxd_id).strip()
    
    # Try exact match first
    if pxd_str in acquisition_from_prep:
        acquisition_methods[pxd_id] = acquisition_from_prep[pxd_str]
        continue
    
    # Try with common suffixes
    for suffix in ['-plasma', '-serum', '-erythrocyte', '-DDA', '-DIA', '-blood_serum', '-blood_plasma']:
        dataset_name = f"{pxd_str}{suffix}"
        if dataset_name in acquisition_from_prep:
            acquisition_methods[pxd_id] = acquisition_from_prep[dataset_name]
            break
    
    # If still not found, try SDRF file
    if pxd_id not in acquisition_methods:
        method = get_acquisition_method_from_sdrf(pxd_id)
        if method:
            acquisition_methods[pxd_id] = method

df_clean['acquisition_method'] = df_clean[pxd_col].map(acquisition_methods)
df_clean['acquisition_method'] = df_clean['acquisition_method'].fillna('Unknown')

# Also add acquisition_method to df_clean_filtered
df_clean_filtered['acquisition_method'] = df_clean_filtered[pxd_col].map(acquisition_methods)
df_clean_filtered['acquisition_method'] = df_clean_filtered['acquisition_method'].fillna('Unknown')

print(f"\nAcquisition methods (original):")
print(df_clean['acquisition_method'].value_counts())
print(f"\nAcquisition methods (filtered):")
print(df_clean_filtered['acquisition_method'].value_counts())

# ============================================
# PLOT SET 1: DDA/DIA Comparison (2 plots)
# ============================================

print("\n" + "=" * 60)
print("PLOT 1: DDA/DIA Comparison")
print("=" * 60)

# Color scheme for acquisition methods
acquisition_colors = {
    'DDA': '#3498db',  # Blue
    'DIA': '#e74c3c',   # Red
    'Unknown': '#6C757D'  # Gray
}

print("\nCreating scatter plot: Reported vs Reanalyzed (by DDA/DIA)")

fig, ax = plt.subplots(figsize=(10, 8))

# Plot scatter points colored by acquisition method
for method in df_clean_filtered['acquisition_method'].unique():
    mask = df_clean_filtered['acquisition_method'] == method
    if mask.sum() > 0:
        subset = df_clean_filtered.loc[mask]
        ax.scatter(subset['reported_proteins'], 
                  subset['reanalyzed_proteins_filtered'],
                  alpha=0.7, s=50, 
                  color=acquisition_colors.get(method, '#6C757D'),
                  edgecolors='black', linewidth=0.5,
                  label=method)
        # Add PXD codes as labels
        for idx, row in subset.iterrows():
            pxd_code = str(row[pxd_col]) if pd.notna(row[pxd_col]) else ''
            ax.annotate(pxd_code, 
                       (row['reported_proteins'], row['reanalyzed_proteins_filtered']),
                       xytext=(3, 3), textcoords='offset points',
                       fontsize=6, alpha=0.7, color='black')

# Add diagonal line (y = x)
max_val = max(df_clean_filtered['reported_proteins'].max(), df_clean_filtered['reanalyzed_proteins_filtered'].max())
ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1.5, alpha=0.5, label='y = x (no improvement)')

ax.set_xlabel('Proteins Reported in Article', fontsize=12, fontweight='bold')
ax.set_ylabel('Proteins Reanalyzed (Filtered - contaminants/entrapments removed)', fontsize=12, fontweight='bold')
ax.set_title('Reported vs Reanalyzed Protein counts', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)

# Calculate net new proteins for each acquisition method
dda_mask = df_clean_filtered['acquisition_method'] == 'DDA'
dia_mask = df_clean_filtered['acquisition_method'] == 'DIA'

if dda_mask.sum() > 0:
    dda_deltas = df_clean_filtered.loc[dda_mask, 'delta_proteins_filtered']
    dda_new = dda_deltas[dda_deltas > 0].sum() if (dda_deltas > 0).any() else 0
    dda_removed = abs(dda_deltas[dda_deltas < 0].sum()) if (dda_deltas < 0).any() else 0
    dda_net = int(dda_new - dda_removed)
else:
    dda_new = 0
    dda_removed = 0
    dda_net = 0

if dia_mask.sum() > 0:
    dia_deltas = df_clean_filtered.loc[dia_mask, 'delta_proteins_filtered']
    dia_new = dia_deltas[dia_deltas > 0].sum() if (dia_deltas > 0).any() else 0
    dia_removed = abs(dia_deltas[dia_deltas < 0].sum()) if (dia_deltas < 0).any() else 0
    dia_net = int(dia_new - dia_removed)
else:
    dia_new = 0
    dia_removed = 0
    dia_net = 0

# Create statistics text box
stats_text = f'Net new proteins:\n'
if dda_mask.sum() > 0:
    stats_text += f'DDA: {dda_net} ({int(dda_new)} new, {int(dda_removed)} removed)\n'
if dia_mask.sum() > 0:
    stats_text += f'DIA: {dia_net} ({int(dia_new)} new, {int(dia_removed)} removed)'

ax.text(0.05, 0.95, stats_text,
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
plt.savefig(output_dir / "reported_vs_reanalyzed_scatter_dda_dia.png", bbox_inches='tight', dpi=300)
plt.close()
print("  Saved: reported_vs_reanalyzed_scatter_dda_dia.png")


# ============================================
# PLOT 2: Plasma/Cell Type Comparison
# ============================================

print("\n" + "=" * 60)
print("PLOT 2: Plasma/Cell Type Comparison")
print("=" * 60)

# Color scheme for tissue categories
tissue_colors = {
    'Plasma/Serum': '#2E86AB',  # Blue
    'Cell Types': '#A23B72',     # Purple
    'Unknown': '#6C757D'         # Gray
}

print("\nCreating scatter plot: Reported vs Reanalyzed (by Tissue)")

fig, ax = plt.subplots(figsize=(10, 8))

# Plot scatter points colored by tissue category
for category in df_clean_filtered['tissue_category'].unique():
    mask = df_clean_filtered['tissue_category'] == category
    if mask.sum() > 0:
        subset = df_clean_filtered.loc[mask]
        ax.scatter(subset['reported_proteins'], 
                  subset['reanalyzed_proteins_filtered'],
                  alpha=0.7, s=50, 
                  color=tissue_colors.get(category, '#6C757D'),
                  edgecolors='black', linewidth=0.5,
                  label=category)
        # Add PXD codes as labels
        for idx, row in subset.iterrows():
            pxd_code = str(row[pxd_col]) if pd.notna(row[pxd_col]) else ''
            ax.annotate(pxd_code, 
                       (row['reported_proteins'], row['reanalyzed_proteins_filtered']),
                       xytext=(3, 3), textcoords='offset points',
                       fontsize=6, alpha=0.7, color='black')

# Add diagonal line (y = x)
max_val = max(df_clean_filtered['reported_proteins'].max(), df_clean_filtered['reanalyzed_proteins_filtered'].max())
ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1.5, alpha=0.5, label='y = x (no improvement)')

ax.set_xlabel('Proteins Reported in Article', fontsize=12, fontweight='bold')
ax.set_ylabel('Proteins Reanalyzed (Filtered - contaminants/entrapments removed)', fontsize=12, fontweight='bold')
ax.set_title('Reported vs Reanalyzed Protein counts', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)

# Calculate net new proteins for each tissue category
plasma_mask = df_clean_filtered['tissue_category'] == 'Plasma/Serum'
cell_mask = df_clean_filtered['tissue_category'] == 'Cell Types'

if plasma_mask.sum() > 0:
    plasma_deltas = df_clean_filtered.loc[plasma_mask, 'delta_proteins_filtered']
    plasma_new = plasma_deltas[plasma_deltas > 0].sum() if (plasma_deltas > 0).any() else 0
    plasma_removed = abs(plasma_deltas[plasma_deltas < 0].sum()) if (plasma_deltas < 0).any() else 0
    plasma_net = int(plasma_new - plasma_removed)
else:
    plasma_new = 0
    plasma_removed = 0
    plasma_net = 0

if cell_mask.sum() > 0:
    cell_deltas = df_clean_filtered.loc[cell_mask, 'delta_proteins_filtered']
    cell_new = cell_deltas[cell_deltas > 0].sum() if (cell_deltas > 0).any() else 0
    cell_removed = abs(cell_deltas[cell_deltas < 0].sum()) if (cell_deltas < 0).any() else 0
    cell_net = int(cell_new - cell_removed)
else:
    cell_new = 0
    cell_removed = 0
    cell_net = 0

# Create statistics text box
stats_text = f'Net new proteins:\n'
if plasma_mask.sum() > 0:
    stats_text += f'Plasma/Serum: {plasma_net} ({int(plasma_new)} new, {int(plasma_removed)} removed)\n'
if cell_mask.sum() > 0:
    stats_text += f'Cell Types: {cell_net} ({int(cell_new)} new, {int(cell_removed)} removed)'

ax.text(0.05, 0.95, stats_text,
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
plt.savefig(output_dir / "reported_vs_reanalyzed_scatter_tissue.png", bbox_inches='tight', dpi=300)
plt.close()
print("  Saved: reported_vs_reanalyzed_scatter_tissue.png")


# ============================================
# Save summary statistics
# ============================================

summary_stats = {
    'total_datasets': len(df_clean_filtered),
    'datasets_with_improvement': (df_clean_filtered['delta_proteins_filtered'] > 0).sum(),
    'pct_with_improvement': (df_clean_filtered['delta_proteins_filtered'] > 0).sum() / len(df_clean_filtered) * 100 if len(df_clean_filtered) > 0 else 0,
    'median_delta': df_clean_filtered['delta_proteins_filtered'].median(),
    'mean_delta': df_clean_filtered['delta_proteins_filtered'].mean(),
    'std_delta': df_clean_filtered['delta_proteins_filtered'].std(),
    'min_delta': df_clean_filtered['delta_proteins_filtered'].min(),
    'max_delta': df_clean_filtered['delta_proteins_filtered'].max(),
    'median_reported': df_clean_filtered['reported_proteins'].median(),
    'median_reanalyzed': df_clean_filtered['reanalyzed_proteins_filtered'].median(),
}

summary_df = pd.DataFrame([summary_stats])
summary_df.to_csv(output_dir / "summary_statistics.csv", index=False)
print("\n  Saved: summary_statistics.csv")

# Save detailed data (msstats/cache only)
columns_to_save = [pxd_col, 'reported_proteins', 'reanalyzed_proteins_filtered', 'delta_proteins_filtered', 'tissue_category', 'acquisition_method']
if tissue_col in df_clean_filtered.columns:
    columns_to_save.append(tissue_col)
df_clean_filtered[columns_to_save].to_csv(output_dir / "detailed_data.csv", index=False)
print("  Saved: detailed_data.csv")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"\nResults saved in: {output_dir}")
print(f"\nSummary (datasets with msstats/cache):")
print(f"  Total datasets analyzed: {summary_stats['total_datasets']}")
print(f"  Datasets with improvement: {summary_stats['datasets_with_improvement']} ({summary_stats['pct_with_improvement']:.1f}%)")
print(f"  Median delta: {summary_stats['median_delta']:.1f} proteins")
print(f"  Mean delta: {summary_stats['mean_delta']:.1f} ± {summary_stats['std_delta']:.1f} proteins")
print(f"\nSummary (Filtered - using B_ script counts, contaminants/entrapments removed):")
print(f"  Total datasets analyzed: {summary_stats['total_datasets']}")
print(f"  Datasets with improvement: {summary_stats['datasets_with_improvement']} ({summary_stats['pct_with_improvement']:.1f}%)")
print(f"  Median delta: {summary_stats['median_delta']:.1f} proteins")
print(f"  Mean delta: {summary_stats['mean_delta']:.1f} ± {summary_stats['std_delta']:.1f} proteins")