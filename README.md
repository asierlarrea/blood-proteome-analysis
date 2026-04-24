# blood-proteome-analysis

A reproducible Python pipeline for large-scale blood proteomics analysis across plasma, serum, and blood-derived cell types, including data curation, tissue/cell-type profiling, disease and sex analyses, and external database benchmarking.

## Scripts

- `A_download_all_datasets.py`: download and organize MSstats + SDRF inputs.
- `A.5_format_analysis.py`: read-only CSV format diagnostics for MSstats files.
- `B_data_preparation_and_filtering.py`: core heavy preprocessing and cache building.
- `C_reported_reanalyzed_proteins.py`: reported vs reanalyzed protein comparisons.
- `D_methodology_analysis.py`: methodology-focused comparisons.
- `E_plasma_and_cell_types.py`: plasma/serum vs cell-type analyses and figures.
- `F_database_comparison.py`: quantm vs external database benchmarking.
- `G_disease_analysis.py`: disease mapping, exports, and disease plots.
- `H_sex_analysis.py`: sex coverage, prevalence analysis, and prediction workflow.
- `tool.py`: optional interactive utility app.

## Recommended Run Order

1. `B_data_preparation_and_filtering.py`
2. `C_reported_reanalyzed_proteins.py`
3. `D_methodology_analysis.py`
4. `E_plasma_and_cell_types.py`
5. `F_database_comparison.py`
6. `G_disease_analysis.py`
7. `H_sex_analysis.py`

Run `A_download_all_datasets.py` only when adding/updating datasets.

## Inputs

Expected root-level input folders:

- `msstats/`
- `sdrf_files/`
- `external_databases/`
- `entry_name_mapping/`

## Core B -> E Logic (Integrated)

`B_data_preparation_and_filtering.py` performs the heavy work once and writes curated cache files used by downstream scripts.

Main responsibilities of `B_`:

- Normalize protein identifiers (including entry-name mapping).
- Filter decoys, contaminants, and non-human proteins.
- Apply peptide/quality filtering and ENTRAP handling.
- Build tissue/cell-type grouped outputs.
- Write cache files under `cache/` plus summary tables in `data_preparation_output/`.

`E_plasma_and_cell_types.py` is designed to be lightweight and consume B outputs:

- Reads curated `cache/*_processed.parquet` and metadata.
- Uses B-generated summaries (`01...`, `02...`) for consistent dataset/tissue scope.
- Produces plasma/serum vs cell-type comparison plots and tables.
- Avoids redoing B’s expensive preprocessing whenever possible.

In short: B does heavy lifting and caching; E (and later scripts) mostly read curated outputs.

## Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

## Notes

- This repository tracks source code and lightweight documentation.
- Large local data, caches, and generated analysis outputs are excluded via `.gitignore`.
