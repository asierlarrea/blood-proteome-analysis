"""
Create a t-SNE map of blood samples colored by disease.

Data source:
- Per-sample protein abundance from B cache (`cache/*_processed.parquet`)
  using feature-based abundance (`PeptideCount`).
- Disease labels from G_disease_analysis sample-disease mapping.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import G_disease_analysis as G


WORK_DIR = Path(r"G:\My Drive\Ikasketak\Postdoc\Cambridge\Cursor\blood_proteome_analysis")
CACHE_DIR = WORK_DIR / "cache"
OUT_DIR = WORK_DIR / "disease_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset_tissue_map() -> dict[str, str]:
    df = pd.read_csv(
        WORK_DIR / "data_preparation_output" / "01_dataset_summary_before_and_after_filtering.csv",
        usecols=["Dataset", "Tissue_CellType"],
    ).drop_duplicates(subset=["Dataset"])
    return dict(zip(df["Dataset"].astype(str), df["Tissue_CellType"].astype(str)))


def get_disease_for_sample(
    dataset_name: str,
    sample_name: str,
    sample_disease_map: dict[str, dict[str, str]],
    lookup_cache: dict[str, tuple[dict[str, str], dict[str, str]]],
) -> str:
    dmap = sample_disease_map.get(dataset_name, {})
    if dataset_name not in lookup_cache:
        lookup_cache[dataset_name] = G._build_disease_alias_lookups(dmap)
    exact_lookup, normalized_lookup = lookup_cache[dataset_name]
    disease = G._lookup_disease_alias(sample_name, exact_lookup, normalized_lookup)

    if disease is None and "_" in dataset_name:
        base = dataset_name.split("_")[0]
        dmap = sample_disease_map.get(base, {})
        if base not in lookup_cache:
            lookup_cache[base] = G._build_disease_alias_lookups(dmap)
        exact_lookup, normalized_lookup = lookup_cache[base]
        disease = G._lookup_disease_alias(sample_name, exact_lookup, normalized_lookup)

    primary = G.pick_primary_disease_label(disease) if disease else None
    return primary if primary else "Unknown"


def build_sample_matrix(plasma_serum_only: bool) -> pd.DataFrame:
    sample_disease_map, _ = G.load_sample_disease_mapping(plasma_serum_only=plasma_serum_only)
    tissue_map = load_dataset_tissue_map()
    rows = []
    lookup_cache: dict[str, tuple[dict[str, str], dict[str, str]]] = {}

    for pq in sorted(CACHE_DIR.glob("*_processed.parquet")):
        dataset_name = pq.name.replace("_processed.parquet", "")
        tissue = tissue_map.get(dataset_name)
        if plasma_serum_only and tissue != "Blood Plasma/Serum":
            continue

        df = pd.read_parquet(pq, columns=["Protein", "Sample", "PeptideCount"])
        if df.empty:
            continue
        df = df.dropna(subset=["Protein", "Sample"])
        df["Sample"] = df["Sample"].astype(str)
        df["Protein"] = df["Protein"].astype(str)
        df["PeptideCount"] = pd.to_numeric(df["PeptideCount"], errors="coerce").fillna(0.0)
        df = df[df["PeptideCount"] > 0]
        if df.empty:
            continue

        grouped = (
            df.groupby(["Sample", "Protein"], as_index=False)["PeptideCount"]
            .sum()
            .rename(columns={"PeptideCount": "abundance"})
        )
        grouped["Dataset"] = dataset_name
        grouped["Disease"] = grouped["Sample"].map(
            lambda s: get_disease_for_sample(dataset_name, s, sample_disease_map, lookup_cache)
        )
        grouped["SampleKey"] = grouped["Dataset"] + "||" + grouped["Sample"]
        rows.append(grouped)

    if not rows:
        raise RuntimeError("No sample-protein abundance rows were found.")

    long_df = pd.concat(rows, ignore_index=True)
    matrix = long_df.pivot_table(
        index=["SampleKey", "Dataset", "Sample", "Disease"],
        columns="Protein",
        values="abundance",
        aggfunc="sum",
        fill_value=0.0,
    )
    matrix = matrix.reset_index()
    return matrix


def run_tsne(matrix_df: pd.DataFrame, max_features: int, perplexity: float, random_state: int) -> pd.DataFrame:
    metadata = matrix_df[["SampleKey", "Dataset", "Sample", "Disease"]].copy()
    X = matrix_df.drop(columns=["SampleKey", "Dataset", "Sample", "Disease"]).to_numpy(dtype=float)

    # Keep the most variable proteins for stable/tractable t-SNE.
    if X.shape[1] > max_features:
        var = np.var(X, axis=0)
        keep_idx = np.argsort(var)[-max_features:]
        X = X[:, keep_idx]

    X = np.log1p(X)
    X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

    pca_dims = min(50, X.shape[1], max(2, X.shape[0] - 1))
    X_pca = PCA(n_components=pca_dims, random_state=random_state).fit_transform(X)

    p = min(perplexity, max(5.0, (X_pca.shape[0] - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        perplexity=p,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
        metric="euclidean",
    )
    emb = tsne.fit_transform(X_pca)
    out = metadata.copy()
    out["tSNE1"] = emb[:, 0]
    out["tSNE2"] = emb[:, 1]
    return out


def plot_tsne(tsne_df: pd.DataFrame, output_png: Path, top_n_diseases: int = 20) -> None:
    counts = tsne_df["Disease"].value_counts()
    top = set(counts.head(top_n_diseases).index.tolist())
    plot_df = tsne_df.copy()
    plot_df["DiseasePlot"] = plot_df["Disease"].where(plot_df["Disease"].isin(top), other="Other")

    diseases = sorted(plot_df["DiseasePlot"].unique(), key=lambda d: (d == "Other", d))
    cmap = plt.get_cmap("tab20")
    color_map = {d: cmap(i % 20) for i, d in enumerate(diseases)}

    plt.figure(figsize=(12, 10))
    for d in diseases:
        sub = plot_df[plot_df["DiseasePlot"] == d]
        plt.scatter(
            sub["tSNE1"],
            sub["tSNE2"],
            s=14,
            alpha=0.75,
            c=[color_map[d]],
            label=f"{d} (n={len(sub)})",
            edgecolors="none",
        )

    plt.title("t-SNE of Blood Samples by Disease (feature-based abundance)", fontsize=14, fontweight="bold")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8, markerscale=1.4, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Create blood sample t-SNE colored by disease.")
    ap.add_argument("--plasma-serum-only", action="store_true", help="Restrict to Blood Plasma/Serum datasets only.")
    ap.add_argument("--max-features", type=int, default=1500, help="Max variable proteins used for t-SNE.")
    ap.add_argument("--perplexity", type=float, default=30.0, help="Target t-SNE perplexity (auto-capped by n).")
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    scope = "plasma_serum_only" if args.plasma_serum_only else "all_blood_datasets"
    matrix_df = build_sample_matrix(plasma_serum_only=args.plasma_serum_only)
    tsne_df = run_tsne(
        matrix_df,
        max_features=args.max_features,
        perplexity=args.perplexity,
        random_state=args.random_state,
    )

    out_png = OUT_DIR / f"tsne_blood_samples_by_disease_{scope}.png"
    out_csv = OUT_DIR / f"tsne_blood_samples_by_disease_{scope}.csv"
    out_meta = OUT_DIR / f"tsne_blood_samples_by_disease_{scope}_meta.json"

    plot_tsne(tsne_df, out_png)
    tsne_df.to_csv(out_csv, index=False)

    meta = {
        "scope": scope,
        "n_samples": int(len(tsne_df)),
        "n_datasets": int(tsne_df["Dataset"].nunique()),
        "n_diseases": int(tsne_df["Disease"].nunique()),
        "max_features": int(args.max_features),
        "perplexity_requested": float(args.perplexity),
        "random_state": int(args.random_state),
    }
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Saved:", out_png)
    print("Saved:", out_csv)
    print("Saved:", out_meta)
    print("Summary:", meta)


if __name__ == "__main__":
    main()
