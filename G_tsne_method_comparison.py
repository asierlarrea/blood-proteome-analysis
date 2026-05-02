"""
Generate t-SNE comparisons for multiple batch-effect mitigation strategies.

Outputs (for each method):
- t-SNE colored by disease
- t-SNE colored by dataset

Saved under: disease_analysis/tsne
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


WORK_DIR = Path(r"G:\My Drive\Ikasketak\Postdoc\Cambridge\Cursor\blood_proteome_analysis")
CACHE_DIR = WORK_DIR / "cache"
OUT_DIR = WORK_DIR / "disease_analysis" / "tsne"
OUT_DIR.mkdir(parents=True, exist_ok=True)

META_CSV = WORK_DIR / "disease_analysis" / "tsne_blood_samples_by_disease_all_blood_datasets.csv"


def load_metadata() -> pd.DataFrame:
    df = pd.read_csv(META_CSV, usecols=["SampleKey", "Dataset", "Sample", "Disease"])
    return df.drop_duplicates(subset=["SampleKey"]).reset_index(drop=True)


def build_sparse_matrix(meta: pd.DataFrame):
    sample_keys = set(meta["SampleKey"].astype(str))
    per_sample = defaultdict(dict)
    protein_dataset_support = defaultdict(set)

    for pq in sorted(CACHE_DIR.glob("*_processed.parquet")):
        dataset = pq.name.replace("_processed.parquet", "")
        df = pd.read_parquet(pq, columns=["Protein", "Sample", "PeptideCount"])
        if df.empty:
            continue
        df = df.dropna(subset=["Protein", "Sample"])
        df["Protein"] = df["Protein"].astype(str)
        df["Sample"] = df["Sample"].astype(str)
        df["PeptideCount"] = pd.to_numeric(df["PeptideCount"], errors="coerce").fillna(0.0)
        df = df[df["PeptideCount"] > 0]
        if df.empty:
            continue
        df["SampleKey"] = dataset + "||" + df["Sample"]
        df = df[df["SampleKey"].isin(sample_keys)]
        if df.empty:
            continue

        grp = df.groupby(["SampleKey", "Protein"], as_index=False)["PeptideCount"].sum()
        for r in grp.itertuples(index=False):
            per_sample[r.SampleKey][r.Protein] = float(r.PeptideCount)
            protein_dataset_support[r.Protein].add(dataset)

    # Keep ordering consistent with metadata
    keys_ordered = [k for k in meta["SampleKey"].astype(str).tolist() if k in per_sample]
    meta2 = meta[meta["SampleKey"].isin(keys_ordered)].copy()
    key_to_row = {k: i for i, k in enumerate(keys_ordered)}
    meta2["__row"] = meta2["SampleKey"].map(key_to_row)
    meta2 = meta2.sort_values("__row").drop(columns=["__row"]).reset_index(drop=True)

    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform([per_sample[k] for k in keys_ordered]).tocsr()
    feature_names = np.array(vec.get_feature_names_out())
    dataset_support = np.array([len(protein_dataset_support.get(p, set())) for p in feature_names], dtype=int)
    return meta2, X, feature_names, dataset_support


def top_variable_indices(X: sparse.csr_matrix, max_features: int) -> np.ndarray:
    # sparse variance: E[x^2] - E[x]^2
    n = X.shape[0]
    mean = np.asarray(X.mean(axis=0)).ravel()
    mean_sq = np.asarray(X.power(2).mean(axis=0)).ravel()
    var = np.maximum(mean_sq - mean**2, 0.0)
    if len(var) <= max_features:
        return np.arange(len(var))
    return np.argsort(var)[-max_features:]


def dataset_zscore(X: np.ndarray, dataset_labels: np.ndarray) -> np.ndarray:
    out = np.zeros_like(X, dtype=float)
    for ds in np.unique(dataset_labels):
        m = dataset_labels == ds
        sub = X[m]
        mu = sub.mean(axis=0)
        sd = sub.std(axis=0)
        sd[sd == 0] = 1.0
        out[m] = (sub - mu) / sd
    return out


def dataset_center(X: np.ndarray, dataset_labels: np.ndarray) -> np.ndarray:
    out = X.copy()
    for ds in np.unique(dataset_labels):
        m = dataset_labels == ds
        out[m] = out[m] - out[m].mean(axis=0)
    return out


def make_embedding(X: np.ndarray, random_state: int = 42) -> np.ndarray:
    Xs = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    pca_dims = min(50, Xs.shape[1], max(2, Xs.shape[0] - 1))
    Xp = PCA(n_components=pca_dims, random_state=random_state).fit_transform(Xs)
    perplexity = min(30.0, max(5.0, (Xp.shape[0] - 1) / 3.0))
    emb = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
        metric="euclidean",
    ).fit_transform(Xp)
    return emb


def plot_by_label(df: pd.DataFrame, label_col: str, title: str, out_png: Path, top_n: int = 25) -> None:
    counts = df[label_col].astype(str).value_counts()
    top = set(counts.head(top_n).index.tolist())
    plot_col = f"{label_col}_plot"
    df = df.copy()
    df[plot_col] = df[label_col].astype(str).where(df[label_col].astype(str).isin(top), other="Other")
    labels = sorted(df[plot_col].unique(), key=lambda x: (x == "Other", x))
    cmap = plt.get_cmap("tab20")
    color_map = {lbl: cmap(i % 20) for i, lbl in enumerate(labels)}

    plt.figure(figsize=(12, 10))
    for lbl in labels:
        sub = df[df[plot_col] == lbl]
        plt.scatter(
            sub["tSNE1"],
            sub["tSNE2"],
            s=12,
            alpha=0.75,
            c=[color_map[lbl]],
            label=f"{lbl} (n={len(sub)})",
            edgecolors="none",
        )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=7, markerscale=1.3, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    print("Loading metadata...")
    meta = load_metadata()
    print(f"  Samples in metadata: {len(meta):,}")

    print("Building sparse sample x protein matrix from cache...")
    meta, X_sparse, feature_names, ds_support = build_sparse_matrix(meta)
    print(f"  Matrix shape: {X_sparse.shape[0]:,} samples x {X_sparse.shape[1]:,} proteins")

    max_features = 1200
    base_idx = top_variable_indices(X_sparse, max_features=max_features)
    X_base = np.log1p(X_sparse[:, base_idx].toarray().astype(np.float32))
    dataset_labels = meta["Dataset"].astype(str).to_numpy()

    # Shared proteins (>=3 datasets), then top variable subset
    shared_mask = ds_support >= 3
    shared_idx_all = np.where(shared_mask)[0]
    if len(shared_idx_all) > 0:
        X_shared_sparse = X_sparse[:, shared_idx_all]
        idx_local = top_variable_indices(X_shared_sparse, max_features=max_features)
        X_shared = np.log1p(X_shared_sparse[:, idx_local].toarray().astype(np.float32))
    else:
        X_shared = X_base.copy()

    methods = {
        "baseline_raw": X_base,
        "dataset_zscore": dataset_zscore(X_base, dataset_labels),
        "shared_proteins_ge3": X_shared,
        "dataset_centered": dataset_center(X_base, dataset_labels),
        "shared_ge3_plus_centered": dataset_center(X_shared, dataset_labels),
    }

    summary = []
    for name, X in methods.items():
        print(f"Running t-SNE: {name}")
        emb = make_embedding(X, random_state=42)
        out_df = meta.copy()
        out_df["tSNE1"] = emb[:, 0]
        out_df["tSNE2"] = emb[:, 1]

        csv_path = OUT_DIR / f"tsne_{name}_coordinates.csv"
        out_df.to_csv(csv_path, index=False)

        plot_by_label(
            out_df,
            label_col="Disease",
            title=f"t-SNE ({name}) - colored by disease",
            out_png=OUT_DIR / f"tsne_{name}_by_disease.png",
            top_n=25,
        )
        plot_by_label(
            out_df,
            label_col="Dataset",
            title=f"t-SNE ({name}) - colored by dataset",
            out_png=OUT_DIR / f"tsne_{name}_by_dataset.png",
            top_n=25,
        )
        summary.append(
            {
                "method": name,
                "n_samples": int(out_df.shape[0]),
                "n_datasets": int(out_df["Dataset"].nunique()),
                "n_diseases": int(out_df["Disease"].nunique()),
            }
        )
        print(f"  Saved plots and coordinates for {name}")

    pd.DataFrame(summary).to_csv(OUT_DIR / "tsne_methods_summary.csv", index=False)
    print("Done. Outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
