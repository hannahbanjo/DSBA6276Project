from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    PROJECT_DIR / "airroi.csv",
    PROJECT_DIR.parent / "airroi.csv",
    PROJECT_DIR.parent.parent / "airroi.csv",
]

CLUSTER_FEATURES = [
    "guests",
    "bedrooms",
    "baths",
    "cleaning_fee",
    "num_reviews_x",
    "rating_overall",
    "superhost",
    "professional_management",
    "ttm_avg_rate",
    "ttm_occupancy",
]

LOG_FEATURES = {"cleaning_fee", "num_reviews_x", "ttm_avg_rate"}

OUTPUT_FILES = [
    "listing_attribute_inventory.csv",
    "listing_level_cluster_input.csv",
    "kmeans_k_evaluation.csv",
    "listing_clusters_k2.csv",
    "cluster_profiles_k2.csv",
    "clustering_summary.md",
]


def resolve_data_path() -> Path:
    for candidate in DATA_CANDIDATES:
        if candidate.exists():
            return candidate
    searched = "\n".join(f"- {candidate}" for candidate in DATA_CANDIDATES)
    raise FileNotFoundError(
        "Could not find airroi.csv. Checked these locations:\n"
        f"{searched}"
    )


def load_listing_level_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.drop_duplicates(subset="listing_id").copy()


def build_attribute_inventory(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        series = df[column]
        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "unique_values": int(series.nunique(dropna=False)),
                "missing_count": int(series.isna().sum()),
                "missing_pct": round(float(series.isna().mean() * 100), 2),
                "sample_value": str(series.iloc[0])[:120] if len(series) else "",
            }
        )
    return pd.DataFrame(rows).sort_values("column").reset_index(drop=True)


def prepare_features(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = df[columns].astype(float).copy()
    for column in LOG_FEATURES.intersection(columns):
        features[column] = np.log1p(features[column])
    features = features.fillna(features.median())
    standardized = (features - features.mean()) / features.std(ddof=0)
    return features, standardized


def initialize_kmeans_pp(data: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    centers = np.empty((k, data.shape[1]), dtype=float)
    centers[0] = data[rng.integers(data.shape[0])]
    closest_dist_sq = np.sum((data - centers[0]) ** 2, axis=1)

    for idx in range(1, k):
        probabilities = closest_dist_sq / closest_dist_sq.sum()
        centers[idx] = data[rng.choice(data.shape[0], p=probabilities)]
        new_dist_sq = np.sum((data - centers[idx]) ** 2, axis=1)
        closest_dist_sq = np.minimum(closest_dist_sq, new_dist_sq)

    return centers


def run_kmeans(
    data: np.ndarray,
    k: int,
    n_init: int = 40,
    max_iter: int = 200,
    seed: int = 42,
) -> tuple[float, np.ndarray, np.ndarray]:
    best_solution = None

    for attempt in range(n_init):
        rng = np.random.default_rng(seed + attempt)
        centers = initialize_kmeans_pp(data, k, rng)
        labels = None

        for _ in range(max_iter):
            squared_distances = ((data[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = squared_distances.argmin(axis=1)

            new_centers = centers.copy()
            for cluster_id in range(k):
                members = data[new_labels == cluster_id]
                if len(members) == 0:
                    new_centers[cluster_id] = data[rng.integers(data.shape[0])]
                else:
                    new_centers[cluster_id] = members.mean(axis=0)

            if labels is not None and np.array_equal(new_labels, labels):
                labels = new_labels
                centers = new_centers
                break

            labels = new_labels
            centers = new_centers

        inertia = float(((data - centers[labels]) ** 2).sum())
        if best_solution is None or inertia < best_solution[0]:
            best_solution = (inertia, labels.copy(), centers.copy())

    return best_solution


def silhouette_score_numpy(data: np.ndarray, labels: np.ndarray) -> float:
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return float("nan")

    squared_norms = np.sum(data**2, axis=1)
    distances = np.sqrt(
        np.maximum(squared_norms[:, None] + squared_norms[None, :] - 2 * data @ data.T, 0)
    )

    scores = np.zeros(len(data))
    for idx in range(len(data)):
        same_cluster = labels == labels[idx]
        same_cluster[idx] = False
        intra_cluster = distances[idx, same_cluster].mean() if same_cluster.any() else 0.0

        nearest_cluster = np.inf
        for cluster_id in unique_labels:
            if cluster_id == labels[idx]:
                continue
            other_cluster = labels == cluster_id
            nearest_cluster = min(nearest_cluster, distances[idx, other_cluster].mean())

        denominator = max(intra_cluster, nearest_cluster)
        scores[idx] = (nearest_cluster - intra_cluster) / denominator if denominator > 0 else 0.0

    return float(scores.mean())


def relabel_clusters(clustered_df: pd.DataFrame) -> pd.DataFrame:
    ordering = (
        clustered_df.groupby("cluster_id")[["ttm_avg_rate", "guests"]]
        .mean()
        .sort_values(["ttm_avg_rate", "guests"])
        .reset_index()
    )
    mapping = {old_id: new_id for new_id, old_id in enumerate(ordering["cluster_id"])}
    relabeled = clustered_df.copy()
    relabeled["cluster_id"] = relabeled["cluster_id"].map(mapping)
    return relabeled


def main() -> None:
    data_path = resolve_data_path()
    raw_df = pd.read_csv(data_path)
    listing_df = raw_df.drop_duplicates(subset="listing_id").copy()

    build_attribute_inventory(listing_df).to_csv(
        PROJECT_DIR / "listing_attribute_inventory.csv",
        index=False,
    )

    _, standardized_features = prepare_features(listing_df, CLUSTER_FEATURES)
    listing_df[["listing_id", "listing_name", "room_type"] + CLUSTER_FEATURES].to_csv(
        PROJECT_DIR / "listing_level_cluster_input.csv",
        index=False,
    )

    data_matrix = standardized_features.to_numpy()
    evaluation_rows = []
    best_by_k = {}

    for k in range(2, 9):
        inertia, labels, centers = run_kmeans(data_matrix, k=k)
        cluster_sizes = pd.Series(labels).value_counts().sort_index()
        silhouette = silhouette_score_numpy(data_matrix, labels)
        evaluation_rows.append(
            {
                "k": k,
                "inertia": round(inertia, 4),
                "silhouette": round(silhouette, 4),
                "min_cluster_size": int(cluster_sizes.min()),
                "max_cluster_size": int(cluster_sizes.max()),
                "cluster_sizes": ",".join(str(x) for x in cluster_sizes.tolist()),
            }
        )
        best_by_k[k] = (labels, centers)

    evaluation_df = pd.DataFrame(evaluation_rows)
    evaluation_df.to_csv(PROJECT_DIR / "kmeans_k_evaluation.csv", index=False)

    optimal_k = int(
        evaluation_df.sort_values(["silhouette", "inertia"], ascending=[False, True]).iloc[0]["k"]
    )
    final_labels, _ = best_by_k[optimal_k]

    clustered = listing_df.copy()
    clustered["cluster_id"] = final_labels
    clustered = relabel_clusters(clustered)
    clustered[["listing_id", "listing_name", "room_type", "cluster_id"] + CLUSTER_FEATURES].to_csv(
        PROJECT_DIR / f"listing_clusters_k{optimal_k}.csv",
        index=False,
    )

    cluster_profile = clustered.groupby("cluster_id")[CLUSTER_FEATURES].agg(["mean", "median", "count"]).round(4)
    cluster_profile.columns = [
        f"{feature}_{stat}"
        for feature, stat in cluster_profile.columns.to_flat_index()
    ]
    cluster_profile.reset_index().to_csv(PROJECT_DIR / f"cluster_profiles_k{optimal_k}.csv", index=False)

    report_lines = [
        "# DSBA6276Project Initial Clustering",
        "",
        f"- Source file: `{data_path}`",
        f"- Raw rows in source file: {len(raw_df)}",
        f"- Listing-level rows used: {len(listing_df)}",
        f"- Selected clustering features: {', '.join(CLUSTER_FEATURES)}",
        f"- Log-transformed before scaling: {', '.join(sorted(LOG_FEATURES))}",
        f"- Optimal k from silhouette scan (2 to 8): {optimal_k}",
        "",
        "## Notes",
        "",
        "- One row per listing is used so repeated monthly snapshots do not dominate the clusters.",
        "- Cluster labels are numeric and re-ordered deterministically so the initial labels stay locked.",
        "- Outputs include the attribute inventory, feature table, k-evaluation table, final assignments, and cluster profiles.",
    ]
    (PROJECT_DIR / "clustering_summary.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
