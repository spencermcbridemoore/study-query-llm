"""
Generate metric grid plots for no-PCA multi-embedding sweep data.

Run from repo root:
  python scripts/plot_no_pca_multi_embedding.py --data-dir experimental_results
  python scripts/plot_no_pca_multi_embedding.py --data-dir experimental_results --out-dir experimental_results/plots/no_pca_metrics

Reads experimental_sweep_*.pkl files (filtered by no-PCA metadata), builds metrics
(objective, dispersion, ari, silhouette from Z or dist), and saves one figure per
(dataset, metric) with subplots: rows=summarizers, cols=embedding engines, x=k 2-20.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

# Repo root for imports when run as script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # No display; save only
import matplotlib.pyplot as plt


def is_no_pca_multi_embedding(data: dict) -> bool:
    meta = data.get("metadata") or {}
    if not isinstance(meta, dict):
        return False
    if "embedding_engine" not in meta:
        return False
    sweep = meta.get("sweep_config") or {}
    if sweep.get("skip_pca") is not True:
        return False
    return True


def load_no_pca_data(data_dir: Path) -> list:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    loaded = []
    for pkl_path in sorted(data_dir.glob("experimental_sweep_*.pkl")):
        try:
            if pkl_path.stat().st_size == 0:
                continue
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"[WARN] Skip {pkl_path.name}: {e}", file=sys.stderr)
            continue
        if not isinstance(data, dict) or not is_no_pca_multi_embedding(data):
            continue
        loaded.append({"path": pkl_path, "data": data})
    return loaded


def _objective_value(entry: dict):
    ob = entry.get("objective")
    if ob is None:
        return None
    if isinstance(ob, (int, float)):
        return float(ob)
    return ob.get("value", ob)


def _silhouette_precomputed(labels, dist, silhouette_score_fn):
    if silhouette_score_fn is None:
        return None
    labels = np.asarray(labels)
    dist = np.asarray(dist)
    if len(np.unique(labels)) < 2:
        return 0.0
    return float(silhouette_score_fn(dist, labels, metric="precomputed"))


from study_query_llm.experiments.result_metrics import dist_from_result as _dist_from_z


def build_grid_data(loaded: list, k_range: list[int] | None = None):
    from collections import defaultdict

    try:
        from sklearn.metrics import adjusted_rand_score, silhouette_score
    except ImportError:
        adjusted_rand_score = None
        silhouette_score = None

    if k_range is None:
        k_range = list(range(2, 21))

    summarizers = []
    engines = []
    by_key = defaultdict(lambda: defaultdict(lambda: ([], [])))

    for item in loaded:
        data = item["data"]
        meta = data.get("metadata") or {}
        result = data.get("result") or {}
        by_k = result.get("by_k") or {}
        dataset = data.get("dataset_name", "unknown")
        summarizer = meta.get("summarizer", "?")
        engine = meta.get("embedding_engine", "?")
        if summarizer not in summarizers:
            summarizers.append(summarizer)
        if engine not in engines:
            engines.append(engine)

        gt = data.get("ground_truth_labels")
        if gt is not None:
            gt = np.asarray(gt)

        dist_arr = result.get("dist")
        if dist_arr is not None:
            dist_arr = np.asarray(dist_arr)
        else:
            dist_arr = _dist_from_z(result)

        n_samples = len(gt) if gt is not None else 0
        if n_samples == 0 and by_k:
            first_labels = by_k.get("2", {}).get("labels", [])
            n_samples = len(first_labels) if first_labels else 0

        for k in k_range:
            k_str = str(k)
            if k_str not in by_k:
                continue
            entry = by_k[k_str]
            labels = entry.get("labels")
            if labels is not None:
                labels = np.asarray(labels)
                n = len(labels)
            else:
                n = n_samples

            ob = _objective_value(entry)
            if ob is not None:
                by_key[(dataset, "objective")][(summarizer, engine)][0].append(k)
                by_key[(dataset, "objective")][(summarizer, engine)][1].append(ob)
            if n and ob is not None:
                by_key[(dataset, "dispersion")][(summarizer, engine)][0].append(k)
                by_key[(dataset, "dispersion")][(summarizer, engine)][1].append(ob / n)

            stab = entry.get("stability")
            if isinstance(stab, dict):
                s_mean = stab.get("silhouette", {}).get("mean")
                if s_mean is not None:
                    by_key[(dataset, "silhouette")][(summarizer, engine)][0].append(k)
                    by_key[(dataset, "silhouette")][(summarizer, engine)][1].append(s_mean)
                for key_name, stab_key in [("stability_ari", "stability_ari"), ("coverage", "coverage")]:
                    v = stab.get(stab_key, {}).get("mean")
                    if v is not None:
                        by_key[(dataset, key_name)][(summarizer, engine)][0].append(k)
                        by_key[(dataset, key_name)][(summarizer, engine)][1].append(v)
            elif dist_arr is not None and labels is not None and len(labels) == dist_arr.shape[0]:
                sil = _silhouette_precomputed(labels, dist_arr, silhouette_score)
                if sil is not None:
                    by_key[(dataset, "silhouette")][(summarizer, engine)][0].append(k)
                    by_key[(dataset, "silhouette")][(summarizer, engine)][1].append(sil)

            if gt is not None and adjusted_rand_score is not None and labels is not None and len(labels) == len(gt):
                ari = adjusted_rand_score(gt, labels)
                by_key[(dataset, "ari")][(summarizer, engine)][0].append(k)
                by_key[(dataset, "ari")][(summarizer, engine)][1].append(ari)

    # Sort each series by k
    for key in list(by_key.keys()):
        for cell_key in list(by_key[key].keys()):
            ks, vals = by_key[key][cell_key]
            order = np.argsort(ks)
            by_key[key][cell_key] = ([ks[i] for i in order], [vals[i] for i in order])

    summarizers_sorted = sorted(summarizers, key=lambda s: (s != "None" and s or "", str(s)))
    engines_sorted = sorted(engines)

    return {
        "by_key": dict(by_key),
        "summarizers": summarizers_sorted,
        "engines": engines_sorted,
        "k_range": k_range,
    }


def plot_and_save(grid_data: dict, out_dir: Path, figsize_per_cell=(2, 2)):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summarizers = grid_data["summarizers"]
    engines = grid_data["engines"]
    by_key = grid_data["by_key"]
    n_rows = len(summarizers)
    n_cols = len(engines)

    if n_rows == 0 or n_cols == 0:
        print("[WARN] No summarizers or engines; no plots.", file=sys.stderr)
        return

    saved = []
    for (dataset, metric_name), series_dict in by_key.items():
        all_vals = []
        for (ks, vals) in series_dict.values():
            all_vals.extend(vals)
        if not all_vals:
            continue
        y_min = min(all_vals)
        y_max = max(all_vals)
        margin = (y_max - y_min) * 0.05 or 0.01
        y_lo = y_min - margin
        y_hi = y_max + margin

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows),
            sharex=True, sharey=True,
        )
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        fig.suptitle(f"{dataset} — {metric_name}", fontsize=12)

        for i, summarizer in enumerate(summarizers):
            for j, engine in enumerate(engines):
                ax = axes[i, j]
                ks, vals = series_dict.get((summarizer, engine), ([], []))
                if ks and vals:
                    order = np.argsort(ks)
                    ks = [ks[o] for o in order]
                    vals = [vals[o] for o in order]
                    ax.plot(ks, vals, "o-", markersize=3)
                ax.set_ylim(y_lo, y_hi)
                ax.set_xlim(1.5, 20.5)
                ax.grid(True, alpha=0.3)
                if i == 0:
                    ax.set_title(engine[:24] + ("..." if len(engine) > 24 else ""), fontsize=8)
                if j == 0:
                    ax.set_ylabel(summarizer if summarizer else "None", fontsize=8)
                if i == n_rows - 1:
                    ax.set_xlabel("k", fontsize=8)

        plt.tight_layout()
        safe_name = f"{dataset}_{metric_name}.png".replace(" ", "_")
        out_path = out_dir / safe_name
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)

    for p in saved:
        print(f"  Saved: {p}")


# Difference rows: (row_label, summarizer_minuend, summarizer_subtrahend) -> plot A - B
DIFF_ROWS = [
    ("gpt4o-mini − None", "gpt-4o-mini", "None"),
    ("gpt5chat − None", "gpt-5-chat", "None"),
    ("gpt5chat − gpt4o-mini", "gpt-5-chat", "gpt-4o-mini"),
]


def build_diff_grid_data(grid_data: dict) -> dict:
    """Build (dataset, metric) -> (diff_row_name, engine) -> (ks, diff_vals)."""
    from collections import defaultdict

    by_key = grid_data["by_key"]
    engines = grid_data["engines"]
    diff_by_key = defaultdict(lambda: defaultdict(lambda: ([], [])))

    for (dataset, metric_name), series_dict in by_key.items():
        for engine in engines:
            for row_label, sum_a, sum_b in DIFF_ROWS:
                ks_a, vals_a = series_dict.get((sum_a, engine), ([], []))
                ks_b, vals_b = series_dict.get((sum_b, engine), ([], []))
                if not ks_a or not ks_b:
                    continue
                # Align by k
                set_a = dict(zip(ks_a, vals_a))
                set_b = dict(zip(ks_b, vals_b))
                common_ks = sorted(set(set_a) & set(set_b))
                if not common_ks:
                    continue
                diffs = [set_a[k] - set_b[k] for k in common_ks]
                key = (dataset, metric_name)
                diff_by_key[key][(row_label, engine)][0].extend(common_ks)
                diff_by_key[key][(row_label, engine)][1].extend(diffs)

    diff_row_names = [r[0] for r in DIFF_ROWS]
    return {
        "by_key": dict(diff_by_key),
        "diff_rows": diff_row_names,
        "engines": engines,
        "k_range": grid_data.get("k_range", list(range(2, 21))),
    }


def plot_and_save_diffs(diff_grid_data: dict, out_dir: Path, figsize_per_cell=(2, 2)):
    """Save one figure per (dataset, metric): rows = diff series, cols = engines."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    diff_rows = diff_grid_data["diff_rows"]
    engines = diff_grid_data["engines"]
    by_key = diff_grid_data["by_key"]
    n_rows = len(diff_rows)
    n_cols = len(engines)

    if n_rows == 0 or n_cols == 0:
        return

    saved = []
    for (dataset, metric_name), series_dict in by_key.items():
        all_vals = []
        for (ks, vals) in series_dict.values():
            all_vals.extend(vals)
        if not all_vals:
            continue
        y_min = min(all_vals)
        y_max = max(all_vals)
        margin = (y_max - y_min) * 0.05 or 0.01
        y_lo = y_min - margin
        y_hi = y_max + margin

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows),
            sharex=True, sharey=True,
        )
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        fig.suptitle(f"{dataset} — {metric_name} (differences)", fontsize=12)

        for i, row_label in enumerate(diff_rows):
            for j, engine in enumerate(engines):
                ax = axes[i, j]
                ks, vals = series_dict.get((row_label, engine), ([], []))
                if ks and vals:
                    order = np.argsort(ks)
                    ks = [ks[o] for o in order]
                    vals = [vals[o] for o in order]
                    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
                    ax.plot(ks, vals, "o-", markersize=3)
                ax.set_ylim(y_lo, y_hi)
                ax.set_xlim(1.5, 20.5)
                ax.grid(True, alpha=0.3)
                if i == 0:
                    ax.set_title(engine[:24] + ("..." if len(engine) > 24 else ""), fontsize=8)
                if j == 0:
                    ax.set_ylabel(row_label, fontsize=8)
                if i == n_rows - 1:
                    ax.set_xlabel("k", fontsize=8)

        plt.tight_layout()
        safe_name = f"{dataset}_{metric_name}_diffs.png".replace(" ", "_")
        out_path = out_dir / safe_name
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)

    for p in saved:
        print(f"  Saved: {p}")


def main():
    parser = argparse.ArgumentParser(description="Plot no-PCA multi-embedding sweep metrics")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "experimental_results",
        help="Directory containing experimental_sweep_*.pkl files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for PNGs (default: data-dir/plots/no_pca_metrics)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / "plots" / "no_pca_metrics"

    print(f"Data dir: {data_dir}")
    print(f"Out dir:  {out_dir}")

    loaded = load_no_pca_data(data_dir)
    if not loaded:
        print("[ERROR] No no-PCA multi-embedding pickle files found.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(loaded)} no-PCA sweep file(s).")

    grid_data = build_grid_data(loaded)
    print(f"Summarizers (rows): {grid_data['summarizers']}")
    print(f"Engines (cols): {grid_data['engines']}")
    print(f"Metrics: {list({k[1] for k in grid_data['by_key'].keys()})}")

    print("Writing value plots...")
    plot_and_save(grid_data, out_dir)

    diff_out = out_dir / "diffs"
    print(f"Writing difference plots (rows: gpt4o-mini-None, gpt5chat-None, gpt5chat-gpt4o-mini) to {diff_out}...")
    diff_grid_data = build_diff_grid_data(grid_data)
    plot_and_save_diffs(diff_grid_data, diff_out)
    print("Done.")


if __name__ == "__main__":
    main()
