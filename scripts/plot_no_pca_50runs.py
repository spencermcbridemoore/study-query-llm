"""
Plot no-PCA 50-runs sweep: mean/stdev and box stats (min, Q1, median, Q3, max) per metric.

Loads no_pca_50runs_*.pkl, computes for each (dataset, summarizer, k) 50 values per metric
(objective, dispersion, silhouette, ari), then mean, stdev, and box stats. Saves:
- Line plots with error bars (mean +/- stdev)
- Box plots (min, low quartile, median, high quartile, max) per k

Usage:
  python scripts/plot_no_pca_50runs.py --data-dir experimental_results
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_50runs_pickles(data_dir: Path) -> list:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    out = []
    for p in sorted(data_dir.glob("no_pca_50runs_*.pkl")):
        try:
            if p.stat().st_size == 0:
                continue
            with open(p, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"[WARN] Skip {p.name}: {e}", file=sys.stderr)
            continue
        if not isinstance(data, dict) or "result" not in data:
            continue
        meta = data.get("metadata") or {}
        if meta.get("n_restarts") != 50 and "n_restarts" not in meta:
            continue
        out.append({"path": p, "data": data})
    return out


from study_query_llm.experiments.result_metrics import dist_from_result as _dist_from_z


def extract_50_values_per_metric(loaded: list, k_range: list[int] | None = None):
    try:
        from sklearn.metrics import adjusted_rand_score, silhouette_score
    except ImportError:
        adjusted_rand_score = None
        silhouette_score = None

    if k_range is None:
        k_range = list(range(2, 21))

    # (dataset, engine, summarizer) -> (metric_name -> (k -> list of 50 values))
    from collections import defaultdict
    by_ds_sum = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for item in loaded:
        data = item["data"]
        result = data.get("result") or {}
        by_k = result.get("by_k") or {}
        dataset = data.get("dataset_name", "unknown")
        meta = data.get("metadata") or {}
        summarizer = meta.get("summarizer", "?")
        engine = meta.get("embedding_engine", "embed-v-4-0")
        gt = data.get("ground_truth_labels")
        if gt is not None:
            gt = np.asarray(gt)
        dist = _dist_from_z(result)
        n_samples = len(gt) if gt is not None else 0
        if n_samples == 0 and by_k.get("2"):
            n_samples = len(by_k["2"].get("labels", []) or by_k["2"].get("objectives", []))

        for k in k_range:
            k_str = str(k)
            if k_str not in by_k:
                continue
            entry = by_k[k_str]
            objectives = entry.get("objectives", [])
            labels_all = entry.get("labels_all")
            if not objectives and (labels_all is None or len(labels_all) == 0):
                continue
            n_vals = len(objectives) if objectives else (len(labels_all) if labels_all else 0)
            if n_vals == 0:
                continue
            n = n_samples or (len(labels_all[0]) if labels_all and labels_all[0] is not None else 0)
            key = (dataset, engine, summarizer)

            for i in range(n_vals):
                ob = objectives[i] if objectives else None
                if ob is not None:
                    by_ds_sum[key]["objective"][k].append(float(ob))
                if n and ob is not None:
                    disp = ob / n
                    by_ds_sum[key]["dispersion"][k].append(disp)
                    # Point-centroid mean cosine similarity = 1 - dispersion (cosine k-means)
                    cos_sim = 1.0 - disp
                    by_ds_sum[key]["cosine_sim"][k].append(cos_sim)
                    by_ds_sum[key]["cosine_sim_norm"][k].append((cos_sim + 1.0) / 2.0)
                if labels_all and i < len(labels_all) and dist is not None and silhouette_score is not None:
                    lab = np.asarray(labels_all[i])
                    if len(lab) == dist.shape[0] and len(np.unique(lab)) >= 2:
                        by_ds_sum[key]["silhouette"][k].append(float(silhouette_score(dist, lab, metric="precomputed")))
                if gt is not None and labels_all and adjusted_rand_score is not None and i < len(labels_all):
                    lab = np.asarray(labels_all[i])
                    if len(lab) == len(gt):
                        by_ds_sum[key]["ari"][k].append(float(adjusted_rand_score(gt, lab)))

    return dict(by_ds_sum), k_range


def summary_stats(vals: list) -> dict:
    if not vals:
        return {"mean": np.nan, "stdev": np.nan, "min": np.nan, "q1": np.nan, "median": np.nan, "q3": np.nan, "max": np.nan}
    a = np.asarray(vals, dtype=float)
    return {
        "mean": float(np.mean(a)),
        "stdev": float(np.std(a)) if len(a) > 1 else 0.0,
        "min": float(np.min(a)),
        "q1": float(np.percentile(a, 25)),
        "median": float(np.median(a)),
        "q3": float(np.percentile(a, 75)),
        "max": float(np.max(a)),
    }


def _engine_safe(engine: str) -> str:
    return engine.replace("-", "_")


def plot_mean_stdev_and_boxes(by_ds_sum: dict, k_range: list[int], out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_line = out_dir / "mean_stdev"
    out_box = out_dir / "box"
    out_line.mkdir(exist_ok=True)
    out_box.mkdir(exist_ok=True)

    # Keys are (dataset, engine, summarizer)
    summarizers = sorted(set(s for _, _, s in by_ds_sum.keys()))
    ds_engines = sorted(set((d, e) for d, e, _ in by_ds_sum.keys()))
    metrics = ["objective", "dispersion", "silhouette", "ari", "cosine_sim", "cosine_sim_norm"]

    for (dataset, engine) in ds_engines:
        engine_safe = _engine_safe(engine)
        for metric in metrics:
            series = {}
            for (d, e, s) in by_ds_sum:
                if d != dataset or e != engine:
                    continue
                m = by_ds_sum[(d, e, s)].get(metric)
                if not m:
                    continue
                ks = sorted(m.keys())
                vals_by_k = [m[k] for k in ks]
                if not any(vals_by_k):
                    continue
                series[s] = (ks, vals_by_k)

            if not series:
                continue

            stats = {}
            for s, (ks, vals_by_k) in series.items():
                stats[s] = [summary_stats(vlist) for vlist in vals_by_k]

            fig, ax = plt.subplots(figsize=(8, 5))
            for s in summarizers:
                if s not in series:
                    continue
                ks, vals_by_k = series[s]
                means = [stats[s][i]["mean"] for i in range(len(ks))]
                stdevs = [stats[s][i]["stdev"] for i in range(len(ks))]
                ax.errorbar(ks, means, yerr=stdevs, label=s or "None", capsize=2, markersize=4)
            ax.set_xlabel("k")
            ax.set_ylabel(metric)
            ax.set_title(f"{dataset} ({engine_safe}) — {metric} (mean ± stdev, n=50)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(1.5, 20.5)
            plt.tight_layout()
            fig.savefig(out_line / f"{dataset}_{engine_safe}_{metric}_mean_stdev.png", dpi=120, bbox_inches="tight")
            plt.close(fig)

            fig, axes = plt.subplots(1, len(summarizers), figsize=(5 * len(summarizers), 5), sharey=True)
            if len(summarizers) == 1:
                axes = [axes]
            for j, s in enumerate(summarizers):
                if s not in series:
                    continue
                ks, vals_by_k = series[s]
                data_for_box = [vals_by_k[i] for i in range(len(ks))]
                bp = axes[j].boxplot(
                    data_for_box,
                    positions=range(len(ks)),
                    widths=0.6,
                    patch_artist=True,
                    showmeans=True,
                )
                axes[j].set_xticks(range(len(ks)))
                axes[j].set_xticklabels(ks)
                axes[j].set_xlabel("k")
                axes[j].set_ylabel(metric)
                axes[j].set_title(s or "None")
                axes[j].grid(True, alpha=0.3, axis="y")
            fig.suptitle(f"{dataset} ({engine_safe}) — {metric} (box: min, Q1, median, Q3, max; n=50)")
            plt.tight_layout()
            fig.savefig(out_box / f"{dataset}_{engine_safe}_{metric}_box.png", dpi=120, bbox_inches="tight")
            plt.close(fig)

    print(f"  Saved mean±stdev to {out_line}")
    print(f"  Saved box plots to {out_box}")


def plot_metrics_grid_red_blue(by_ds_sum: dict, k_range: list[int], out_dir: Path):
    """
    One figure per (dataset, engine): columns = metrics, rows = None | gpt5 | gpt5-None.
    - Rows None and gpt5: red dots (mean), light red shaded stdev, blue boxplots, blue line through medians.
    - Row gpt5-None: red dots (mean diff), light red shaded (combined stdev); no blue.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_dir = out_dir / "metrics_grid"
    grid_dir.mkdir(exist_ok=True)

    metrics = ["objective", "dispersion", "silhouette", "ari", "cosine_sim", "cosine_sim_norm"]

    ds_engines = sorted(set((d, e) for d, e, _ in by_ds_sum.keys()))
    for (dataset, engine) in ds_engines:
        engine_safe = _engine_safe(engine)
        has_none = (dataset, engine, "None") in by_ds_sum
        has_gpt5 = (dataset, engine, "gpt-5-chat") in by_ds_sum
        if not has_none or not has_gpt5:
            continue

        n_rows, n_cols = 3, len(metrics)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3.5 * n_rows), sharex="col", sharey="col")

        for col, metric in enumerate(metrics):
            m_none = by_ds_sum.get((dataset, engine, "None"), {}).get(metric)
            m_gpt5 = by_ds_sum.get((dataset, engine, "gpt-5-chat"), {}).get(metric)
            if not m_none or not m_gpt5:
                for r in range(3):
                    axes[r, col].set_visible(False)
                continue

            ks = sorted(m_none.keys())
            if not ks:
                continue

            # Row 0: None
            vals_by_k = [m_none[k] for k in ks]
            stats = [summary_stats(v) for v in vals_by_k]
            means = [s["mean"] for s in stats]
            stdevs = [s["stdev"] for s in stats]
            medians = [s["median"] for s in stats]
            ax = axes[0, col]
            ax.fill_between(ks, np.array(means) - np.array(stdevs), np.array(means) + np.array(stdevs), color="red", alpha=0.2)
            ax.plot(ks, means, "o-", color="red", markersize=4, label="mean")
            bp = ax.boxplot([vals_by_k[i] for i in range(len(ks))], positions=ks, widths=0.5, patch_artist=True, showfliers=False)
            for patch in bp["boxes"]:
                patch.set_facecolor("lightblue")
                patch.set_edgecolor("blue")
            for el in ["whiskers", "caps", "medians"]:
                for line in bp.get(el, []):
                    line.set_color("blue")
            ax.plot(ks, medians, "-", color="blue", linewidth=1.5, label="median")
            if col == 0:
                ax.set_ylabel("None\n" + metric)
            ax.set_title(metric)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(1.5, 20.5)

            # Row 1: gpt5
            vals_by_k = [m_gpt5[k] for k in ks]
            stats = [summary_stats(v) for v in vals_by_k]
            means = [s["mean"] for s in stats]
            stdevs = [s["stdev"] for s in stats]
            medians = [s["median"] for s in stats]
            ax = axes[1, col]
            ax.fill_between(ks, np.array(means) - np.array(stdevs), np.array(means) + np.array(stdevs), color="red", alpha=0.2)
            ax.plot(ks, means, "o-", color="red", markersize=4)
            bp = ax.boxplot([vals_by_k[i] for i in range(len(ks))], positions=ks, widths=0.5, patch_artist=True, showfliers=False)
            for patch in bp["boxes"]:
                patch.set_facecolor("lightblue")
                patch.set_edgecolor("blue")
            for el in ["whiskers", "caps", "medians"]:
                for line in bp.get(el, []):
                    line.set_color("blue")
            ax.plot(ks, medians, "-", color="blue", linewidth=1.5)
            if col == 0:
                ax.set_ylabel("gpt5\n" + metric)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(1.5, 20.5)

            # Row 2: gpt5-None (mean diff, combined stdev; red only)
            stats_none = [summary_stats(m_none[k]) for k in ks]
            stats_gpt5 = [summary_stats(m_gpt5[k]) for k in ks]
            diff_means = [stats_gpt5[i]["mean"] - stats_none[i]["mean"] for i in range(len(ks))]
            # Combined stdev for difference: sqrt(s1^2 + s2^2)
            diff_stdevs = [np.sqrt(stats_none[i]["stdev"] ** 2 + stats_gpt5[i]["stdev"] ** 2) for i in range(len(ks))]
            ax = axes[2, col]
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
            ax.fill_between(ks, np.array(diff_means) - np.array(diff_stdevs), np.array(diff_means) + np.array(diff_stdevs), color="red", alpha=0.2)
            ax.plot(ks, diff_means, "o-", color="red", markersize=4)
            if col == 0:
                ax.set_ylabel("gpt5-None\n" + metric)
            ax.set_xlabel("k")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(1.5, 20.5)

        for c in range(n_cols):
            axes[2, c].set_xlabel("k")
            axes[0, c].set_title(metrics[c])
        plt.tight_layout()
        out_path = grid_dir / f"{dataset}_{engine_safe}_metrics_grid.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")

    if grid_dir.exists() and any(grid_dir.iterdir()):
        print(f"  Metrics grid (red mean±stdev, blue box+median) -> {grid_dir}")


def main():
    parser = argparse.ArgumentParser(description="Plot no-PCA 50-runs: mean/stdev and box stats")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "experimental_results")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / "plots" / "no_pca_50runs"

    print(f"Data dir: {data_dir}")
    print(f"Out dir:  {out_dir}")

    loaded = load_50runs_pickles(data_dir)
    if not loaded:
        print("[ERROR] No no_pca_50runs_*.pkl files found.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(loaded)} 50runs pickle(s).")

    by_ds_sum, k_range = extract_50_values_per_metric(loaded)
    if not by_ds_sum:
        print("[ERROR] No (dataset, summarizer) data with 50 values.", file=sys.stderr)
        sys.exit(1)

    print("Writing mean/stdev and box plots...")
    plot_mean_stdev_and_boxes(by_ds_sum, k_range, out_dir)
    print("Writing metrics grid (red mean+stdev, blue box+median, gpt5-None red only)...")
    plot_metrics_grid_red_blue(by_ds_sum, k_range, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
