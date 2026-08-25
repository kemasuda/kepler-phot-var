"""Diagnostics for weighted-KIC joint and conditional flow fits."""

from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path

import jax
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jax import random

from likelihood_wkic_common.flowutils import (
    conditional_log_prob,
    conditional_sample,
    jsonable,
)

__all__ = [
    "library_versions",
    "run_conditional_flow_diagnostics",
    "run_joint_flow_diagnostics",
]


def library_versions():
    """Return library versions that affect flow serialization or likelihoods."""
    packages = ["flowjax", "jax", "jaxlib", "equinox", "numpy", "pandas"]
    versions = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    versions["jax_default_backend"] = jax.default_backend()
    return versions


def _as_1d(values):
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr


def _weights_or_none(weights, n):
    if weights is None:
        return None
    weights = _as_1d(weights)
    if weights.shape[0] != int(n):
        raise ValueError("weights must have the same length as values.")
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights contain non-finite values.")
    if np.any(weights < 0.0):
        raise ValueError("weights must be non-negative.")
    if not np.any(weights > 0.0):
        raise ValueError("at least one weight must be positive.")
    return weights


def _weighted_mean(values, weights=None):
    values = np.asarray(values, dtype=float)
    if weights is None:
        return float(np.mean(values))
    weights = np.asarray(weights, dtype=float)
    total = np.sum(weights)
    if not np.isfinite(total) or total <= 0.0:
        return float(np.mean(values))
    return float(np.sum(weights * values) / total)


def _log_prob_summary(name, logp, *, weights=None):
    logp = _as_1d(logp)
    weights = _weights_or_none(weights, logp.shape[0])
    finite = np.isfinite(logp)
    row = {
        "split": name,
        "n": int(logp.shape[0]),
        "n_finite": int(np.sum(finite)),
        "n_nonfinite": int(np.sum(~finite)),
    }
    if not np.any(finite):
        row.update(
            {
                "mean_log_prob": np.nan,
                "weighted_mean_log_prob": np.nan,
                "negative_log_likelihood": np.nan,
                "weighted_negative_log_likelihood": np.nan,
                "std_log_prob": np.nan,
                "p01_log_prob": np.nan,
                "p05_log_prob": np.nan,
                "p16_log_prob": np.nan,
                "p50_log_prob": np.nan,
                "p84_log_prob": np.nan,
                "p95_log_prob": np.nan,
                "p99_log_prob": np.nan,
            }
        )
        return row

    finite_logp = logp[finite]
    finite_weights = None if weights is None else weights[finite]
    percentiles = np.percentile(finite_logp, [1, 5, 16, 50, 84, 95, 99])
    row.update(
        {
            "mean_log_prob": float(np.mean(finite_logp)),
            "weighted_mean_log_prob": _weighted_mean(finite_logp, finite_weights),
            "negative_log_likelihood": float(-np.mean(finite_logp)),
            "weighted_negative_log_likelihood": float(
                -_weighted_mean(finite_logp, finite_weights)
            ),
            "std_log_prob": float(np.std(finite_logp)),
            "p01_log_prob": float(percentiles[0]),
            "p05_log_prob": float(percentiles[1]),
            "p16_log_prob": float(percentiles[2]),
            "p50_log_prob": float(percentiles[3]),
            "p84_log_prob": float(percentiles[4]),
            "p95_log_prob": float(percentiles[5]),
            "p99_log_prob": float(percentiles[6]),
        }
    )
    return row


def _sample_rows(values, *, n_points, seed):
    values = np.asarray(values)
    n = values.shape[0]
    n_points = min(int(n_points), int(n))
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=n_points, replace=False)
    return values[idx], idx


def _normalize_hist_weights(weights):
    if weights is None:
        return None
    weights = np.asarray(weights, dtype=float)
    total = np.sum(weights)
    if not np.isfinite(total) or total <= 0.0:
        return None
    return weights / total


def _loss_series(losses):
    if isinstance(losses, dict):
        series = {}
        preferred = ["train", "val", "validation", "test"]
        keys = [key for key in preferred if key in losses]
        keys.extend(key for key in losses if key not in keys)
        for key in keys:
            try:
                values = np.asarray(losses[key], dtype=float).reshape(-1)
            except (TypeError, ValueError):
                continue
            series[str(key)] = values
        return series

    if isinstance(losses, (list, tuple)) and losses and isinstance(losses[0], dict):
        keys = sorted({key for row in losses for key in row})
        series = {}
        for key in keys:
            values = [row.get(key, np.nan) for row in losses]
            try:
                series[str(key)] = np.asarray(values, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                continue
        return series

    return {"loss": np.asarray(losses, dtype=float).reshape(-1)}


def _summarize_loss_values(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    return {
        "n_loss_values": int(values.size),
        "n_finite_loss_values": int(finite.size),
        "loss_initial": float(values[0]) if values.size else None,
        "loss_final": float(values[-1]) if values.size else None,
        "loss_min": float(np.min(finite)) if finite.size else None,
    }


def _plot_loss_curve(losses, path):
    series = _loss_series(losses)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for label, values in series.items():
        ax.plot(np.arange(values.size), values, lw=1.5, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.25)
    if len(series) > 1:
        ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    if set(series) == {"loss"}:
        return _summarize_loss_values(series["loss"])
    return {label: _summarize_loss_values(values) for label, values in series.items()}


def _plot_rows(values, weights=None):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    finite = np.all(np.isfinite(values), axis=1)
    if weights is None:
        return values[finite], None
    weights = _weights_or_none(weights, values.shape[0])
    return values[finite], weights[finite]


def _plot_flow_check(
    reference,
    samples,
    keys,
    path,
    *,
    reference_label="training sample",
    sample_label="samples from normalizing flow model",
    reference_weights=None,
    sample_weights=None,
    bins=50,
):
    try:
        import corner
    except Exception:
        return {"created": False, "reason": "corner is not installed"}

    reference, reference_weights = _plot_rows(reference, reference_weights)
    samples, sample_weights = _plot_rows(samples, sample_weights)
    if reference.shape[1] != samples.shape[1]:
        return {"created": False, "reason": "reference/sample dimensions differ"}
    if min(len(reference), len(samples)) < 2:
        return {"created": False, "reason": "fewer than two finite rows"}
    labels = list(keys)

    try:
        fig = corner.corner(
            reference,
            labels=labels,
            color="C0",
            show_titles=True,
            title_fmt=None,
            hist_kwargs={"density": True},
            plot_datapoints=False,
            weights=reference_weights,
            bins=bins,
        )
        corner.corner(
            samples,
            color="C1",
            hist_kwargs={"density": True},
            plot_datapoints=False,
            weights=sample_weights,
            bins=bins,
            fig=fig,
        )
        handles = [
            mlines.Line2D([], [], color="C0", label=reference_label),
            mlines.Line2D([], [], color="C1", label=sample_label),
        ]
        plt.legend(handles=handles, bbox_to_anchor=(0.8, len(labels)), fontsize=25)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        return {"created": False, "reason": str(exc)}
    return {
        "created": True,
        "n_reference": int(len(reference)),
        "n_flow": int(len(samples)),
        "reference_weighted": bool(reference_weights is not None),
        "sample_weighted": bool(sample_weights is not None),
        "bins": int(bins),
        "reference_label": reference_label,
        "sample_label": sample_label,
    }


def _binned_target_quantiles(x, y, *, weights=None, n_bins=12, min_count=20):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        finite &= np.isfinite(weights) & (weights >= 0.0)
    x = x[finite]
    y = y[finite]
    if weights is not None:
        weights = weights[finite]
    if x.size == 0:
        return {
            "x": np.array([]),
            "p16": np.array([]),
            "median": np.array([]),
            "p84": np.array([]),
            "count": np.array([], dtype=int),
        }

    edges = np.unique(np.quantile(x, np.linspace(0.02, 0.98, int(n_bins) + 1)))
    if edges.size < 3:
        edges = np.linspace(np.min(x), np.max(x), min(int(n_bins), x.size) + 1)
    if edges.size < 2:
        return {
            "x": np.array([]),
            "p16": np.array([]),
            "median": np.array([]),
            "p84": np.array([]),
            "count": np.array([], dtype=int),
        }

    rows = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if i == len(edges) - 2:
            mask = (x >= lo) & (x <= hi)
        else:
            mask = (x >= lo) & (x < hi)
        if int(np.sum(mask)) < int(min_count):
            continue
        weights_bin = None if weights is None else weights[mask]
        if weights_bin is not None and not np.any(weights_bin > 0.0):
            weights_bin = None
        p16, p50, p84 = _weighted_quantile(
            y[mask],
            [0.16, 0.50, 0.84],
            weights_bin,
        )
        x_mid = _weighted_quantile(x[mask], [0.50], weights_bin)[0]
        rows.append((x_mid, p16, p50, p84, int(np.sum(mask))))

    if not rows:
        return {
            "x": np.array([]),
            "p16": np.array([]),
            "median": np.array([]),
            "p84": np.array([]),
            "count": np.array([], dtype=int),
        }

    arr = np.asarray(rows, dtype=float)
    return {
        "x": arr[:, 0],
        "p16": arr[:, 1],
        "median": arr[:, 2],
        "p84": arr[:, 3],
        "count": arr[:, 4].astype(int),
    }


def _plot_conditional_panels(
    target_data,
    target_flow,
    condition,
    target_key,
    condition_keys,
    path,
    *,
    weights_data=None,
    weights_flow=None,
    n_bins=12,
    min_count=20,
):
    condition_keys = list(condition_keys)
    condition = np.asarray(condition, dtype=float)
    target_data = _as_1d(target_data)
    target_flow = _as_1d(target_flow)
    ncols = min(2, len(condition_keys))
    nrows = int(np.ceil(len(condition_keys) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.0 * ncols, 3.4 * nrows),
        squeeze=False,
    )
    summary = {}

    for ax, i in zip(axes.ravel(), range(len(condition_keys))):
        x = condition[:, i]
        data_stats = _binned_target_quantiles(
            x,
            target_data,
            weights=weights_data,
            n_bins=n_bins,
            min_count=min_count,
        )
        flow_stats = _binned_target_quantiles(
            x,
            target_flow,
            weights=weights_flow,
            n_bins=n_bins,
            min_count=min_count,
        )
        if data_stats["x"].size:
            ax.fill_between(
                data_stats["x"],
                data_stats["p16"],
                data_stats["p84"],
                color="C0",
                alpha=0.18,
                linewidth=0,
            )
            ax.plot(data_stats["x"], data_stats["median"], color="C0", label="KIC data")
        if flow_stats["x"].size:
            ax.fill_between(
                flow_stats["x"],
                flow_stats["p16"],
                flow_stats["p84"],
                color="C1",
                alpha=0.18,
                linewidth=0,
            )
            ax.plot(
                flow_stats["x"],
                flow_stats["median"],
                color="C1",
                label="flow at KIC conditions",
            )
        ax.set_xlabel(condition_keys[i])
        ax.set_ylabel(target_key)
        ax.grid(alpha=0.2)
        summary[condition_keys[i]] = {
            "n_data_bins": int(data_stats["x"].size),
            "n_flow_bins": int(flow_stats["x"].size),
            "median_bin_count": (
                float(np.median(data_stats["count"]))
                if data_stats["count"].size
                else None
            ),
        }

    for ax in axes.ravel()[len(condition_keys):]:
        ax.axis("off")
    axes.ravel()[0].legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return summary


def _weighted_quantile(values, quantiles, weights=None):
    values = np.asarray(values, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    finite = np.isfinite(values)
    finite_values = values[finite]
    if weights is None:
        return np.quantile(finite_values, quantiles)

    weights = np.asarray(weights, dtype=float)
    finite &= np.isfinite(weights) & (weights >= 0.0)
    values = values[finite]
    weights = weights[finite]
    if values.size == 0 or not np.any(weights > 0.0):
        return np.quantile(finite_values, quantiles)

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights)
    cdf = cdf / cdf[-1]
    return np.interp(quantiles, cdf, values)


def _condition_location_scale(condition, meta):
    loc = np.asarray(meta.get("condition_locs", np.nanmean(condition, axis=0)), dtype=float)
    scale = np.asarray(meta.get("condition_scales", np.nanstd(condition, axis=0)), dtype=float)
    bad = ~np.isfinite(scale) | (scale <= 0.0)
    if np.any(bad):
        scale = scale.copy()
        scale[bad] = 1.0
    return loc, scale


def _select_condition_slice_indices(
    condition,
    *,
    weights=None,
    loc,
    scale,
    n_slices=8,
):
    condition = np.asarray(condition, dtype=float)
    n_slices = max(1, min(int(n_slices), 10, condition.shape[0]))
    finite = np.all(np.isfinite(condition), axis=1)
    if not np.any(finite):
        return np.array([], dtype=int)

    idx_all = np.flatnonzero(finite)
    condition_finite = condition[idx_all]
    weights_finite = None if weights is None else np.asarray(weights, dtype=float)[idx_all]
    z = (condition_finite - loc) / scale
    quantiles = np.array([0.5]) if n_slices == 1 else np.linspace(0.05, 0.95, n_slices)
    target_first = _weighted_quantile(condition_finite[:, 0], quantiles, weights_finite)
    target_z_first = (target_first - loc[0]) / scale[0]
    other_penalty = np.zeros(condition_finite.shape[0])
    if condition_finite.shape[1] > 1:
        other_penalty = np.sqrt(np.mean(np.square(z[:, 1:]), axis=1))

    chosen = []
    used = set()
    for target_z in target_z_first:
        score = np.abs(z[:, 0] - target_z) + 0.15 * other_penalty
        for pos in np.argsort(score):
            idx = int(idx_all[pos])
            if idx not in used:
                chosen.append(idx)
                used.add(idx)
                break

    return np.asarray(sorted(chosen, key=lambda i: condition[i, 0]), dtype=int)


def _nearest_condition_indices(condition, center, *, loc, scale, n_neighbors):
    condition = np.asarray(condition, dtype=float)
    center = np.asarray(center, dtype=float)
    finite = np.all(np.isfinite(condition), axis=1)
    idx_all = np.flatnonzero(finite)
    if idx_all.size == 0:
        return np.array([], dtype=int)

    z = (condition[idx_all] - loc) / scale
    z_center = (center - loc) / scale
    dist2 = np.sum(np.square(z - z_center), axis=1)
    n_neighbors = max(1, min(int(n_neighbors), idx_all.size))
    nearest_pos = np.argpartition(dist2, n_neighbors - 1)[:n_neighbors]
    nearest_pos = nearest_pos[np.argsort(dist2[nearest_pos])]
    return idx_all[nearest_pos]


def _format_condition_title(condition_row, condition_keys):
    parts = [
        f"{key}={value:.4g}"
        for key, value in zip(condition_keys, np.asarray(condition_row, dtype=float))
    ]
    if len(parts) <= 2:
        return ", ".join(parts)
    return ", ".join(parts[:2]) + "\n" + ", ".join(parts[2:])


def _plot_conditional_density_slices(
    flow,
    target,
    condition,
    meta,
    path,
    *,
    weights=None,
    seed=123,
    n_slices=8,
    n_neighbors=500,
    n_flow_samples=4000,
    bins=50,
):
    target_key = meta["target_key"]
    condition_keys = list(meta["condition_keys"])
    target = np.asarray(target, dtype=float).reshape(-1)
    condition = np.asarray(condition, dtype=float)
    if condition.shape[0] != target.shape[0]:
        raise ValueError("target and condition must have the same number of rows")
    weights = _weights_or_none(weights, target.shape[0])
    loc, scale = _condition_location_scale(condition, meta)
    slice_idx = _select_condition_slice_indices(
        condition,
        weights=weights,
        loc=loc,
        scale=scale,
        n_slices=n_slices,
    )
    if slice_idx.size == 0:
        return {"created": False, "reason": "no finite condition rows"}

    ncols = min(2, int(slice_idx.size))
    nrows = int(np.ceil(slice_idx.size / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.4 * ncols, 3.3 * nrows),
        squeeze=False,
    )
    summary = []

    for panel, (ax, row_idx) in enumerate(zip(axes.ravel(), slice_idx)):
        center = condition[row_idx]
        neighbor_idx = _nearest_condition_indices(
            condition,
            center,
            loc=loc,
            scale=scale,
            n_neighbors=n_neighbors,
        )
        data_values = target[neighbor_idx]
        data_weights = None if weights is None else weights[neighbor_idx]
        finite_data = np.isfinite(data_values)
        data_values = data_values[finite_data]
        if data_weights is not None:
            data_weights = data_weights[finite_data]
            if not np.any(data_weights > 0.0):
                data_weights = None

        flow_condition = np.repeat(center[None, :], int(n_flow_samples), axis=0)
        flow_target = np.asarray(
            conditional_sample(
                flow,
                random.fold_in(random.key(seed), int(panel)),
                flow_condition,
                meta,
            )
        ).reshape(-1)
        flow_target = flow_target[np.isfinite(flow_target)]

        combined = np.concatenate([data_values, flow_target])
        finite_combined = combined[np.isfinite(combined)]
        if finite_combined.size >= 2:
            lo, hi = np.percentile(finite_combined, [0.5, 99.5])
            if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                lo, hi = float(np.min(finite_combined)), float(np.max(finite_combined))
            hist_bins = np.linspace(lo, hi, int(bins) + 1) if lo < hi else int(bins)
        else:
            hist_bins = int(bins)

        ax.hist(
            data_values,
            bins=hist_bins,
            density=True,
            histtype="step",
            lw=1.7,
            color="C0",
            weights=data_weights,
            label="local weighted KIC",
        )
        ax.hist(
            flow_target,
            bins=hist_bins,
            density=True,
            histtype="step",
            lw=1.7,
            color="C1",
            label=f"flow p({target_key} | cond)",
        )
        ax.set_title(_format_condition_title(center, condition_keys), fontsize=10)
        ax.set_xlabel(target_key)
        ax.set_ylabel("density")
        ax.grid(alpha=0.2)
        if panel == 0:
            ax.legend(loc="best", fontsize=9)

        summary.append(
            {
                "row_index": int(row_idx),
                "condition": {
                    key: float(value) for key, value in zip(condition_keys, center)
                },
                "n_neighbors": int(neighbor_idx.size),
                "n_data_finite": int(data_values.size),
                "n_flow": int(flow_target.size),
                "data_weighted": bool(data_weights is not None),
                "data_mean": _weighted_mean(data_values, data_weights)
                if data_values.size
                else None,
                "flow_mean": float(np.mean(flow_target)) if flow_target.size else None,
                "data_std": float(np.std(data_values)) if data_values.size else None,
                "flow_std": float(np.std(flow_target)) if flow_target.size else None,
            }
        )

    for ax in axes.ravel()[slice_idx.size:]:
        ax.axis("off")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {
        "created": True,
        "n_slices": int(slice_idx.size),
        "n_neighbors": int(min(int(n_neighbors), condition.shape[0])),
        "n_flow_samples_per_slice": int(n_flow_samples),
        "bins": int(bins),
        "slices": summary,
    }


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")


def _write_logprob_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def run_joint_flow_diagnostics(
    flow,
    theta_fit,
    meta,
    out_dir,
    *,
    losses=None,
    weights_fit=None,
    theta_validation=None,
    weights_validation=None,
    seed=123,
    n_points=8000,
    bins=50,
):
    """Write diagnostics for a joint flow and return a JSON-serializable report."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = list(meta["keys"])
    theta_fit = np.asarray(theta_fit, dtype=float)
    weights_fit = _weights_or_none(weights_fit, theta_fit.shape[0])

    logp_fit = np.asarray(flow.log_prob(theta_fit))
    rows = [_log_prob_summary("fit", logp_fit, weights=weights_fit)]
    if theta_validation is not None:
        theta_validation = np.asarray(theta_validation, dtype=float)
        weights_validation = _weights_or_none(weights_validation, theta_validation.shape[0])
        logp_validation = np.asarray(flow.log_prob(theta_validation))
        rows.append(
            _log_prob_summary(
                "validation",
                logp_validation,
                weights=weights_validation,
            )
        )
    else:
        logp_validation = None

    logprob_summary_path = out_dir / "logprob_summary.csv"
    _write_logprob_csv(logprob_summary_path, rows)

    plot_summary = {}
    paths = {"logprob_summary": logprob_summary_path}
    if losses is not None:
        paths["loss_curve"] = out_dir / "loss_curve.png"
        plot_summary["loss_curve"] = _plot_loss_curve(losses, paths["loss_curve"])

    n_flow = min(int(n_points), int(theta_fit.shape[0]))
    theta_flow = np.asarray(flow.sample(random.key(seed), (n_flow,)))
    paths["check_corner"] = out_dir / "check_corner.png"
    plot_summary["check_corner"] = _plot_flow_check(
        theta_fit,
        theta_flow,
        keys,
        paths["check_corner"],
        reference_label="weighted KIC" if weights_fit is not None else "training sample",
        reference_weights=weights_fit,
        bins=bins,
    )

    report = {
        "flow_kind": "joint",
        "library_versions": library_versions(),
        "keys": keys,
        "n_fit": int(theta_fit.shape[0]),
        "n_validation": 0 if theta_validation is None else int(theta_validation.shape[0]),
        "logprob_rows": rows,
        "paths": paths,
        "plots": plot_summary,
        "seed": int(seed),
        "n_flow_plot_points": int(n_flow),
    }
    diagnostics_path = out_dir / "diagnostics.json"
    _write_json(diagnostics_path, report)
    report["paths"]["diagnostics"] = diagnostics_path
    return report


def run_conditional_flow_diagnostics(
    flow,
    target_fit,
    condition_fit,
    meta,
    out_dir,
    *,
    losses=None,
    weights_fit=None,
    target_validation=None,
    condition_validation=None,
    weights_validation=None,
    seed=123,
    n_points=8000,
    bins=50,
    n_bins=12,
    min_count=20,
    n_condition_slices=8,
    n_slice_neighbors=500,
    n_slice_flow_samples=4000,
):
    """Write diagnostics for a conditional flow and return a JSON report."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_key = meta["target_key"]
    condition_keys = list(meta["condition_keys"])
    target_fit = np.asarray(target_fit, dtype=float).reshape(-1, 1)
    condition_fit = np.asarray(condition_fit, dtype=float)
    weights_fit = _weights_or_none(weights_fit, target_fit.shape[0])

    logp_fit = np.asarray(conditional_log_prob(flow, target_fit, condition_fit, meta))
    rows = [_log_prob_summary("fit", logp_fit, weights=weights_fit)]
    if target_validation is not None and condition_validation is not None:
        target_validation = np.asarray(target_validation, dtype=float).reshape(-1, 1)
        condition_validation = np.asarray(condition_validation, dtype=float)
        weights_validation = _weights_or_none(weights_validation, target_validation.shape[0])
        logp_validation = np.asarray(
            conditional_log_prob(flow, target_validation, condition_validation, meta)
        )
        rows.append(
            _log_prob_summary(
                "validation",
                logp_validation,
                weights=weights_validation,
            )
        )
    else:
        target_validation = None
        condition_validation = None
        logp_validation = None

    logprob_summary_path = out_dir / "logprob_summary.csv"
    _write_logprob_csv(logprob_summary_path, rows)

    plot_summary = {}
    paths = {"logprob_summary": logprob_summary_path}
    if losses is not None:
        paths["loss_curve"] = out_dir / "loss_curve.png"
        plot_summary["loss_curve"] = _plot_loss_curve(losses, paths["loss_curve"])

    condition_plot, idx = _sample_rows(condition_fit, n_points=n_points, seed=seed)
    target_plot = target_fit[idx, 0]
    flow_weights = None if weights_fit is None else weights_fit[idx]
    target_flow = np.asarray(
        conditional_sample(flow, random.key(seed), condition_plot, meta)
    )[..., 0]

    paths["target_condition_panels"] = out_dir / "target_condition_panels.png"
    plot_summary["target_condition_panels"] = _plot_conditional_panels(
        target_plot,
        target_flow,
        condition_plot,
        target_key,
        condition_keys,
        paths["target_condition_panels"],
        weights_data=flow_weights,
        weights_flow=flow_weights,
        n_bins=n_bins,
        min_count=min_count,
    )

    paths["conditional_density_slices"] = out_dir / "conditional_density_slices.png"
    plot_summary["conditional_density_slices"] = _plot_conditional_density_slices(
        flow,
        target_fit,
        condition_fit,
        meta,
        paths["conditional_density_slices"],
        weights=weights_fit,
        seed=seed,
        n_slices=n_condition_slices,
        n_neighbors=n_slice_neighbors,
        n_flow_samples=n_slice_flow_samples,
        bins=bins,
    )

    data_corner = np.column_stack([target_fit[:, 0], condition_fit])
    flow_corner = np.column_stack([target_flow, condition_plot])
    paths["check_corner"] = out_dir / "check_corner.png"
    plot_summary["check_corner"] = _plot_flow_check(
        data_corner,
        flow_corner,
        [target_key] + condition_keys,
        paths["check_corner"],
        reference_label="weighted KIC" if weights_fit is not None else "training sample",
        sample_label="flow at KIC conditions",
        reference_weights=weights_fit,
        sample_weights=flow_weights,
        bins=bins,
    )

    report = {
        "flow_kind": "conditional",
        "library_versions": library_versions(),
        "target_key": target_key,
        "condition_keys": condition_keys,
        "n_fit": int(target_fit.shape[0]),
        "n_validation": 0 if target_validation is None else int(target_validation.shape[0]),
        "logprob_rows": rows,
        "paths": paths,
        "plots": plot_summary,
        "condition_columns_are_empirical": True,
        "seed": int(seed),
        "n_flow_plot_points": int(condition_plot.shape[0]),
    }
    diagnostics_path = out_dir / "diagnostics.json"
    _write_json(diagnostics_path, report)
    report["paths"]["diagnostics"] = diagnostics_path
    return report
