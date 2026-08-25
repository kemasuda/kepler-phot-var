"""Shared flow utilities for weighted-KIC joint and conditional likelihoods."""

from __future__ import annotations

import json
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import paramax
from jax import random

from flowjax.bijections import Affine, Invert, RationalQuadraticSpline
from flowjax.distributions import Normal, Transformed
from flowjax.flows import masked_autoregressive_flow
from flowjax.train import fit_to_data

__all__ = [
    "WeightedMaximumLikelihoodLoss",
    "WeightedConditionalMaximumLikelihoodLoss",
    "add_standard_columns",
    "catalog_to_target_condition",
    "catalog_to_theta",
    "conditional_log_prob",
    "conditional_sample",
    "condition_scaled",
    "jsonable",
    "load_conditional_flow_r",
    "load_flow_theta",
    "save_conditional_flow_r",
    "save_flow_theta",
    "tau_g98",
    "train_conditional_flow_r",
    "train_flow_theta",
    "write_metadata",
]


def tau_g98(teff):
    """Convective turnover timescale used by the KIC/KOI analyses."""
    teff = np.asarray(teff)
    return 314.24 * np.exp(-(teff / 1952.5) - (teff / 6250.0) ** 18) + 0.002


def add_standard_columns(df):
    """Add derived columns used by the flow specs when possible."""
    df = df.copy()
    if "logp" not in df.columns and "Prot" in df.columns:
        df["logp"] = np.log10(df["Prot"])
    if "lograd" not in df.columns and "rad" in df.columns:
        df["lograd"] = np.log10(df["rad"])
    if "logRo" not in df.columns and {"Teff", "Prot"}.issubset(df.columns):
        df["tau_g98"] = tau_g98(df["Teff"])
        df["logRo"] = np.log10(df["Prot"] / df["tau_g98"])
    return df


def _require_columns(df, columns):
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def _validate_finite_array(values, name):
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _validate_scales(scales, name, min_scale=1e-12):
    arr = _validate_finite_array(scales, name)
    bad = arr <= min_scale
    if np.any(bad):
        raise ValueError(
            f"{name} must be finite and larger than {min_scale}; got {arr[bad]}"
        )
    return arr


def _validate_weights(weights, n_samples):
    weights_np = np.asarray(weights, dtype=float)
    if weights_np.ndim != 1 or weights_np.shape[0] != n_samples:
        raise ValueError("weights must have shape (n_samples,)")
    if not np.all(np.isfinite(weights_np)):
        raise ValueError("weights must be finite")
    if np.any(weights_np < 0.0):
        raise ValueError("weights must be non-negative")
    if not np.any(weights_np > 0.0):
        raise ValueError("at least one weight must be positive")
    return jnp.asarray(weights_np)


def catalog_to_theta(df, keys):
    """Return joint-flow columns in the requested order."""
    df = add_standard_columns(df)
    keys = list(keys)
    _require_columns(df, keys)
    return df[keys].to_numpy(dtype=float)


class WeightedMaximumLikelihoodLoss:
    """Unconditional weighted negative log-likelihood."""

    @eqx.filter_jit
    def __call__(self, params, static, x, weights, key=None):
        dist = paramax.unwrap(eqx.combine(params, static))
        logp = dist.log_prob(x)
        weights = jnp.asarray(weights)
        return -jnp.sum(weights * logp) / jnp.sum(weights)


def _make_flow_theta_skeleton(
    *,
    key,
    n_dim,
    locs,
    scales,
    knots=16,
    interval=8,
):
    flow_x = masked_autoregressive_flow(
        key,
        base_dist=Normal(jnp.zeros(n_dim)),
        transformer=RationalQuadraticSpline(knots=knots, interval=interval),
        invert=True,
    )
    preprocess = Affine(-locs / scales, 1.0 / scales)
    return Transformed(flow_x, Invert(preprocess))


def train_flow_theta(
    theta_samples,
    *,
    weights=None,
    key=random.key(0),
    learning_rate=3e-4,
    max_epochs=500,
    max_patience=50,
    batch_size=1024,
    knots=16,
    interval=8,
    keys=None,
    weight_col=None,
):
    """Train a joint flow that evaluates raw theta-space log probability."""
    theta_np = _validate_finite_array(theta_samples, "theta_samples")
    if theta_np.ndim != 2:
        raise ValueError("theta_samples must have shape (n_samples, n_dim)")
    n_samples, n_dim = theta_np.shape
    z = jnp.asarray(theta_np)

    locs = jnp.mean(z, axis=0)
    scales = jnp.std(z, axis=0)
    _validate_scales(np.asarray(scales), "scales")
    x_samples = (z - locs) / scales

    weights_for_loss = None
    if weights is not None:
        weights_for_loss = _validate_weights(weights, n_samples)

    key, init_key, train_key, sample_key = random.split(key, 4)
    flow_x = masked_autoregressive_flow(
        init_key,
        base_dist=Normal(jnp.zeros(n_dim)),
        transformer=RationalQuadraticSpline(knots=knots, interval=interval),
        invert=True,
    )

    if weights_for_loss is None:
        flow_x, losses = fit_to_data(
            train_key,
            flow_x,
            x_samples,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            max_patience=max_patience,
            batch_size=batch_size,
        )
    else:
        flow_x, losses = fit_to_data(
            train_key,
            flow_x,
            data=(x_samples, weights_for_loss),
            loss_fn=WeightedMaximumLikelihoodLoss(),
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            max_patience=max_patience,
            batch_size=batch_size,
        )

    preprocess = Affine(-locs / scales, 1.0 / scales)
    flow_theta = Transformed(flow_x, Invert(preprocess))
    draws = flow_theta.sample(sample_key, (n_samples,))

    meta = {
        "flow_kind": "joint",
        "n_dim": int(n_dim),
        "keys": None if keys is None else list(keys),
        "locs": np.asarray(locs).tolist(),
        "scales": np.asarray(scales).tolist(),
        "knots": int(knots),
        "interval": float(interval),
        "learning_rate": float(learning_rate),
        "max_epochs": int(max_epochs),
        "max_patience": int(max_patience),
        "batch_size": int(batch_size),
        "weighted": bool(weights is not None),
        "weight_col": weight_col,
    }
    return flow_theta, losses, draws, meta


def save_flow_theta(path, flow_theta, meta):
    """Save metadata as the first JSON line followed by Equinox leaves."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write((json.dumps(jsonable(meta)) + "\n").encode())
        eqx.tree_serialise_leaves(f, flow_theta)


def load_flow_theta(path):
    """Load a joint flow saved by :func:`save_flow_theta`."""
    path = Path(path)
    with path.open("rb") as f:
        meta = json.loads(f.readline().decode())
        flow_like = _make_flow_theta_skeleton(
            key=random.key(0),
            n_dim=meta["n_dim"],
            locs=jnp.asarray(meta["locs"]),
            scales=jnp.asarray(meta["scales"]),
            knots=meta["knots"],
            interval=meta["interval"],
        )
        flow_theta = eqx.tree_deserialise_leaves(f, flow_like)
    return flow_theta, meta


def _as_target_2d(r_raw):
    r = jnp.asarray(r_raw)
    if r.ndim == 1:
        r = r[:, None]
    if r.ndim != 2 or r.shape[1] != 1:
        raise ValueError("target samples must have shape (n_samples, 1)")
    return r


def _as_condition_2d(condition_raw, n_cond, n_samples=None):
    condition = jnp.asarray(condition_raw)
    if condition.ndim == 1:
        if n_cond == 1 and (n_samples is None or condition.shape[0] == n_samples):
            condition = condition[:, None]
        elif condition.shape[0] == n_cond and (n_samples is None or n_samples == 1):
            condition = condition[None, :]
        else:
            raise ValueError(
                "1D condition input is ambiguous; pass shape (n_samples, n_cond)"
            )
    if condition.ndim != 2 or condition.shape[1] != n_cond:
        raise ValueError(f"condition samples must have shape (n_samples, {n_cond})")
    if n_samples is not None and condition.shape[0] != n_samples:
        raise ValueError("target and condition must have the same number of rows")
    return condition


def catalog_to_target_condition(df, target_key, condition_keys):
    """Return raw target and condition arrays from a catalog."""
    df = add_standard_columns(df)
    condition_keys = list(condition_keys)
    _require_columns(df, [target_key] + condition_keys)
    target = df[target_key].to_numpy(dtype=float)[:, None]
    condition = df[condition_keys].to_numpy(dtype=float)
    return target, condition


def condition_scaled(condition_raw, meta):
    """Standardize condition columns using saved conditional-flow metadata."""
    n_cond = int(meta["n_cond"])
    condition = _as_condition_2d(condition_raw, n_cond)
    locs = jnp.asarray(meta["condition_locs"])
    scales = jnp.asarray(meta["condition_scales"])
    return (condition - locs) / scales


def _target_scaled(r_raw, meta):
    target = _as_target_2d(r_raw)
    loc = jnp.asarray(meta["target_loc"])
    scale = jnp.asarray(meta["target_scale"])
    return (target - loc) / scale


class WeightedConditionalMaximumLikelihoodLoss:
    """Conditional weighted negative log-likelihood."""

    @eqx.filter_jit
    def __call__(self, params, static, r, condition, weights, key=None):
        dist = paramax.unwrap(eqx.combine(params, static))
        logp = dist.log_prob(r, condition)
        weights = jnp.asarray(weights)
        return -jnp.sum(weights * logp) / jnp.sum(weights)


def _make_scaled_conditional_flow(
    *,
    key,
    n_cond,
    knots=16,
    interval=8,
    flow_layers=8,
    nn_width=50,
    nn_depth=1,
):
    return masked_autoregressive_flow(
        key,
        base_dist=Normal(jnp.zeros(1)),
        transformer=RationalQuadraticSpline(knots=knots, interval=interval),
        cond_dim=int(n_cond),
        flow_layers=int(flow_layers),
        nn_width=int(nn_width),
        nn_depth=int(nn_depth),
        invert=True,
    )


def _wrap_raw_target(flow_scaled, *, target_loc, target_scale):
    loc = jnp.asarray([target_loc])
    scale = jnp.asarray([target_scale])
    preprocess = Affine(-loc / scale, 1.0 / scale)
    return Transformed(flow_scaled, Invert(preprocess))


def _make_conditional_flow_skeleton(
    *,
    key,
    n_cond,
    target_loc,
    target_scale,
    knots=16,
    interval=8,
    flow_layers=8,
    nn_width=50,
    nn_depth=1,
    target_transform_in_flow=True,
):
    flow_scaled = _make_scaled_conditional_flow(
        key=key,
        n_cond=n_cond,
        knots=knots,
        interval=interval,
        flow_layers=flow_layers,
        nn_width=nn_width,
        nn_depth=nn_depth,
    )
    if target_transform_in_flow:
        return _wrap_raw_target(
            flow_scaled,
            target_loc=target_loc,
            target_scale=target_scale,
        )
    return flow_scaled


def train_conditional_flow_r(
    r_samples,
    condition_samples,
    *,
    weights=None,
    key=random.key(0),
    learning_rate=3e-4,
    max_epochs=500,
    max_patience=50,
    batch_size=1024,
    knots=16,
    interval=8,
    flow_layers=8,
    nn_width=50,
    nn_depth=1,
    target_key="logr",
    condition_keys=None,
    weight_col=None,
    target_transform_in_flow=True,
):
    """Train a conditional flow for raw target values given raw conditions."""
    r_samples = _as_target_2d(r_samples)
    n_samples = r_samples.shape[0]
    if condition_keys is None:
        raise ValueError("condition_keys is required")
    condition_keys = list(condition_keys)
    condition_samples = _as_condition_2d(
        condition_samples,
        len(condition_keys),
        n_samples=n_samples,
    )

    r_np = _validate_finite_array(r_samples, "r_samples")
    condition_np = _validate_finite_array(condition_samples, "condition_samples")

    target_loc = float(np.mean(r_np[:, 0]))
    target_scale = float(np.std(r_np[:, 0]))
    _validate_scales([target_scale], "target_scale")
    condition_locs = np.mean(condition_np, axis=0)
    condition_scales = np.std(condition_np, axis=0)
    _validate_scales(condition_scales, "condition_scales")

    r_scaled = (jnp.asarray(r_np) - target_loc) / target_scale
    c_scaled = (jnp.asarray(condition_np) - jnp.asarray(condition_locs)) / jnp.asarray(
        condition_scales
    )

    weights_for_loss = None
    if weights is not None:
        weights_for_loss = _validate_weights(weights, n_samples)

    key, init_key, train_key, sample_key = random.split(key, 4)
    flow_scaled = _make_scaled_conditional_flow(
        key=init_key,
        n_cond=len(condition_keys),
        knots=knots,
        interval=interval,
        flow_layers=flow_layers,
        nn_width=nn_width,
        nn_depth=nn_depth,
    )

    if weights_for_loss is None:
        flow_scaled, losses = fit_to_data(
            train_key,
            flow_scaled,
            data=(r_scaled, c_scaled),
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            max_patience=max_patience,
            batch_size=batch_size,
        )
    else:
        flow_scaled, losses = fit_to_data(
            train_key,
            flow_scaled,
            data=(r_scaled, c_scaled, weights_for_loss),
            loss_fn=WeightedConditionalMaximumLikelihoodLoss(),
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            max_patience=max_patience,
            batch_size=batch_size,
        )

    if target_transform_in_flow:
        flow = _wrap_raw_target(
            flow_scaled,
            target_loc=target_loc,
            target_scale=target_scale,
        )
    else:
        flow = flow_scaled

    sample_meta = {
        "target_loc": target_loc,
        "target_scale": target_scale,
        "condition_locs": condition_locs.tolist(),
        "condition_scales": condition_scales.tolist(),
        "n_cond": len(condition_keys),
        "target_transform_in_flow": bool(target_transform_in_flow),
    }
    draws = conditional_sample(flow, sample_key, condition_np, sample_meta)

    meta = {
        "flow_kind": "conditional",
        "target_key": target_key,
        "condition_keys": condition_keys,
        "target_loc": target_loc,
        "target_scale": target_scale,
        "condition_locs": condition_locs.tolist(),
        "condition_scales": condition_scales.tolist(),
        "n_cond": int(len(condition_keys)),
        "knots": int(knots),
        "interval": float(interval),
        "flow_layers": int(flow_layers),
        "nn_width": int(nn_width),
        "nn_depth": int(nn_depth),
        "target_transform_in_flow": bool(target_transform_in_flow),
        "learning_rate": float(learning_rate),
        "max_epochs": int(max_epochs),
        "max_patience": int(max_patience),
        "batch_size": int(batch_size),
        "weighted": bool(weights is not None),
        "weight_col": weight_col,
    }
    return flow, losses, draws, meta


def conditional_log_prob(flow, r_raw, condition_raw, meta):
    """Evaluate raw-target log p(target | condition)."""
    r = _as_target_2d(r_raw)
    condition = _as_condition_2d(condition_raw, int(meta["n_cond"]), r.shape[0])
    c_scaled = condition_scaled(condition, meta)
    if meta.get("target_transform_in_flow", True):
        logp = flow.log_prob(r, c_scaled)
    else:
        logp = flow.log_prob(_target_scaled(r, meta), c_scaled)
        logp = logp - jnp.log(jnp.asarray(meta["target_scale"]))
    return jnp.reshape(logp, (r.shape[0],))


def conditional_sample(flow, key, condition_raw, meta, sample_shape=()):
    """Sample raw target values for raw condition rows."""
    c_scaled = condition_scaled(condition_raw, meta)
    samples = flow.sample(key, sample_shape=sample_shape, condition=c_scaled)
    if not meta.get("target_transform_in_flow", True):
        samples = samples * jnp.asarray(meta["target_scale"]) + jnp.asarray(
            meta["target_loc"]
        )
    return samples


def save_conditional_flow_r(path, flow, meta):
    """Save metadata as the first JSON line followed by Equinox leaves."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write((json.dumps(jsonable(meta)) + "\n").encode())
        eqx.tree_serialise_leaves(f, flow)


def load_conditional_flow_r(path):
    """Load a conditional flow saved by :func:`save_conditional_flow_r`."""
    path = Path(path)
    with path.open("rb") as f:
        meta = json.loads(f.readline().decode())
        flow_like = _make_conditional_flow_skeleton(
            key=random.key(0),
            n_cond=meta["n_cond"],
            target_loc=meta["target_loc"],
            target_scale=meta["target_scale"],
            knots=meta["knots"],
            interval=meta["interval"],
            flow_layers=meta.get("flow_layers", 8),
            nn_width=meta.get("nn_width", 50),
            nn_depth=meta.get("nn_depth", 1),
            target_transform_in_flow=meta.get("target_transform_in_flow", True),
        )
        flow = eqx.tree_deserialise_leaves(f, flow_like)
    return flow, meta


def write_metadata(path, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(meta), indent=2, sort_keys=True) + "\n")


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return value.tolist()
    return value

