"""Alpha inference utilities for weighted-KIC joint flow likelihoods."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import arviz as az
import corner
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from jax import random
from numpyro.infer import MCMC, NUTS, init_to_value
from tinygp import GaussianProcess, kernels

LIKELIHOOD_DIR = Path(__file__).resolve().parent
REPO_ROOT = LIKELIHOOD_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from likelihood_wkic_common.flowutils import (  # noqa: E402
    add_standard_columns,
    catalog_to_theta,
    jsonable,
    load_flow_theta,
)

try:
    from numpyro_inferutils import find_map_svi
except Exception:
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoDelta
    from numpyro.optim import Adam

    def find_map_svi(model, rng_key, step_size=5e-2, num_steps=2000):
        guide = AutoDelta(model)
        svi = SVI(model, guide, Adam(step_size), Trace_ELBO())
        state = svi.init(rng_key)
        for _ in range(int(num_steps)):
            state, _ = svi.update(state)
        params = svi.get_params(state)
        return guide.median(params)


DEFAULT_KOI_PATH = REPO_ROOT / "data_m15" / "m15_koi.csv"
DEFAULT_STELLAR_RADIUS_PATH = REPO_ROOT / "data" / "DR2PapTable1.txt"
DEFAULT_KOI_CANDIDATE_PATH = REPO_ROOT / "data" / "koi_candidates.csv"

DEFAULT_ALPHA_BOUNDS = (-0.2, 0.2)
DEFAULT_ALPHA_MEAN_PRIOR = "uniform"
DEFAULT_LOG_SIGMA_BOUNDS = (np.log(5e-3), np.log(0.5))
DEFAULT_LOG_ELL_BOUNDS = (np.log(0.05), np.log(5.0))

__all__ = [
    "DEFAULT_KOI_CANDIDATE_PATH",
    "DEFAULT_ALPHA_MEAN_PRIOR",
    "DEFAULT_KOI_PATH",
    "DEFAULT_STELLAR_RADIUS_PATH",
    "LIKELIHOOD_DIR",
    "REPO_ROOT",
    "alpha_grid_samples",
    "alpha_grid_summary",
    "build_interp_matrix",
    "flow_model_path",
    "inference_output_dir",
    "load_koi_catalog",
    "make_alpha_model_quasisep",
    "plot_alpha_grid",
    "plot_corner",
    "plot_svi_alpha_quicklook",
    "prepare_ykoi",
    "run_alpha_inference",
    "run_and_save_alpha_inference",
    "save_inference_outputs",
]


def _load_flow_theta_for_inference(flow_path):
    try:
        return load_flow_theta(flow_path)
    except RuntimeError as exc:
        message = str(exc)
        if "changed shape" not in message:
            raise
        raise RuntimeError(
            "\n".join(
                [
                    f"Could not load joint flow model: {flow_path}",
                    "",
                    "The saved Equinox/FlowJAX tree does not match the current",
                    "joint-flow skeleton. Re-run train_kic_flow.ipynb in this",
                    "environment, or load the model in the environment that trained it.",
                    "",
                    "Original deserialisation error:",
                    message,
                ]
            )
        ) from exc


def flow_model_path(flow_id, *, likelihood_dir=LIKELIHOOD_DIR):
    return Path(likelihood_dir) / "flows" / flow_id / "model.eqx"


def inference_output_dir(alpha_name, flow_id, *, root=None):
    if root is None:
        root = LIKELIHOOD_DIR / "inference"
    return Path(root) / alpha_name / flow_id


def _query_needs_column(query, column):
    return query is not None and column in query


def _merge_koi_period(dkoi, koi_candidate_path):
    dp = pd.read_csv(koi_candidate_path, comment="#")
    dp_inner = (
        dp.sort_values("koi_period")
        .drop_duplicates("kepid", keep="first")
        .reset_index(drop=True)
    )
    return pd.merge(dkoi, dp_inner[["kepid", "koi_period"]], on="kepid")


def load_koi_catalog(
    *,
    koi_path=DEFAULT_KOI_PATH,
    stellar_radius_path=DEFAULT_STELLAR_RADIUS_PATH,
    koi_candidate_path=DEFAULT_KOI_CANDIDATE_PATH,
    query=None,
):
    dkoi = pd.read_csv(koi_path)
    if stellar_radius_path is not None and "rad" not in dkoi.columns:
        dkicrad = pd.read_csv(stellar_radius_path, delimiter="&")
        dkicrad.columns = dkicrad.columns.str.strip()
        dkoi = pd.merge(dkoi, dkicrad[["KIC", "rad"]], on="KIC")
    if (
        "koi_period" not in dkoi.columns
        and koi_candidate_path is not None
        and _query_needs_column(query, "koi_period")
    ):
        dkoi = _merge_koi_period(dkoi, koi_candidate_path)
    dkoi = add_standard_columns(dkoi)
    if query:
        dkoi = dkoi.query(query).copy()
    return dkoi.reset_index(drop=True)


def prepare_ykoi(dkoi, keys):
    required_cols = list(keys)
    dkoi_used = (
        add_standard_columns(dkoi)
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=required_cols)
        .reset_index(drop=True)
    )
    if len(dkoi_used) == 0:
        raise ValueError("No KOI rows remain after finite-value cuts.")
    ykoi = jnp.asarray(catalog_to_theta(dkoi_used.copy(), keys))
    return ykoi, dkoi_used


def _normalize_bounds(bounds, name):
    if bounds is None:
        return None
    if len(bounds) != 2:
        raise ValueError(f"{name} must be a length-2 tuple: (lower, upper).")
    lower, upper = map(float, bounds)
    if not np.isfinite(lower) or not np.isfinite(upper) or not lower < upper:
        raise ValueError(f"{name} must be finite and satisfy lower < upper.")
    return lower, upper


def _normalize_alpha_mean_prior(alpha_mean_prior):
    if alpha_mean_prior is None:
        return "fixed"
    prior = str(alpha_mean_prior).lower()
    if prior in {"fixed", "none", "zero"}:
        return "fixed"
    if prior in {"uniform", "normal"}:
        return prior
    raise ValueError(
        "alpha_mean_prior must be one of None, 'fixed', 'uniform', or 'normal'."
    )


def build_interp_matrix(x, x_grid):
    """Piecewise-linear interpolation matrix W such that f(x) ~= W @ f(x_grid)."""
    n = x.shape[0]
    m = x_grid.shape[0]
    idx = jnp.searchsorted(x_grid, x, side="right") - 1
    idx = jnp.clip(idx, 0, m - 2)
    x0 = x_grid[idx]
    x1 = x_grid[idx + 1]
    t = jnp.clip((x - x0) / (x1 - x0), 0.0, 1.0)
    W = jnp.zeros((n, m))
    W = W.at[jnp.arange(n), idx].set(1.0 - t)
    W = W.at[jnp.arange(n), idx + 1].add(t)
    return W


def make_alpha_model_quasisep(
    *,
    ykoi,
    flow,
    x_name,
    x_index=0,
    shift_index=1,
    n_grid=20,
    jitter=1e-6,
    alpha_bounds=DEFAULT_ALPHA_BOUNDS,
    log_sigma_bounds=DEFAULT_LOG_SIGMA_BOUNDS,
    log_ell_bounds=DEFAULT_LOG_ELL_BOUNDS,
    alpha_mean_prior=DEFAULT_ALPHA_MEAN_PRIOR,
    alpha_mean_bounds=DEFAULT_ALPHA_BOUNDS,
    alpha_mean_loc=0.0,
    alpha_mean_scale=0.1,
):
    if n_grid < 2:
        raise ValueError("n_grid must be at least 2.")
    alpha_mean_prior = _normalize_alpha_mean_prior(alpha_mean_prior)
    alpha_bounds = _normalize_bounds(alpha_bounds, "alpha_bounds")
    log_sigma_bounds = _normalize_bounds(log_sigma_bounds, "log_sigma_bounds")
    log_ell_bounds = _normalize_bounds(log_ell_bounds, "log_ell_bounds")
    if alpha_bounds is None:
        raise ValueError("alpha_bounds is required.")
    if log_sigma_bounds is None:
        raise ValueError("log_sigma_bounds is required.")
    if log_ell_bounds is None:
        raise ValueError("log_ell_bounds is required.")
    alpha_mean_bounds = _normalize_bounds(alpha_mean_bounds, "alpha_mean_bounds")
    if alpha_mean_prior == "uniform" and alpha_mean_bounds is None:
        raise ValueError(
            "alpha_mean_bounds is required when alpha_mean_prior='uniform'."
        )
    alpha_mean_loc = float(alpha_mean_loc)
    alpha_mean_scale = float(alpha_mean_scale)
    if not np.isfinite(alpha_mean_loc):
        raise ValueError("alpha_mean_loc must be finite.")
    if not np.isfinite(alpha_mean_scale) or alpha_mean_scale <= 0.0:
        raise ValueError("alpha_mean_scale must be finite and positive.")

    x = ykoi[:, x_index]
    x_min = jnp.min(x)
    x_max = jnp.max(x)
    if not bool(x_min < x_max):
        raise ValueError("x values must span a non-zero range.")
    x_scaled = (x - x_min) / (x_max - x_min)
    x_grid_scaled = jnp.linspace(0.0, 1.0, n_grid)
    x_grid_phys = x_min + x_grid_scaled * (x_max - x_min)
    W_data = build_interp_matrix(x_scaled, x_grid_scaled)

    alpha_lo, alpha_hi = alpha_bounds
    log_sigma_lo, log_sigma_hi = log_sigma_bounds
    log_ell_lo, log_ell_hi = log_ell_bounds
    if alpha_mean_bounds is None:
        alpha_mean_lo, alpha_mean_hi = None, None
    else:
        alpha_mean_lo, alpha_mean_hi = alpha_mean_bounds

    def model():
        if alpha_mean_prior == "fixed":
            alpha_mean = 0.0
        elif alpha_mean_prior == "uniform":
            alpha_mean = numpyro.sample(
                "alpha_mean",
                dist.Uniform(jnp.asarray(alpha_mean_lo), jnp.asarray(alpha_mean_hi)),
            )
        elif alpha_mean_prior == "normal":
            alpha_mean = numpyro.sample(
                "alpha_mean",
                dist.Normal(jnp.asarray(alpha_mean_loc), jnp.asarray(alpha_mean_scale)),
            )

        alpha = numpyro.sample(
            "alpha",
            dist.Uniform(alpha_lo * jnp.ones(n_grid), alpha_hi * jnp.ones(n_grid)),
        )
        log_sigma = numpyro.sample(
            "log_sigma_alpha",
            dist.Uniform(jnp.asarray(log_sigma_lo), jnp.asarray(log_sigma_hi)),
        )
        log_ell = numpyro.sample(
            "log_ell",
            dist.Uniform(jnp.asarray(log_ell_lo), jnp.asarray(log_ell_hi)),
        )

        sigma_alpha = jnp.exp(log_sigma)
        ell = jnp.exp(log_ell)
        kernel = kernels.quasisep.Matern32(scale=ell, sigma=sigma_alpha)
        gp = GaussianProcess(kernel, x_grid_scaled, diag=jitter)
        numpyro.factor("gpprior", gp.log_probability(alpha - alpha_mean))

        alpha_each = W_data @ alpha
        numpyro.deterministic("alpha_each", alpha_each)
        y_shift = ykoi.at[:, shift_index].add(-alpha_each)
        flow_loglike = jnp.sum(flow.log_prob(y_shift))
        numpyro.deterministic("flow_loglike_value", flow_loglike)
        numpyro.factor("flow_loglike", flow_loglike)

    aux = {
        "x_name": x_name,
        "x_index": int(x_index),
        "shift_index": int(shift_index),
        "x_min": x_min,
        "x_max": x_max,
        "x_grid_scaled": x_grid_scaled,
        "x_grid_phys": x_grid_phys,
        "W_data": W_data,
        "n_grid": int(n_grid),
        "jitter": float(jitter),
        "alpha_bounds": list(alpha_bounds),
        "log_sigma_bounds": list(log_sigma_bounds),
        "log_ell_bounds": list(log_ell_bounds),
        "alpha_mean_prior": alpha_mean_prior,
        "alpha_mean_bounds": (
            None if alpha_mean_bounds is None else list(alpha_mean_bounds)
        ),
        "alpha_mean_loc": float(alpha_mean_loc),
        "alpha_mean_scale": float(alpha_mean_scale),
    }
    return model, aux


def run_alpha_inference(
    *,
    flow_id=None,
    flow_path=None,
    x_name=None,
    x_index=0,
    shift_index=1,
    koi_path=DEFAULT_KOI_PATH,
    stellar_radius_path=DEFAULT_STELLAR_RADIUS_PATH,
    koi_candidate_path=DEFAULT_KOI_CANDIDATE_PATH,
    koi_query=None,
    n_grid=20,
    jitter=1e-6,
    alpha_bounds=DEFAULT_ALPHA_BOUNDS,
    log_sigma_bounds=DEFAULT_LOG_SIGMA_BOUNDS,
    log_ell_bounds=DEFAULT_LOG_ELL_BOUNDS,
    alpha_mean_prior=DEFAULT_ALPHA_MEAN_PRIOR,
    alpha_mean_bounds=DEFAULT_ALPHA_BOUNDS,
    alpha_mean_loc=0.0,
    alpha_mean_scale=0.1,
    num_warmup=500,
    num_samples=500,
    num_chains=2,
    rng_seed=0,
    svi_seed=0,
    svi_step_size=5e-2,
    svi_num_steps=2000,
    dense_mass=True,
    max_tree_depth=8,
    target_accept_prob=0.8,
    progress_bar=True,
    print_summary=True,
    svi_quicklook_path=None,
    svi_quicklook_csv_path=None,
    svi_quicklook_x_label=None,
    svi_quicklook_ylabel=r"$r_\mathrm{KOI}-r_\mathrm{KIC}$",
    svi_map_path=None,
):
    if flow_path is None:
        if flow_id is None:
            raise ValueError("Either flow_id or flow_path is required.")
        flow_path = flow_model_path(flow_id)

    numpyro.set_host_device_count(num_chains)
    flow, flow_meta = _load_flow_theta_for_inference(flow_path)
    keys = list(flow_meta["keys"])
    if x_name is None:
        x_name = keys[x_index]

    dkoi = load_koi_catalog(
        koi_path=koi_path,
        stellar_radius_path=stellar_radius_path,
        koi_candidate_path=koi_candidate_path,
        query=koi_query,
    )
    ykoi, dkoi_used = prepare_ykoi(dkoi, keys)

    model, aux = make_alpha_model_quasisep(
        ykoi=ykoi,
        flow=flow,
        x_name=x_name,
        x_index=x_index,
        shift_index=shift_index,
        n_grid=n_grid,
        jitter=jitter,
        alpha_bounds=alpha_bounds,
        log_sigma_bounds=log_sigma_bounds,
        log_ell_bounds=log_ell_bounds,
        alpha_mean_prior=alpha_mean_prior,
        alpha_mean_bounds=alpha_mean_bounds,
        alpha_mean_loc=alpha_mean_loc,
        alpha_mean_scale=alpha_mean_scale,
    )

    popt = find_map_svi(
        model,
        rng_key=random.PRNGKey(svi_seed),
        step_size=svi_step_size,
        num_steps=svi_num_steps,
    )
    _write_svi_map(popt, svi_map_path)
    if svi_quicklook_path is not None or svi_quicklook_csv_path is not None:
        fig, _ = plot_svi_alpha_quicklook(
            popt,
            aux,
            xlabel=svi_quicklook_x_label,
            ylabel=svi_quicklook_ylabel,
            path=svi_quicklook_path,
            csv_path=svi_quicklook_csv_path,
        )
        if fig is not None:
            plt.close(fig)
    kernel = NUTS(
        model,
        dense_mass=dense_mass,
        init_strategy=init_to_value(values=popt),
        max_tree_depth=max_tree_depth,
        target_accept_prob=target_accept_prob,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )
    mcmc.run(random.PRNGKey(rng_seed))
    if print_summary:
        mcmc.print_summary()

    return {
        "mcmc": mcmc,
        "aux": aux,
        "flow_meta": flow_meta,
        "flow_path": Path(flow_path),
        "flow_id": flow_id,
        "svi_map": popt,
        "keys": keys,
        "ykoi": ykoi,
        "dkoi": dkoi_used,
        "run_config": {
            "koi_path": str(koi_path),
            "stellar_radius_path": (
                None if stellar_radius_path is None else str(stellar_radius_path)
            ),
            "koi_candidate_path": (
                None if koi_candidate_path is None else str(koi_candidate_path)
            ),
            "koi_query": koi_query,
            "x_name": x_name,
            "x_index": int(x_index),
            "shift_index": int(shift_index),
            "n_grid": int(n_grid),
            "jitter": float(jitter),
            "alpha_bounds": list(alpha_bounds),
            "log_sigma_bounds": list(log_sigma_bounds),
            "log_ell_bounds": list(log_ell_bounds),
            "alpha_mean_prior": (
                None if alpha_mean_prior is None else str(alpha_mean_prior)
            ),
            "alpha_mean_bounds": (
                None if alpha_mean_bounds is None else list(alpha_mean_bounds)
            ),
            "alpha_mean_loc": float(alpha_mean_loc),
            "alpha_mean_scale": float(alpha_mean_scale),
            "num_warmup": int(num_warmup),
            "num_samples": int(num_samples),
            "num_chains": int(num_chains),
            "rng_seed": int(rng_seed),
            "svi_seed": int(svi_seed),
            "svi_step_size": float(svi_step_size),
            "svi_num_steps": int(svi_num_steps),
            "dense_mass": bool(dense_mass),
            "max_tree_depth": int(max_tree_depth),
            "target_accept_prob": float(target_accept_prob),
            "svi_quicklook_path": (
                None if svi_quicklook_path is None else str(svi_quicklook_path)
            ),
            "svi_quicklook_csv_path": (
                None if svi_quicklook_csv_path is None else str(svi_quicklook_csv_path)
            ),
            "svi_map_path": None if svi_map_path is None else str(svi_map_path),
            "jax_device_count": int(jax.local_device_count()),
            "n_koi_input": int(len(dkoi)),
            "n_koi": int(len(dkoi_used)),
        },
    }


def alpha_grid_samples(mcmc):
    return np.asarray(mcmc.get_samples()["alpha"])


def alpha_grid_summary(mcmc, percentiles=(16, 50, 84)):
    return np.percentile(alpha_grid_samples(mcmc), percentiles, axis=0)


def plot_alpha_grid(
    mcmc,
    aux,
    *,
    xlabel=None,
    ylabel=r"$r_\mathrm{KOI}-r_\mathrm{KIC}$",
    path=None,
):
    alpha_lo, alpha_med, alpha_hi = alpha_grid_summary(mcmc)
    x_grid_phys = np.asarray(aux["x_grid_phys"])
    fig, ax = plt.subplots()
    ax.fill_between(x_grid_phys, alpha_lo, alpha_hi, alpha=0.3)
    ax.plot(x_grid_phys, alpha_med, ".-")
    ax.set_xlim(x_grid_phys[0], x_grid_phys[-1])
    ax.axhline(y=0, color="k", ls="dashed")
    ax.set_xlabel(xlabel or aux["x_name"])
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig, ax


def plot_svi_alpha_quicklook(
    svi_map,
    aux,
    *,
    xlabel=None,
    ylabel=r"$r_\mathrm{KOI}-r_\mathrm{KIC}$",
    path=None,
    csv_path=None,
):
    if svi_map is None or "alpha" not in svi_map:
        return None, None
    x_grid_phys = np.asarray(aux["x_grid_phys"])
    alpha = np.asarray(svi_map["alpha"])
    fig, ax = plt.subplots()
    ax.plot(x_grid_phys, alpha, ".-", color="C3", label="SVI MAP")
    ax.axhline(y=0, color="k", ls="dashed")
    ax.set_xlim(x_grid_phys[0], x_grid_phys[-1])
    ax.set_xlabel(xlabel or aux["x_name"])
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    fig.tight_layout()
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    if csv_path is not None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({aux["x_name"]: x_grid_phys, "alpha_svi_map": alpha}).to_csv(
            csv_path,
            index=False,
        )
    return fig, ax


def _write_svi_map(svi_map, path):
    if path is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(svi_map), indent=2, sort_keys=True) + "\n")
    return path


def plot_corner(
    mcmc,
    *,
    var_names=("alpha", "alpha_mean", "log_sigma_alpha", "log_ell"),
    path=None,
):
    idata = az.from_numpyro(mcmc)
    posterior_vars = set(idata.posterior.data_vars)
    var_names = [name for name in var_names if name in posterior_vars]
    if len(var_names) == 0:
        return None
    fig = corner.corner(idata, var_names=var_names, show_titles=True)
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def _aux_for_json(aux):
    out = {}
    for key, value in aux.items():
        if key == "W_data":
            out[f"{key}_shape"] = list(np.asarray(value).shape)
        else:
            out[key] = value
    return out


def _posterior_samples_for_npz(mcmc):
    samples = mcmc.get_samples()
    keep = {}
    for name in [
        "alpha",
        "alpha_mean",
        "log_sigma_alpha",
        "log_ell",
        "flow_loglike_value",
    ]:
        if name in samples:
            keep[name] = np.asarray(samples[name])
    return keep


def _write_convergence_diagnostics(mcmc, path):
    path = Path(path)
    try:
        summary = az.summary(az.from_numpyro(mcmc), kind="diagnostics")
        summary.to_json(path)
    except Exception as exc:
        path.write_text(json.dumps({"error": str(exc)}, indent=2) + "\n")


def save_inference_outputs(
    result,
    output_dir,
    *,
    alpha_name,
    x_label=None,
    ylabel=r"$r_\mathrm{KOI}-r_\mathrm{KIC}$",
    corner_var_names=("alpha", "alpha_mean", "log_sigma_alpha", "log_ell"),
    run_spec=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mcmc = result["mcmc"]
    aux = result["aux"]

    plot_alpha_grid(mcmc, aux, xlabel=x_label, ylabel=ylabel, path=output_dir / "alpha.png")
    plot_svi_alpha_quicklook(
        result.get("svi_map"),
        aux,
        xlabel=x_label,
        ylabel=ylabel,
        path=output_dir / "svi_alpha.png",
        csv_path=output_dir / "svi_alpha.csv",
    )
    svi_map_path = _write_svi_map(result.get("svi_map"), output_dir / "svi_map.json")
    plot_corner(mcmc, var_names=corner_var_names, path=output_dir / "corner.png")

    idata = az.from_numpyro(mcmc)
    posterior_path = output_dir / "posterior.nc"
    try:
        idata.to_netcdf(posterior_path)
    except Exception:
        posterior_path = output_dir / "posterior.json"
        idata.to_json(posterior_path)

    alpha_lo, alpha_med, alpha_hi = alpha_grid_summary(mcmc)
    alpha_summary = pd.DataFrame(
        {
            result["run_config"]["x_name"]: np.asarray(aux["x_grid_phys"]),
            "alpha_p16": alpha_lo,
            "alpha_median": alpha_med,
            "alpha_p84": alpha_hi,
            "alpha_err68": 0.5 * (alpha_hi - alpha_lo),
        }
    )
    alpha_summary_path = output_dir / "alpha_summary.csv"
    alpha_summary.to_csv(alpha_summary_path, index=False)

    samples_path = output_dir / "posterior_samples.npz"
    np.savez(samples_path, **_posterior_samples_for_npz(mcmc))

    convergence_path = output_dir / "convergence_diagnostics.json"
    _write_convergence_diagnostics(mcmc, convergence_path)

    aux_path = output_dir / "aux.json"
    aux_path.write_text(json.dumps(jsonable(_aux_for_json(aux)), indent=2, sort_keys=True) + "\n")

    metadata = {
        "alpha_name": alpha_name,
        "flow_id": result.get("flow_id"),
        "flow_path": result.get("flow_path"),
        "flow_meta": result.get("flow_meta"),
        "svi_map": result.get("svi_map"),
        "svi_map_path": svi_map_path,
        "keys": result.get("keys"),
        "n_koi": len(result["dkoi"]),
        "output_dir": output_dir,
        "posterior_path": posterior_path,
        "alpha_summary_path": alpha_summary_path,
        "posterior_samples_path": samples_path,
        "convergence_diagnostics_path": convergence_path,
        "run_config": result.get("run_config"),
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(jsonable(metadata), indent=2, sort_keys=True) + "\n")

    run_spec_path = output_dir / "run_spec.json"
    run_spec_path.write_text(json.dumps(jsonable(run_spec or {}), indent=2, sort_keys=True) + "\n")

    return {
        "output_dir": output_dir,
        "posterior_path": posterior_path,
        "alpha_path": output_dir / "alpha.png",
        "svi_alpha_path": output_dir / "svi_alpha.png",
        "svi_alpha_csv_path": output_dir / "svi_alpha.csv",
        "svi_map_path": output_dir / "svi_map.json",
        "corner_path": output_dir / "corner.png",
        "aux_path": aux_path,
        "metadata_path": metadata_path,
        "alpha_summary_path": alpha_summary_path,
        "posterior_samples_path": samples_path,
        "convergence_diagnostics_path": convergence_path,
        "run_spec_path": run_spec_path,
    }


def run_and_save_alpha_inference(spec, **overrides):
    original_spec = dict(spec)
    spec = dict(spec)
    spec.update(overrides)

    alpha_name = spec.pop("alpha_name")
    flow_id = spec.get("flow_id")
    x_label = spec.pop("x_label", None)
    ylabel = spec.pop("ylabel", r"$r_\mathrm{KOI}-r_\mathrm{KIC}$")
    output_dir = spec.pop("output_dir", None)
    inference_root = spec.pop("inference_root", None)

    if output_dir is None:
        output_dir = inference_output_dir(
            alpha_name,
            flow_id,
            root=inference_root,
        )
    output_dir = Path(output_dir)

    spec.setdefault("svi_quicklook_path", output_dir / "svi_alpha.png")
    spec.setdefault("svi_quicklook_csv_path", output_dir / "svi_alpha.csv")
    spec.setdefault("svi_quicklook_x_label", x_label)
    spec.setdefault("svi_quicklook_ylabel", ylabel)
    spec.setdefault("svi_map_path", output_dir / "svi_map.json")

    result = run_alpha_inference(**spec)
    run_spec = dict(original_spec)
    run_spec.update(overrides)
    paths = save_inference_outputs(
        result,
        output_dir,
        alpha_name=alpha_name,
        x_label=x_label,
        ylabel=ylabel,
        run_spec=run_spec,
    )
    result["paths"] = paths
    return result
