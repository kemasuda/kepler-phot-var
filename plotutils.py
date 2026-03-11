from __future__ import annotations

__all__ = ['plot']

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation as mad


def weighted_median(x, w):
    x = np.asarray(x)
    w = np.asarray(w)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[m], w[m]
    if x.size == 0:
        return np.nan
    idx = np.argsort(x)
    x, w = x[idx], w[idx]
    cw = np.cumsum(w)
    return x[np.searchsorted(cw, 0.5 * cw[-1])]


def weighted_mad(x, w):
    med = weighted_median(x, w)
    return weighted_median(np.abs(x - med), w)


def binned_logr_stats(
    df,
    teff_bins,
    *,
    use_weight=False,
    weight_col="pdet",
):
    d = df.dropna(subset=["Teff", "logr"]).copy()

    if use_weight:
        if weight_col not in d.columns:
            raise KeyError(f"{weight_col} not found in DataFrame")
        d = d.dropna(subset=[weight_col])
        d = d[d[weight_col] > 0]

    d["Teff_bin"] = pd.cut(d["Teff"], bins=teff_bins)

    rows = []
    for i, (_, g) in enumerate(d.groupby("Teff_bin", observed=True)):
        x = g["logr"].to_numpy()

        if use_weight:
            w = g[weight_col].to_numpy()
            logr_med = weighted_median(x, w)
            logr_mad = weighted_mad(x, w)
            N = np.sum(w)
        else:
            logr_med = np.median(x)
            logr_mad = mad(x)
            N = len(x)

        rows.append(
            dict(
                teff_bin=0.5 * (teff_bins[i + 1] + teff_bins[i]),
                logr_med=logr_med,
                logr_mad=logr_mad,
                N=N,
            )
        )

    return pd.DataFrame(rows)


def gaussian_weighted_median(
    teff, x, t0, sigma,
    sample_weight=None
):
    teff = np.asarray(teff)
    x = np.asarray(x)

    if sample_weight is None:
        sample_weight = np.ones_like(x, dtype=float)
    else:
        sample_weight = np.asarray(sample_weight, dtype=float)

    m = (
        np.isfinite(teff)
        & np.isfinite(x)
        & np.isfinite(sample_weight)
        & (sample_weight > 0)
    )
    teff, x, sample_weight = teff[m], x[m], sample_weight[m]

    w = sample_weight * np.exp(-0.5 * ((teff - t0) / sigma) ** 2)
    return weighted_median(x, w)


def smooth_logr_vs_teff(
    df,
    teff_grid,
    sigma,
    *,
    stat='median',
    use_weight=False,
    weight_col="pdet",
):
    teff = df["Teff"].to_numpy()
    logr = df["logr"].to_numpy()

    if use_weight:
        if weight_col not in df.columns:
            raise KeyError(f"{weight_col} not found in DataFrame")
        sample_weight = df[weight_col].to_numpy()
    else:
        sample_weight = None

    return np.array([
        gaussian_weighted_median(
            teff, logr, t0, sigma,
            sample_weight=sample_weight
        )
        for t0 in teff_grid
    ])


def plot(
    dkic, dkoi, df_list=None,
    label_list='resampled',
    ckic='gray', ckoi='salmon', ckic2='C1',
    teff_bins=np.arange(3500, 6750, 250),
    teff_grid=np.linspace(3500, 6500, 100),
    teff_sigma=100,
    skip_bins=False,
    *,
    use_weight=False,
    rs_method="smooth",   # "smooth" or "binned"
):
    dkic_bin = binned_logr_stats(dkic, teff_bins)
    dkoi_bin = binned_logr_stats(dkoi, teff_bins)

    fig = plt.figure(figsize=(14, 9))
    plt.xlabel("$T_\\mathrm{eff}$ (K)")
    plt.ylabel("$\\log_{10} R$ (ppm)")
    plt.ylim(2.3, 5.2)
    plt.xlim(teff_grid[0], teff_grid[-1])

    if not skip_bins:
        plt.plot(dkic.Teff, dkic.logr, ',', color=ckic)
        plt.plot(
            dkic_bin.teff_bin, dkic_bin.logr_med, 'o',
            mfc='white', color=ckic, mew=2, lw=1,
            label=f'KIC ({len(dkic)}), binned median'
        )

        plt.plot(dkoi.Teff, dkoi.logr, '.', color=ckoi, markersize=2)
        plt.plot(
            dkoi_bin.teff_bin, dkoi_bin.logr_med, 'o',
            mfc='white', color=ckoi, mew=2, lw=1,
            label=f'KOI ({len(dkoi)}), binned median'
        )

    logr_med_kic = smooth_logr_vs_teff(dkic, teff_grid, teff_sigma)
    plt.plot(
        teff_grid, logr_med_kic, '-', color=ckic, lw=1,
        label=f'KIC, windowed median ($\\sigma={teff_sigma:.0f}\\,\\mathrm{{K}}$)'
    )

    logr_med_koi = smooth_logr_vs_teff(dkoi, teff_grid, teff_sigma)
    plt.plot(
        teff_grid, logr_med_koi, '-', color=ckoi, lw=1,
        label=f'KOI, windowed median ($\\sigma={teff_sigma:.0f}\\,\\mathrm{{K}}$)'
    )

    if df_list is not None:
        if rs_method not in ("smooth", "binned"):
            raise ValueError("rs_method must be 'smooth' or 'binned'")

        if rs_method == "smooth":
            y_list = []
            for dkic_rs in df_list:
                y_list.append(
                    smooth_logr_vs_teff(
                        dkic_rs, teff_grid, teff_sigma, use_weight=use_weight
                    )
                )
            y_arr = np.asarray(y_list)

            x = teff_grid
            rs_med = np.median(y_arr, axis=0)
            rs_sigma = 1.4826 * mad(y_arr, axis=0)
            plt.plot(x, rs_med, '-', color=ckic2,
                     ls='dashed', label=label_list)

        else:
            y_list = []
            x = None
            for dkic_rs in df_list:
                b = binned_logr_stats(
                    dkic_rs, teff_bins, use_weight=use_weight
                )
                if x is None:
                    x = np.asarray(b.teff_bin)
                y_list.append(np.asarray(b.logr_med))
            y_arr = np.asarray(y_list)

            rs_med = np.median(y_arr, axis=0)
            rs_sigma = 1.4826 * mad(y_arr, axis=0)
            plt.plot(
                x, rs_med, 'o-', color=ckic2, ls='dashed',
                mfc='white', label=label_list, mew=2, lw=1
            )

        plt.fill_between(
            x, rs_med - rs_sigma, rs_med + rs_sigma,
            color=ckic2, alpha=0.2
        )

    plt.legend(loc='best')
    return fig
