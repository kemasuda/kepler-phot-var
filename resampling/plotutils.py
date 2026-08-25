from __future__ import annotations

__all__ = ["plot_list"]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation as mad


def binned_logr_stats(df, teff_bins):
    d = df.dropna(subset=["Teff", "logr"]).copy()
    d["Teff_bin"] = pd.cut(d["Teff"], bins=teff_bins)

    rows = []
    for i, (_, g) in enumerate(d.groupby("Teff_bin", observed=True)):
        x = g["logr"].to_numpy()
        rows.append(
            dict(
                teff_bin=0.5 * (teff_bins[i + 1] + teff_bins[i]),
                logr_med=np.median(x),
                logr_mad=mad(x),
                N=len(x),
            )
        )

    return pd.DataFrame(rows)


def plot_list(
    dkic,
    dkoi,
    dkic_list=None,
    label_list="Resampled KIC",
    ckic="gray",
    ckoi="salmon",
    ckic2="C1",
    teff_bins=np.arange(3500, 6750, 250),
    skip_bins=False,
    use_pct=True,
    p_lo=16.,
    p_hi=84.,
    figsize=(14, 9),
):
    def _binned_on_full_grid(df):
        b = binned_logr_stats(df, teff_bins)
        x_full = 0.5 * (teff_bins[1:] + teff_bins[:-1])
        y_full = np.full(len(x_full), np.nan, dtype=float)

        if len(b) == 0:
            return x_full, y_full

        x_b = np.asarray(b["teff_bin"], dtype=float)
        y_b = np.asarray(b["logr_med"], dtype=float)

        for xb, yb in zip(x_b, y_b):
            idx = np.argmin(np.abs(x_full - xb))
            if np.isclose(x_full[idx], xb):
                y_full[idx] = yb

        return x_full, y_full

    dkic_bin = binned_logr_stats(dkic, teff_bins)
    dkoi_bin = binned_logr_stats(dkoi, teff_bins)

    fig = plt.figure(figsize=figsize)
    plt.xlabel("$T_\\mathrm{eff}$ (K)")
    plt.ylabel("$\\log_{10} R$ (ppm)")
    plt.ylim(2.3, 5.2)
    plt.xlim(teff_bins[0], teff_bins[-1])

    if not skip_bins:
        plt.plot(dkic.Teff, dkic.logr, ",", color=ckic)
        plt.plot(
            dkic_bin.teff_bin,
            dkic_bin.logr_med,
            "o-",
            mfc="white",
            color=ckic,
            mew=2,
            lw=1,
            label=f"KIC ({len(dkic)})",
        )

        plt.plot(dkoi.Teff, dkoi.logr, ".", color=ckoi, markersize=2)
        plt.plot(
            dkoi_bin.teff_bin,
            dkoi_bin.logr_med,
            "s-",
            mfc="white",
            color=ckoi,
            mew=2,
            lw=1,
            label=f"KOI ({len(dkoi)})",
        )

    if dkic_list is not None:
        if len(dkic_list) == 0:
            raise ValueError("dkic_list is empty")

        x = 0.5 * (teff_bins[1:] + teff_bins[:-1])
        y_list = []

        for dtmp in dkic_list:
            _, y = _binned_on_full_grid(dtmp)
            y_list.append(np.asarray(y, dtype=float))

        y_arr = np.asarray(y_list, dtype=float)

        rs_med = np.nanmedian(y_arr, axis=0)
        if use_pct:
            rs_lo = np.nanpercentile(y_arr, p_lo, axis=0)
            rs_hi = np.nanpercentile(y_arr, p_hi, axis=0)
        else:
            rs_sigma = 1.4826 * mad(y_arr, axis=0, nan_policy="omit")
            rs_lo = rs_med - rs_sigma
            rs_hi = rs_med + rs_sigma

        plt.plot(
            x,
            rs_med,
            "o--",
            color=ckic2,
            mfc="white",
            mew=2,
            lw=1,
            label=label_list,
        )
        plt.fill_between(x, rs_lo, rs_hi, color=ckic2, alpha=0.2)

    handles, labels = plt.gca().get_legend_handles_labels()

    if dkic_list is not None:
        order = [0, 2, 1]
        plt.legend([handles[i] for i in order], [labels[i]
                   for i in order], loc="best")
    else:
        plt.legend(loc="best")
    return fig
