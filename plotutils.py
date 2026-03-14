from __future__ import annotations

__all__ = ['plot', 'plot_list']

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
    weight_col="w",
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
    use_weight=False,
    weight_col="w",
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


'''
def plot(
    dkic,
    dkoi,
    weight_samples=None,
    label_list='weighted KOI',
    ckic='gray',
    ckoi='salmon',
    ckic2='C1',
    teff_bins=np.arange(3500, 6750, 250),
    teff_grid=np.linspace(3500, 6500, 100),
    teff_sigma=100,
    skip_bins=False,
    *,
    use_weight=False,
    weight_col="w",
    rs_method="smooth",   # "smooth" or "binned"
):
    """
    Plot KIC vs KOI logr-Teff relation.

    Parameters
    ----------
    dkic : DataFrame
        KIC (= R sample). Always plotted unweighted.
    dkoi : DataFrame
        KOI (= DTR sample). If use_weight=True and weight_col exists,
        the single KOI curve is plotted with those weights.
    weight_samples : ndarray, optional
        Array of shape (len(dkoi), n_draws). Each column is one draw of
        KOI weights. If provided, a dashed summary curve and scatter band
        across draws are plotted.
    """
    # KIC always unweighted
    use_weight_kic = False

    # Single KOI curve can be weighted if requested
    use_weight_koi = use_weight and (weight_col in dkoi.columns)

    dkic_bin = binned_logr_stats(
        dkic, teff_bins,
        use_weight=use_weight_kic,
        weight_col=weight_col,
    )
    dkoi_bin = binned_logr_stats(
        dkoi, teff_bins,
        use_weight=use_weight_koi,
        weight_col=weight_col,
    )

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
            label=(
                f'KOI ({len(dkoi)}), '
                + ('weighted ' if use_weight_koi else '')
                + 'binned median'
            )
        )

    logr_med_kic = smooth_logr_vs_teff(
        dkic, teff_grid, teff_sigma,
        use_weight=False,
        weight_col=weight_col,
    )
    plt.plot(
        teff_grid, logr_med_kic, '-', color=ckic, lw=1,
        label=f'KIC, windowed median ($\\sigma={teff_sigma:.0f}\\,\\mathrm{{K}}$)'
    )

    logr_med_koi = smooth_logr_vs_teff(
        dkoi, teff_grid, teff_sigma,
        use_weight=use_weight_koi,
        weight_col=weight_col,
    )
    plt.plot(
        teff_grid, logr_med_koi, '-', color=ckoi, lw=1,
        label=(
            'KOI, '
            + ('weighted ' if use_weight_koi else '')
            + f'windowed median ($\\sigma={teff_sigma:.0f}\\,\\mathrm{{K}}$)'
        )
    )

    if weight_samples is not None:
        weight_samples = np.asarray(weight_samples, dtype=float)

        if weight_samples.ndim != 2:
            raise ValueError("weight_samples must be a 2D array")
        if weight_samples.shape[0] != len(dkoi):
            raise ValueError(
                "weight_samples must have shape (len(dkoi), n_draws)"
            )
        if rs_method not in ("smooth", "binned"):
            raise ValueError("rs_method must be 'smooth' or 'binned'")

        y_list = []

        if rs_method == "smooth":
            x = teff_grid
            for s in range(weight_samples.shape[1]):
                dtmp = dkoi.copy()
                dtmp[weight_col] = weight_samples[:, s]
                y_list.append(
                    smooth_logr_vs_teff(
                        dtmp, teff_grid, teff_sigma,
                        use_weight=True,
                        weight_col=weight_col,
                    )
                )
        else:
            x = None
            for s in range(weight_samples.shape[1]):
                dtmp = dkoi.copy()
                dtmp[weight_col] = weight_samples[:, s]
                b = binned_logr_stats(
                    dtmp, teff_bins,
                    use_weight=True,
                    weight_col=weight_col,
                )
                if x is None:
                    x = np.asarray(b.teff_bin)
                y_list.append(np.asarray(b.logr_med))

        y_arr = np.asarray(y_list)
        rs_med = np.median(y_arr, axis=0)
        rs_sigma = 1.4826 * mad(y_arr, axis=0)

        if rs_method == "smooth":
            plt.plot(
                x, rs_med, '-', color=ckic2,
                ls='dashed', label=label_list
            )
        else:
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
'''


def plot(
    dkic,
    dkoi,
    weight_samples=None,
    label_list="weighted draws",
    ckic="gray",
    ckoi="salmon",
    ckic2="C1",
    teff_bins=np.arange(3500, 6750, 250),
    teff_grid=np.linspace(3500, 6500, 100),
    teff_sigma=100,
    skip_bins=False,
    *,
    use_weight=False,
    weight_col="w",
    weight_side="koi",   # "koi" or "kic"
    rs_method="smooth",  # "smooth" or "binned"
):
    if weight_side not in ("koi", "kic"):
        raise ValueError("weight_side must be 'koi' or 'kic'")

    use_weight_kic = (
        use_weight
        and (weight_side == "kic")
        and (weight_col in dkic.columns)
    )
    use_weight_koi = (
        use_weight
        and (weight_side == "koi")
        and (weight_col in dkoi.columns)
    )

    dkic_bin = binned_logr_stats(
        dkic,
        teff_bins,
        use_weight=use_weight_kic,
        weight_col=weight_col,
    )
    dkoi_bin = binned_logr_stats(
        dkoi,
        teff_bins,
        use_weight=use_weight_koi,
        weight_col=weight_col,
    )

    fig = plt.figure(figsize=(14, 9))
    plt.xlabel("$T_\\mathrm{eff}$ (K)")
    plt.ylabel("$\\log_{10} R$ (ppm)")
    plt.ylim(2.3, 5.2)
    plt.xlim(teff_grid[0], teff_grid[-1])

    if not skip_bins:
        plt.plot(dkic.Teff, dkic.logr, ",", color=ckic)
        plt.plot(
            dkic_bin.teff_bin,
            dkic_bin.logr_med,
            "o",
            mfc="white",
            color=ckic,
            mew=2,
            lw=1,
            label=(
                f"KIC ({len(dkic)}), "
                + ("weighted " if use_weight_kic else "")
                + "binned median"
            ),
        )

        plt.plot(dkoi.Teff, dkoi.logr, ".", color=ckoi, markersize=2)
        plt.plot(
            dkoi_bin.teff_bin,
            dkoi_bin.logr_med,
            "o",
            mfc="white",
            color=ckoi,
            mew=2,
            lw=1,
            label=(
                f"KOI ({len(dkoi)}), "
                + ("weighted " if use_weight_koi else "")
                + "binned median"
            ),
        )

    logr_med_kic = smooth_logr_vs_teff(
        dkic,
        teff_grid,
        teff_sigma,
        use_weight=use_weight_kic,
        weight_col=weight_col,
    )
    plt.plot(
        teff_grid,
        logr_med_kic,
        "-",
        color=ckic,
        lw=1,
        label=(
            "KIC, "
            + ("weighted " if use_weight_kic else "")
            + f"windowed median ($\\sigma={teff_sigma:.0f}\\,\\mathrm{{K}}$)"
        ),
    )

    logr_med_koi = smooth_logr_vs_teff(
        dkoi,
        teff_grid,
        teff_sigma,
        use_weight=use_weight_koi,
        weight_col=weight_col,
    )
    plt.plot(
        teff_grid,
        logr_med_koi,
        "-",
        color=ckoi,
        lw=1,
        label=(
            "KOI, "
            + ("weighted " if use_weight_koi else "")
            + f"windowed median ($\\sigma={teff_sigma:.0f}\\,\\mathrm{{K}}$)"
        ),
    )

    if weight_samples is not None:
        weight_samples = np.asarray(weight_samples, dtype=float)

        base_df = dkoi if weight_side == "koi" else dkic

        if weight_samples.ndim != 2:
            raise ValueError("weight_samples must be a 2D array")
        if weight_samples.shape[0] != len(base_df):
            raise ValueError(
                f"weight_samples must have shape (len({weight_side}), n_draws)"
            )
        if rs_method not in ("smooth", "binned"):
            raise ValueError("rs_method must be 'smooth' or 'binned'")

        y_list = []

        if rs_method == "smooth":
            x = teff_grid
            for s in range(weight_samples.shape[1]):
                dtmp = base_df.copy()
                dtmp[weight_col] = weight_samples[:, s]
                y = smooth_logr_vs_teff(
                    dtmp,
                    teff_grid,
                    teff_sigma,
                    use_weight=True,
                    weight_col=weight_col,
                )
                y_list.append(y)
        else:
            x = None
            for s in range(weight_samples.shape[1]):
                dtmp = base_df.copy()
                dtmp[weight_col] = weight_samples[:, s]
                b = binned_logr_stats(
                    dtmp,
                    teff_bins,
                    use_weight=True,
                    weight_col=weight_col,
                )
                if x is None:
                    x = np.asarray(b.teff_bin)
                y_list.append(np.asarray(b.logr_med))

        y_arr = np.asarray(y_list)
        rs_med = np.median(y_arr, axis=0)
        rs_sigma = 1.4826 * mad(y_arr, axis=0)

        if rs_method == "smooth":
            plt.plot(
                x,
                rs_med,
                "-",
                color=ckic2,
                ls="dashed",
                label=label_list,
            )
        else:
            plt.plot(
                x,
                rs_med,
                "o-",
                color=ckic2,
                ls="dashed",
                mfc="white",
                label=label_list,
                mew=2,
                lw=1,
            )

        plt.fill_between(
            x,
            rs_med - rs_sigma,
            rs_med + rs_sigma,
            color=ckic2,
            alpha=0.2,
        )

    plt.legend(loc="best")
    return fig


def plot_list(
    dkic,
    dkoi,
    dkic_list=None,
    label_list="resampled KIC draws",
    ckic="gray",
    ckoi="salmon",
    ckic2="C1",
    teff_bins=np.arange(3500, 6750, 250),
    teff_grid=np.linspace(3500, 6500, 100),
    teff_sigma=100,
    skip_bins=False,
    use_pct=False,
    p_lo=5.,
    p_hi=95.,
    *,
    rs_method="smooth",  # "smooth" or "binned"
):
    """
    Plot KIC vs KOI logr-Teff relations, with optional uncertainty band
    from a list of resampled KIC catalogs.

    Parameters
    ----------
    dkic : DataFrame
        Original KIC catalog. Must contain columns 'Teff' and 'logr'.
    dkoi : DataFrame
        KOI catalog. Must contain columns 'Teff' and 'logr'.
    dkic_list : list of DataFrame or None
        List of resampled KIC catalogs. Each element must contain
        columns 'Teff' and 'logr'. If provided, the median curve and
        5--95 percentile band across the list are shown.
    label_list : str
        Label for the resampled KIC summary curve.
    ckic, ckoi, ckic2 : str
        Colors for KIC, KOI, and resampled KIC summary.
    teff_bins : array-like
        Bin edges for binned statistics.
    teff_grid : array-like
        Grid for smoothed statistics.
    teff_sigma : float
        Gaussian width in Teff for smooth_logr_vs_teff.
    skip_bins : bool
        If True, skip scatter and binned-median plotting for dkic/dkoi.
    rs_method : {"smooth", "binned"}
        If "smooth", summarize each resampled catalog with smooth_logr_vs_teff.
        If "binned", summarize each resampled catalog with binned_logr_stats.
    """
    if rs_method not in ("smooth", "binned"):
        raise ValueError("rs_method must be 'smooth' or 'binned'")

    def _binned_on_full_grid(df):
        """
        Return binned logr median on the full bin-center grid, inserting NaN
        for bins with no data.
        """
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

    fig = plt.figure(figsize=(14, 9))
    plt.xlabel("$T_\\mathrm{eff}$ (K)")
    plt.ylabel("$\\log_{10} R$ (ppm)")
    plt.ylim(2.3, 5.2)
    plt.xlim(teff_grid[0], teff_grid[-1])

    if not skip_bins:
        plt.plot(dkic.Teff, dkic.logr, ",", color=ckic)
        plt.plot(
            dkic_bin.teff_bin,
            dkic_bin.logr_med,
            "o",
            mfc="white",
            color=ckic,
            mew=2,
            lw=1,
            label=f"KIC ({len(dkic)}), binned median",
        )

        plt.plot(dkoi.Teff, dkoi.logr, ".", color=ckoi, markersize=2)
        plt.plot(
            dkoi_bin.teff_bin,
            dkoi_bin.logr_med,
            "o",
            mfc="white",
            color=ckoi,
            mew=2,
            lw=1,
            label=f"KOI ({len(dkoi)}), binned median",
        )

    logr_med_kic = smooth_logr_vs_teff(
        dkic,
        teff_grid,
        teff_sigma,
    )
    plt.plot(
        teff_grid,
        logr_med_kic,
        "-",
        color=ckic,
        lw=1,
        label=f"KIC, windowed median ($\\sigma={teff_sigma:.0f}\\,\\mathrm{{K}}$)",
    )

    logr_med_koi = smooth_logr_vs_teff(
        dkoi,
        teff_grid,
        teff_sigma,
    )
    plt.plot(
        teff_grid,
        logr_med_koi,
        "-",
        color=ckoi,
        lw=1,
        label=f"KOI, windowed median ($\\sigma={teff_sigma:.0f}\\,\\mathrm{{K}}$)",
    )

    if dkic_list is not None:
        if len(dkic_list) == 0:
            raise ValueError("dkic_list is empty")

        y_list = []

        if rs_method == "smooth":
            x = np.asarray(teff_grid, dtype=float)
            for dtmp in dkic_list:
                y = smooth_logr_vs_teff(
                    dtmp,
                    teff_grid,
                    teff_sigma,
                )
                y_list.append(np.asarray(y, dtype=float))
        else:
            x = 0.5 * (teff_bins[1:] + teff_bins[:-1])
            for dtmp in dkic_list:
                _, y = _binned_on_full_grid(dtmp)
                y_list.append(np.asarray(y, dtype=float))

        y_arr = np.asarray(y_list, dtype=float)

        if use_pct:
            rs_med = np.nanmedian(y_arr, axis=0)
            rs_lo = np.nanpercentile(y_arr, p_lo, axis=0)
            rs_hi = np.nanpercentile(y_arr, p_hi, axis=0)
        else:
            rs_med = np.median(y_arr, axis=0)
            rs_sigma = 1.4826 * mad(y_arr, axis=0)
            rs_lo = rs_med - rs_sigma
            rs_hi = rs_med + rs_sigma

        if rs_method == "smooth":
            plt.plot(
                x,
                rs_med,
                "-",
                color=ckic2,
                ls="dashed",
                label=label_list,
            )
        else:
            plt.plot(
                x,
                rs_med,
                "o-",
                color=ckic2,
                ls="dashed",
                mfc="white",
                label=label_list,
                mew=2,
                lw=1,
            )

        plt.fill_between(
            x,
            rs_lo,
            rs_hi,
            color=ckic2,
            alpha=0.2,
        )

    plt.legend(loc="best")
    return fig
