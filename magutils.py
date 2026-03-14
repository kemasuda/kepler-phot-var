
__all__ = ['sample_near_dkoi', 'sample_near_dkoiR']

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def sample_near_dkoi(
    dkic,
    dkoi,
    teff_bins,
    N=None,
    *,
    size_like="dkoi",   # "dkoi" or "dkic"
    m_col="kepmag",
    x_col="Teff",
    k=32,
    bw_m=0.1,
    base_weight_col=None,
    random_state=None,
    shuffle=True,
):
    """
    Sample rows from dkic so that, within each Teff bin, the kepmag
    distribution matches that of dkoi.

    Parameters
    ----------
    dkic : pandas.DataFrame
        Source catalog to sample from.
    dkoi : pandas.DataFrame
        Target catalog whose kepmag distribution is matched within each Teff bin.
    teff_bins : array-like
        Bin edges for Teff.
    N : int or None, optional
        Total number of draws. If None:
        - size_like="dkoi" -> total per-bin counts are exactly those of dkoi
        - size_like="dkic" -> total draws = len(dkic), allocated across bins
                              in proportion to dkoi bin counts
    size_like : {"dkoi", "dkic"}
        Controls total sample size only when N is None.
    m_col, x_col : str
        Columns for magnitude and Teff.
    k : int
        Number of nearest neighbors in kepmag space to consider within each bin.
    bw_m : float
        Bandwidth for kepmag distance scaling.
    base_weight_col : str or None
        Optional pre-existing weight column on dkic.
    random_state : int or None
        Random seed.
    shuffle : bool
        If True, shuffle the concatenated sampled rows before returning.

    Returns
    -------
    sampled : pandas.DataFrame
        Sampled rows from dkic.
    """
    src = dkic.dropna(subset=[m_col, x_col]).copy()
    tgt = dkoi.dropna(subset=[m_col, x_col]).copy()

    src["_teff_bin"] = pd.cut(src[x_col], bins=teff_bins, include_lowest=True)
    tgt["_teff_bin"] = pd.cut(tgt[x_col], bins=teff_bins, include_lowest=True)

    src = src.dropna(subset=["_teff_bin"]).copy()
    tgt = tgt.dropna(subset=["_teff_bin"]).copy()

    if len(tgt) == 0:
        raise ValueError("No target rows remain in dkoi after binning.")
    if len(src) == 0:
        raise ValueError("No source rows remain in dkic after binning.")

    rng = np.random.default_rng(random_state)
    bins = src["_teff_bin"].cat.categories

    tgt_counts = (
        tgt["_teff_bin"]
        .value_counts(sort=False)
        .reindex(bins, fill_value=0)
        .to_numpy(int)
    )

    # Decide only the number of draws in each Teff bin.
    # Everything else is identical to the previous version.
    if N is None and size_like == "dkoi":
        draw_counts = tgt_counts.copy()
    else:
        if N is None:
            if size_like == "dkic":
                Ntot = len(src)
            elif size_like == "dkoi":
                Ntot = len(tgt)
            else:
                raise ValueError("size_like must be 'dkoi' or 'dkic'")
        else:
            Ntot = int(N)

        p = tgt_counts.astype(float)
        if p.sum() <= 0:
            raise ValueError("All dkoi bin counts are zero.")
        p /= p.sum()
        draw_counts = rng.multinomial(Ntot, p)

    sampled_list = []

    for b, n_draw in zip(bins, draw_counts):
        if n_draw == 0:
            continue

        src_bin = src.loc[src["_teff_bin"] == b].reset_index(drop=True)
        tgt_bin = tgt.loc[src_bin["_teff_bin"].iloc[0]
                          if False else tgt["_teff_bin"] == b].reset_index(drop=True)

        if len(src_bin) == 0:
            raise ValueError(f"No dkic rows found in Teff bin {b}.")
        if len(tgt_bin) == 0:
            continue

        # Keep the previous behavior unchanged except for the number of draws.
        if N is None and size_like == "dkoi":
            tgt_sel = tgt_bin
        else:
            idx = rng.choice(len(tgt_bin), size=n_draw, replace=True)
            tgt_sel = tgt_bin.iloc[idx].reset_index(drop=True)

        z_src = (src_bin[m_col].to_numpy(float) / bw_m)[:, None]
        z_tgt = (tgt_sel[m_col].to_numpy(float) / bw_m)[:, None]

        if base_weight_col is None:
            base_w = np.ones(len(src_bin), dtype=float)
        else:
            base_w = src_bin[base_weight_col].to_numpy(float)

        nn = NearestNeighbors(n_neighbors=min(k, len(src_bin)))
        nn.fit(z_src)
        dist, ind = nn.kneighbors(z_tgt)

        picked = np.empty(len(tgt_sel), dtype=int)

        for j in range(len(tgt_sel)):
            ii = ind[j]
            dd = dist[j]

            w = np.exp(-0.5 * dd**2) * base_w[ii]
            if (not np.all(np.isfinite(w))) or (w.sum() <= 0):
                w = np.ones_like(w, dtype=float)
            w = w / w.sum()

            picked[j] = rng.choice(ii, p=w)

        sampled_bin = src_bin.iloc[picked].copy()
        sampled_list.append(sampled_bin)

    if len(sampled_list) == 0:
        return src.iloc[[]].copy().drop(columns="_teff_bin", errors="ignore")

    sampled = pd.concat(sampled_list, axis=0, ignore_index=True)
    sampled = sampled.drop(columns="_teff_bin", errors="ignore")

    if shuffle:
        sampled = sampled.sample(
            frac=1, random_state=random_state).reset_index(drop=True)

    return sampled


def sample_near_dkoiR(
    dkic,
    dkoi,
    teff_bins,
    N=None,
    *,
    size_like="dkoi",   # "dkoi" or "dkic"
    m_col="kepmag",
    rad_col="rad",
    x_col="Teff",
    k=32,
    bw_m=0.1,
    bw_rad=0.05,
    base_weight_col=None,
    random_state=None,
    shuffle=True,
):
    """
    Sample rows from dkic so that, within each Teff bin, the joint
    (kepmag, rad) distribution matches that of dkoi.

    Parameters
    ----------
    dkic : pandas.DataFrame
        Source catalog to sample from.
    dkoi : pandas.DataFrame
        Target catalog whose (kepmag, rad) distribution is matched
        within each Teff bin.
    teff_bins : array-like
        Bin edges for Teff.
    N : int or None, optional
        Total number of draws. If None:
        - size_like="dkoi" -> total per-bin counts are exactly those of dkoi
        - size_like="dkic" -> total draws = len(dkic), allocated across bins
                              in proportion to dkoi bin counts
    size_like : {"dkoi", "dkic"}
        Controls total sample size only when N is None.
    m_col : str
        Column for magnitude (e.g., kepmag).
    rad_col : str
        Column for stellar radius (or other second matching variable).
    x_col : str
        Column for Teff.
    k : int
        Number of nearest neighbors in 2D (kepmag, rad) space
        to consider within each bin.
    bw_m : float
        Bandwidth for kepmag distance scaling.
    bw_rad : float
        Bandwidth for rad distance scaling.
    base_weight_col : str or None
        Optional pre-existing weight column on dkic.
    random_state : int or None
        Random seed.
    shuffle : bool
        If True, shuffle the concatenated sampled rows before returning.

    Returns
    -------
    sampled : pandas.DataFrame
        Sampled rows from dkic.
    """
    need_cols = [m_col, rad_col, x_col]

    src = dkic.dropna(subset=need_cols).copy()
    tgt = dkoi.dropna(subset=need_cols).copy()

    src["_teff_bin"] = pd.cut(src[x_col], bins=teff_bins, include_lowest=True)
    tgt["_teff_bin"] = pd.cut(tgt[x_col], bins=teff_bins, include_lowest=True)

    src = src.dropna(subset=["_teff_bin"]).copy()
    tgt = tgt.dropna(subset=["_teff_bin"]).copy()

    if len(tgt) == 0:
        raise ValueError("No target rows remain in dkoi after binning.")
    if len(src) == 0:
        raise ValueError("No source rows remain in dkic after binning.")

    rng = np.random.default_rng(random_state)
    bins = src["_teff_bin"].cat.categories

    tgt_counts = (
        tgt["_teff_bin"]
        .value_counts(sort=False)
        .reindex(bins, fill_value=0)
        .to_numpy(int)
    )

    # Decide only the number of draws in each Teff bin.
    if N is None and size_like == "dkoi":
        draw_counts = tgt_counts.copy()
    else:
        if N is None:
            if size_like == "dkic":
                Ntot = len(src)
            elif size_like == "dkoi":
                Ntot = len(tgt)
            else:
                raise ValueError("size_like must be 'dkoi' or 'dkic'")
        else:
            Ntot = int(N)

        p = tgt_counts.astype(float)
        if p.sum() <= 0:
            raise ValueError("All dkoi bin counts are zero.")
        p /= p.sum()
        draw_counts = rng.multinomial(Ntot, p)

    sampled_list = []

    for b, n_draw in zip(bins, draw_counts):
        if n_draw == 0:
            continue

        src_bin = src.loc[src["_teff_bin"] == b].reset_index(drop=True)
        tgt_bin = tgt.loc[tgt["_teff_bin"] == b].reset_index(drop=True)

        if len(src_bin) == 0:
            raise ValueError(f"No dkic rows found in Teff bin {b}.")
        if len(tgt_bin) == 0:
            continue

        if N is None and size_like == "dkoi":
            tgt_sel = tgt_bin
        else:
            idx = rng.choice(len(tgt_bin), size=n_draw, replace=True)
            tgt_sel = tgt_bin.iloc[idx].reset_index(drop=True)

        z_src = np.column_stack([
            src_bin[m_col].to_numpy(float) / bw_m,
            src_bin[rad_col].to_numpy(float) / bw_rad,
        ])
        z_tgt = np.column_stack([
            tgt_sel[m_col].to_numpy(float) / bw_m,
            tgt_sel[rad_col].to_numpy(float) / bw_rad,
        ])

        if base_weight_col is None:
            base_w = np.ones(len(src_bin), dtype=float)
        else:
            base_w = src_bin[base_weight_col].to_numpy(float)

        nn = NearestNeighbors(n_neighbors=min(k, len(src_bin)))
        nn.fit(z_src)
        dist, ind = nn.kneighbors(z_tgt)

        picked = np.empty(len(tgt_sel), dtype=int)

        for j in range(len(tgt_sel)):
            ii = ind[j]
            dd = dist[j]

            # dd is Euclidean distance in the scaled 2D space:
            # dd^2 = (Δm/bw_m)^2 + (Δrad/bw_rad)^2
            w = np.exp(-0.5 * dd**2) * base_w[ii]

            if (not np.all(np.isfinite(w))) or (w.sum() <= 0):
                w = np.ones_like(w, dtype=float)
            w = w / w.sum()

            picked[j] = rng.choice(ii, p=w)

        sampled_bin = src_bin.iloc[picked].copy()
        sampled_list.append(sampled_bin)

    if len(sampled_list) == 0:
        return src.iloc[[]].copy().drop(columns="_teff_bin", errors="ignore")

    sampled = pd.concat(sampled_list, axis=0, ignore_index=True)
    sampled = sampled.drop(columns="_teff_bin", errors="ignore")

    if shuffle:
        sampled = sampled.sample(
            frac=1, random_state=random_state
        ).reset_index(drop=True)

    return sampled
