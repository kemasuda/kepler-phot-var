
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
__all__ = ['PdetSampler', 'PdetSamplerR', 'CDPPInterpolatorByKIC',
           'p_det', 'plot_pdet_contour']

from dataclasses import dataclass
import pandas as pd
import re
import jax.numpy as jnp
from jax.scipy.special import gamma, gammainc
import numpy as np
from sklearn.neighbors import NearestNeighbors


class PdetSampler:
    """
    Sample pdet by drawing from kNN of (Teff, logr, kepmag), with anisotropic
    Gaussian weights:
        w ~ exp(-0.5*(dTeff/sigma_T)**2
                -0.5*(dlogr/sigma_logr)**2
                -0.5*(dkepmag/sigma_kepmag)**2)

    Inputs:
        - Teff_q, logr_q, kepmag_q: either scalars, or 1D arrays with the same length N.
    Outputs:
        - sample(...): (N,)
        - sample_n(..., n_draws): (n_draws, N)
    """

    def __init__(self, dkic, k=100, *, seed=0, sigma_T=100., sigma_logr=0.1, sigma_kepmag=0.2):
        self.k = int(k)
        self.rng = np.random.default_rng(seed)

        T = np.asarray(dkic["Teff"], float)
        L = np.asarray(dkic["logr"], float)
        M = np.asarray(dkic["kepmag"], float)
        p = np.asarray(dkic["pdet"], float)

        ok = np.isfinite(T) & np.isfinite(L) & np.isfinite(M) & np.isfinite(p)
        self.T = T[ok]
        self.L = L[ok]
        self.M = M[ok]
        self.p = p[ok]
        self.sigma_T = sigma_T
        self.sigma_logr = sigma_logr
        self.sigma_kepmag = sigma_kepmag

        X = np.column_stack([self.T, self.L, self.M])
        self.nn = NearestNeighbors(n_neighbors=self.k, algorithm="auto")
        self.nn.fit(X)

    @staticmethod
    def _as_1d_triplet(Teff_q, logr_q, kepmag_q):
        Tq = np.asarray(Teff_q, float)
        Lq = np.asarray(logr_q, float)
        Mq = np.asarray(kepmag_q, float)

        if Tq.ndim == 0:
            Tq = Tq[None]
        if Lq.ndim == 0:
            Lq = Lq[None]
        if Mq.ndim == 0:
            Mq = Mq[None]

        if Tq.ndim != 1 or Lq.ndim != 1 or Mq.ndim != 1:
            raise ValueError(
                "Teff_q, logr_q, and kepmag_q must be scalars or 1D arrays.")
        if not (len(Tq) == len(Lq) == len(Mq)):
            raise ValueError(
                f"Teff_q, logr_q, and kepmag_q must have the same length. "
                f"Got {len(Tq)}, {len(Lq)}, and {len(Mq)}."
            )
        return Tq, Lq, Mq

    def sample(self, logr_q, kepmag_q, Teff_q):
        Tq, Lq, Mq = self._as_1d_triplet(Teff_q, logr_q, kepmag_q)
        Xq = np.column_stack([Tq, Lq, Mq])  # (N,3)

        _, idx = self.nn.kneighbors(Xq, return_distance=True)

        sigma_T = float(self.sigma_T)
        sigma_logr = float(self.sigma_logr)
        sigma_kepmag = float(self.sigma_kepmag)

        Tn = self.T[idx]  # (N,k)
        Ln = self.L[idx]  # (N,k)
        Mn = self.M[idx]  # (N,k)

        dT = (Tn - Tq[:, None]) / sigma_T
        dL = (Ln - Lq[:, None]) / sigma_logr
        dM = (Mn - Mq[:, None]) / sigma_kepmag

        w = np.exp(-0.5 * (dT * dT + dL * dL + dM * dM))
        wsum = w.sum(axis=1, keepdims=True)
        w = w / np.maximum(wsum, 1e-300)

        u = self.rng.random(size=(idx.shape[0], 1))
        cdf = np.cumsum(w, axis=1)
        pick = (cdf < u).sum(axis=1)

        chosen = idx[np.arange(idx.shape[0]), pick]
        return self.p[chosen]  # (N,)

    def sample_n(self, logr_q, kepmag_q, Teff_q, n_draws):
        Tq, Lq, Mq = self._as_1d_triplet(Teff_q, logr_q, kepmag_q)
        Xq = np.column_stack([Tq, Lq, Mq])  # (N,3)

        _, idx = self.nn.kneighbors(Xq, return_distance=True)

        sigma_T = float(self.sigma_T)
        sigma_logr = float(self.sigma_logr)
        sigma_kepmag = float(self.sigma_kepmag)

        Tn = self.T[idx]  # (N,k)
        Ln = self.L[idx]  # (N,k)
        Mn = self.M[idx]  # (N,k)

        dT = (Tn - Tq[:, None]) / sigma_T
        dL = (Ln - Lq[:, None]) / sigma_logr
        dM = (Mn - Mq[:, None]) / sigma_kepmag

        w = np.exp(-0.5 * (dT * dT + dL * dL + dM * dM))
        wsum = w.sum(axis=1, keepdims=True)
        w = w / np.maximum(wsum, 1e-300)
        cdf = np.cumsum(w, axis=1)  # (N,k)

        n_draws = int(n_draws)
        u = self.rng.random(size=(n_draws, idx.shape[0], 1))  # (D,N,1)
        pick = (cdf[None, :, :] < u).sum(axis=2)              # (D,N)

        chosen = idx[np.arange(idx.shape[0])[None, :], pick]  # (D,N)
        return self.p[chosen]                                  # (D,N)


class PdetSamplerR:
    """
    Sample pdet by drawing from kNN of (Teff, logr, kepmag, rad), with
    anisotropic Gaussian weights:
        w ~ exp(-0.5*(dTeff/sigma_T)**2
                -0.5*(dlogr/sigma_logr)**2
                -0.5*(dkepmag/sigma_kepmag)**2
                -0.5*(drad/sigma_rad)**2)

    Inputs:
        - Teff_q, logr_q, kepmag_q, rad_q:
          either scalars, or 1D arrays with the same length N.
    Outputs:
        - sample(...): (N,)
        - sample_n(..., n_draws): (n_draws, N)
    """

    def __init__(
        self,
        dkic,
        k=100,
        *,
        seed=0,
        sigma_T=100.0,
        sigma_logr=0.1,
        sigma_kepmag=0.2,
        sigma_rad=0.1,
        rad_col="rad",
    ):
        self.rng = np.random.default_rng(seed)

        T = np.asarray(dkic["Teff"], float)
        L = np.asarray(dkic["logr"], float)
        M = np.asarray(dkic["kepmag"], float)
        R = np.asarray(dkic[rad_col], float)
        p = np.asarray(dkic["pdet"], float)

        ok = np.isfinite(T) & np.isfinite(L) & np.isfinite(
            M) & np.isfinite(R) & np.isfinite(p)
        self.T = T[ok]
        self.L = L[ok]
        self.M = M[ok]
        self.R = R[ok]
        self.p = p[ok]

        if len(self.p) == 0:
            raise ValueError(
                "No finite rows remain after filtering Teff/logr/kepmag/rad/pdet.")

        self.k = min(int(k), len(self.p))
        self.sigma_T = float(sigma_T)
        self.sigma_logr = float(sigma_logr)
        self.sigma_kepmag = float(sigma_kepmag)
        self.sigma_rad = float(sigma_rad)

        X = np.column_stack([self.T, self.L, self.M, self.R])
        self.nn = NearestNeighbors(n_neighbors=self.k, algorithm="auto")
        self.nn.fit(X)

    @staticmethod
    def _as_1d_quadruplet(Teff_q, logr_q, kepmag_q, rad_q):
        Tq = np.asarray(Teff_q, float)
        Lq = np.asarray(logr_q, float)
        Mq = np.asarray(kepmag_q, float)
        Rq = np.asarray(rad_q, float)

        if Tq.ndim == 0:
            Tq = Tq[None]
        if Lq.ndim == 0:
            Lq = Lq[None]
        if Mq.ndim == 0:
            Mq = Mq[None]
        if Rq.ndim == 0:
            Rq = Rq[None]

        if Tq.ndim != 1 or Lq.ndim != 1 or Mq.ndim != 1 or Rq.ndim != 1:
            raise ValueError(
                "Teff_q, logr_q, kepmag_q, and rad_q must be scalars or 1D arrays."
            )
        if not (len(Tq) == len(Lq) == len(Mq) == len(Rq)):
            raise ValueError(
                "Teff_q, logr_q, kepmag_q, and rad_q must have the same length. "
                f"Got {len(Tq)}, {len(Lq)}, {len(Mq)}, and {len(Rq)}."
            )

        return Tq, Lq, Mq, Rq

    def _compute_weights_and_idx(self, logr_q, kepmag_q, Teff_q, rad_q):
        Tq, Lq, Mq, Rq = self._as_1d_quadruplet(
            Teff_q, logr_q, kepmag_q, rad_q)
        Xq = np.column_stack([Tq, Lq, Mq, Rq])  # (N,4)

        _, idx = self.nn.kneighbors(Xq, return_distance=True)

        Tn = self.T[idx]  # (N,k)
        Ln = self.L[idx]  # (N,k)
        Mn = self.M[idx]  # (N,k)
        Rn = self.R[idx]  # (N,k)

        dT = (Tn - Tq[:, None]) / self.sigma_T
        dL = (Ln - Lq[:, None]) / self.sigma_logr
        dM = (Mn - Mq[:, None]) / self.sigma_kepmag
        dR = (Rn - Rq[:, None]) / self.sigma_rad

        w = np.exp(-0.5 * (dT * dT + dL * dL + dM * dM + dR * dR))
        wsum = w.sum(axis=1, keepdims=True)
        w = w / np.maximum(wsum, 1e-300)

        return idx, w

    def sample(self, logr_q, kepmag_q, Teff_q, rad_q):
        idx, w = self._compute_weights_and_idx(logr_q, kepmag_q, Teff_q, rad_q)

        u = self.rng.random(size=(idx.shape[0], 1))   # (N,1)
        cdf = np.cumsum(w, axis=1)                    # (N,k)
        # avoid pick == k from FP roundoff
        cdf[:, -1] = 1.0
        pick = (cdf < u).sum(axis=1)                  # (N,)
        pick = np.minimum(pick, idx.shape[1] - 1)

        chosen = idx[np.arange(idx.shape[0]), pick]
        return self.p[chosen]                         # (N,)

    def sample_n(self, logr_q, kepmag_q, Teff_q, rad_q, n_draws):
        idx, w = self._compute_weights_and_idx(logr_q, kepmag_q, Teff_q, rad_q)

        cdf = np.cumsum(w, axis=1)                    # (N,k)
        # avoid pick == k from FP roundoff
        cdf[:, -1] = 1.0

        n_draws = int(n_draws)
        u = self.rng.random(size=(n_draws, idx.shape[0], 1))  # (D,N,1)
        pick = (cdf[None, :, :] < u).sum(axis=2)              # (D,N)
        pick = np.minimum(pick, idx.shape[1] - 1)

        chosen = idx[np.arange(idx.shape[0])[None, :], pick]  # (D,N)
        return self.p[chosen]                                  # (D,N)


def _parse_ts_cols(df: pd.DataFrame):
    cols, ts = [], []
    for c in df.columns:
        m = re.match(r"rrmscdpp(\d+)p(\d+)", c)
        if m:
            t = float(m.group(1)) + float(m.group(2)) / 10.0
            cols.append(c)
            ts.append(t)
    ts = np.asarray(ts, float)
    order = np.argsort(ts)
    return ts[order], [cols[i] for i in order]


@dataclass
class CDPPInterpolatorByKIC:
    """
    CDPP(dur) for a single KIC ID.
    - lookup by KIC once
    - log-log interpolation over dur
    - output in linear CDPP
    """
    ts: np.ndarray          # (Nts,)
    log_ts: np.ndarray      # (Nts,)
    logY: np.ndarray        # (Nstar, Nts)
    kic_index: pd.Index

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        kic_col: str = "KIC",
        dtype=np.float32,
        y_floor: float = 1e-30,
    ):
        ts, cols = _parse_ts_cols(df)

        Y = df[cols].to_numpy(dtype=float)
        Y = np.where(Y > y_floor, Y, y_floor)

        logY = np.log(Y).astype(dtype)
        log_ts = np.log(ts)

        kic_index = pd.Index(df[kic_col].to_numpy(), name=kic_col)

        return cls(
            ts=ts,
            log_ts=log_ts,
            logY=logY,
            kic_index=kic_index,
        )

    def __call__(self, kic_id: int, dur):
        """
        Parameters
        ----------
        kic_id : int
            single KIC ID
        dur : float or array-like
            transit duration(s)

        Returns
        -------
        cdpp : float or ndarray
            CDPP at given duration(s)
        """
        # lookup
        idx = self.kic_index.get_loc(kic_id)

        logYrow = self.logY[idx]          # (Nts,)

        T = np.asarray(dur, float)
        scalar = (T.ndim == 0)

        # clamp + log
        T = np.clip(T, self.ts[0], self.ts[-1])
        logT = np.log(T)

        # interpolation indices
        i1 = np.searchsorted(self.log_ts, logT, side="right")
        i1 = np.clip(i1, 1, len(self.log_ts) - 1)
        i0 = i1 - 1

        lt0 = self.log_ts[i0]
        lt1 = self.log_ts[i1]
        w = (logT - lt0) / (lt1 - lt0)

        ly0 = logYrow[i0]
        ly1 = logYrow[i1]

        log_cdpp = (1.0 - w) * ly0 + w * ly1
        cdpp = np.exp(log_cdpp)

        return cdpp.item() if scalar else cdpp


def p_det(s, k=17.56, l=1., theta=0.49, clamp=True):
    s = jnp.asarray(s, dtype=float)
    x = (s - l) / theta
    if clamp:
        x = jnp.maximum(x, 0.0)

    return gammainc(k, x)


def plot_pdet_contour(
    dkic,
    xcol="Teff",
    x_edges=None,
    logr_edges=np.arange(2.4, 5.2, 0.05),
    min_cnt=3,
    levels=6,
    xlabel=None,
):
    # --- defaults ---
    if x_edges is None:
        if xcol == "Teff":
            x_edges = np.arange(3500, 6501, 100)
        elif xcol == "kepmag":
            x_edges = np.arange(8, 17.1, 0.2)
        else:
            raise ValueError(
                "Please provide x_edges for xcol != 'Teff' or 'kepmag'.")

    if xlabel is None:
        if xcol == "Teff":
            xlabel = r"$T_\mathrm{eff}$ (K)"
        elif xcol == "kepmag":
            xlabel = "Kepler magnitude"
        else:
            xlabel = xcol

    # --- inputs ---
    X = dkic[xcol].to_numpy()
    L = dkic["logr"].to_numpy()
    P = dkic["pdet"].to_numpy()

    ok = np.isfinite(X) & np.isfinite(L) & np.isfinite(P)
    X, L, P = X[ok], L[ok], P[ok]

    # --- binning ---
    nx = len(x_edges) - 1
    nl = len(logr_edges) - 1

    xi = np.digitize(X, x_edges) - 1
    li = np.digitize(L, logr_edges) - 1
    ok = (xi >= 0) & (xi < nx) & (li >= 0) & (li < nl)

    xi, li, P = xi[ok], li[ok], P[ok]

    sumP = np.zeros((nx, nl))
    cnt = np.zeros((nx, nl))
    np.add.at(sumP, (xi, li), P)
    np.add.at(cnt,  (xi, li), 1)

    Z = np.full((nx, nl), np.nan)
    m = cnt > 0
    Z[m] = sumP[m] / cnt[m]

    # --- optional smoothing ---
    Zs = median_filter(Z, size=3)

    # --- mask sparse bins ---
    Zplot = Zs.copy()
    Zplot[cnt < min_cnt] = np.nan

    # --- bin centers for contour ---
    Xc = 0.5 * (x_edges[:-1] + x_edges[1:])
    Lc = 0.5 * (logr_edges[:-1] + logr_edges[1:])
    XX, LL = np.meshgrid(Xc, Lc, indexing="ij")

    # --- plot ---
    fig, ax = plt.subplots(figsize=(10, 6))

    pcm = ax.pcolormesh(
        x_edges,
        logr_edges,
        Zplot.T,
        shading="flat",
    )
    cbar = plt.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label(r"$p(D|TR)$")

    cs = ax.contour(
        XX, LL, Zplot,
        levels=levels,
        colors="k",
        linewidths=0.7,
    )
    ax.clabel(cs, fontsize=12)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\log_{10} r$ (ppm)")
    ax.grid(False)

    # 必要なら
    # if xcol == "Teff":
    #     ax.invert_xaxis()

    plt.tight_layout()
    plt.show()

    return fig
