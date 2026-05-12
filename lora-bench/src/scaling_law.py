"""Fit merging scaling law from arXiv:2509.24244.

   L(k) = L_inf + A / (k + b)

Three-point fit suffices; we use scipy.optimize.curve_fit on >=3 (k, L) pairs.

Optional cross-size variant fits per-size N then aggregates:
   L_inf(N) = L_star + B * N^(-beta)
   A(N)     = A0 * N^(-gamma)

CLI lives in `src.cli.fit_law`.
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.optimize import curve_fit
except ImportError as e:
    raise SystemExit("scipy required: uv pip install scipy") from e


def _law(k, L_inf, A, b):
    return L_inf + A / (k + b)


def fit_k_law(ks: list[int], losses: list[float]) -> dict:
    if len(ks) < 3:
        raise ValueError("need >=3 (k, L) points")
    x = np.asarray(ks, dtype=float)
    y = np.asarray(losses, dtype=float)
    p0 = [y.min(), (y.max() - y.min()) * (x[0] + 1.0), 1.0]
    popt, _ = curve_fit(_law, x, y, p0=p0, maxfev=10_000, bounds=([0, 0, 0], [10, 50, 50]))
    y_hat = _law(x, *popt)
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return {"L_inf": float(popt[0]), "A": float(popt[1]), "b": float(popt[2]), "r2": r2}


def _size_law(N, L_star, B, beta):
    return L_star + B * np.power(N, -beta)


def _A_law(N, A0, gamma):
    return A0 * np.power(N, -gamma)


def fit_size_law(per_size: dict[str, dict]) -> dict:
    """per_size: {size_str: {'L_inf':..., 'A':..., 'b':...}}. size_str = N in params."""
    Ns = np.asarray([float(s) for s in per_size], dtype=float)
    Linf = np.asarray([per_size[s]["L_inf"] for s in per_size], dtype=float)
    As = np.asarray([per_size[s]["A"] for s in per_size], dtype=float)
    pL, _ = curve_fit(_size_law, Ns, Linf, p0=[Linf.min(), 1.0, 0.3], maxfev=10_000)
    pA, _ = curve_fit(_A_law, Ns, As, p0=[As.max(), 0.01], maxfev=10_000)
    return {
        "L_star": float(pL[0]), "B": float(pL[1]), "beta": float(pL[2]),
        "A0": float(pA[0]), "gamma": float(pA[1]),
    }


