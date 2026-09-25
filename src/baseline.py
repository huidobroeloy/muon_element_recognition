"""
baseline.py
===========
Re-implementation of the peak-fitting analysis used in the ISIS / Oxford
Roman-coin muXES studies, so the neural network can be compared with it on
identical spectra.

Published procedure
-------------------
* Hampshire et al., Heritage 2 (2019): Ag 4f-3d (304.5, 308.2 keV) and Cu 3d-2p
  (330.6, 336.4 keV) peaks in the 300-340 keV window fitted with Gaussians
  "confining the width between them"; the sum of the fitted integrals per
  element, weighted by a calibration factor from a known standard, gives the
  (atomic) composition. Binary Ag-Cu alloy assumed; Pb and Al peaks ignored.
* Green et al., J. Archaeol. Sci. 134 (2021): Au peaks around 400-410 keV,
  Ag at 305/310 keV, Cu near 330 keV; calibration with MAC 1/2 standards
  and a 70:30 Ag/Au mixture; ~1 wt% detection limit.

This module does exactly that: Poisson-weighted least squares with a common
width and common energy shift, linear background per window, relative
sensitivities fitted on standards, areas -> atomic fractions -> wt%.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from config import LE_BIN, LE_EMIN, W_OF
from physics import transition_energy_kev
from simulate import LE_GRID

BASELINE_ELEMENTS = ("Au", "Ag", "Cu")
PEAKS = {
    "Ag": ["4f7/2-3d5/2", "4f5/2-3d3/2"],
    "Cu": ["3d5/2-2p3/2", "3d3/2-2p1/2"],
    "Au": ["5g9/2-4f7/2", "5g7/2-4f5/2"],
}


def peak_energies():
    return {el: [transition_energy_kev(el, t) for t in trs] for el, trs in PEAKS.items()}


def _windows():
    pe = peak_energies()
    agcu = sorted(pe["Ag"] + pe["Cu"])
    au = sorted(pe["Au"])
    return [((agcu[0] - 8, agcu[-1] + 8), ["Ag", "Cu"]), ((au[0] - 8, au[-1] + 8), ["Au"])]


def fit_window(counts, lo, hi, elements, pe):
    sel = (LE_GRID >= lo) & (LE_GRID <= hi)
    x, y = LE_GRID[sel], counts[sel].astype(float)
    centers = [(el, e) for el in elements for e in pe[el]]
    npk = len(centers)
    w = 1.0 / np.sqrt(np.maximum(y, 1.0))
    x0 = 0.5 * (lo + hi)

    def model(p):
        amps, shift, sig, b0, b1 = p[:npk], p[npk], p[npk + 1], p[npk + 2], p[npk + 3]
        m = b0 + b1 * (x - x0)
        for a, (_, c) in zip(amps, centers):
            m = m + a * LE_BIN / (np.sqrt(2 * np.pi) * sig) * np.exp(-0.5 * ((x - c - shift) / sig) ** 2)
        return m

    bg0 = float(np.median(y))
    a0 = [max(float(y[np.argmin(abs(x - c))]) - bg0, 1.0) * 2.0 for _, c in centers]
    p0 = np.array(a0 + [0.0, 0.45, bg0, 0.0])
    lb = [0.0] * npk + [-1.5, 0.2, -np.inf, -np.inf]
    ub = [np.inf] * npk + [1.5, 2.0, np.inf, np.inf]
    res = least_squares(lambda p: (model(p) - y) * w, p0, bounds=(lb, ub), x_scale="jac")
    J = res.jac
    dof = max(len(y) - len(p0), 1)
    chi2 = float((res.fun ** 2).sum() / dof)
    try:
        cov = np.linalg.inv(J.T @ J) * max(chi2, 1.0)
        err = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        err = np.full(len(p0), np.nan)
    areas = {}
    for k, (el, _) in enumerate(centers):
        a, e = areas.get(el, (0.0, 0.0))
        areas[el] = (a + res.x[k], np.hypot(e, err[k]))
    return areas, chi2


def peak_areas(le_counts):
    pe = peak_energies()
    out = {}
    for (lo, hi), els in _windows():
        a, _ = fit_window(le_counts, lo, hi, els, pe)
        out.update(a)
    return out


class PeakFitQuantifier:
    """Relative sensitivities s_el (area per atomic fraction) from standards."""

    def __init__(self):
        self.s = {el: 1.0 for el in BASELINE_ELEMENTS}

    def calibrate(self, standards):
        """standards: list of (le_counts, wt dict). Log-space least squares on
        area ratios, with Ag fixed as the reference."""
        rows, rhs = [], []
        idx = {"Au": 0, "Cu": 1}
        for counts, wt in standards:
            A = peak_areas(counts)
            x = _wt_to_at({k: v for k, v in wt.items() if k in BASELINE_ELEMENTS})
            for el in ("Au", "Cu"):
                if x.get(el, 0) > 0.005 and x.get("Ag", 0) > 0.005 and A[el][0] > 0 and A["Ag"][0] > 0:
                    r = np.zeros(2)
                    r[idx[el]] = 1.0
                    rows.append(r)
                    rhs.append(np.log(A[el][0] / x[el]) - np.log(A["Ag"][0] / x["Ag"]))
        sol, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
        self.s = {"Ag": 1.0, "Au": float(np.exp(sol[0])), "Cu": float(np.exp(sol[1]))}
        return self.s

    def quantify(self, le_counts):
        """-> wt fractions over {Au, Ag, Cu} and 1-sigma errors (counting only)."""
        A = peak_areas(le_counts)
        n = {el: max(A[el][0], 0.0) / self.s[el] for el in BASELINE_ELEMENTS}
        # singular covariance (e.g. an absent element) -> fall back to sqrt(N)
        dn = {el: (A[el][1] if np.isfinite(A[el][1]) else np.sqrt(max(A[el][0], 1.0))) / self.s[el]
              for el in BASELINE_ELEMENTS}
        tot_at = sum(n.values())
        if tot_at <= 0:
            return {el: 0.0 for el in BASELINE_ELEMENTS}, {el: 1.0 for el in BASELINE_ELEMENTS}
        x = {el: n[el] / tot_at for el in BASELINE_ELEMENTS}
        wt = _at_to_wt(x)
        # propagate counting errors by Monte Carlo on the areas
        rng = np.random.default_rng(0)
        draws = []
        for _ in range(200):
            nn_ = {el: max(n[el] + dn[el] * rng.standard_normal(), 0.0) for el in BASELINE_ELEMENTS}
            t = sum(nn_.values()) or 1.0
            draws.append([_at_to_wt({e: nn_[e] / t for e in BASELINE_ELEMENTS})[e] for e in BASELINE_ELEMENTS])
        sd = np.std(np.array(draws), axis=0)
        return wt, {el: float(s) for el, s in zip(BASELINE_ELEMENTS, sd)}


def _wt_to_at(wt):
    n = {k: v / W_OF[k] for k, v in wt.items() if v > 0}
    t = sum(n.values())
    return {k: v / t for k, v in n.items()}


def _at_to_wt(at):
    m = {k: v * W_OF[k] for k, v in at.items()}
    t = sum(m.values()) or 1.0
    return {k: v / t for k, v in m.items()}


def at_to_wt(at):
    return _at_to_wt(at)


def wt_to_at(wt):
    return _wt_to_at(wt)
