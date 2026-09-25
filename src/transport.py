"""
transport.py
============
Where do the muons stop, and how much of each emitted X-ray escapes?

* Muon stopping: Bethe formula (mean excitation energies from config.I_OF,
  Bragg additivity for mixtures) with a smooth low-energy regularisation of
  the stopping logarithm. One global range-scale factor is fitted to the
  implantation depths the ISIS/Oxford coin papers report from SRIM
  (literature.DEPTH_BENCHMARKS), which absorbs shell/Barkas corrections.
* Beam: Gaussian momentum bite (10 % FWHM, Green et al. 2021) plus 3 % range
  straggling; the beam first crosses an air gap and the aluminium sample
  packet (16 um in Green et al. 2021), both of which also stop muons at low
  momentum and give N/O/Al lines -- exactly as seen in the published spectra.
* Objects are stacks of layers (thickness in um, composition in wt fractions,
  density); depth-graded profiles such as the surface enrichment of Roman
  silver coinage are discretised into thin layers.
* Photon attenuation: Klein-Nishina Compton + a Z^4/E^2.6 photoelectric term
  with K-edges, calibrated to NIST XCOM for Pb (within ~10-15 % for Cu/Ag/Au
  between 100 keV and 1 MeV). Used for self-absorption of muonic X-rays on
  their way to the upstream detector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from config import I_OF, MOMENTUM_FWHM_FRAC, W_OF, Z_OF

M_MU = 105.6583755
M_E_EV = 510998.95
K_BETHE = 0.307075          # MeV cm^2 / mol
N_A = 6.02214076e23
R_E2 = (2.8179403262e-13) ** 2

ELEMENT_DENSITY = {  # g/cm^3 of the pure element (solids) -- for alloy densities
    "Na": 0.97, "Mg": 1.74, "Al": 2.70, "Si": 2.33, "P": 1.82, "S": 2.07, "Cl": 1.8, "K": 0.86,
    "Ca": 1.55, "Ti": 4.51, "Cr": 7.19, "Mn": 7.21, "Fe": 7.87, "Co": 8.90, "Ni": 8.91,
    "Cu": 8.96, "Zn": 7.13, "As": 5.73, "Se": 4.81, "Ag": 10.49, "Cd": 8.65, "Sn": 7.29,
    "Sb": 6.69, "Ba": 3.51, "Pt": 21.45, "Au": 19.30, "Hg": 13.53, "Pb": 11.34, "Bi": 9.78,
    "C": 2.0, "N": 1.0, "O": 1.4,
}
K_EDGE_KEV = {"Ag": 25.51, "Cd": 26.71, "Sn": 29.20, "Sb": 30.49, "Ba": 37.44, "Pt": 78.39,
              "Au": 80.72, "Hg": 83.10, "Pb": 88.00, "Bi": 90.53}

AIR = {"N": 0.7553, "O": 0.2318, "C": 0.0129}    # (Ar folded into N)
AIR_DENSITY = 1.205e-3


def alloy_density(wt: dict) -> float:
    inv = sum(w / ELEMENT_DENSITY.get(s, 2.5) for s, w in wt.items())
    return 1.0 / inv


@dataclass
class Layer:
    thickness_um: float
    wt: dict                     # element -> weight fraction (sums to 1)
    density: float | None = None
    region: str = "object"       # "air", "packet", "object", "backing", "collimator"

    def rho(self) -> float:
        return self.density if self.density is not None else alloy_density(self.wt)


# ---------------------------------------------------------------------------
# Stopping power and range
# ---------------------------------------------------------------------------
_T_GRID = np.logspace(-3.0, math.log10(60.0), 600)   # muon kinetic energy, MeV


def _z_over_a_and_lni(wt: dict):
    za = sum(w * Z_OF[s] / W_OF[s] for s, w in wt.items())
    lni = sum(w * Z_OF[s] / W_OF[s] * math.log(I_OF[s]) for s, w in wt.items()) / za
    return za, math.exp(lni)


def stopping_power(T: np.ndarray, wt: dict) -> np.ndarray:
    """Mass stopping power (MeV cm^2/g) of a mu- with kinetic energy T (MeV)."""
    za, I = _z_over_a_and_lni(wt)
    gamma = 1.0 + T / M_MU
    b2 = 1.0 - 1.0 / gamma ** 2
    arg = 2 * M_E_EV * b2 * gamma ** 2 / I
    L = 0.5 * np.log1p(arg ** 2) - b2           # -> ln(arg) - b2 at high energy, ->0 smoothly
    return K_BETHE * za / b2 * np.maximum(L, 1e-3)


_RANGE_SCALE = [1.0]


def _range_table(wt_key):
    wt = dict(wt_key)
    S = stopping_power(_T_GRID, wt)
    inv = 1.0 / S
    R = np.concatenate([[0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(_T_GRID))])
    R += _T_GRID[0] * inv[0] * 0.5          # residual range below the grid
    return R                                 # g/cm^2 (before global scale)


@lru_cache(maxsize=4096)
def _range_table_cached(wt_key):
    return _range_table(wt_key)


def _key(wt):
    return tuple(sorted((s, round(w, 4)) for s, w in wt.items() if w > 0))


def csda_range(T: np.ndarray, wt: dict) -> np.ndarray:
    """Areal range (g/cm^2) for kinetic energy T (MeV)."""
    R = _range_table_cached(_key(wt)) * _RANGE_SCALE[0]
    return np.interp(T, _T_GRID, R)


def energy_from_range(R: np.ndarray, wt: dict) -> np.ndarray:
    Rt = _range_table_cached(_key(wt)) * _RANGE_SCALE[0]
    return np.interp(R, Rt, _T_GRID, left=0.0)


def kinetic_energy(p_mevc):
    return np.sqrt(np.asarray(p_mevc) ** 2 + M_MU ** 2) - M_MU


# Upstream material in the RIKEN-RAL coin measurements: ~100 mm of air
# (Green et al. 2021, Fig. 2) and a ~16 um aluminium packet.
ISIS_UPSTREAM = [(AIR, AIR_DENSITY, 100e3), ({"Al": 1.0}, 2.70, 16.0)]


def mean_depth_um(p_mevc: float, wt: dict, upstream=ISIS_UPSTREAM) -> float:
    """Depth (um) at which a muon of central momentum p stops in material wt,
    after crossing the upstream layers."""
    T = float(kinetic_energy(p_mevc))
    for uwt, urho, t_um in upstream:
        R = float(csda_range(T, uwt)) - urho * t_um * 1e-4
        if R <= 0:
            return 0.0
        T = float(energy_from_range(R, uwt))
    return float(csda_range(T, wt) / alloy_density(wt) * 1e4)


def calibrate_range_scale(verbose=False):
    """Fit the single global range factor to the published SRIM depths
    (log-space weighted least squares), then report residuals."""
    from literature import DEPTH_BENCHMARKS
    _RANGE_SCALE[0] = 1.0
    for _ in range(4):                      # upstream losses depend on the scale: iterate
        logs = []
        for wt, p, d, sig, _src in DEPTH_BENCHMARKS:
            logs.append((math.log(d / max(mean_depth_um(p, wt), 1e-3)), (sig / d) ** -2))
        k = math.exp(sum(l * w for l, w in logs) / sum(w for _, w in logs))
        _RANGE_SCALE[0] *= k
    if verbose:
        print(f"range scale factor = {_RANGE_SCALE[0]:.3f}")
        for wt, p, d, sig, src in DEPTH_BENCHMARKS:
            print(f"  {str(wt):>26} {p:5.1f} MeV/c: model {mean_depth_um(p, wt):7.1f} um | "
                  f"published {d:.0f} +/- {sig:.0f} um  ({src})")
    return _RANGE_SCALE[0]


calibrate_range_scale()


# ---------------------------------------------------------------------------
# Stopping distribution in a layered target
# ---------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _quantile_normals(n: int):
    """Deterministic standard-normal quantiles (momentum, straggling), so the
    expected spectrum is smooth and reproducible."""
    from scipy.special import ndtri
    z1 = ndtri((np.arange(n) + 0.5) / n)
    z2 = np.random.default_rng(1).permutation(z1)
    return z1, z2


def stopping_fractions(layers: list[Layer], p0: float, n_mu: int = 6000,
                       rng: np.random.Generator | None = None,
                       momentum_fwhm: float = MOMENTUM_FWHM_FRAC, straggle: float = 0.03,
                       return_depth: bool = False):
    """
    Fraction of incident muons stopping in each layer (last entry: transmitted).
    Deterministic quantiles are used when rng is None (smooth, reproducible).
    With return_depth=True also returns the mean stopping depth (um) inside
    each layer, measured from that layer's front face.
    """
    sig = momentum_fwhm / 2.3548
    if rng is None:
        z1, z2 = _quantile_normals(n_mu)
    else:
        z1 = rng.standard_normal(n_mu)
        z2 = rng.standard_normal(n_mu)
    p = p0 * (1.0 + sig * z1)
    T = kinetic_energy(np.clip(p, 1.0, None))
    strag = np.clip(1.0 + straggle * z2, 0.5, 1.5)
    alive = np.ones(n_mu, dtype=bool)
    counts = np.zeros(len(layers) + 1)
    depth = np.zeros(len(layers))
    for i, lay in enumerate(layers):
        if not alive.any():
            break
        areal = lay.rho() * lay.thickness_um * 1e-4
        R = csda_range(T, lay.wt) * strag
        stop = alive & (R <= areal)
        counts[i] = stop.sum()
        if counts[i] > 0:
            depth[i] = float(R[stop].mean()) / lay.rho() * 1e4
        alive &= ~stop
        T = np.where(alive, energy_from_range((R - areal) / strag, lay.wt), 0.0)
    counts[-1] = alive.sum()
    if return_depth:
        return counts / n_mu, depth
    return counts / n_mu


def object_stop_depths(p0: float, obj_wt: dict, upstream=ISIS_UPSTREAM, n_mu: int = 6000,
                       momentum_fwhm: float = MOMENTUM_FWHM_FRAC, straggle: float = 0.03):
    """Per-muon stopping depths (um) inside a thick object of composition
    obj_wt, for the muons that reach it (deterministic quantiles). Used for
    smooth, differentiable depth-profile forward models."""
    z1, z2 = _quantile_normals(n_mu)
    p = p0 * (1.0 + momentum_fwhm / 2.3548 * z1)
    T = kinetic_energy(np.clip(p, 1.0, None))
    strag = np.clip(1.0 + straggle * z2, 0.5, 1.5)
    alive = np.ones(n_mu, dtype=bool)
    for uwt, urho, t_um in upstream:
        R = csda_range(T, uwt) * strag
        areal = urho * t_um * 1e-4
        alive &= R > areal
        T = np.where(alive, energy_from_range((R - areal) / strag, uwt), 0.0)
    R = csda_range(T[alive], obj_wt) * strag[alive]
    return R / alloy_density(obj_wt) * 1e4


# ---------------------------------------------------------------------------
# Photon attenuation
# ---------------------------------------------------------------------------
def _klein_nishina_cm2(E_kev: np.ndarray) -> np.ndarray:
    k = np.asarray(E_kev) / 511.0
    t1 = (1 + k) / k ** 2 * (2 * (1 + k) / (1 + 2 * k) - np.log1p(2 * k) / k)
    t2 = np.log1p(2 * k) / (2 * k) - (1 + 3 * k) / (1 + 2 * k) ** 2
    return 2 * np.pi * R_E2 * (t1 + t2)


def mass_attenuation(E_kev: np.ndarray, wt: dict) -> np.ndarray:
    """mu/rho (cm^2/g) of a mixture (Compton + photoelectric)."""
    E = np.asarray(E_kev, dtype=float)
    kn = _klein_nishina_cm2(E)
    out = np.zeros_like(E)
    for s, w in wt.items():
        Z, A = Z_OF[s], W_OF[s]
        compton = N_A * Z / A * kn
        pe = 3.91 * Z ** 4 / (A * E ** 2.6)
        ek = K_EDGE_KEV.get(s, 13.6e-3 * (Z - 1) ** 2)
        pe = np.where(E < ek, pe / 5.0, pe)
        out += w * (compton + pe)
    return out


def escape_probability(E_kev: np.ndarray, layers_above: list[Layer], cos_theta: float = 0.7071):
    """Transmission of photons of energy E through a stack of layers at angle theta."""
    E = np.asarray(E_kev, dtype=float)
    mu_t = np.zeros_like(E)
    for lay in layers_above:
        mu_t += mass_attenuation(E, lay.wt) * lay.rho() * lay.thickness_um * 1e-4
    return np.exp(-mu_t / cos_theta)


if __name__ == "__main__":
    calibrate_range_scale(verbose=True)
    print("\nMean implantation depth (um) vs momentum:")
    for name, wt in [("Au", {"Au": 1.0}), ("Ag62Cu38", {"Ag": 0.62, "Cu": 0.38}), ("Cu", {"Cu": 1.0})]:
        row = " ".join(f"{p:>3.0f}:{mean_depth_um(p, wt):7.1f}" for p in (17, 18, 20, 22, 25, 30, 35, 40, 50, 60))
        print(f"  {name:>9}  {row}")
    print("\nmu/rho (cm^2/g) at 100/300/1000 keV (NIST XCOM: Cu .458/.107/.059, Ag 1.47/.15/.063, Pb 5.55/.403/.071)")
    for s in ("Cu", "Ag", "Pb"):
        print(f"  {s}: " + " ".join(f"{v:.3f}" for v in mass_attenuation(np.array([100., 300., 1000.]), {s: 1.0})))
    air = Layer(100e3, AIR, AIR_DENSITY, "air")
    foil = Layer(16.0, {"Al": 1.0}, 2.70, "packet")
    coin = Layer(1000.0, {"Au": 1.0}, None, "object")
    for p in (17, 18, 20, 25, 40):
        f = stopping_fractions([air, foil, coin], p)
        print(f"  p={p} MeV/c: air {f[0]:.3f} packet {f[1]:.3f} coin {f[2]:.3f} through {f[3]:.3f}")
