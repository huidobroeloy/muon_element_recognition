"""
simulate.py
===========
Forward model: object (layer stack) + beam momentum + detector set-up ->
expected count spectra in a low-energy planar HPGe (20-1000 keV, 0.5 keV
bins) and a high-energy coaxial HPGe (0.1-8 MeV, 2 keV bins), as in the
RIKEN-RAL Port 4 arrangement used for the Roman-coin studies.

Per spectrum:
  1. Muon stopping in air gap / Al packet / object layers / backing
     (transport.stopping_fractions, 10 % FWHM momentum bite).
  2. Atomic capture shared between elements of a layer with a Z-law
     (Fermi-Teller, exponent ~1, jittered to represent calibration error).
  3. Muonic cascade line lists from physics.cascade_lines (Dirac energies;
     cascade parameters randomised per element), electronic K X-rays of the
     (Z-1) atom, and nuclear-capture / (n,n') gammas.
  4. Self-absorption of every line on its way out through the layers above
     the stopping layer (upstream detectors, random take-off angle).
  5. Detector response: efficiency, resolution, bin-integrated Gaussians,
     Compton continuum, single/double escape peaks, gain/offset drift.
  6. Background: continuum + 511 keV, Ge(n,n') 596/691 keV (with recoil
     tails), 40K, 208Tl, p(n,g)d; optional collimator Pb (Hampshire et al.
     2019); rare random unexplained peaks for robustness.
Returns the *expected* spectra (Poisson noise is drawn in dataset.py, so each
stored spectrum can be re-noised at a different exposure every epoch).
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field

import numpy as np
from scipy.special import erf

from config import (ABSENT_WT, HE_BIN, HE_EMIN, HE_NBINS, LE_BIN, LE_EMIN, LE_NBINS,
                    PRESENT_WT, SIM_ELEMENTS, TARGET_INDEX, TARGETS, NUM_TARGETS, W_OF, Z_OF)
from literature import BACKGROUND_LINES, CAPTURE_GAMMAS, ELECTRONIC_KX
from materials import ObjectModel, sample_object
from physics import CACHE_DIR, CASCADE_ALPHAS, CASCADE_AUGER_E0, cascade_lines
from transport import AIR, AIR_DENSITY, Layer, escape_probability, mass_attenuation, stopping_fractions

LE_GRID = LE_EMIN + LE_BIN * (np.arange(LE_NBINS) + 0.5)
HE_GRID = HE_EMIN + HE_BIN * (np.arange(HE_NBINS) + 0.5)
LINE_CACHE = CACHE_DIR / "cascade_lines.pkl"


# ---------------------------------------------------------------------------
# Line tables: per element, fixed energies, intensities for each cascade model
# ---------------------------------------------------------------------------
def build_line_tables(force=False, verbose=True):
    if LINE_CACHE.exists() and not force:
        with open(LINE_CACHE, "rb") as f:
            return pickle.load(f)
    tables = {}
    combos = [(a, e) for a in CASCADE_ALPHAS for e in CASCADE_AUGER_E0]
    for sym in SIM_ELEMENTS:
        per = []
        for a, e0 in combos:
            per.append({lab: (E, I) for E, I, lab in cascade_lines(sym, a, e0, min_intensity=2e-5)})
        labels = sorted(set().union(*[set(p) for p in per]))
        E = np.array([next(p[l][0] for p in per if l in p) for l in labels])
        I = np.array([[p[l][1] if l in p else 0.0 for l in labels] for p in per])
        keep = I.max(axis=0) >= 3e-4
        tables[sym] = {"E": E[keep], "I": I[:, keep], "labels": [l for l, k in zip(labels, keep) if k]}
        if verbose:
            print(f"  cascade {sym:>2}: {keep.sum():4d} lines, Kalpha yield "
                  f"{I[:, [i for i, l in enumerate(labels) if l.startswith('2p') and l.endswith('1s1/2')]].sum(1).mean():.3f}/mu",
                  flush=True)
    with open(LINE_CACHE, "wb") as f:
        pickle.dump(tables, f)
    return tables


_TABLES = None


def line_tables():
    global _TABLES
    if _TABLES is None:
        _TABLES = build_line_tables()
    return _TABLES


# ---------------------------------------------------------------------------
# Experimental set-up (everything that is not the object)
# ---------------------------------------------------------------------------
@dataclass
class Setup:
    momentum: float                    # MeV/c
    air_mm: float = 100.0
    packet_um: float = 16.0
    collimator_pb: float = 0.0         # extra Pb muonic yield per stopped muon
    counts_le: float = 3e5
    counts_he_ratio: float = 0.5
    cos_le: float = 0.75
    cos_he: float = 0.75
    zlaw_gamma: float = 1.0
    cascade_choice: dict = field(default_factory=dict)
    ekx_yield: float = 0.25            # electronic K X-rays per muon (heavy elements)
    capture_gamma_scale: float = 1.0
    # detector
    le_res: tuple = (0.55, 0.0012)
    he_res: tuple = (1.3, 0.0018)
    le_eff: tuple = (200.0, 1.7)
    he_eff: float = 0.85
    le_pt: float = 200.0
    he_pt: float = 350.0
    escape: tuple = (0.5, 0.4)
    le_cal: tuple = (1.0, 0.0)
    he_cal: tuple = (1.0, 0.0)
    bg: dict = field(default_factory=dict)
    extra_peaks: list = field(default_factory=list)


def random_setup(rng, momentum=None) -> Setup:
    ntab = len(CASCADE_ALPHAS) * len(CASCADE_AUGER_E0)
    s = Setup(
        momentum=float(momentum if momentum is not None else 10 ** rng.uniform(math.log10(16.5), math.log10(60))),
        air_mm=rng.uniform(40, 130),
        packet_um=rng.uniform(10, 25) if rng.random() < 0.8 else 0.0,
        collimator_pb=10 ** rng.uniform(-3, -1.7) if rng.random() < 0.4 else 0.0,
        counts_le=10 ** rng.uniform(4.5, 6.5),
        counts_he_ratio=rng.uniform(0.2, 1.0),
        cos_le=rng.uniform(0.55, 0.95),
        cos_he=rng.uniform(0.55, 0.95),
        zlaw_gamma=rng.normal(1.0, 0.04),
        cascade_choice={e: int(rng.integers(ntab)) for e in SIM_ELEMENTS},
        ekx_yield=rng.uniform(0.1, 0.4),
        capture_gamma_scale=rng.uniform(0.4, 1.6),
        le_res=(rng.uniform(0.45, 0.75), rng.uniform(0.0007, 0.0017)),
        he_res=(rng.uniform(1.0, 1.7), rng.uniform(0.0012, 0.0026)),
        le_eff=(rng.uniform(140, 280), rng.uniform(1.3, 2.1)),
        he_eff=rng.uniform(0.7, 1.0),
        le_pt=rng.uniform(140, 320),
        he_pt=rng.uniform(250, 480),
        escape=(rng.uniform(0.3, 0.8), rng.uniform(0.2, 0.8)),
        le_cal=(1 + rng.normal(0, 1.5e-4), rng.normal(0, 0.06)),
        he_cal=(1 + rng.normal(0, 2e-4), rng.normal(0, 0.3)),
        bg=dict(frac_le=rng.uniform(0.05, 0.35), frac_he=rng.uniform(0.1, 0.5),
                slope_le=rng.uniform(80, 400), slope_he=rng.uniform(400, 2500),
                pow_le=rng.uniform(0.5, 2.0), pow_he=rng.uniform(0.5, 2.0), mix=rng.uniform(0, 1),
                line_scale=10 ** rng.uniform(-3.0, -1.3), gen_scale=10 ** rng.uniform(-1, 0.5)),
        extra_peaks=[(rng.uniform(25, 7800), 10 ** rng.uniform(-3.5, -2.3))
                     for _ in range(int(rng.poisson(0.4)))],
    )
    return s


def nominal_setup(momentum, counts_le=3e5) -> Setup:
    """Mid-range, deterministic set-up (used for case studies/baselines)."""
    ntab = len(CASCADE_ALPHAS) * len(CASCADE_AUGER_E0)
    mid = ntab // 2
    return Setup(momentum=momentum, counts_le=counts_le, cascade_choice={e: mid for e in SIM_ELEMENTS},
                 bg=dict(frac_le=0.15, frac_he=0.25, slope_le=200, slope_he=1200, pow_le=1.2, pow_he=1.2,
                         mix=0.5, line_scale=0.01, gen_scale=1.0))


# ---------------------------------------------------------------------------
# Detector response helpers
# ---------------------------------------------------------------------------
def eff_le(E, p):
    E1, p1 = p
    return (1 - np.exp(-(E / 18.0) ** 3)) / (1 + (E / E1) ** p1)


def eff_he(E, p2):
    return (1 - np.exp(-(E / 80.0) ** 3)) * (1 + E / 200.0) ** (-p2)


def fwhm(E, res):
    a, b = res
    return np.sqrt(a * a + b * E)


def peak_to_total(E, ept, k):
    return 1.0 / (1.0 + (E / ept) ** k)


def render_gaussians(E, A, fw, emin, binw, nb):
    """Bin-integrated Gaussians, vectorised with bincount."""
    ok = (A > 0) & (E > emin - 10 * fw) & (E < emin + binw * nb + 10 * fw)
    E, A, fw = E[ok], A[ok], fw[ok]
    if E.size == 0:
        return np.zeros(nb)
    sig = fw / 2.3548
    W = int(math.ceil(5 * sig.max() / binw)) + 1
    c = np.floor((E - emin) / binw).astype(int)
    offs = np.arange(-W, W + 1)
    idx = c[:, None] + offs[None, :]
    lo = emin + idx * binw
    s2 = math.sqrt(2) * sig[:, None]
    vals = 0.5 * A[:, None] * (erf((lo + binw - E[:, None]) / s2) - erf((lo - E[:, None]) / s2))
    m = (idx >= 0) & (idx < nb)
    return np.bincount(idx[m], weights=vals[m], minlength=nb)


def compton_continuum(E, A_cont, emin, binw, nb):
    """Flat Compton continuum from ~0 up to the Compton edge of each line."""
    ec = E * (1 - 1 / (1 + 2 * E / 511.0))
    i1 = np.clip(((ec - emin) / binw).astype(int), 0, nb)
    width = np.maximum(ec - emin, binw)
    h = A_cont / (width / binw)
    ok = (ec > emin) & (A_cont > 0)
    d = np.zeros(nb + 1)
    np.add.at(d, np.zeros(ok.sum(), dtype=int), h[ok])
    np.add.at(d, i1[ok], -h[ok])
    return np.cumsum(d)[:nb]


# ---------------------------------------------------------------------------
# Main forward model
# ---------------------------------------------------------------------------
def _capture_shares(wt: dict, gamma: float) -> dict:
    x = {s: w / W_OF[s] for s, w in wt.items() if w > 0}
    t = sum(x.values())
    c = {s: (v / t) * Z_OF[s] ** gamma for s, v in x.items()}
    tc = sum(c.values())
    return {s: v / tc for s, v in c.items()}


def build_stack(obj: ObjectModel, setup: Setup):
    stack = [Layer(setup.air_mm * 1e3, dict(AIR), AIR_DENSITY, "air")]
    if setup.packet_um > 0:
        stack.append(Layer(setup.packet_um, {"Al": 1.0}, 2.70, "packet"))
    for lay in obj.layers:
        stack.append(Layer(lay.thickness_um, lay.wt, lay.density, "object"))
    if setup.packet_um > 0:
        stack.append(Layer(setup.packet_um, {"Al": 1.0}, 2.70, "backing"))
    stack.append(Layer(2e6, dict(AIR), AIR_DENSITY, "backing"))
    return stack


def sampled_composition(obj: ObjectModel, setup: Setup):
    """Muon-weighted object composition (all elements) and object stop fraction."""
    stack = build_stack(obj, setup)
    f, depth = stopping_fractions(stack, setup.momentum, n_mu=3000, return_depth=True)
    f = f[:-1]
    comp, fobj = {}, 0.0
    for lay, fi in zip(stack, f):
        if lay.region != "object" or fi <= 0:
            continue
        fobj += fi
        for s, w in lay.wt.items():
            comp[s] = comp.get(s, 0.0) + fi * w
    if fobj > 0:
        comp = {s: v / fobj for s, v in comp.items()}
    return comp, fobj, stack, f, depth


def target_fractions(comp: dict) -> np.ndarray:
    """wt fractions over the predicted (Z >= 11) elements, summing to 1."""
    y = np.zeros(NUM_TARGETS)
    for s, v in comp.items():
        if s in TARGET_INDEX:
            y[TARGET_INDEX[s]] = v
    t = y.sum()
    return y / t if t > 0 else y


def labels_from_fractions(frac: np.ndarray):
    present = (frac >= PRESENT_WT).astype(np.float32)
    mask = ((frac >= PRESENT_WT) | (frac < ABSENT_WT)).astype(np.float32)
    return present, mask


def render(obj: ObjectModel, setup: Setup, rng: np.random.Generator | None = None):
    tabs = line_tables()
    comp, fobj, stack, f, depth = sampled_composition(obj, setup)

    # --- per-element emission lists (cascade + electronic K X-rays + gammas)
    active = [i for i, fi in enumerate(f) if fi > 1e-6]
    elements = sorted({s for i in active for s in stack[i].wt})
    if setup.collimator_pb > 0 and "Pb" not in elements:
        elements.append("Pb")
    E_parts, I_parts, sl = [np.zeros(0)], [np.zeros(0)], {}
    off = 0
    for sym in elements:
        t = tabs[sym]
        E = t["E"]
        I = t["I"][setup.cascade_choice.get(sym, 0)]
        if sym in ELECTRONIC_KX:
            E = np.concatenate([E, ELECTRONIC_KX[sym]])
            I = np.concatenate([I, setup.ekx_yield * np.array([0.28, 0.52, 0.20])])
        g = CAPTURE_GAMMAS.get(sym, [])
        if g:
            E = np.concatenate([E, [e for e, _ in g]])
            I = np.concatenate([I, [y * setup.capture_gamma_scale for _, y in g]])
        E_parts.append(E); I_parts.append(I)
        sl[sym] = slice(off, off + len(E))
        off += len(E)
    E_u = np.concatenate(E_parts)
    I_u = np.concatenate(I_parts)

    # --- attenuation: cumulative areal mu over object layers (upstream detectors)
    obj_idx = [i for i, l in enumerate(stack) if l.region == "object"]
    mu_cum = np.zeros_like(E_u)          # attenuation of all object layers above the current one
    mu_of = {}
    for i in obj_idx:
        lay = stack[i]
        mu_of[i] = mass_attenuation(E_u, lay.wt) * lay.rho() * 1e-4        # per um
    A_le = np.zeros_like(E_u)
    A_he = np.zeros_like(E_u)
    for i, (lay, fi) in enumerate(zip(stack, f)):
        if lay.region == "object":
            mu_here = mu_cum + mu_of[i] * depth[i]
            mu_cum = mu_cum + mu_of[i] * lay.thickness_um
        elif lay.region == "backing":
            mu_here = mu_cum                  # photons cross the whole object
        else:
            mu_here = None
        if fi <= 1e-6:
            continue
        w_le = np.zeros_like(E_u)
        for sym, sh in _capture_shares(lay.wt, setup.zlaw_gamma).items():
            w_le[sl[sym]] = fi * sh
        contrib = w_le * I_u
        if mu_here is None:
            A_le += contrib
            A_he += contrib
        else:
            A_le += contrib * np.exp(-mu_here / setup.cos_le)
            A_he += contrib * np.exp(-mu_here / setup.cos_he)

    # collimator lead (Hampshire et al. 2019 ignore these peaks; we model them)
    E_le, E_he = E_u, E_u
    if setup.collimator_pb > 0:
        t = tabs["Pb"]
        extra_E = np.concatenate([t["E"], [72.80, 74.97, 84.94]])
        extra_A = setup.collimator_pb * np.concatenate([t["I"][setup.cascade_choice.get("Pb", 0)], [0.1, 0.17, 0.07]])
        E_le = E_he = np.concatenate([E_u, extra_E])
        A_le = np.concatenate([A_le, extra_A])
        A_he = np.concatenate([A_he, extra_A])

    le = _detector_le(E_le, A_le, setup)
    he = _detector_he(E_he, A_he, setup)
    le, he = _add_background(le, he, setup)
    le *= setup.counts_le / max(le.sum(), 1e-12)
    he *= setup.counts_le * setup.counts_he_ratio / max(he.sum(), 1e-12)

    frac = target_fractions(comp)
    present, mask = labels_from_fractions(frac)
    info = dict(comp=comp, f_object=fobj, stop_fractions=f, family=obj.family, profile=obj.profile)
    return le.astype(np.float32), he.astype(np.float32), frac.astype(np.float32), present, mask, info


def _detector_le(E, A, s: Setup):
    gain, off = s.le_cal
    Em = E * gain + off
    Ad = A * eff_le(E, s.le_eff)
    pt = peak_to_total(E, s.le_pt, 1.3)
    spec = render_gaussians(Em, Ad, fwhm(E, s.le_res), LE_EMIN, LE_BIN, LE_NBINS)
    spec += compton_continuum(Em, Ad * (1 / pt - 1), LE_EMIN, LE_BIN, LE_NBINS)
    return spec


def _detector_he(E, A, s: Setup):
    gain, off = s.he_cal
    Em = E * gain + off
    Ad = A * eff_he(E, s.he_eff)
    pt = peak_to_total(E, s.he_pt, 1.1)
    fw = fwhm(E, s.he_res)
    spec = render_gaussians(Em, Ad, fw, HE_EMIN, HE_BIN, HE_NBINS)
    hi = E > 1100
    if hi.any():
        x = 1 - 1022.0 / E[hi]
        spec += render_gaussians(Em[hi] - 511.0, Ad[hi] * s.escape[0] * x, fw[hi], HE_EMIN, HE_BIN, HE_NBINS)
        spec += render_gaussians(Em[hi] - 1022.0, Ad[hi] * s.escape[1] * x ** 1.5, fw[hi], HE_EMIN, HE_BIN, HE_NBINS)
        # annihilation photons escaping from pair events in surroundings
        spec += render_gaussians(np.array([511.0 * gain + off]), np.array([Ad[hi].sum() * 0.05]),
                                 fwhm(np.array([511.0]), s.he_res), HE_EMIN, HE_BIN, HE_NBINS)
    spec += compton_continuum(Em, Ad * (1 / pt - 1), HE_EMIN, HE_BIN, HE_NBINS)
    return spec


def _asym_peak(E0, amp, res, emin, binw, nb, cal):
    # Ge(n,n') peaks: sharp edge + tail to higher energy (recoil)
    offs = np.array([0.0, 1.5, 3.0, 5.0, 8.0, 12.0])
    w = np.array([0.3, 0.22, 0.17, 0.13, 0.1, 0.08])
    E = (E0 + offs) * cal[0] + cal[1]
    return render_gaussians(E, amp * w, fwhm(E, res), emin, binw, nb)


def _add_background(le, he, s: Setup):
    b = s.bg
    for spec, grid, frac, slope, pw, res, emin, binw, nb, cal in (
        (le, LE_GRID, b["frac_le"], b["slope_le"], b["pow_le"], s.le_res, LE_EMIN, LE_BIN, LE_NBINS, s.le_cal),
        (he, HE_GRID, b["frac_he"], b["slope_he"], b["pow_he"], s.he_res, HE_EMIN, HE_BIN, HE_NBINS, s.he_cal),
    ):
        tot = max(spec.sum(), 1e-12)
        shape = b["mix"] * np.exp(-grid / slope) + (1 - b["mix"]) * (grid / 100.0) ** (-pw)
        shape *= frac * tot / shape.sum() / max(1 - frac, 0.05)
        spec += shape
        lamp = b["line_scale"] * tot
        for e0, name in BACKGROUND_LINES:
            if "Ge(n" in name:
                spec += _asym_peak(e0, lamp * 2.0, res, emin, binw, nb, cal)
            else:
                spec += render_gaussians(np.array([e0 * cal[0] + cal[1]]), np.array([lamp * (3.0 if e0 == 511.0 else 1.0)]),
                                         fwhm(np.array([e0]), res), emin, binw, nb)
        for e0, a in s.extra_peaks:
            spec += render_gaussians(np.array([e0]), np.array([a * tot]), fwhm(np.array([e0]), res), emin, binw, nb)
    return le, he


# ---------------------------------------------------------------------------
# Random training sample
# ---------------------------------------------------------------------------
def generate_sample(rng: np.random.Generator, p_roman: float = 0.75):
    for _ in range(20):
        obj = sample_object(rng, p_roman)
        setup = random_setup(rng)
        out = render(obj, setup, rng)
        if out[5]["f_object"] >= 0.6:
            le, he, frac, present, mask, info = out
            info["momentum"] = setup.momentum
            info["counts_le"] = setup.counts_le
            return le, he, frac, present, mask, info
    return le, he, frac, present, mask, info


if __name__ == "__main__":
    import time
    tabs = line_tables()
    rng = np.random.default_rng(0)
    t0 = time.time()
    n = 40
    for _ in range(n):
        generate_sample(rng)
    print(f"{(time.time() - t0) / n * 1e3:.1f} ms per sample")
    le, he, frac, present, mask, info = generate_sample(np.random.default_rng(3))
    print(info["family"], info["profile"], f"p={info['momentum']:.1f}", "f_obj=%.2f" % info["f_object"])
    print({TARGETS[i]: round(float(v), 4) for i, v in enumerate(frac) if v > 0})
