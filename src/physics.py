"""
physics.py
==========
Muonic-atom physics from first principles, calibrated and validated against
measured muonic X-ray energies (see literature.py).

What changed with respect to v1 of this repo
--------------------------------------------
v1 used the point-nucleus Sommerfeld formula and rescaled the K lines with an
empirical factor; L/M lines were left point-like and only six hand-picked
Delta-n = 1 transitions existed, with hand-set intensities. That model cannot
reproduce the lines the ISIS/Oxford Roman-coin work actually uses
(Ag 4f-3d at 304.5/308.2 keV, Cu 3d-2p at 330.6/336.4 keV, Au 5g-4f near
405-420 keV) to better than a few keV, and it had no 5g-4f lines at all.

This module instead:
  1. Solves the radial Dirac equation numerically for every (n, kappa) level
     up to n = N_DIRAC for a muon (with the muon-nucleus reduced mass) in the
     potential of a two-parameter Fermi nuclear charge distribution plus the
     Uehling vacuum-polarisation potential folded over that distribution --
     the same physics as MuDirac (Sturniolo & Hillier, X-Ray Spectrom. 50
     (2021) 180). Levels n > N_DIRAC use the analytic point-nucleus Dirac
     energy (finite size is negligible there).
  2. Adds one small empirical shift to the 1s level only, interpolated in Z
     from measured 2p3/2-1s energies ("anchor" lines). This absorbs nuclear
     polarisation and higher-order QED (MuDirac reports a ~ -50 keV residual
     for Pb K lines without them). Nothing else is fitted.
  3. Runs a muonic cascade from n = 14 with a modified-statistical initial
     l-distribution, E1 radiative rates (hydrogenic dipole matrix elements,
     true Dirac transition energies) competing with an Auger channel, and
     splits every (n l -> n' l') line into its fine-structure components with
     6j-symbol line strengths. This gives *all* lines (Lyman, Balmer,
     Paschen, ... series, Delta-n > 1 included) with physically motivated
     relative intensities. The two poorly known cascade parameters (initial
     l-distribution slope and Auger crossover energy) are exposed so the
     simulator can randomise over them (domain randomisation).

Run `python physics.py` to build the level cache (a few minutes, once) and
print the validation table.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from config import A_OF, RMS_OF, SYM, Z_OF

# ---------------------------------------------------------------------------
# Constants (CODATA 2018)
# ---------------------------------------------------------------------------
HBARC = 197.3269804           # MeV fm
ALPHA = 1.0 / 137.035999084
M_MU = 105.6583755            # MeV
M_E = 0.51099895              # MeV
AMU = 931.49410242            # MeV
LAMBDA_E = HBARC / M_E        # reduced electron Compton wavelength, fm

N_DIRAC = 10                  # levels n <= N_DIRAC are solved numerically
N_CASCADE = 14                # muons are captured around n ~ 14 (Green et al. 2021)
SPECTRO = "spdfghiklmnoqrtuv"

CACHE_DIR = Path(__file__).resolve().parent.parent / "data"
LEVEL_CACHE = CACHE_DIR / "dirac_levels.json"
CACHE_VERSION = 3


def reduced_mass(A: float) -> float:
    m_nuc = A * AMU
    return M_MU * m_nuc / (M_MU + m_nuc)


def kappa_of(l: int, j2: int) -> int:
    """Dirac quantum number kappa for orbital l and total 2j."""
    return -(l + 1) if j2 == 2 * l + 1 else l


def level_name(n: int, l: int, j2: int) -> str:
    return f"{n}{SPECTRO[l]}{j2}/2"


def point_dirac_energy(Z: int, m: float, n: int, kappa: int) -> float:
    """Analytic point-nucleus Dirac binding energy (MeV, negative)."""
    za = Z * ALPHA
    ak = abs(kappa)
    gamma = math.sqrt(ak * ak - za * za)
    denom = n - ak + gamma
    return m * (1.0 / math.sqrt(1.0 + (za / denom) ** 2) - 1.0)


# ---------------------------------------------------------------------------
# Nuclear charge distribution and potentials
# ---------------------------------------------------------------------------
class Nucleus:
    """Two-parameter Fermi (or uniform sphere) charge distribution."""

    def __init__(self, Z: int, r_rms: float, model: str = "fermi", t_skin: float = 2.30):
        self.Z = Z
        if model == "fermi":
            a = t_skin / (4.0 * math.log(3.0))
            c2 = 5.0 / 3.0 * (r_rms ** 2 - 7.0 / 5.0 * math.pi ** 2 * a ** 2)
            c = math.sqrt(max(c2, 0.25))
            rmax = c + 25.0 * a
            rg = np.linspace(0.0, rmax, 1600)
            shape = 1.0 / (1.0 + np.exp((rg - c) / a))
        elif model == "sphere":
            R = math.sqrt(5.0 / 3.0) * r_rms
            rmax = R * 1.0001
            rg = np.linspace(0.0, rmax, 1600)
            shape = (rg <= R).astype(float)
        else:
            raise ValueError(model)
        norm = np.trapezoid(4 * np.pi * rg ** 2 * shape, rg)
        self.rg = rg
        self.rho = Z * shape / norm                                   # e / fm^3
        f_enc = 4 * np.pi * rg ** 2 * self.rho
        f_out = 4 * np.pi * rg * self.rho
        self.q_enc = _cumtrapz(f_enc, rg)
        out_cum = _cumtrapz(f_out, rg)
        self.q_out = out_cum[-1] - out_cum                            # int_r^inf
        self.rmax = rmax

    def coulomb(self, r: np.ndarray) -> np.ndarray:
        q = np.interp(r, self.rg, self.q_enc, right=self.Z)
        o = np.interp(r, self.rg, self.q_out, right=0.0)
        return -ALPHA * HBARC * (q / r + o)

    def uehling(self, r: np.ndarray) -> np.ndarray:
        """Uehling potential of the extended charge (MeV).
        V(r) = (2 pi / r) int r' rho(r') [W(r+r') - W(|r-r'|)] dr'
        with W(s) = int_0^s s' U_point(s') ds' for a unit point charge."""
        out = np.empty_like(r)
        rp = self.rg
        wts = self.rho * rp
        for i0 in range(0, len(r), 1500):
            rr = r[i0:i0 + 1500, None]
            diff = _W(rr + rp[None, :]) - _W(np.abs(rr - rp[None, :]))
            out[i0:i0 + 1500] = 2 * np.pi / rr[:, 0] * np.trapezoid(wts[None, :] * diff, rp, axis=1)
        return out


def _cumtrapz(f, x):
    out = np.zeros_like(f)
    out[1:] = np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(x))
    return out


def _build_w_table():
    th, wt = np.polynomial.legendre.leggauss(96)
    th = 0.25 * np.pi * (th + 1.0)          # map [-1,1] -> [0, pi/2]
    wt = 0.25 * np.pi * wt
    u = np.sin(th)
    s = np.logspace(-8, 6.5, 3000)
    g = (LAMBDA_E / 2) * (-np.expm1(-2 * s[:, None] / (LAMBDA_E * u[None, :]))) \
        * (1 + u[None, :] ** 2 / 2) * np.cos(th[None, :]) ** 2
    W = -(2 * ALPHA ** 2 * HBARC / (3 * np.pi)) * (g @ wt)
    return np.log(s), W


_LOGS, _WTAB = _build_w_table()


def _W(s: np.ndarray) -> np.ndarray:
    return np.interp(np.log(np.maximum(s, 1e-12)), _LOGS, _WTAB, left=0.0)


# ---------------------------------------------------------------------------
# Vectorised radial Dirac shooting solver
# ---------------------------------------------------------------------------
def solve_dirac_levels(Z: int, A: float, r_rms: float, states, *, model="fermi",
                       uehling=True, n_steps=6000, r0=1e-3, k_trial=16, sweeps=13):
    """
    Eigen-energies (MeV, negative) for a list of (n, l, j2) states.

    Method: outward RK4 integration on a logarithmic grid; the number of
    nodes of the large component P(r) (including the sign flip of a
    diverging tail) is monotonic in the trial energy and equals n-l-1 just
    below the eigenvalue, so the eigenvalue is bracketed by node counting.
    All states and K trial energies per state are integrated simultaneously.
    """
    m = reduced_mass(A)
    nuc = Nucleus(Z, r_rms, model=model)
    a_bohr = HBARC / (m * ALPHA)                     # fm, for Z = 1
    nmax = max(s[0] for s in states)
    rmax = 4.5 * nmax ** 2 * a_bohr / Z + 30.0
    x = np.linspace(math.log(r0), math.log(rmax), 2 * n_steps + 1)
    r = np.exp(x)
    V = nuc.coulomb(r)
    if uehling:
        V = V + nuc.uehling(r)
    h = x[2] - x[0]

    kap = np.array([kappa_of(l, j2) for (_, l, j2) in states], dtype=float)
    n_r = np.array([n - l - 1 for (n, l, _) in states])
    e_pt = np.array([point_dirac_energy(Z, m, n, int(k)) for (n, _, _), k in zip(states, kap)])
    lo = e_pt * 1.08          # more bound than any physical correction
    hi = e_pt * 0.12          # far less bound (finite size can halve 1s binding)

    cA = r * (2 * m - V) / HBARC        # coefficient pieces, energy-independent
    cB = -r * V / HBARC
    V0 = V[0]
    M = len(states)
    frac = np.arange(1, k_trial + 1) / (k_trial + 1)

    for _ in range(sweeps):
        E = lo[:, None] + (hi - lo)[:, None] * frac[None, :]
        kk = np.repeat(kap, k_trial)
        nodes = _count_nodes(E.ravel(), kk, r, cA, cB, V0, m, h, n_steps).reshape(M, k_trial)
        above = nodes >= (n_r[:, None] + 1)
        # highest trial energy still below the eigenvalue / lowest above
        E_below = np.where(~above, E, -np.inf).max(axis=1)
        E_above = np.where(above, E, np.inf).min(axis=1)
        lo = np.maximum(lo, E_below)
        hi = np.minimum(hi, E_above)
    return 0.5 * (lo + hi)


def _count_nodes(e, kap, r, cA, cB, V0, m, h, n_steps):
    e_h = e / HBARC
    r_0 = r[0]
    P = np.empty_like(e)
    Q = np.empty_like(e)
    neg = kap < 0
    a = np.abs(kap)
    # regular solution at the origin of a finite nucleus
    P[neg] = r_0 ** a[neg]
    Q[neg] = -(e[neg] - V0) / HBARC * r_0 ** (a[neg] + 1) / (1 - 2 * kap[neg])
    Q[~neg] = r_0 ** a[~neg]
    P[~neg] = (2 * m + e[~neg] - V0) / HBARC * r_0 ** (a[~neg] + 1) / (2 * kap[~neg] + 1)
    nodes = np.zeros(e.shape, dtype=np.int32)
    hh = 0.5 * h
    for i in range(n_steps):
        j = 2 * i
        r0_, r1_, r2_ = r[j], r[j + 1], r[j + 2]
        a0 = cA[j] + r0_ * e_h; b0 = r0_ * e_h + cB[j]
        a1 = cA[j + 1] + r1_ * e_h; b1 = r1_ * e_h + cB[j + 1]
        a2 = cA[j + 2] + r2_ * e_h; b2 = r2_ * e_h + cB[j + 2]
        k1p = -kap * P + a0 * Q;            k1q = kap * Q - b0 * P
        P2 = P + hh * k1p; Q2 = Q + hh * k1q
        k2p = -kap * P2 + a1 * Q2;          k2q = kap * Q2 - b1 * P2
        P3 = P + hh * k2p; Q3 = Q + hh * k2q
        k3p = -kap * P3 + a1 * Q3;          k3q = kap * Q3 - b1 * P3
        P4 = P + h * k3p; Q4 = Q + h * k3q
        k4p = -kap * P4 + a2 * Q4;          k4q = kap * Q4 - b2 * P4
        Pn = P + h / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
        Q = Q + h / 6 * (k1q + 2 * k2q + 2 * k3q + k4q)
        nodes += (Pn * P < 0)
        P = Pn
        if i % 16 == 0:
            s = np.maximum(np.abs(P), np.abs(Q))
            big = s > 1e100
            if big.any():
                P[big] /= s[big]
                Q[big] /= s[big]
    return nodes


# ---------------------------------------------------------------------------
# Level table with cache
# ---------------------------------------------------------------------------
def all_states(nmax: int):
    out = []
    for n in range(1, nmax + 1):
        for l in range(n):
            out.append((n, l, 2 * l + 1))
            if l > 0:
                out.append((n, l, 2 * l - 1))
    return out


def _compute_element_levels(sym: str) -> dict:
    Z, A, rms = Z_OF[sym], A_OF[sym], RMS_OF[sym]
    states = all_states(N_DIRAC)
    E = solve_dirac_levels(Z, A, rms, states)
    return {level_name(*s): float(e) * 1e3 for s, e in zip(states, E)}  # keV


def load_level_table(recompute: bool = False, verbose: bool = True) -> dict:
    """{symbol: {level_name: binding energy keV (negative)}} for n <= N_DIRAC."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    table = {}
    if LEVEL_CACHE.exists() and not recompute:
        blob = json.loads(LEVEL_CACHE.read_text())
        if blob.get("version") == CACHE_VERSION:
            table = blob["levels"]
    missing = [s for s in SYM if s not in table]
    if missing:
        import time
        for s in missing:
            t0 = time.time()
            table[s] = _compute_element_levels(s)
            if verbose:
                print(f"  solved Dirac levels for {s:>2} (Z={Z_OF[s]}) in {time.time() - t0:.1f}s", flush=True)
            LEVEL_CACHE.write_text(json.dumps({"version": CACHE_VERSION, "levels": table}))
    return table


@lru_cache(maxsize=1)
def _levels():
    return load_level_table(verbose=True)


# ---------------------------------------------------------------------------
# Residual 1s correction calibrated to measured 2p3/2-1s energies
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _k_shift_curve():
    from literature import MEASURED_LINES
    lev = _levels()
    zs, ds = [1.0], [0.0]
    for sym, tr, e_meas, _sig, role, _src in MEASURED_LINES:
        if role != "anchor":
            continue
        e_calc = lev[sym]["2p3/2"] - lev[sym]["1s1/2"]
        zs.append(float(Z_OF[sym]))
        ds.append(e_meas - e_calc)
    order = np.argsort(zs)
    return PchipInterpolator(np.array(zs)[order], np.array(ds)[order], extrapolate=False), max(zs)


def k_shift_kev(Z: int) -> float:
    """Extra 1s binding (keV) from the anchor calibration."""
    curve, zmax = _k_shift_curve()
    return float(curve(min(float(Z), zmax)))


def level_energy_kev(sym: str, n: int, l: int, j2: int) -> float:
    """Binding energy (keV, negative) of a muonic level, calibrated."""
    if n <= N_DIRAC:
        e = _levels()[sym][level_name(n, l, j2)]
    else:
        e = point_dirac_energy(Z_OF[sym], reduced_mass(A_OF[sym]), n, kappa_of(l, j2)) * 1e3
    if n == 1:
        e -= k_shift_kev(Z_OF[sym])
    return e


def parse_level(name: str):
    n = int(name[0]) if not name[1].isdigit() else int(name[:2])
    rest = name[len(str(n)):]
    l = SPECTRO.index(rest[0])
    j2 = int(rest[1:].split("/")[0])
    return n, l, j2


def transition_energy_kev(sym: str, transition: str, calibrated: bool = True) -> float:
    up, lo_ = transition.split("-")
    nu, lu, ju = parse_level(up)
    nl, ll, jl = parse_level(lo_)
    if calibrated:
        return level_energy_kev(sym, nu, lu, ju) - level_energy_kev(sym, nl, ll, jl)
    lev = _levels()[sym]
    return lev[level_name(nu, lu, ju)] - lev[level_name(nl, ll, jl)]


# ---------------------------------------------------------------------------
# Muonic cascade -> line list with intensities per stopped muon
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _hydrogenic_dipoles():
    """<n l | r | n' l'> for hydrogen (a = Z = 1), all n, n' <= N_CASCADE."""
    from scipy.special import eval_genlaguerre, gammaln
    r = np.linspace(1e-6, 900.0, 60000)
    R = {}
    for n in range(1, N_CASCADE + 1):
        for l in range(n):
            rho = 2 * r / n
            lognorm = 0.5 * (3 * math.log(2 / n) + gammaln(n - l) - math.log(2 * n) - gammaln(n + l + 1))
            R[(n, l)] = math.exp(lognorm) * np.exp(-r / n) * rho ** l * eval_genlaguerre(n - l - 1, 2 * l + 1, rho)
    D = {}
    for (n, l), Ru in R.items():
        for (n2, l2), Rl in R.items():
            if n2 < n and abs(l - l2) == 1:
                D[(n, l, n2, l2)] = float(np.trapezoid(Ru * Rl * r ** 3, r))
    return D


@lru_cache(maxsize=None)
def _six_j_weights(l_u: int, l_l: int):
    """Relative strengths of fine-structure components of an l_u -> l_l E1 line."""
    from sympy.physics.wigner import wigner_6j
    from sympy import Rational
    out = []
    for ju in ([2 * l_u + 1, 2 * l_u - 1] if l_u > 0 else [1]):
        for jl in ([2 * l_l + 1, 2 * l_l - 1] if l_l > 0 else [1]):
            if abs(ju - jl) > 2:
                continue
            w = float(wigner_6j(l_u, Rational(ju, 2), Rational(1, 2), Rational(jl, 2), l_l, 1)) ** 2
            w *= (ju + 1) * (jl + 1)
            if w > 0:
                out.append((ju, jl, w))
    tot = sum(w for *_, w in out)
    return tuple((ju, jl, w / tot) for ju, jl, w in out)


def electron_binding_kev(Z: int, shell: str) -> float:
    """Screened-hydrogenic estimate of K/L electron binding (keV)."""
    if shell == "K":
        return 13.6e-3 * (Z - 1) ** 2
    return 13.6e-3 * max(Z - 4.15, 1) ** 2 / 4


def cascade_lines(sym: str, alpha_l: float = 0.15, auger_e0_kev: float = 18.0,
                  min_intensity: float = 2e-4, min_energy_kev: float = 15.0):
    """
    Line list [(E_keV, intensity per stopped muon, label)] for element `sym`.

    alpha_l      : initial l-distribution P(l) ~ (2l+1) exp(alpha_l * l) at n=14
    auger_e0_kev : Auger/radiative crossover energy for Z = 26; scales as
                   (Z/26)^0.7. Auger rate is taken energy-independent and
                   proportional to the same dipole matrix element.
    """
    Z = Z_OF[sym]
    D = _hydrogenic_dipoles()
    a_scale = HBARC / (reduced_mass(A_OF[sym]) * ALPHA) / Z   # fm per hydrogenic unit

    # (n,l)-averaged energies (keV)
    def e_nl(n, l):
        if l == 0:
            return level_energy_kev(sym, n, 0, 1)
        ea = level_energy_kev(sym, n, l, 2 * l + 1)
        eb = level_energy_kev(sym, n, l, 2 * l - 1)
        return ((2 * l + 2) * ea + 2 * l * eb) / (4 * l + 2)

    E = {(n, l): e_nl(n, l) for n in range(1, N_CASCADE + 1) for l in range(n)}
    e_auger = auger_e0_kev * (Z / 26.0) ** 0.7
    b_l = electron_binding_kev(Z, "L")

    pop = {k: 0.0 for k in E}
    ls = np.arange(N_CASCADE)
    p0 = (2 * ls + 1) * np.exp(alpha_l * ls)
    p0 /= p0.sum()
    for l in range(N_CASCADE):
        pop[(N_CASCADE, l)] = p0[l]

    lines = {}
    order = sorted(E.keys(), key=lambda k: -E[k])    # least bound first
    for (n, l) in order:
        p = pop[(n, l)]
        if p <= 0 or n == 1:
            continue
        chans = []
        for (n2, l2) in ((n2, l2) for n2 in range(1, n) for l2 in (l - 1, l + 1) if 0 <= l2 < n2):
            de = E[(n, l)] - E[(n2, l2)]
            if de <= 0:
                continue
            d = D[(n, l, n2, l2)] * a_scale
            ang = max(l, l2) / (2 * l + 1)
            a_rad = de ** 3 * d * d * ang
            a_aug = (e_auger ** 3 * d * d * ang) if de > b_l else 0.0
            chans.append((n2, l2, de, a_rad, a_aug))
        if not chans:            # 2s: E1 to 1s forbidden -> feeds 2p (2s-2p + 2p-1s)
            pop[(2, 1)] = pop.get((2, 1), 0.0) + p
            continue
        tot = sum(c[3] + c[4] for c in chans)
        for n2, l2, de, a_rad, a_aug in chans:
            br = (a_rad + a_aug) / tot
            pop[(n2, l2)] += p * br
            lines[(n, l, n2, l2)] = lines.get((n, l, n2, l2), 0.0) + p * a_rad / tot

    out = []
    for (n, l, n2, l2), inten in lines.items():
        if inten < min_intensity:
            continue
        for ju, jl, w in _six_j_weights(l, l2):
            ek = level_energy_kev(sym, n, l, ju) - level_energy_kev(sym, n2, l2, jl)
            if ek >= min_energy_kev and inten * w >= min_intensity * 0.25:
                out.append((ek, inten * w, f"{level_name(n, l, ju)}-{level_name(n2, l2, jl)}"))
    out.sort()
    return out


# Grid of cascade parameters used for domain randomisation in simulate.py
CASCADE_ALPHAS = (0.0, 0.12, 0.25)
CASCADE_AUGER_E0 = (10.0, 18.0, 30.0)


def validation_table():
    from literature import MEASURED_LINES
    rows = []
    for sym, tr, e_meas, sig, role, src in MEASURED_LINES:
        e_raw = transition_energy_kev(sym, tr, calibrated=False)
        e_cal = transition_energy_kev(sym, tr, calibrated=True)
        rows.append((sym, tr, role, e_meas, sig, e_raw, e_cal, src))
    return rows


if __name__ == "__main__":
    import time
    t0 = time.time()
    # Self-test of the solver against the analytic point-nucleus Dirac formula
    m = reduced_mass(197)
    test_states = [(1, 0, 1), (2, 1, 3), (3, 2, 5), (5, 4, 9)]
    num = solve_dirac_levels(79, 197, 0.002, test_states, model="sphere", uehling=False)
    print("Solver self-test (Au, ~point nucleus, no VP): numeric vs analytic (keV)")
    for s, e in zip(test_states, num):
        ea = point_dirac_energy(79, m, s[0], kappa_of(s[1], s[2]))
        print(f"  {level_name(*s):>7}: {e * 1e3:14.4f} {ea * 1e3:14.4f}  diff {abs(e - ea) * 1e3:.2e} keV")

    load_level_table()
    print(f"\nLevel table ready ({time.time() - t0:.0f}s).")
    print(f"\n{'el':>3} {'transition':>13} {'role':>6} {'measured':>10} {'Dirac+VP':>10} {'+K-cal':>10} {'resid':>8}")
    for sym, tr, role, em, sig, er, ec, src in validation_table():
        print(f"{sym:>3} {tr:>13} {role:>6} {em:10.3f} {er:10.3f} {ec:10.3f} {ec - em:+8.3f}")
    print("\nAu cascade (alpha_l=0.12, E0=18 keV), strongest lines:")
    for e, i, lab in sorted(cascade_lines("Au", 0.12, 18.0), key=lambda t: -t[1])[:14]:
        print(f"  {e:9.2f} keV  {i:.3f}/mu  {lab}")
