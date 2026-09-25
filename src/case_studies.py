"""
case_studies.py
===============
Literature-replica benchmarks for Roman coinage (all spectra simulated with
the validated physics; the published *raw* spectra are not public).

A. Neural network vs. the published peak-fit analysis (baseline.py) on the
   same fixed instrument, both calibrated with the same certified standards
   (MAC 1-3, Ag/Au 70:30, Ag/Cu 50:50), on random Au-Ag-Cu Roman coins.
B. The six gold coins of Green et al. (2021, 2022): XRF compositions used as
   ground truth, measured at 40 MeV/c ("core") and 18 MeV/c ("near surface").
C. Julia Domna denarius (Hampshire et al. 2019): a surface-enriched Ag-Cu
   profile consistent with the published points, measured at the six
   published momenta; depth profile recovered (i) the published way
   (exponential fitted to point estimates placed at the mean implantation
   depth) and (ii) by a forward-model inversion through the full muon
   stopping distribution.

    python case_studies.py --checkpoint ../checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

import materials as M
from baseline import PeakFitQuantifier, at_to_wt, wt_to_at
from config import SIM_ELEMENTS, TARGETS
from inference import Analyzer
from literature import GOLD_COINS, JULIA_DOMNA, MAC_STANDARDS, PUBLISHED_PERFORMANCE
from materials import ObjectModel
from physics import CASCADE_ALPHAS, CASCADE_AUGER_E0
from simulate import nominal_setup, render, sampled_composition
from transport import Layer, mean_depth_um, object_stop_depths

TRIO = ("Au", "Ag", "Cu")


# ---------------------------------------------------------------------------
# A fixed "instrument": nominal detectors + one hidden, fixed choice of the
# cascade parameters for every element (nature is fixed; we just don't know it)
# ---------------------------------------------------------------------------
def instrument(momentum, counts):
    s = nominal_setup(momentum, counts)
    rng = np.random.default_rng(2024)
    ntab = len(CASCADE_ALPHAS) * len(CASCADE_AUGER_E0)
    s.cascade_choice = {e: int(rng.integers(ntab)) for e in SIM_ELEMENTS}
    return s


def measure(obj, momentum, counts, rng):
    le, he, frac, *_ , info = render(obj, instrument(momentum, counts))
    return rng.poisson(le).astype(float), rng.poisson(he).astype(float), frac, info


def homog(wt, th=1500.0):
    t = sum(wt.values())
    return ObjectModel("custom", "homogeneous", [Layer(th, {k: v / t for k, v in wt.items() if v > 0})])


def standards():
    std = [(k, {e: v / 100 for e, v in wt.items()}) for k, wt in MAC_STANDARDS.items()]
    std.append(("Ag70Au30", {"Ag": 0.70, "Au": 0.30}))
    std.append(("Ag50Cu50(at)", at_to_wt({"Ag": 0.5, "Cu": 0.5})))
    return std


def calibrate(an, pf, momentum, rng):
    specs, comps = [], []
    for name, wt in standards():
        le, he, _, _ = measure(homog(wt), momentum, 1e6, rng)
        specs.append((le, he, momentum)); comps.append(wt)
    pf.calibrate([(s[0], c) for s, c in zip(specs, comps)])
    corr = an.calibrate_with_standards(specs, comps)
    return corr


def trio_norm(d):
    t = sum(d.get(e, 0.0) for e in TRIO) or 1.0
    return {e: d.get(e, 0.0) / t for e in TRIO}


def nn_trio(an, le, he, p):
    prob, comp, sig = an.analyse(le, he, p)
    d = {e: float(comp[0, TARGETS.index(e)]) for e in TRIO}
    s = {e: float(sig[0, TARGETS.index(e)]) for e in TRIO}
    return trio_norm(d), s, prob[0]


# ---------------------------------------------------------------------------
def part_a(an, n=240, seed=11):
    rng = np.random.default_rng(seed)
    fams = [M.aureus_pure, M.aureus_debased, M.denarius_fine, M.denarius_debased, M.antoninianus]
    res = {"baseline": [], "nn": [], "truth": [], "family": []}
    cal_cache = {}
    for i in range(n):
        f = fams[i % len(fams)]
        wt = f(rng)
        p = 40.0 if f in (M.aureus_pure, M.aureus_debased) else 35.0
        if p not in cal_cache:
            pf_p = PeakFitQuantifier()
            calibrate(an, pf_p, p, np.random.default_rng(99))
            cal_cache[p] = (pf_p, np.array(an.correction))
        pf_p, corr = cal_cache[p]
        an.correction = corr
        counts = 10 ** rng.uniform(5.0, 6.0)
        le, he, frac, info = measure(homog(wt), p, counts, rng)
        truth = trio_norm(info["comp"])
        b, _ = pf_p.quantify(le)
        nn, _, _ = nn_trio(an, le, he, p)
        res["baseline"].append([b[e] for e in TRIO]); res["nn"].append([nn[e] for e in TRIO])
        res["truth"].append([truth[e] for e in TRIO]); res["family"].append(f.__name__)
    for k in ("baseline", "nn", "truth"):
        res[k] = np.array(res[k])
    mae_b = 100 * np.abs(res["baseline"] - res["truth"]).mean(0)
    mae_n = 100 * np.abs(res["nn"] - res["truth"]).mean(0)
    p95_b = 100 * np.percentile(np.abs(res["baseline"] - res["truth"]), 95, axis=0)
    p95_n = 100 * np.percentile(np.abs(res["nn"] - res["truth"]), 95, axis=0)
    print(f"\n[A] Fixed instrument, standards-calibrated, Au/Ag/Cu wt% (normalised to Au+Ag+Cu), n={n}")
    print(f"{'':>22} {'Au':>7} {'Ag':>7} {'Cu':>7}")
    print(f"{'peak-fit MAE':>22} " + " ".join(f"{v:7.2f}" for v in mae_b))
    print(f"{'neural net MAE':>22} " + " ".join(f"{v:7.2f}" for v in mae_n))
    print(f"{'peak-fit 95th pct':>22} " + " ".join(f"{v:7.2f}" for v in p95_b))
    print(f"{'neural net 95th pct':>22} " + " ".join(f"{v:7.2f}" for v in p95_n))
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    for j, (ax, e) in enumerate(zip(axs, TRIO)):
        ax.plot(100 * res["truth"][:, j], 100 * res["baseline"][:, j], "o", ms=3, alpha=0.5,
                label=f"peak fit (MAE {mae_b[j]:.2f})")
        ax.plot(100 * res["truth"][:, j], 100 * res["nn"][:, j], "o", ms=3, alpha=0.5,
                label=f"neural net (MAE {mae_n[j]:.2f})")
        ax.plot([0, 100], [0, 100], "k--", lw=0.8); ax.set_title(e); ax.set_xlabel("true wt%"); ax.set_ylabel("estimated wt%")
        ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(OUT / "A_nn_vs_peakfit.png", dpi=110); plt.close()
    return dict(n=n, mae_peakfit=mae_b.tolist(), mae_nn=mae_n.tolist(), p95_peakfit=p95_b.tolist(), p95_nn=p95_n.tolist())


# ---------------------------------------------------------------------------
def part_b(an, reps=20, seed=12):
    rng = np.random.default_rng(seed)
    rows = []
    print(f"\n[B] Gold coins (Green et al. 2021, 2022): truth = published XRF; mean +/- sd over {reps} simulated runs")
    for p in (40.0, 18.0):
        pf = PeakFitQuantifier()
        calibrate(an, pf, p, np.random.default_rng(99))
        for name, comp, pub, src in GOLD_COINS:
            wt = {k: v / 100 for k, v in comp.items()}
            nn_r, b_r, sig_r = [], [], []
            for _ in range(reps):
                le, he, frac, info = measure(homog(wt, 1000.0), p, 3e5, rng)
                nn, s, _ = nn_trio(an, le, he, p)
                b, _ = pf.quantify(le)
                nn_r.append([nn[e] for e in TRIO]); b_r.append([b[e] for e in TRIO]); sig_r.append([s[e] for e in TRIO])
            nn_r, b_r = 100 * np.array(nn_r), 100 * np.array(b_r)
            truth = trio_norm(wt)
            row = dict(coin=name, momentum=p, truth={e: 100 * truth[e] for e in TRIO}, published_muXES=pub,
                       nn_mean=dict(zip(TRIO, nn_r.mean(0).round(2))), nn_sd=dict(zip(TRIO, nn_r.std(0).round(2))),
                       nn_sigma=dict(zip(TRIO, (100 * np.array(sig_r).mean(0)).round(2))),
                       pf_mean=dict(zip(TRIO, b_r.mean(0).round(2))), pf_sd=dict(zip(TRIO, b_r.std(0).round(2))))
            rows.append(row)
            print(f"  {p:4.0f} MeV/c {name[:32]:<32} true Au/Ag/Cu {truth['Au'] * 100:6.2f}/{truth['Ag'] * 100:5.2f}/{truth['Cu'] * 100:5.2f}"
                  f" | NN {nn_r[:, 0].mean():6.2f}+/-{nn_r[:, 0].std():.2f} / {nn_r[:, 1].mean():5.2f}+/-{nn_r[:, 1].std():.2f}"
                  f" / {nn_r[:, 2].mean():5.2f}+/-{nn_r[:, 2].std():.2f}"
                  f" | peak-fit {b_r[:, 0].mean():6.2f}+/-{b_r[:, 0].std():.2f} / {b_r[:, 1].mean():5.2f}+/-{b_r[:, 1].std():.2f}"
                  f" / {b_r[:, 2].mean():5.2f}+/-{b_r[:, 2].std():.2f} | published: {pub}")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    r40 = [r for r in rows if r["momentum"] == 40.0]
    x = np.arange(len(r40))
    for k, (el, axx) in enumerate(zip(["Ag", "Cu"], ax)):
        axx.bar(x - 0.27, [r["truth"][el] for r in r40], 0.27, label="truth (XRF)", color="0.6")
        axx.bar(x, [r["nn_mean"][el] for r in r40], 0.27, yerr=[r["nn_sd"][el] for r in r40], label="neural net", color="C1")
        axx.bar(x + 0.27, [r["pf_mean"][el] for r in r40], 0.27, yerr=[r["pf_sd"][el] for r in r40], label="peak fit", color="C0")
        axx.set_xticks(x); axx.set_xticklabels([r["coin"].split(" (")[0] for r in r40], rotation=25, fontsize=8)
        axx.set_ylabel(f"{el} wt%"); axx.set_title(f"{el} in Roman gold coins, 40 MeV/c, 3e5 counts"); axx.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(OUT / "B_gold_coins.png", dpi=110); plt.close()
    return rows


# ---------------------------------------------------------------------------
def domna_layers(xs, xb, lam, thickness=1500.0):
    """Surface-enriched Ag-Cu coin, exponential in ATOMIC Ag fraction
    (Hampshire et al. report atomic composition)."""
    edges = [0.0]
    step = max(lam / 12, 0.5)
    while edges[-1] < 6 * lam:
        edges.append(edges[-1] + step); step *= 1.15
    layers = []
    for z0, z1 in zip(edges[:-1], edges[1:]):
        x = xb + (xs - xb) * np.exp(-0.5 * (z0 + z1) / lam)
        layers.append(Layer(z1 - z0, at_to_wt({"Ag": x, "Cu": 1 - x})))
    layers.append(Layer(thickness - edges[-1], at_to_wt({"Ag": xb, "Cu": 1 - xb})))
    return ObjectModel("denarius", "surface_enriched", layers)


_DEPTHS = {}


def _stop_depths(p, bulk_wt):
    key = (float(p), tuple(sorted(bulk_wt.items())))
    if key not in _DEPTHS:
        _DEPTHS[key] = object_stop_depths(float(p), bulk_wt)
    return _DEPTHS[key]


def model_at_frac(xs, xb, lam, momenta, bulk_wt=None):
    """Smooth forward model: muon-weighted Ag fraction (atomic) for an
    exponential atomic-fraction profile, averaged over the exact per-muon
    stopping depths (wt fractions are averaged, as in the measurement)."""
    bulk_wt = bulk_wt or at_to_wt({"Ag": 0.49, "Cu": 0.51})
    wAg, wCu = 107.868, 63.546
    out = []
    for p in momenta:
        z = _stop_depths(p, bulk_wt)
        x = xb + (xs - xb) * np.exp(-z / lam)
        w = x * wAg / (x * wAg + (1 - x) * wCu)          # wt fraction of Ag at each stop
        wm = w.mean()
        out.append((wm / wAg) / (wm / wAg + (1 - wm) / wCu))
    return np.array(out)


def _clean(meas, sig):
    meas = np.clip(np.nan_to_num(meas, nan=0.5), 0.0, 1.0)
    sig = np.where(np.isfinite(sig) & (sig > 0), sig, 0.05)
    return meas, sig


def invert_forward(meas, sig, momenta):
    meas, sig = _clean(meas, sig)

    def r(th):
        xs, xb, loglam = th
        return (model_at_frac(xs, xb, np.exp(loglam), momenta) - meas) / sig
    sol = least_squares(r, [0.95, 0.5, np.log(40.0)], bounds=([0.3, 0.05, np.log(3)], [1.0, 0.95, np.log(400)]))
    return sol.x[0], sol.x[1], float(np.exp(sol.x[2]))


def invert_naive(meas, sig, depths):
    meas, sig = _clean(meas, sig)

    def r(th):
        xs, xb, loglam = th
        return (xb + (xs - xb) * np.exp(-depths / np.exp(loglam)) - meas) / sig
    sol = least_squares(r, [0.95, 0.5, np.log(40.0)], bounds=([0.3, 0.05, np.log(3)], [1.0, 0.95, np.log(400)]))
    return sol.x[0], sol.x[1], float(np.exp(sol.x[2]))


def layer_thickness(xs, xb, lam, tol=0.01):
    return lam * np.log(max((xs - xb) / tol, 1.0001))


def part_c(an, reps=12, seed=13):
    rng = np.random.default_rng(seed)
    xs_t, xb_t = 1.0, JULIA_DOMNA["bulk_at_frac_Ag"]
    d2, x2, _ = JULIA_DOMNA["surface_points"][1]
    lam_t = d2 / np.log((xs_t - xb_t) / (x2 - xb_t))          # 49.7 um from the published points
    truth_t = layer_thickness(xs_t, xb_t, lam_t)
    obj = domna_layers(xs_t, xb_t, lam_t)
    pf = PeakFitQuantifier()
    sets = {"published momenta": JULIA_DOMNA["momenta_MeVc"],
            "extended momenta": [17, 18, 19, 20, 21, 22, 24, 26, 29, 35, 45]}
    bulk_wt = at_to_wt({"Ag": xb_t, "Cu": 1 - xb_t})
    summary = {}
    print(f"\n[C] Julia Domna replica: truth x_surface={xs_t:.2f}, x_bulk={xb_t:.2f} (atomic Ag), "
          f"lambda={lam_t:.1f} um -> enriched layer {truth_t:.0f} um (published ~190 um)")
    for label, momenta in sets.items():
        momenta = np.array(momenta, dtype=float)
        calib = {}
        for p in momenta:
            pf_p = PeakFitQuantifier()
            calibrate(an, pf_p, p, np.random.default_rng(99))
            calib[p] = (pf_p, np.array(an.correction))
        depths = np.array([mean_depth_um(p, bulk_wt) for p in momenta])
        # truth for the plotted points: the full layered simulation
        true_pts = []
        for p in momenta:
            comp, *_ = sampled_composition(obj, instrument(p, 1e5))
            true_pts.append(wt_to_at({"Ag": comp.get("Ag", 0), "Cu": comp.get("Cu", 0)})["Ag"])
        true_pts = np.array(true_pts)
        print(f"    forward-model check (smooth vs layered sim): max |diff| "
              f"{np.abs(model_at_frac(xs_t, xb_t, lam_t, momenta) - true_pts).max() * 100:.2f} at%")
        est = {k: [] for k in ("nn_forward", "nn_naive", "pf_forward", "pf_naive")}
        pts_nn, pts_pf = [], []
        for _ in range(reps):
            m_nn, s_nn, m_pf, s_pf = [], [], [], []
            for p in momenta:
                pf_p, corr = calib[p]
                an.correction = corr
                le, he, _, _ = measure(obj, p, 2e5, rng)
                nn, s, _ = nn_trio(an, le, he, p)
                at = wt_to_at({"Ag": nn["Ag"], "Cu": nn["Cu"]})["Ag"]
                m_nn.append(at); s_nn.append(max(s["Ag"], 0.005))
                b, bs = pf_p.quantify(le)
                m_pf.append(wt_to_at({"Ag": b["Ag"], "Cu": b["Cu"]})["Ag"]); s_pf.append(max(bs["Ag"], 0.005))
            m_nn, s_nn, m_pf, s_pf = map(np.array, (m_nn, s_nn, m_pf, s_pf))
            pts_nn.append(m_nn); pts_pf.append(m_pf)
            est["nn_forward"].append(invert_forward(m_nn, s_nn, momenta))
            est["nn_naive"].append(invert_naive(m_nn, s_nn, depths))
            est["pf_forward"].append(invert_forward(m_pf, s_pf, momenta))
            est["pf_naive"].append(invert_naive(m_pf, s_pf, depths))
        res = {}
        for k, v in est.items():
            v = np.array(v)
            thick = np.array([layer_thickness(*t) for t in v])
            res[k] = dict(x_bulk=(float(v[:, 1].mean()), float(v[:, 1].std())),
                          x_surface=(float(v[:, 0].mean()), float(v[:, 0].std())),
                          lambda_um=(float(v[:, 2].mean()), float(v[:, 2].std())),
                          layer_um=(float(thick.mean()), float(thick.std())))
        summary[label] = dict(momenta=momenta.tolist(), depths_um=depths.round(1).tolist(), true_points=true_pts.tolist(),
                              nn_points=np.mean(pts_nn, 0).tolist(), pf_points=np.mean(pts_pf, 0).tolist(), fits=res)
        print(f"  {label}: {momenta.tolist()}")
        for k, r in res.items():
            print(f"    {k:>11}: bulk Ag {100 * r['x_bulk'][0]:5.1f}+/-{100 * r['x_bulk'][1]:.1f} at% "
                  f"(true {100 * xb_t:.0f}) | lambda {r['lambda_um'][0]:6.1f}+/-{r['lambda_um'][1]:.1f} um (true {lam_t:.1f})"
                  f" | layer {r['layer_um'][0]:6.0f}+/-{r['layer_um'][1]:.0f} um (true {truth_t:.0f})")
        if label == "published momenta":
            zz = np.linspace(0, 450, 300)
            fig, ax = plt.subplots(figsize=(8, 4.6))
            ax.plot(zz, 100 * (xb_t + (xs_t - xb_t) * np.exp(-zz / lam_t)), "k-", lw=2, label="true profile")
            for k, c, ls in (("nn_forward", "C1", "-"), ("nn_naive", "C1", ":"), ("pf_naive", "C0", ":")):
                xs, xb, lam = (res[k]["x_surface"][0], res[k]["x_bulk"][0], res[k]["lambda_um"][0])
                ax.plot(zz, 100 * (xb + (xs - xb) * np.exp(-zz / lam)), color=c, ls=ls, label=k.replace("_", ": "))
            ax.errorbar(depths, 100 * np.mean(pts_nn, 0), yerr=100 * np.std(pts_nn, 0), fmt="o", color="C1", label="NN estimates at mean depth")
            ax.errorbar(depths, 100 * np.mean(pts_pf, 0), yerr=100 * np.std(pts_pf, 0), fmt="s", mfc="none", color="C0", label="peak-fit estimates at mean depth")
            pub = JULIA_DOMNA["surface_points"] + [(JULIA_DOMNA["bulk_depth_um"], xb_t, JULIA_DOMNA["bulk_at_frac_Ag_err"])]
            ax.errorbar([d for d, *_ in pub], [100 * x for _, x, _ in pub], yerr=[100 * e for *_, e in pub], fmt="kD",
                        mfc="none", label="Hampshire et al. 2019 (published)")
            ax.set_xlabel("depth (um)"); ax.set_ylabel("Ag (at%)"); ax.set_title("Julia Domna denarius: surface-enrichment profile")
            ax.legend(fontsize=7); plt.tight_layout(); plt.savefig(OUT / "C_julia_domna_profile.png", dpi=110); plt.close()
    return dict(truth=dict(x_surface=xs_t, x_bulk=xb_t, lambda_um=lam_t, layer_um=truth_t), sets=summary)


OUT = Path("../results")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="../checkpoints/best.pt")
    ap.add_argument("--out", default="../results")
    ap.add_argument("--parts", default="ABC")
    ap.add_argument("--quick", action="store_true", help="tiny sample sizes (smoke test)")
    a = ap.parse_args()
    OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
    an = Analyzer(a.checkpoint)
    results = {"published_reference": PUBLISHED_PERFORMANCE}
    if "A" in a.parts:
        results["A"] = part_a(an, n=10 if a.quick else 240)
    if "B" in a.parts:
        results["B"] = part_b(an, reps=2 if a.quick else 20)
    if "C" in a.parts:
        results["C"] = part_c(an, reps=1 if a.quick else 12)
    json.dump(results, open(OUT / "case_studies.json", "w"), indent=1, default=float)
    print(f"\nwritten {OUT / 'case_studies.json'}")
