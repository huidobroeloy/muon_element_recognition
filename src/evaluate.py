"""
evaluate.py
===========
Held-out test-set evaluation of a trained checkpoint:
  * per-element precision / recall / F1 (tuned thresholds), micro/macro F1
  * composition error (TVD, MAE on major elements) and uncertainty calibration
  * detection-limit curves: recall vs true concentration
  * per coin-family breakdown
  * figures: per-element F1, detection limits, parity plots, example spectra

    python evaluate.py --checkpoint ../checkpoints/best.pt --test ../data/test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import TARGETS
from dataset import FAMILY_NAMES, SpectrumDataset
from inference import Analyzer
from simulate import HE_GRID, LE_GRID
from train import metrics, predict

KEY = ["Au", "Ag", "Cu", "Pb", "Sn", "Zn", "Fe", "Hg", "Bi", "Pt"]
BINS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 1.0001]


def detection_curves(prob, frac, thr):
    out = {}
    for el in KEY:
        j = TARGETS.index(el)
        rows = []
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            m = (frac[:, j] >= lo) & (frac[:, j] < hi)
            if m.sum() >= 8:
                rows.append((lo, hi, int(m.sum()), float((prob[m, j] >= thr[j]).mean())))
        absent = frac[:, j] == 0
        out[el] = dict(bins=rows, false_positive_rate=float((prob[absent, j] >= thr[j]).mean()) if absent.any() else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="../checkpoints/best.pt")
    ap.add_argument("--test", default="../data/test")
    ap.add_argument("--out", default="../results")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    an = Analyzer(a.checkpoint)
    ds = SpectrumDataset(a.test, train=False)
    prob, comp, sig, frac, present, mask = predict(an.model, DataLoader(ds, batch_size=128, num_workers=2))
    thr = an.thresholds
    m = metrics(prob, comp, sig, frac, present, mask, thr)
    fam = ds.meta[:, 2].astype(int)

    print(f"TEST  macro-F1 {m['macro_f1']:.4f}  micro-F1 {m['micro_f1']:.4f}  exact-set {m['exact']:.3f}  "
          f"TVD {m['tvd']:.4f}  MAE(major>=5wt%) {m['mae_major_wt']:.2f} wt%  1-sigma coverage {m['sigma_coverage_1s']:.2f}")
    print(f"\n{'el':>3} {'F1':>6} {'prec':>6} {'rec':>6} {'n+':>5}")
    for el, r in m["per_element"].items():
        print(f"{el:>3} {r['f1']:6.3f} {r['precision']:6.3f} {r['recall']:6.3f} {r['support']:5d}")

    fam_rows = {}
    for k, name in enumerate(FAMILY_NAMES):
        sel = fam == k
        if sel.sum() < 20:
            continue
        mk = metrics(prob[sel], comp[sel], sig[sel], frac[sel], present[sel], mask[sel], thr)
        fam_rows[name] = dict(n=int(sel.sum()), micro_f1=mk["micro_f1"], exact=mk["exact"], tvd=mk["tvd"],
                              mae_major_wt=mk["mae_major_wt"])
    print(f"\n{'family':>22} {'n':>5} {'microF1':>8} {'exact':>6} {'TVD':>7} {'MAEmaj':>7}")
    for k, r in fam_rows.items():
        print(f"{k:>22} {r['n']:5d} {r['micro_f1']:8.3f} {r['exact']:6.3f} {r['tvd']:7.4f} {r['mae_major_wt']:7.2f}")

    det = detection_curves(prob, frac, thr)
    print("\nDetection probability vs true concentration (wt%):")
    for el, d in det.items():
        s = "  ".join(f"[{100 * lo:.1f}-{100 * hi:.1f}):{r:.2f}" for lo, hi, n, r in d["bins"])
        print(f"  {el:>2} FPR={d['false_positive_rate']:.3f}  {s}")

    # ------------------------------------------------------------- figures
    fig, ax = plt.subplots(figsize=(11, 3.4))
    f1 = [m["per_element"][e]["f1"] for e in TARGETS]
    ax.bar(TARGETS, f1, color="#3b6fb6")
    ax.set_ylim(0, 1.02); ax.set_ylabel("F1 (test)"); ax.set_title("Element presence (>= 1 wt%), held-out test set")
    plt.tight_layout(); plt.savefig(out / "per_element_f1.png", dpi=110); plt.close()

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for el in ["Au", "Ag", "Cu", "Pb", "Sn", "Zn"]:
        rows = det[el]["bins"]
        if rows:
            x = [100 * np.sqrt(lo * hi) for lo, hi, _, _ in rows]
            ax.plot(x, [r for *_, r in rows], "o-", label=el)
    ax.axvline(1.0, color="k", ls="--", lw=0.8)
    ax.text(1.05, 0.05, "published muXES\ndetection limit ~1 wt%", fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("true concentration in sampled volume (wt%)"); ax.set_ylabel("detection probability")
    ax.legend(); ax.set_title("Detection limit")
    plt.tight_layout(); plt.savefig(out / "detection_limit.png", dpi=110); plt.close()

    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    for axx, el in zip(axs, ["Au", "Ag", "Cu"]):
        j = TARGETS.index(el)
        sel = frac[:, j] > 0.005
        axx.errorbar(100 * frac[sel, j], 100 * comp[sel, j], yerr=100 * sig[sel, j], fmt=".", ms=3, alpha=0.4, elinewidth=0.5)
        axx.plot([0, 100], [0, 100], "k--", lw=0.8)
        mae = 100 * np.abs(comp[sel, j] - frac[sel, j]).mean()
        axx.set_title(f"{el}: MAE {mae:.2f} wt% (n={sel.sum()})"); axx.set_xlabel("true wt%"); axx.set_ylabel("predicted wt%")
    plt.tight_layout(); plt.savefig(out / "parity_Au_Ag_Cu.png", dpi=110); plt.close()

    # example spectra
    rng = np.random.default_rng(5)
    idx = rng.choice(len(ds), 4, replace=False)
    fig, axs = plt.subplots(4, 2, figsize=(14, 11), gridspec_kw=dict(width_ratios=[1.4, 1]))
    for r, i in enumerate(idx):
        le, he = ds.noisy(i, np.random.default_rng(10_000_019 + int(i)))
        true = [f"{TARGETS[j]} {100 * frac[i, j]:.1f}" for j in np.argsort(-frac[i]) if frac[i, j] >= 0.005]
        pred = [f"{TARGETS[j]} {100 * comp[i, j]:.1f}+/-{100 * sig[i, j]:.1f}" for j in np.argsort(-comp[i])
                if prob[i, j] >= thr[j]]
        axs[r, 0].semilogy(LE_GRID, np.maximum(le, 0.5), lw=0.5)
        axs[r, 1].semilogy(HE_GRID, np.maximum(he, 0.5), lw=0.5, color="C1")
        axs[r, 0].set_title(f"{FAMILY_NAMES[fam[i]]} p={ds.meta[i, 0]:.1f} MeV/c | true: {', '.join(true)}", fontsize=8)
        axs[r, 1].set_title(f"pred: {', '.join(pred)}", fontsize=8)
    for axx in axs[-1]:
        axx.set_xlabel("keV")
    plt.tight_layout(); plt.savefig(out / "example_predictions.png", dpi=100); plt.close()

    json.dump(dict(test=m, families=fam_rows, detection=det), open(out / "test_metrics.json", "w"), indent=1)
    print(f"\nfigures + test_metrics.json written to {out.resolve()}")


if __name__ == "__main__":
    main()
