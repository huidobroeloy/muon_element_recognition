"""
validate_physics.py
===================
Figures + JSON documenting how the physics layer compares with measurements:
  * transition energies vs literature (anchor and independent lines)
  * implantation depth vs momentum vs the published SRIM depths

    python validate_physics.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from literature import DEPTH_BENCHMARKS
from physics import validation_table
from transport import mean_depth_um

OUT = Path("../results")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = validation_table()
    data = [dict(element=s, transition=t, role=r, measured_keV=em, sigma_keV=sg, dirac_vp_keV=er,
                 calibrated_keV=ec, residual_keV=ec - em, source=src) for s, t, r, em, sg, er, ec, src in rows]
    val = [d for d in data if d["role"] == "valid"]
    print(f"{len(val)} independent validation lines: median |residual| "
          f"{np.median([abs(d['residual_keV']) for d in val]):.3f} keV, max {max(abs(d['residual_keV']) for d in val):.3f} keV; "
          f"median relative {np.median([abs(d['residual_keV']) / d['measured_keV'] for d in val]) * 100:.4f} %")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for d in data:
        c = "C3" if d["role"] == "anchor" else "C0"
        ax[0].errorbar(d["measured_keV"], d["residual_keV"], yerr=d["sigma_keV"], fmt="o", color=c, ms=4)
        if d["role"] == "valid":
            ax[0].annotate(f"{d['element']} {d['transition']}", (d["measured_keV"], d["residual_keV"]), fontsize=6,
                           xytext=(3, 3), textcoords="offset points")
    ax[0].axhline(0, color="k", lw=0.7)
    ax[0].set_xscale("log"); ax[0].set_xlabel("measured energy (keV)"); ax[0].set_ylabel("model - measured (keV)")
    ax[0].set_title("Muonic transition energies (red: 1s anchors, blue: independent)")

    ps = np.linspace(16, 60, 100)
    for wt, lab in (({"Au": 1.0}, "gold"), ({"Ag": 0.62, "Cu": 0.38}, "Ag62Cu38 (Julia Domna bulk)")):
        ax[1].plot(ps, [mean_depth_um(p, wt) for p in ps], label=lab)
    for wt, p, dep, sig, src in DEPTH_BENCHMARKS:
        ax[1].errorbar(p, dep, yerr=sig, fmt="kD", mfc="none")
        ax[1].annotate(src.split(":")[0].split("(")[0], (p, dep), fontsize=6, xytext=(4, -10), textcoords="offset points")
    ax[1].set_yscale("log"); ax[1].set_xlabel("muon momentum (MeV/c)"); ax[1].set_ylabel("mean implantation depth (um)")
    ax[1].set_title("Depth model vs published SRIM depths (after 100 mm air + 16 um Al)"); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(OUT / "physics_validation.png", dpi=110); plt.close()
    json.dump(data, open(OUT / "physics_validation.json", "w"), indent=1)
    print("written", OUT / "physics_validation.png")


if __name__ == "__main__":
    main()
