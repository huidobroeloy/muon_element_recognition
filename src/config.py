"""
config.py
=========
Element table, nuclear/stopping parameters, detector binning and global
constants shared by the physics, simulation and learning code.

Element rows carry everything the physics needs:
  symbol, Z, standard atomic weight, mass number of the isotope used for the
  muonic-level calculation, nuclear rms charge radius (fm), and the mean
  excitation energy I (eV) used in the Bethe stopping-power formula.

Sources
-------
* rms charge radii: I. Angeli & K. P. Marinova, "Table of experimental nuclear
  ground state charge radii: An update", At. Data Nucl. Data Tables 99 (2013)
  69-95 (values transcribed for the most abundant isotope; Pd-106 from
  Saito et al., arXiv:2204.03233). Where a value is not listed we fall back on
  r_rms = 0.836 A^(1/3) + 0.570 fm (the same paper's empirical fit).
* Mean excitation energies: ICRU Report 37/49 values (as used by NIST
  ESTAR/PSTAR); elements without a tabulated value use I ~ 10 Z eV.

Only elements with Z >= 11 are *targets* of the classifier. C, N and O are
simulated (air gap, organics, corrosion oxides) but not predicted, because in
real ISIS spectra their muonic lines are dominated by muons stopping in the
air in front of the sample (Green et al. 2021, Fig. 5a).
"""

from __future__ import annotations

# (symbol, Z, atomic weight, isotope A, r_rms [fm], I [eV])
ELEMENT_TABLE = [
    ("C",  6, 12.011,  12, 2.4702,  78.0),
    ("N",  7, 14.007,  14, 2.5582,  82.0),
    ("O",  8, 15.999,  16, 2.6991,  95.0),
    ("Na", 11, 22.990, 23, 2.9936, 149.0),
    ("Mg", 12, 24.305, 24, 3.0570, 156.0),
    ("Al", 13, 26.982, 27, 3.0610, 166.0),
    ("Si", 14, 28.085, 28, 3.1224, 173.0),
    ("P",  15, 30.974, 31, 3.1889, 173.0),
    ("S",  16, 32.06,  32, 3.2611, 180.0),
    ("Cl", 17, 35.45,  35, 3.3654, 174.0),
    ("K",  19, 39.098, 39, 3.4349, 190.0),
    ("Ca", 20, 40.078, 40, 3.4776, 191.0),
    ("Ti", 22, 47.867, 48, 3.5921, 233.0),
    ("Cr", 24, 51.996, 52, 3.6452, 257.0),
    ("Mn", 25, 54.938, 55, 3.7057, 272.0),
    ("Fe", 26, 55.845, 56, 3.7377, 286.0),
    ("Co", 27, 58.933, 59, 3.7875, 297.0),
    ("Ni", 28, 58.693, 58, 3.7757, 311.0),
    ("Cu", 29, 63.546, 63, 3.8823, 322.0),
    ("Zn", 30, 65.38,  64, 3.9283, 330.0),
    ("As", 33, 74.922, 75, 4.0968, 347.0),
    ("Se", 34, 78.971, 80, 4.1400, 348.0),
    ("Pd", 46, 106.42, 106, 4.5070, 470.0),   # validation anchor only
    ("Ag", 47, 107.868, 107, 4.5454, 470.0),
    ("Cd", 48, 112.414, 114, 4.6120, 469.0),
    ("Sn", 50, 118.710, 120, 4.6519, 488.0),
    ("Sb", 51, 121.760, 121, 4.6802, 487.0),
    ("Ba", 56, 137.327, 138, 4.8378, 491.0),
    ("Nd", 60, 144.242, 144, 4.9254, 591.0),  # validation anchor only
    ("Pt", 78, 195.084, 195, 5.4236, 790.0),
    ("Au", 79, 196.967, 197, 5.4371, 790.0),
    ("Hg", 80, 200.592, 202, 5.4648, 800.0),
    ("Pb", 82, 207.2,  208, 5.5012, 823.0),
    ("Bi", 83, 208.980, 209, 5.5211, 823.0),
]

SYM = [row[0] for row in ELEMENT_TABLE]
Z_OF = {row[0]: row[1] for row in ELEMENT_TABLE}
W_OF = {row[0]: row[2] for row in ELEMENT_TABLE}      # atomic weight
A_OF = {row[0]: row[3] for row in ELEMENT_TABLE}      # isotope mass number
RMS_OF = {row[0]: row[4] for row in ELEMENT_TABLE}
I_OF = {row[0]: row[5] for row in ELEMENT_TABLE}

# Elements the simulator can put into an object (everything except the
# anchor-only rows) and the subset the network predicts.
ANCHOR_ONLY = {"Pd", "Nd"}
SIM_ELEMENTS = [s for s in SYM if s not in ANCHOR_ONLY]
LIGHT_CONTEXT = ["C", "N", "O"]
TARGETS = [s for s in SIM_ELEMENTS if s not in LIGHT_CONTEXT]
NUM_TARGETS = len(TARGETS)
TARGET_INDEX = {s: i for i, s in enumerate(TARGETS)}

# ---------------------------------------------------------------------------
# Detector binning. The RIKEN-RAL Port 4 set-up (Hampshire et al. 2019; Green
# et al. 2021) pairs a low-energy high-resolution HPGe with a high-energy
# coaxial HPGe on each side of the sample. We model one of each.
# ---------------------------------------------------------------------------
LE_EMIN, LE_EMAX, LE_BIN = 20.0, 1000.0, 0.5      # keV
HE_EMIN, HE_EMAX, HE_BIN = 100.0, 8000.0, 2.0     # keV
LE_NBINS = int(round((LE_EMAX - LE_EMIN) / LE_BIN))   # 1960
HE_NBINS = int(round((HE_EMAX - HE_EMIN) / HE_BIN))   # 3950

# Ground-truth labelling (wt% inside the muon-sampled volume)
PRESENT_WT = 0.01     # >= 1 wt%  -> present (Green et al. 2021 detection limit)
ABSENT_WT = 0.002     # <  0.2 wt% -> absent; in between is masked in the loss

# ISIS beam: 10 % FWHM momentum bite (Green et al. 2021)
MOMENTUM_FWHM_FRAC = 0.10
