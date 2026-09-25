"""
literature.py
=============
Every number in this project that comes from a publication lives here, with
its source, so that calibration, validation and the Roman-coin benchmarks
are traceable.

Transition labels use spectroscopic (n, l, j) notation, e.g. "4f7/2-3d5/2".
X-ray notation used in some papers maps as K1=1s1/2, L2=2p1/2, L3=2p3/2,
M4=3d3/2, M5=3d5/2, N6=4f5/2, N7=4f7/2, O8=5g7/2, O9=5g9/2.
"""

# ---------------------------------------------------------------------------
# 1. Measured muonic transition energies (keV)
#    role = "anchor": used to calibrate the residual 1s shift (nuclear
#           polarisation + higher-order QED that the Dirac+Uehling model lacks)
#    role = "valid" : never used in any fit, only to validate
# ---------------------------------------------------------------------------
MEASURED_LINES = [
    # symbol, transition, E_keV, sigma_keV, role, source
    ("Ti", "2p3/2-1s1/2", 932.4, 1.0, "anchor", "ALCAP, natural Ti (value carried over from v1 of this repo)"),
    ("Fe", "2p3/2-1s1/2", 1257.19, 1.0, "anchor", "ENSDF 56Fe muonic X-rays (2006Me08)"),
    ("Pd", "2p3/2-1s1/2", 3084.0, 0.3, "anchor", "Saito et al., arXiv:2204.03233, 106Pd"),
    ("Nd", "2p3/2-1s1/2", 4336.38, 1.0, "anchor", "144Nd, 70MB1 (via MuDirac validation set)"),
    ("Au", "2p3/2-1s1/2", 5763.1, 2.0, "anchor", "ENSDF muonic 197Au (2007Me09); lit. spread 5762.5-5764.9"),
    ("Pb", "2p3/2-1s1/2", 5963.3, 0.5, "anchor", "Sturniolo & Hillier, X-Ray Spectrom. 50 (2021) Table 2"),
    # --- independent validation lines -------------------------------------
    ("Mg", "3d3/2-2p1/2", 56.392, 0.01, "valid", "MuDirac Table 2 (exp.)"),
    ("Mg", "3d5/2-2p3/2", 56.216, 0.01, "valid", "MuDirac Table 2 (exp.)"),
    ("Pd", "2p1/2-1s1/2", 3050.3, 0.5, "valid", "Saito et al. 2022, 106Pd"),
    ("Pd", "3d5/2-2p3/2", 833.4, 0.3, "valid", "Saito et al. 2022, 106Pd"),
    ("Pd", "3d3/2-2p1/2", 863.8, 0.3, "valid", "Saito et al. 2022, 106Pd"),
    ("Pd", "4d5/2-2p3/2", 1124.8, 0.4, "valid", "Saito et al. 2022, 106Pd"),
    ("Pd", "4f7/2-3d5/2", 291.7, 0.2, "valid", "Saito et al. 2022, 106Pd"),
    ("Pd", "4f5/2-3d3/2", 295.1, 0.3, "valid", "Saito et al. 2022, 106Pd"),
    ("Sn", "4f5/2-3d3/2", 349.975, 0.01, "valid", "MuDirac Table 2 (exp.)"),
    ("Sn", "4f7/2-3d5/2", 345.254, 0.01, "valid", "MuDirac Table 2 (exp.)"),
    ("Pb", "5g7/2-4f5/2", 437.749, 0.01, "valid", "MuDirac Table 2 (exp.)"),
    ("Pb", "5g9/2-4f7/2", 431.328, 0.01, "valid", "MuDirac Table 2 (exp.)"),
    ("Pb", "2p1/2-1s1/2", 5778.9, 0.5, "valid", "MuDirac Table 2 (exp.)"),
    ("Au", "2p1/2-1s1/2", 5594.0, 2.0, "valid", "lit. spread 5592.8-5595.0 keV"),
    # Lines actually used for Roman-coin quantification at ISIS
    ("Ag", "4f7/2-3d5/2", 304.5, 0.3, "valid", "Hampshire et al., Heritage 2 (2019) 400"),
    ("Ag", "4f5/2-3d3/2", 308.2, 0.3, "valid", "Hampshire et al. 2019"),
    ("Cu", "3d5/2-2p3/2", 330.6, 0.3, "valid", "Hampshire et al. 2019"),
    ("Cu", "3d3/2-2p1/2", 336.4, 0.3, "valid", "Hampshire et al. 2019"),
    ("Al", "2p3/2-1s1/2", 346.8, 1.0, "valid", "Green et al., J. Archaeol. Sci. 134 (2021): 'At 346 keV ... aluminium'"),
]

# ---------------------------------------------------------------------------
# 2. Nuclear-capture gamma rays seen in coin spectra (keV). After mu- capture
#    the (Z-1) daughter is left excited; Hampshire et al. 2019 (Fig. 4) see Ni
#    lines from Cu and Pd lines from Ag; Sturniolo & Hillier 2021 (Fig. 6) see
#    196Pt lines from Au. Energies are standard nuclear level differences.
#    Yields per stopped muon are rough (order 1-10 %) and are randomised.
# ---------------------------------------------------------------------------
CAPTURE_GAMMAS = {
    "Cu": [(1172.9, 0.06), (1345.8, 0.03)],                 # 62Ni, 64Ni 2+ -> 0+
    "Ag": [(511.9, 0.04), (433.9, 0.03)],                   # 106Pd, 108Pd 2+ -> 0+
    "Au": [(355.7, 0.08), (333.0, 0.03), (521.2, 0.03)],    # 196Pt 2+1->0+, 2+2->2+1, 4+->2+
    "Si": [(843.8, 0.05), (1014.5, 0.04)],                  # 27Al
    "Al": [(1808.7, 0.05)],                                 # 26Mg
    # Capture-neutron inelastic scattering on the sample itself
    "Fe": [(846.8, 0.03)],                                  # 56Fe(n,n') 2+ -> 0+
    "Pb": [(569.7, 0.02), (803.1, 0.02)],                   # 207Pb, 206Pb (n,n')
}

# Environmental / beam-related background lines (keV)
BACKGROUND_LINES = [
    (511.0, "e+e- annihilation"),
    (595.8, "74Ge(n,n') -- asymmetric"),
    (691.3, "72Ge(n,n') -- asymmetric"),
    (1460.8, "40K"),
    (2223.2, "p(n,g)d"),
    (2614.5, "208Tl"),
]

# Electronic K X-rays emitted after the muon has reached 1s: the electrons see
# charge Z-1, so element Z emits the X-rays of element Z-1 (keV: Ka2, Ka1, Kb1)
ELECTRONIC_KX = {
    "Ag": (21.02, 21.18, 23.82),   # Pd-like
    "Cd": (21.99, 22.16, 24.94),   # Ag-like
    "Sn": (24.00, 24.21, 27.28),   # In-like
    "Sb": (25.04, 25.27, 28.49),   # Sn-like
    "Ba": (30.63, 30.97, 34.99),   # Cs-like
    "Pt": (63.00, 65.12, 73.36),   # Ir-like
    "Au": (65.12, 66.83, 75.75),   # Pt-like
    "Hg": (66.99, 68.80, 77.98),   # Au-like
    "Pb": (70.83, 72.87, 82.58),   # Tl-like
    "Bi": (72.80, 74.97, 84.94),   # Pb-like
}

# ---------------------------------------------------------------------------
# 3. Muon implantation depths reported for Roman coins at RIKEN-RAL
# ---------------------------------------------------------------------------
DEPTH_BENCHMARKS = [
    # material wt fractions, momentum MeV/c, depth um, sigma um, source
    ({"Au": 1.0}, 18.0, 10.0, 5.0, "Green et al. 2021: ~0.01 mm in gold coins (SRIM)"),
    ({"Au": 1.0}, 40.0, 400.0, 60.0, "Green et al. 2021; Green, Ishida & Hillier 2022: ~0.4 mm"),
    # Julia Domna denarius, bulk 49 at% Ag / 51 at% Cu -> 62 wt% Ag
    ({"Ag": 0.62, "Cu": 0.38}, 35.0, 402.0, 61.0, "Hampshire et al. 2019 (SRIM)"),
]

# ---------------------------------------------------------------------------
# 4. Roman coin compositions (wt%) used as benchmark scenarios
# ---------------------------------------------------------------------------
GOLD_COINS = [
    # name, XRF surface composition (wt%), muXES result, source
    ("Tiberius aureus (AD 14-37)", {"Au": 99.73, "Ag": 0.27}, "Au >99, Ag <1",
     "Green et al., J. Archaeol. Sci. 134 (2021) 105470, Table 1"),
    ("Hadrian aureus (AD 134-138)", {"Au": 99.55, "Ag": 0.45}, "Au >99, Ag <1", "Green et al. 2021"),
    ("Julian II solidus (AD 361-363)", {"Au": 95.58, "Ag": 4.18, "Cu": 0.24}, "Au 96(1), Ag 4(1)",
     "Green et al. 2021"),
    ("Galba aureus #11 (AD 68/9)", {"Au": 89.3, "Ag": 4.9, "Cu": 5.8}, "Au 89.4(0.8)",
     "Green, Ishida & Hillier, Archaeol. Anthropol. Sci. 14 (2022) 165"),
    ("Otho aureus #16 (AD 69)", {"Au": 83.7, "Ag": 10.9, "Cu": 5.4}, "Au 84.1(1.2)", "Green et al. 2022"),
    ("Vespasian aureus #24 (AD 69-79)", {"Au": 93.9, "Ag": 4.0, "Cu": 2.1}, "Au 94.0(1.1)", "Green et al. 2022"),
]
# Cu in the 2022 coins is the balance of Au+Ag, as quoted in that paper's summary.

# Julia Domna denarius (Hampshire et al. 2019): atomic composition vs depth
JULIA_DOMNA = {
    "momenta_MeVc": [17.0, 18.0, 19.0, 20.0, 22.0, 35.0],
    "bulk_at_frac_Ag": 0.49, "bulk_at_frac_Ag_err": 0.019,
    "bulk_depth_um": 402.0,
    "surface_points": [(1.8, 1.00, 0.075), (54.6, 0.66, 0.027)],  # (depth um, at-frac Ag, err)
    "enriched_layer_um": 190.0,
    "source": "Hampshire et al., Heritage 2 (2019) 400-407",
}

# Certified standards used at ISIS (Micro-Analysis Consultants), wt%
MAC_STANDARDS = {
    "MAC1": {"Au": 94.0, "Ag": 4.5, "Cu": 1.0},
    "MAC2": {"Au": 75.0, "Ag": 19.0, "Cu": 5.0},
    "MAC3": {"Au": 59.0, "Ag": 30.0, "Cu": 9.0},
}

# Published muXES performance to beat
PUBLISHED_PERFORMANCE = {
    "detection_limit_wt": 1.0,          # "minimum composition detectable was approximately 1%"
    "gold_coin_Ag_uncertainty_wt": 1.0,  # Julian II: Ag 4(1) wt%
    "denarius_bulk_uncertainty_at": 1.9,
}
