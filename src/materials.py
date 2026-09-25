"""
materials.py
============
Random, archaeologically grounded objects expressed as layer stacks for
transport.py. Roman coinage dominates (it is the application the ISIS /
Oxford muXES work targets); a set of broader heritage materials from v1 of
this repo is kept so the model does not assume every object is a coin.

Composition ranges (wt%) follow the published literature the muXES papers
build on:
  * aurei / solidi: Au 97-99.9 with Ag balance; AD 68/9 and late-Roman issues
    debased to 83-96 Au with Ag and (deliberately added) Cu
    (Green et al. 2021, 2022; Callu et al. 1985).
  * denarii: Ag 85-98 under the Julio-Claudians falling to 40-60 Ag in the
    Severan period, Cu balance, Pb/Au/Bi traces (Butcher & Ponting 2014),
    with a deliberately silver-enriched surface ~100-200 um deep
    (Hampshire et al. 2019).
  * 3rd-century antoniniani: 2-45 Ag billon, Sn/Pb/Zn additions.
  * fourrees: silver foil over a copper core; gilded bronzes; orichalcum
    (Cu-Zn) sestertii and dupondii; copper asses; leaded late-Roman nummi.
Corrosion and soil crusts are added as extra surface layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from transport import Layer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _draw(rng, spec: dict, balance: str, min_balance: float = 5.0) -> dict:
    """spec: element -> (lo, hi) wt%; log-uniform for small upper bounds.
    Balance element takes the remainder. Returns wt fractions."""
    for _ in range(100):
        wt = {}
        for s, (lo, hi) in spec.items():
            if hi <= 0:
                continue
            if lo <= 0 or hi / max(lo, 1e-9) > 20:
                v = 10 ** rng.uniform(np.log10(max(lo, hi * 1e-3)), np.log10(hi))
                if lo <= 0 and rng.random() < 0.3:
                    v = 0.0              # trace often absent altogether
            else:
                v = rng.uniform(lo, hi)
            if v > 0:
                wt[s] = v
        rest = 100.0 - sum(wt.values())
        if rest >= min_balance:
            wt[balance] = wt.get(balance, 0.0) + rest
            return {k: v / 100.0 for k, v in wt.items()}
    raise RuntimeError("could not satisfy composition spec")


def _norm(wt: dict) -> dict:
    t = sum(wt.values())
    return {k: v / t for k, v in wt.items() if v > 0}


@dataclass
class ObjectModel:
    family: str
    profile: str
    layers: list                       # list[Layer], object layers only, surface first
    notes: dict = field(default_factory=dict)


def _enriched_profile(bulk: dict, rich: str, lean: str, surf_frac: float,
                      decay_um: float, thickness_um: float) -> list:
    """Exponential surface enrichment of element `rich` at the expense of
    `lean` (Hampshire et al. 2019 fit an exponential to their depth profile).
    Discretised into geometrically growing layers."""
    layers = []
    edges = [0.0]
    step = max(decay_um / 12, 0.5)
    while edges[-1] < 6 * decay_um:
        edges.append(edges[-1] + step)
        step *= 1.15
    edges = np.array(edges)
    for z0, z1 in zip(edges[:-1], edges[1:]):
        zc = 0.5 * (z0 + z1)
        wt = dict(bulk)
        pair = bulk.get(rich, 0) + bulk.get(lean, 0)
        f_rich_bulk = bulk.get(rich, 0) / pair
        f_rich = f_rich_bulk + (surf_frac - f_rich_bulk) * np.exp(-zc / decay_um)
        wt[rich] = pair * f_rich
        wt[lean] = pair * (1 - f_rich)
        layers.append(Layer(z1 - z0, _norm(wt)))
    layers.append(Layer(max(thickness_um - edges[-1], 10.0), _norm(bulk)))
    return layers


# ---------------------------------------------------------------------------
# Roman coin families
# ---------------------------------------------------------------------------
def aureus_pure(rng):
    return _draw(rng, {"Ag": (0.05, 2.5), "Cu": (0.0, 0.4), "Pt": (0.0, 0.08)}, "Au")


def aureus_debased(rng):
    return _draw(rng, {"Ag": (1.5, 12.0), "Cu": (0.1, 8.0), "Pt": (0.0, 0.05)}, "Au")


def denarius_fine(rng):
    return _draw(rng, {"Cu": (1.5, 15.0), "Pb": (0.0, 1.0), "Au": (0.0, 0.8), "Bi": (0.0, 0.2),
                       "Zn": (0.0, 0.3)}, "Ag")


def denarius_debased(rng):
    return _draw(rng, {"Ag": (35.0, 80.0), "Pb": (0.0, 1.5), "Au": (0.0, 0.6), "Sn": (0.0, 1.5),
                       "Zn": (0.0, 1.0), "Bi": (0.0, 0.1)}, "Cu")


def antoninianus(rng):
    return _draw(rng, {"Ag": (2.0, 45.0), "Sn": (0.0, 5.0), "Pb": (0.0, 4.0), "Zn": (0.0, 3.0),
                       "Fe": (0.0, 0.5)}, "Cu")


def copper_core(rng):
    return _draw(rng, {"Sn": (0.0, 6.0), "Pb": (0.0, 5.0), "Zn": (0.0, 2.0), "Fe": (0.0, 0.8),
                       "As": (0.0, 0.8), "Sb": (0.0, 0.5), "Ni": (0.0, 0.3)}, "Cu")


def orichalcum(rng):
    return _draw(rng, {"Zn": (10.0, 28.0), "Sn": (0.0, 2.5), "Pb": (0.0, 2.0), "Fe": (0.0, 0.6)}, "Cu")


def copper_as(rng):
    return _draw(rng, {"Sn": (0.0, 2.0), "Pb": (0.0, 2.0), "Fe": (0.0, 0.5), "Zn": (0.0, 1.0)}, "Cu")


def nummus(rng):
    return _draw(rng, {"Sn": (1.0, 10.0), "Pb": (2.0, 25.0), "Ag": (0.0, 3.0), "Zn": (0.0, 1.0),
                       "Fe": (0.0, 0.5)}, "Cu")


def silver_foil(rng):
    return _draw(rng, {"Cu": (1.0, 10.0), "Au": (0.0, 0.8), "Pb": (0.0, 0.8)}, "Ag")


def gilding(rng):
    return _draw(rng, {"Hg": (0.0, 15.0), "Ag": (0.5, 8.0), "Cu": (0.0, 3.0)}, "Au")


# Corrosion products (wt fractions incl. light elements; H omitted)
CUPRITE = {"Cu": 0.888, "O": 0.112}
ATACAMITE = {"Cu": 0.595, "Cl": 0.166, "O": 0.239}
MALACHITE = {"Cu": 0.575, "C": 0.054, "O": 0.371}
CHLORARGYRITE = {"Ag": 0.753, "Cl": 0.247}
ACANTHITE = {"Ag": 0.871, "S": 0.129}
CERUSSITE = {"Pb": 0.775, "C": 0.045, "O": 0.180}
GOETHITE = {"Fe": 0.629, "O": 0.371}
SOIL = {"Si": 0.30, "Al": 0.08, "Fe": 0.05, "Ca": 0.05, "K": 0.02, "O": 0.50}


def _corrosion_for(base: str, rng):
    if base == "Cu":
        opts = [(CUPRITE, 2.7), (ATACAMITE, 3.8), (MALACHITE, 4.0)]
    elif base == "Ag":
        opts = [(CHLORARGYRITE, 5.6), (ACANTHITE, 7.2)]
    elif base == "Pb":
        opts = [(CERUSSITE, 6.6)]
    elif base == "Fe":
        opts = [(GOETHITE, 4.3)]
    else:
        opts = [(SOIL, 1.8)]
    wt, dens = opts[rng.integers(len(opts))]
    return dict(wt), dens


def _coin(rng, family, bulk, thickness_um, base, allow_enrich=None, plating=None):
    profile = "homogeneous"
    layers = []
    notes = {}
    if plating is not None:
        plate_wt, plate_um, pname = plating
        layers.append(Layer(plate_um, plate_wt))
        layers.append(Layer(max(thickness_um - plate_um, 20.0), bulk))
        profile = pname
        notes["plate_um"] = plate_um
    elif allow_enrich and rng.random() < allow_enrich:
        rich, lean = "Ag", "Cu"
        surf = rng.uniform(0.85, 1.0)
        lam = 10 ** rng.uniform(np.log10(15.0), np.log10(120.0))
        layers = _enriched_profile(bulk, rich, lean, surf, lam, thickness_um)
        profile = "surface_enriched"
        notes.update(surf_frac=surf, decay_um=lam)
    else:
        layers.append(Layer(thickness_um, bulk))
    if rng.random() < 0.35:          # corrosion crust
        cwt, dens = _corrosion_for(base, rng)
        t = 10 ** rng.uniform(np.log10(1.0), np.log10(60.0))
        layers.insert(0, Layer(t, cwt, dens))
        notes["crust_um"] = t
    if rng.random() < 0.12:          # adhering soil
        t = 10 ** rng.uniform(0.0, np.log10(30.0))
        layers.insert(0, Layer(t, dict(SOIL), 1.8))
        notes["soil_um"] = t
    return ObjectModel(family, profile, layers, notes)


def sample_roman_coin(rng) -> ObjectModel:
    fam = rng.choice(
        ["aureus", "aureus_debased", "denarius_fine", "denarius_debased", "antoninianus",
         "fourree", "orichalcum", "as", "nummus", "gilded_bronze", "plated_gold_forgery"],
        p=[0.10, 0.12, 0.10, 0.15, 0.10, 0.07, 0.08, 0.06, 0.08, 0.07, 0.07])
    th = rng.uniform(800.0, 2500.0)
    if fam == "aureus":
        return _coin(rng, fam, aureus_pure(rng), th, "Au")
    if fam == "aureus_debased":
        return _coin(rng, fam, aureus_debased(rng), th, "Au")
    if fam == "denarius_fine":
        return _coin(rng, fam, denarius_fine(rng), th, "Ag", allow_enrich=0.3)
    if fam == "denarius_debased":
        return _coin(rng, fam, denarius_debased(rng), th, "Cu", allow_enrich=0.8)
    if fam == "antoninianus":
        return _coin(rng, fam, antoninianus(rng), th, "Cu", allow_enrich=0.7)
    if fam == "fourree":
        t = 10 ** rng.uniform(np.log10(20.0), np.log10(200.0))
        return _coin(rng, fam, copper_core(rng), th, "Cu", plating=(silver_foil(rng), t, "plated"))
    if fam == "orichalcum":
        return _coin(rng, fam, orichalcum(rng), rng.uniform(1500, 4000), "Cu")
    if fam == "as":
        return _coin(rng, fam, copper_as(rng), rng.uniform(1500, 3500), "Cu")
    if fam == "nummus":
        return _coin(rng, fam, nummus(rng), rng.uniform(800, 2000), "Cu", allow_enrich=0.2)
    if fam == "gilded_bronze":
        t = 10 ** rng.uniform(0.0, np.log10(15.0))
        return _coin(rng, fam, copper_core(rng), th, "Cu", plating=(gilding(rng), t, "gilded"))
    # plated gold forgery: gold foil over copper or silver core
    core = copper_core(rng) if rng.random() < 0.6 else silver_foil(rng)
    t = 10 ** rng.uniform(np.log10(5.0), np.log10(80.0))
    return _coin(rng, fam, core, th, "Au", plating=(aureus_pure(rng), t, "plated"))


# ---------------------------------------------------------------------------
# Broader heritage materials (the v1 presets, with densities)
# ---------------------------------------------------------------------------
GENERAL_PRESETS = {
    "tin_bronze":         ({"Cu": 0.86, "Sn": 0.12, "Pb": 0.01, "Zn": 0.01}, None),
    "leaded_bronze":      ({"Cu": 0.72, "Sn": 0.10, "Pb": 0.16, "Zn": 0.02}, None),
    "brass":              ({"Cu": 0.64, "Zn": 0.34, "Pb": 0.02}, None),
    "wrought_iron":       ({"Fe": 0.97, "Mn": 0.02, "Si": 0.01}, None),
    "steel_alloy":        ({"Fe": 0.95, "Mn": 0.02, "Ni": 0.02, "Cr": 0.01}, None),
    "silver_alloy":       ({"Ag": 0.90, "Cu": 0.08, "Pb": 0.02}, None),
    "gold_alloy":         ({"Au": 0.75, "Ag": 0.15, "Cu": 0.10}, None),
    "lead_object":        ({"Pb": 0.96, "Sb": 0.02, "Sn": 0.02}, None),
    "pewter":             ({"Sn": 0.85, "Pb": 0.10, "Cu": 0.05}, None),
    "terracotta_ceramic": ({"Si": 0.26, "Al": 0.09, "Fe": 0.05, "Ca": 0.06, "K": 0.03, "Ti": 0.005,
                            "Mg": 0.015, "Na": 0.01, "O": 0.48}, 2.0),
    "lead_glass":         ({"Si": 0.18, "Pb": 0.35, "K": 0.05, "Ca": 0.03, "Na": 0.04, "O": 0.35}, 3.5),
    "soda_lime_glass":    ({"Si": 0.33, "Na": 0.11, "Ca": 0.06, "K": 0.01, "Mg": 0.01, "Al": 0.01,
                            "O": 0.47}, 2.5),
    "pigment_azurite":    ({"Cu": 0.55, "C": 0.07, "O": 0.37, "Ca": 0.01}, 3.8),
    "pigment_cinnabar":   ({"Hg": 0.86, "S": 0.14}, 8.1),
    "pigment_lead_white": ({"Pb": 0.80, "C": 0.03, "O": 0.17}, 6.1),
    "pigment_orpiment":   ({"As": 0.61, "S": 0.39}, 3.5),
    "marble_limestone":   ({"Ca": 0.39, "Mg": 0.01, "C": 0.12, "O": 0.48}, 2.7),
    "burial_soil_patina": ({"Fe": 0.10, "Si": 0.25, "Al": 0.07, "Ca": 0.05, "K": 0.02, "O": 0.51}, 1.9),
    "fire_gilded_bronze": ({"Cu": 0.80, "Sn": 0.10, "Au": 0.06, "Hg": 0.04}, None),
}
GENERAL_NAMES = list(GENERAL_PRESETS)


def sample_general(rng) -> ObjectModel:
    names = rng.choice(GENERAL_NAMES, size=int(rng.choice([1, 2], p=[0.7, 0.3])), replace=False)
    comp, dens = {}, []
    for nm in names:
        base, d = GENERAL_PRESETS[str(nm)]
        scale = rng.uniform(0.5, 1.5)
        for s, w in base.items():
            comp[s] = comp.get(s, 0) + w * scale * rng.uniform(0.7, 1.3)
        dens.append(d)
    if rng.random() < 0.3:            # random trace
        from config import TARGETS
        s = str(rng.choice(TARGETS))
        comp[s] = comp.get(s, 0) + sum(comp.values()) * 10 ** rng.uniform(-2.7, -1.3)
    comp = _norm(comp)
    density = None if all(d is None for d in dens) else float(np.mean([d or 8.0 for d in dens]))
    th = rng.uniform(2000.0, 10000.0)
    return ObjectModel("general:" + "+".join(map(str, names)), "homogeneous", [Layer(th, comp, density)])


def sample_object(rng, p_roman: float = 0.75) -> ObjectModel:
    return sample_roman_coin(rng) if rng.random() < p_roman else sample_general(rng)
