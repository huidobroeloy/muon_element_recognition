"""
inference.py
============
Load a trained checkpoint and analyse count spectra (simulated or real).
Real data only needs to be re-binned onto the two detector grids in config.py
(LE: 20-1000 keV / 0.5 keV, HE: 100-8000 keV / 2 keV) and energy-calibrated.
"""

from __future__ import annotations

import numpy as np
import torch

from config import TARGETS
from dataset import aux_features, transform
from model import MuonNet


class Analyzer:
    def __init__(self, checkpoint: str, threads: int = 3):
        torch.set_num_threads(threads)
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = MuonNet(**ck["model_config"])
        self.model.load_state_dict(ck["state"])
        self.model.eval()
        self.thresholds = np.array(ck["thresholds"])
        self.targets = ck["targets"]
        self.ck = ck
        self.correction = np.ones(len(self.targets))     # optional standards calibration

    @torch.no_grad()
    def raw(self, le_counts, he_counts, momentum):
        le = torch.from_numpy(transform(np.atleast_2d(le_counts)))
        he = torch.from_numpy(transform(np.atleast_2d(he_counts)))
        aux = torch.from_numpy(np.stack([aux_features(momentum, a.sum(), b.sum())
                                         for a, b in zip(np.atleast_2d(le_counts), np.atleast_2d(he_counts))]))
        p, c, s = self.model(le, he, aux)
        prob = torch.sigmoid(p).numpy()
        comp = torch.softmax(c, 1).numpy()
        sig = s.exp().numpy()
        return prob, comp, sig

    def analyse(self, le_counts, he_counts, momentum):
        prob, comp, sig = self.raw(le_counts, he_counts, momentum)
        comp = comp * self.correction[None, :]
        comp /= comp.sum(1, keepdims=True)
        return prob, comp, sig

    def calibrate_with_standards(self, spectra, compositions, elements=("Au", "Ag", "Cu")):
        """Per-element multiplicative correction of predicted wt fractions from
        certified standards measured on the same instrument (the same data a
        peak-fit analysis uses for its calibration factors)."""
        idx = [self.targets.index(e) for e in elements]
        logs = {i: [] for i in idx}
        for (le, he, p), wt in zip(spectra, compositions):
            _, comp, _ = self.raw(le, he, p)
            comp = comp[0]
            for e, i in zip(elements, idx):
                if wt.get(e, 0) >= 0.01 and comp[i] > 1e-4:
                    logs[i].append(np.log(wt[e] / comp[i]))
        self.correction = np.ones(len(self.targets))
        for i, v in logs.items():
            if v:
                self.correction[i] = float(np.exp(np.mean(v)))
        return {self.targets[i]: self.correction[i] for i in idx}

    def report(self, le_counts, he_counts, momentum, min_wt=0.002):
        prob, comp, sig = self.analyse(le_counts, he_counts, momentum)
        rows = []
        for j, el in enumerate(self.targets):
            if prob[0, j] >= self.thresholds[j] or comp[0, j] >= min_wt:
                rows.append((el, float(prob[0, j]), 100 * float(comp[0, j]), 100 * float(sig[0, j]),
                             bool(prob[0, j] >= self.thresholds[j])))
        return sorted(rows, key=lambda r: -r[2])
