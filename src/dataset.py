"""
dataset.py
==========
Pre-generates *expected* (noise-free) spectra with the physics simulator into
memory-mapped .npy files, then serves Poisson-noised, re-exposed copies.

Why not purely on-the-fly (as in v1)? The v2 simulator is far more detailed
(Dirac line lists, transport through layered coins, two detectors), so it
costs ~10-20 ms per spectrum. Storing expected spectra and re-drawing the
counting noise at a random exposure every time a sample is read keeps
unlimited noise augmentation at negligible cost.

    python dataset.py --n 60000 --seed 1 --out ../data/train
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from config import HE_NBINS, LE_NBINS, NUM_TARGETS

def _family_code(name: str) -> int:
    base = name.split(":")[0]
    return {"aureus": 0, "aureus_debased": 1, "denarius_fine": 2, "denarius_debased": 3, "antoninianus": 4,
            "fourree": 5, "orichalcum": 6, "as": 7, "nummus": 8, "gilded_bronze": 9,
            "plated_gold_forgery": 10, "general": 11}.get(base, 11)


PROFILE_CODES = {"homogeneous": 0, "surface_enriched": 1, "plated": 2, "gilded": 3}
FAMILY_NAMES = ["aureus", "aureus_debased", "denarius_fine", "denarius_debased", "antoninianus", "fourree",
                "orichalcum", "as", "nummus", "gilded_bronze", "plated_gold_forgery", "general"]


def _worker(args):
    start, count, seed, out = args
    from simulate import generate_sample, line_tables
    line_tables()
    rng = np.random.default_rng([seed, start])
    le = np.load(out / "le.npy", mmap_mode="r+")
    he = np.load(out / "he.npy", mmap_mode="r+")
    fr = np.load(out / "frac.npy", mmap_mode="r+")
    pr = np.load(out / "present.npy", mmap_mode="r+")
    mk = np.load(out / "mask.npy", mmap_mode="r+")
    meta = np.load(out / "meta.npy", mmap_mode="r+")
    for i in range(start, start + count):
        a, b, f, p, m, info = generate_sample(rng)
        le[i], he[i], fr[i], pr[i], mk[i] = a, b, f, p, m
        meta[i] = (info["momentum"], info["counts_le"], _family_code(info["family"]),
                   PROFILE_CODES.get(info["profile"], 0), info["f_object"])
    for arr in (le, he, fr, pr, mk, meta):
        arr.flush()
    return count


def generate_dataset(n: int, seed: int, out: str | Path, workers: int = 3, chunk: int = 250):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    from numpy.lib.format import open_memmap
    open_memmap(out / "le.npy", "w+", np.float32, (n, LE_NBINS))
    open_memmap(out / "he.npy", "w+", np.float32, (n, HE_NBINS))
    open_memmap(out / "frac.npy", "w+", np.float32, (n, NUM_TARGETS))
    open_memmap(out / "present.npy", "w+", np.float32, (n, NUM_TARGETS))
    open_memmap(out / "mask.npy", "w+", np.float32, (n, NUM_TARGETS))
    open_memmap(out / "meta.npy", "w+", np.float32, (n, 5))
    jobs = [(s, min(chunk, n - s), seed, out) for s in range(0, n, chunk)]
    t0 = time.time()
    done = 0
    with Pool(workers) as pool:
        for c in pool.imap_unordered(_worker, jobs):
            done += c
            if done % (chunk * 8) == 0 or done == n:
                el = time.time() - t0
                print(f"  {done}/{n} spectra, {el:.0f}s elapsed, ETA {el / done * (n - done):.0f}s", flush=True)
    (out / "info.json").write_text(json.dumps({"n": n, "seed": seed}))


# ---------------------------------------------------------------------------
# Input transform shared by training, evaluation and inference on real data
# ---------------------------------------------------------------------------
def transform(counts: np.ndarray) -> np.ndarray:
    """log1p compression then per-spectrum standardisation."""
    x = np.log1p(counts.astype(np.float32))
    m = x.mean(axis=-1, keepdims=True)
    s = x.std(axis=-1, keepdims=True) + 1e-6
    return (x - m) / s


def aux_features(momentum, counts_le_obs, counts_he_obs):
    return np.array([math.log(momentum / 30.0), math.log10(max(counts_le_obs, 1.0)) - 5.0,
                     math.log10(max(counts_he_obs, 1.0)) - 5.0], dtype=np.float32)


try:
    import torch
    from torch.utils.data import Dataset

    class SpectrumDataset(Dataset):
        """
        train=True : fresh Poisson noise at a random exposure (x0.3..x3) per read
        train=False: deterministic noise (seeded by index) at the stored exposure
        """

        def __init__(self, root, train: bool, limit: int | None = None, exposure=(0.3, 3.0)):
            self.root = Path(root)
            self.train = train
            self.exposure = exposure
            self._le = self._he = None           # memmaps opened lazily (cheap to pickle to workers)
            self.frac = np.load(self.root / "frac.npy")
            self.present = np.load(self.root / "present.npy")
            self.mask = np.load(self.root / "mask.npy")
            self.meta = np.load(self.root / "meta.npy")
            self.n = len(self.frac) if limit is None else min(limit, len(self.frac))

        def __len__(self):
            return self.n

        @property
        def le(self):
            if self._le is None:
                self._le = np.load(self.root / "le.npy", mmap_mode="r")
            return self._le

        @property
        def he(self):
            if self._he is None:
                self._he = np.load(self.root / "he.npy", mmap_mode="r")
            return self._he

        def __getstate__(self):
            state = dict(self.__dict__)
            state["_le"] = state["_he"] = None
            return state

        def noisy(self, i, rng):
            scale = 10 ** rng.uniform(math.log10(self.exposure[0]), math.log10(self.exposure[1])) if self.train else 1.0
            le = rng.poisson(np.asarray(self.le[i], dtype=np.float64) * scale)
            he = rng.poisson(np.asarray(self.he[i], dtype=np.float64) * scale)
            return le, he

        def __getitem__(self, i):
            rng = np.random.default_rng() if self.train else np.random.default_rng(10_000_019 + i)
            le, he = self.noisy(i, rng)
            aux = aux_features(float(self.meta[i, 0]), le.sum(), he.sum())
            return (torch.from_numpy(transform(le)), torch.from_numpy(transform(he)), torch.from_numpy(aux),
                    torch.from_numpy(self.frac[i]), torch.from_numpy(self.present[i]), torch.from_numpy(self.mask[i]))
except ImportError:          # physics-only use without torch
    pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    generate_dataset(a.n, a.seed, a.out, a.workers)
