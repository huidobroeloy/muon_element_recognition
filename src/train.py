"""
train.py
========
Train MuonNet on pre-generated expected spectra (see dataset.py).

    python train.py --train ../data/train --val ../data/val --epochs 12 --out ../checkpoints/best.pt

Selection metric ("score") = macro-F1 of presence  -  mean total-variation
distance of the predicted composition (0.5 * sum |f_pred - f_true|).
After training, per-element decision thresholds are tuned on the validation
set (maximum F1) and stored in the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from config import HE_NBINS, LE_NBINS, NUM_TARGETS, TARGETS
from dataset import SpectrumDataset
from model import MuonNet, multitask_loss

DEFAULTS = dict(lr=2e-3, weight_decay=2e-4, width=16, embed=192, hidden=320, dropout=0.15,
                w_comp=3.0, w_nll=0.1, batch=48)


@torch.no_grad()
def predict(model, loader, device="cpu"):
    model.eval()
    P, C, S, F, Y, M = [], [], [], [], [], []
    for le, he, aux, frac, present, mask in loader:
        p, c, s = model(le.to(device), he.to(device), aux.to(device))
        P.append(torch.sigmoid(p).cpu()); C.append(torch.softmax(c, 1).cpu()); S.append(s.exp().cpu())
        F.append(frac); Y.append(present); M.append(mask)
    cat = lambda t: torch.cat(t).numpy()
    return cat(P), cat(C), cat(S), cat(F), cat(Y), cat(M)


def metrics(prob, comp, sig, frac, present, mask, thresholds=None):
    thr = np.full(prob.shape[1], 0.5) if thresholds is None else np.asarray(thresholds)
    pred = (prob >= thr).astype(int)
    # masked elements (0.2-1 wt%) are excluded from the presence metrics
    yt = np.where(mask > 0, present, -1)
    f1s, precs, recs, support = [], [], [], []
    for j in range(prob.shape[1]):
        m = yt[:, j] >= 0
        t, p = yt[m, j], pred[m, j]
        tp = int(((t == 1) & (p == 1)).sum()); fp = int(((t == 0) & (p == 1)).sum()); fn = int(((t == 1) & (p == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        precs.append(prec); recs.append(rec); support.append(int((t == 1).sum()))
    support = np.array(support)
    valid = support >= 5
    macro_f1 = float(np.mean(np.array(f1s)[valid]))
    m = yt >= 0
    tp = ((yt == 1) & (pred == 1) & m).sum(); fp = ((yt == 0) & (pred == 1) & m).sum(); fn = ((yt == 1) & (pred == 0) & m).sum()
    micro_f1 = float(2 * tp / max(2 * tp + fp + fn, 1))
    exact = float(np.all((pred == present) | (mask == 0), axis=1).mean())
    tvd = float(0.5 * np.abs(comp - frac).sum(1).mean())
    mae_major = float(np.abs(comp - frac)[frac >= 0.05].mean()) if (frac >= 0.05).any() else float("nan")
    z = np.abs(comp - frac) / np.maximum(sig, 1e-6)
    cover = float((z[frac >= 0.01] <= 1.0).mean())
    return dict(macro_f1=macro_f1, micro_f1=micro_f1, exact=exact, tvd=tvd, mae_major_wt=100 * mae_major,
                sigma_coverage_1s=cover, score=macro_f1 - tvd,
                per_element={TARGETS[j]: dict(f1=f1s[j], precision=precs[j], recall=recs[j], support=int(support[j]))
                             for j in range(prob.shape[1])})


def tune_thresholds(prob, present, mask):
    thr = np.full(prob.shape[1], 0.5)
    grid = np.linspace(0.1, 0.9, 33)
    for j in range(prob.shape[1]):
        m = mask[:, j] > 0
        if present[m, j].sum() < 5:
            continue
        best = max(grid, key=lambda t: f1_score(present[m, j], prob[m, j] >= t, zero_division=0))
        thr[j] = float(best)
    return thr


def train(cfg: dict, train_dir, val_dir, epochs: int, out: str | None = None, train_limit=None,
          val_limit=None, workers: int = 1, threads: int = 3, trial=None, verbose=True, seed=0,
          samples_per_epoch: int | None = None):
    cfg = {**DEFAULTS, **cfg}
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(threads)
    tr = SpectrumDataset(train_dir, train=True, limit=train_limit)
    va = SpectrumDataset(val_dir, train=False, limit=val_limit)
    # an "epoch" can be a random subset (fresh Poisson noise every draw)
    sampler = torch.utils.data.RandomSampler(tr, replacement=False, num_samples=min(samples_per_epoch, len(tr))) \
        if samples_per_epoch else None
    tl = DataLoader(tr, batch_size=cfg["batch"], shuffle=sampler is None, sampler=sampler, num_workers=workers,
                    drop_last=True, persistent_workers=workers > 0)
    vl = DataLoader(va, batch_size=128, shuffle=False, num_workers=workers, persistent_workers=workers > 0)
    model = MuonNet(NUM_TARGETS, LE_NBINS, HE_NBINS, width=cfg["width"], embed=cfg["embed"],
                    hidden=cfg["hidden"], dropout=cfg["dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=epochs * len(tl),
                                                pct_start=0.15, div_factor=10, final_div_factor=100)
    best, best_state, history = -1e9, None, []
    for ep in range(1, epochs + 1):
        model.train()
        t0, run, nb = time.time(), 0.0, 0
        for le, he, aux, frac, present, mask in tl:
            out_ = model(le, he, aux)
            loss, parts = multitask_loss(out_, frac, present, mask, cfg["w_comp"], cfg["w_nll"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item(); nb += 1
        prob, comp, sig, frac, present, mask = predict(model, vl)
        m = metrics(prob, comp, sig, frac, present, mask)
        history.append(dict(epoch=ep, loss=run / nb, **{k: v for k, v in m.items() if k != "per_element"}))
        if verbose:
            print(f"epoch {ep:2d}/{epochs} loss {run / nb:.4f} | val macroF1 {m['macro_f1']:.4f} microF1 {m['micro_f1']:.4f} "
                  f"exact {m['exact']:.3f} | TVD {m['tvd']:.4f} MAE(major) {m['mae_major_wt']:.2f} wt% "
                  f"| 1-sigma cover {m['sigma_coverage_1s']:.2f} | {time.time() - t0:.0f}s", flush=True)
        if m["score"] > best:
            best = m["score"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if trial is not None:
            trial.report(m["score"], ep)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()
    model.load_state_dict(best_state)
    if out is not None:
        prob, comp, sig, frac, present, mask = predict(model, vl)
        thr = tune_thresholds(prob, present, mask)
        m = metrics(prob, comp, sig, frac, present, mask, thr)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(state=model.state_dict(), model_config=model.config, train_config=cfg, targets=TARGETS,
                        thresholds=thr.tolist(), history=history, val_metrics=m), out)
        if verbose:
            print(f"\nsaved {out}  (tuned thresholds: val macroF1 {m['macro_f1']:.4f}, microF1 {m['micro_f1']:.4f})")
    return best, model, history


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="../data/train")
    ap.add_argument("--val", default="../data/val")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--out", default="../checkpoints/best.pt")
    ap.add_argument("--config", default=None, help="JSON file with hyper-parameters (e.g. from optimize.py)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--samples-per-epoch", type=int, default=None)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text()) if a.config else {}
    print("config:", {**DEFAULTS, **cfg})
    train(cfg, a.train, a.val, a.epochs, a.out, train_limit=a.train_limit, workers=a.workers, threads=a.threads,
          samples_per_epoch=a.samples_per_epoch)
