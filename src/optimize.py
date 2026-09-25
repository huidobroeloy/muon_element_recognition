"""
optimize.py
===========
Hyper-parameter search with Optuna (TPE sampler + median pruning) on short
training budgets, then writes the best configuration to JSON for train.py.

    python optimize.py --trials 7 --epochs 3 --samples-per-epoch 4800 --out ../checkpoints/best_config.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import optuna

from train import train


def objective(trial, a):
    cfg = dict(
        lr=trial.suggest_float("lr", 5e-4, 5e-3, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        width=trial.suggest_categorical("width", [12, 16]),       # CPU budget: see README
        embed=trial.suggest_categorical("embed", [128, 192]),
        hidden=trial.suggest_categorical("hidden", [256, 384]),
        dropout=trial.suggest_float("dropout", 0.0, 0.3),
        w_comp=trial.suggest_float("w_comp", 1.0, 6.0, log=True),
    )
    score, _, _ = train(cfg, a.train, a.val, a.epochs, out=None, samples_per_epoch=a.samples_per_epoch,
                        val_limit=a.val_limit, workers=a.workers, threads=a.threads, trial=trial, verbose=True)
    return score


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="../data/train")
    ap.add_argument("--val", default="../data/val")
    ap.add_argument("--trials", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--samples-per-epoch", type=int, default=4800)
    ap.add_argument("--val-limit", type=int, default=800)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--out", default="../checkpoints/best_config.json")
    a = ap.parse_args()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1))
    # start from the hand-tuned defaults so the search can only improve on them
    study.enqueue_trial(dict(lr=2e-3, weight_decay=2e-4, width=16, embed=192, hidden=384, dropout=0.15,
                             w_comp=3.0))
    study.optimize(lambda t: objective(t, a), n_trials=a.trials)
    print("\nbest score %.4f with" % study.best_value, study.best_params)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(study.best_params, indent=2))
    rows = [dict(number=t.number, value=t.value, state=str(t.state), **t.params) for t in study.trials]
    Path(a.out).with_name("optuna_trials.json").write_text(json.dumps(rows, indent=2))
