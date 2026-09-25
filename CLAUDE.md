# CLAUDE.md

Muonic X-ray (µXES) simulation + neural network for non-destructive analysis of Roman coins.
README.md has the science, results and limitations; this file is the working notes.

## Environment

- Windows 10, dual-core i3 (4 threads), 7.8 GB RAM, **no GPU**. Everything is CPU-bound; budget time accordingly.
- Python 3.12 venv at `.venv/` (not the system Python — there is none on PATH). Run scripts from `src/`:
  `..\.venv\Scripts\python.exe train.py ...` (PowerShell) — all paths in scripts are relative to `src/`.
- Git/gh were installed via winget; open a new shell if `git`/`gh` aren't found. Remote: private repo
  `huidobroeloy/muon_element_recognition`, branch `main`.

## Pipeline order and cost

1. `physics.py` → `data/dirac_levels.json` (~12 min; committed, so normally cached)
2. `simulate.py` → `data/cascade_lines.pkl` (committed)
3. `dataset.py` → `data/{train,val,test}` memmaps (60k/4k/4k, ~10 min on 3 workers, 1.6 GB, **gitignored**)
4. `optimize.py` (~40 min) → `checkpoints/best_config.json`
5. `train.py --config ../checkpoints/best_config.json --epochs 9 --samples-per-epoch 24000` (~95 min)
6. `evaluate.py`, `case_studies.py` (~20 min), `validate_physics.py` → `results/`

Bump `CACHE_VERSION` in `physics.py` whenever the level physics changes, and delete
`data/cascade_lines.pkl` whenever cascade/line physics changes; neither is invalidated automatically.
Regenerate datasets after any change to `simulate.py`, `materials.py`, `transport.py` or `config.py` labels.

## Decisions (and why)

- **Physics is solved, not fitted.** Radial Dirac (Fermi nucleus + Uehling VP) replaced v1's point-nucleus +
  scale factor. Only two things are fitted: a 1s shift from 6 anchor lines, and one range-scale factor from
  3 published SRIM depths. The 19 lines tagged `valid` in `literature.py` must never be used in a fit.
- **All literature numbers live in `literature.py` with sources.** Don't hard-code published values elsewhere.
- **Cascade intensities are uncertain → randomised** (9 variants per element) rather than trusted.
- **Labels = muon-weighted composition of the sampled volume, metal basis (Z ≥ 11)**; present ≥ 1 wt%
  (published detection limit), absent < 0.2 wt%, the band between is masked from the loss.
  C/N/O are simulated but not predicted (air gap dominates them in real ISIS spectra).
- **Store expected spectra, re-draw Poisson noise every read** (simulator is ~25 ms/spectrum; too slow for on-the-fly).
- **CoordConv + position-aware flatten, no global pooling**: element identity is peak *energy*; v1's global
  average pooling threw that away.
- **Width 16 network**: chosen for CPU cost (1.1 s per batch of 48). Optuna's best trial was the hand-tuned defaults.
- **`baseline.py` is a faithful replica of the published peak fit** (Hampshire 2019, Green 2021). Keep it
  faithful — it is the comparison target, not something to improve silently.
- Hampshire et al. report **atomic** %; the network predicts **wt%**. Convert (`baseline.at_to_wt/wt_to_at`).

## Gotchas hit so far

- Windows DataLoader workers pickle the dataset: memmaps must be opened lazily (`SpectrumDataset.le/.he`
  properties + `__getstate__`), or pickling ~1.4 GB crashes with `OSError: [Errno 22]`.
- High momentum + thin object → no muons stop anywhere → empty line list; `render` must tolerate it
  (`generate_sample` rejects samples with < 60 % of stops in the object).
- The 18 MeV/c "~0.01 mm" depth is measured *after* 100 mm air + 16 µm Al; depth calibration must include
  the upstream layers (`ISIS_UPSTREAM`).
- Layer-discretised forward models are piecewise-constant in the fit parameters → least_squares stalls at
  the start value. Use the smooth per-muon version (`transport.object_stop_depths`).
- Peak-fit covariance can be singular for absent elements → NaN errors; guarded in `baseline.quantify` and `case_studies._clean`.
- Au capture γ at 333.0 keV sits between the Cu 3d-2p lines (330.9/336.5) — a real confounder for Cu in gold coins.
- Don't run heavy scripts while training: on this CPU they double epoch times.

## Current results (simulated data only)

- Physics: 19 independent lines, median error 0.17 keV (0.026 %), worst 1.1 keV (Pd 4d-2p).
- Test set: micro-F1 0.989, exact element set 94.9 %, major-element MAE 1.30 wt%, ≥ 92 % detection at 1–2 wt% for Ag/Cu/Sn/Zn/Hg.
- vs published peak fit: **the peak fit is more accurate** on clean Au-Ag-Cu (MAE 0.22/0.65/0.58 vs NN 0.58/1.25/0.77 wt%);
  the NN over-estimates Ag by 2–3 wt% in debased gold coins even after standards calibration.
  The NN's value is blind multi-element identification and robustness. Report this honestly.
- Julia Domna: forward-model inversion ≈ published mean-depth fit for this gradual profile; 11 momenta cut bulk error ~3× vs 6.

## Still to do

- [ ] Hybrid quantification: NN identifies elements → multi-line physics template fit quantifies (likely beats both).
- [ ] Fix NN Ag bias in debased gold coins (check calibration nonlinearity; `Analyzer.calibrate_with_standards` is only a per-element scale).
- [ ] Rare trace elements (P, Co, Se, Cd, Ba, Pt, Bi) have F1 ≈ 0: add a trace-rich training population.
- [ ] 1-σ coverage is 0.84 (slightly conservative); consider recalibrating σ on validation.
- [ ] Test on real spectra: needs real ISIS standards / pure-element spectra re-binned to the LE/HE grids.
- [ ] Case study for sharp profiles (thin fourrée plating, gilding) where the forward-model inversion should beat the mean-depth fit.
- [ ] Natural isotope mixtures (currently dominant isotope only; matters at keV level for heavy K lines).
- [ ] Unverified inputs worth checking against primary sources: rms radii transcribed from Angeli & Marinova 2013;
  Ti 932.4 and Au 5763.1 keV anchors (carried over from v1; Au literature spread 5762.5–5764.9).
- [ ] The architecture map (claude.ai artifact "Muon Coin Pipeline Map") is not in the repo.
