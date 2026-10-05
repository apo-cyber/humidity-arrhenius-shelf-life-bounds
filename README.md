# Closed-form Bayesian shelf-life bounds under temperature and humidity — code

Code for the paper *Closed-form Bayesian shelf-life bounds from temperature–humidity accelerated stability data: one-sided
coverage, design precision and the cost of an imported prior* (Y. Arai, 2026). Twelve self-contained Python scripts
(NumPy + SciPy; Matplotlib for the figures) produce every table and figure of the paper. Sections 2–5 use no measured data:
they run on the generating model in `model.py` (first-order loss, humidity-corrected Arrhenius, the truth of the companion
study). Section 6 uses the published supplementary tables of Tamura et al. (2020), transcribed in `tamura2020.py`, and the
real-time values digitised from their Figure 7 (`fig7_digitise.py`). `model.py` is the single source of the model, the three
Stage 2 estimators and the closed forms (design covariance, variance decomposition, rank-1 update, bias budget, coverage,
inflation).

| Script | Paper | What it does |
|---|---|---|
| `model.py` | §2 (all equations) | Truth, Stage 1 OLS, Stage 2 NIG update with observed / fitted / $k$-space Gauss–Newton weights, one-sided bound, closed forms; the comparator bounds of Table 1 |
| `prop3_check.py` | §2.5, Proposition 3 | Numerical check of the profile-information identity with two nuisance parameters per condition (intercept and curvature): the Schur complement of the full Fisher information equals the Stage 2 information with Stage 1 precisions as weights (relative difference $10^{-15}$) |
| `pathways.py` | §5.5, Table 9, Proposition 4 | Two-pathway generating models (parallel: sum of Arrhenius rates; consecutive: reciprocal sum), estimator applied unchanged; coverage, median, simulated / noise-free / closed-form displacement of $\ln k$ at 25 °C, on the ICH 3×2 and ASAP-5 designs |
| `bias.py` | §2.5 | Closed-form second-order bias of the observed-weight (and log-scale) regression, $\sigma^2 M^{-1}X^\top(\tfrac32\mathbf 1 - 2h)$, checked against simulation for four designs × three noise levels |
| `compare.py` | Table 1; §2.5, §7 | One-sided bounds on the same Stage 1 summaries: working weights with NIG–$t$ (this paper), one-stage NLS + delta + $t$ (AccelStab-type; exact from the summaries after eliminating per-condition intercepts), FBH (4000 draws from the multivariate $t$), ASAP-type unweighted $\ln t_{\rm iso}$ regression + Monte Carlo; the two observed-weight variants are also printed for the record (text of §2.5) |
| `simulate.py` | Tables 3, 4, 5; §3.3, §4 | Estimator × design × noise (4000 replicates); Stage B with imported prior over a grid of prior displacements, closed-form vs simulated coverage; inflation of the prior and the resulting bound. Adds to the original five designs the ICH 0/3/6 grid and a 0/1/3 × 1-batch Stage B |
| `nomogram.py` | Table 2, §3.2–3.4, Figure 1 | Geometry-only precision (uniform SE), variance decomposition into intercept / temperature / humidity / cross terms, value of one more condition (rank-1), bound-vs-precision nomogram with the simulated values overlaid |
| `budget.py` | Figure 2, §4.2–4.3 | Coverage vs prior displacement with the bias budget marked; bound after inflating the prior, for three Stage B precisions |
| `tamura2020.py` | §6 | Supplementary Tables S1–S3 of Tamura et al. (2020) *Chem Pharm Bull* 68:1049 (doi 10.1248/cpb.c20-00443), transcribed; running it checks the slopes against the authors' |
| `illustration.py` | §6, Table 9, Figure 3 | Stage 1 on the real data (residual SD pooled within temperature), 60-condition and per-level Stage 2, design decomposition on real precisions, prior transfer between MgSt levels with the budget, comparison with the 25 °C / 60 % RH real-time values digitised from the authors' Figure 7 |
| `fig7_digitise.py` | §6 | How the real-time values and the authors' model line were digitised from Figure 7 (axis calibration on the tick marks, marker bounding boxes); the article PDF is not redistributed |
| `robustness.py` | Tables 7–8, §5.3–5.4 | Batch-to-batch random effect (pooled vs batch-as-unit, Stage A and B); kinetic order $n \in \{0,1,2\}$ read as first order / known / profiled ; numerical identity of the isoconversion entry |

Time grids enter only through $S_{tt,i}$, $n_i$ and $\mathrm{RSS}_i$; any grid (unequal spacing, different grids per condition, replicates) can be passed to `stage1`.

## Run

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
./run_all.sh        # ~1 min; writes output/*.txt and ../figures/fig{1,2,3}_*.{png,pdf,tiff} (TIFF: 1200 dpi line art, 600 dpi for fig3)
```

The `.txt` files in `output/` are the runs used for the paper (seeds fixed in the scripts: 20260919 for Stage A, 1 for Stage B).
Script docstrings and some printed headers are in Japanese (the working language of the project). "paper-A" and "paper-B" in
the comments are the companion studies (github.com/apo-cyber/bayesian-shelf-life-arrhenius and apo-cyber/arrhenius-r2-criterion).

Set `PAPERC_FIG_DIR` to write the figures elsewhere. `./export.sh` builds the standalone copy that is published (scripts,
`output/`, `figures/` as PNG/PDF, `LICENSE`, `CITATION.cff`).

## Data

`tamura2020.py` holds the 60-condition data of Tamura, Ono, Kawabe & Yonemochi (2020) *Chem. Pharm. Bull.* 68(11):1049–1054,
https://doi.org/10.1248/cpb.c20-00443, transcribed from the open-access supplementary material (Tables S1–S3); running the
script checks the transcription against the authors' rate constants. The article itself is not redistributed; the real-time
values in `illustration.py` were digitised from its Figure 7 as documented in `fig7_digitise.py`.

## Licence and citation

MIT licence (see `LICENSE`). Please cite the paper and the Zenodo archive of this repository (`CITATION.cff`).
