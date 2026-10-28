# Closed-form one-sided shelf-life bounds under temperature and humidity — code

Code for the paper *Closed-form one-sided shelf-life bounds from temperature–humidity accelerated stability data*
(Y. Arai, 2026). Version 1.1.0 corresponds to the revised manuscript (flat prior by default, per-batch intercepts, 54-truth sensitivity grid, Supplementary S13–S16). Sixteen self-contained Python scripts
(NumPy + SciPy; Matplotlib for the figures) produce every table and figure of the paper. Sections 2–5 use no measured data:
they run on the generating model in `model.py` (first-order loss, humidity-corrected Arrhenius with $E_a$ = 80 kJ/mol,
$B$ = 0.04 per % RH and a shelf life of 30 months at 25 °C / 60 % RH), and `sensitivity_truth.py` varies that truth. Section 6 uses the published supplementary tables of Tamura et al. (2020), transcribed in `tamura2020.py`, and the
real-time values digitised from their Figure 7 (`fig7_digitise.py`). `model.py` is the single source of the model, the three
Stage 2 estimators and the closed forms (design covariance, variance decomposition, rank-1 update, bias budget, coverage,
inflation).

| Script | Paper | What it does |
|---|---|---|
| `model.py` | §2 (all equations) | Truth, Stage 1 OLS (batches: per-batch intercepts, common slope), Stage 2 NIG update with observed / fitted / $k$-space Gauss–Newton weights, flat prior and $a_0 = 0$ by default (residual degrees of freedom net of the effective number of coefficients, $p_{\rm eff}$; $N - n - 3$ when flat), the weak prior `WEAK_SD` as an option, one-sided bound, the decoupled single-condition Stage B (`stageB_decoupled`), closed forms; the comparator bounds of Table 1 |
| `prop3_check.py` | Remark 2, Supplementary S2 | Numerical check of the profile-information identity with two nuisance parameters per condition (intercept and curvature): the Schur complement of the full Fisher information equals the Stage 2 information with Stage 1 precisions as weights (relative difference $10^{-15}$) |
| `pathways.py` | §4.2, §5.4, Proposition 2, Supplementary S6 | Two-pathway generating models (parallel: sum of Arrhenius rates; consecutive: reciprocal sum), estimator applied unchanged; coverage, median, simulated / noise-free / closed-form displacement of $\ln k$ at 25 °C, on the ICH 3×2 and ASAP-5 designs |
| `bias.py` | §2.5, Proposition 1, Supplementary S13 | Closed-form second-order bias of the observed-weight (and log-scale) regression, $\sigma^2 M^{-1}X^\top(\tfrac32\mathbf 1 - 2h)$, checked against simulation for four designs × three noise levels, and for a temperature-only three-temperature design |
| `compare.py` | Table 1; §2.5, §7 | One-sided bounds on the same Stage 1 summaries: working weights with NIG–$t$ (this paper; flat prior, identical to the one-stage fit) and the same with a weak prior, one-stage NLS + delta + $t$ (AccelStab-type; exact from the summaries after eliminating per-condition intercepts), FBH (4000 draws from the multivariate $t$), a reconstruction of the ASAP analysis (unweighted $\ln t_{\rm iso}$ regression + Monte Carlo) with three choices of the Stage 1 standard error (Supplementary S16); the two observed-weight variants are also printed for the record (text of §2.5) |
| `simulate.py` | Table 3; §2.3, §3.3, §4; Supplementary S1, S3, S4, S9 | Estimator × design × noise (4000 replicates); Stage B with imported prior over a grid of prior displacements, closed-form vs simulated coverage; inflation of the prior and the resulting bound. Adds to the original five designs the ICH 0/3/6 grid and a 0/1/3 × 1-batch Stage B |
| `nomogram.py` | Table 2, §3.2–3.4, Figure 1 | Geometry-only precision (uniform SE), time to a 0.4 % limit at the ASAP conditions, variance decomposition into intercept / temperature / humidity / cross terms, value of one more condition (rank-1), bound-vs-precision nomogram with the simulated values overlaid |
| `budget.py` | Figure 2, §4.2–4.3 | Coverage vs prior displacement with the bias budget marked; bound after inflating the prior, for three Stage B precisions |
| `tamura2020.py` | §6 | Supplementary Tables S1–S3 of Tamura et al. (2020) *Chem Pharm Bull* 68:1049 (doi 10.1248/cpb.c20-00443), transcribed; running it checks the slopes against the authors' |
| `illustration.py` | §6, Table 5, Supplementary S10, Figure S1 | Stage 1 on the real data (residual SD pooled within temperature), 60-condition and per-level Stage 2, design decomposition on real precisions, prior transfer between MgSt levels with the budget, comparison with the 25 °C / 60 % RH real-time values digitised from the authors' Figure 7, against the bound on the mean and a one-sided 95 % prediction bound for an individual measurement (Supplementary S10b) |
| `fig7_digitise.py` | §6 | How the real-time values and the authors' model line were digitised from Figure 7 (axis calibration on the tick marks, marker bounding boxes); the article PDF is not redistributed |
| `robustness.py` | §5.3–5.4, §7.3, Supplementary S5, S14 | Batch-to-batch random effect (pooled vs batch-as-unit, Stage A and B, and a common intercept for comparison); residual SD differing between conditions; kinetic order $n \in \{0,1,2\}$ read as first order / known / profiled ; numerical identity of the isoconversion entry |
| `worked_example.py` | Supplementary S15 | One simulated data set (ICH-type 3×2, assay precision) worked through Stage 1, the Stage 2 iterations and the one-sided bound, with the one-stage fit for comparison |
| `m2_sigma_mismatch.py` | §4.1, Supplementary S8 | Stage B when the development and QC laboratories differ in precision ($\sigma_B/\sigma_A$ = 1/4–4): the NIG analysis (coefficient prior scaled by $\sigma^2$, variance carried over) against the decoupled one (prior in absolute units, variance from Stage B, Welch–Satterthwaite $t$) |
| `sensitivity_truth.py` | §5.1, Table 4, Supplementary S7 | Sensitivity grid: $E_a$ ∈ {40, 50, 60, 80, 100, 120} kJ/mol × $B$ ∈ {0.02, 0.04, 0.06} × shelf life ∈ {18, 30, 60} months, three designs, two precisions; coverage, bound, closed form vs simulation, observed-weight bias, and the weak prior for comparison |
| `prior_checks.py` | §2.3, §5.1, Supplementary S17 | The informative-prior branch: weakly informative prior with $\sigma_{\mathrm{ref}}$ at 1/2, 1 and 2 times the generating $\sigma$; informative priors at the truth with $p_{\mathrm{eff}}$ subtracted from the degrees of freedom or not |
| `slow_cells.py` | §2.3, Supplementary S7 | The slow-product cell ($E_a$ 40, $B$ 0.02, 60 months, ICH 3×2, assay precision): share of bounds near zero, and runaway Gauss–Newton under the code's additive step and under the working-response form of the Algorithm |
| `supplement_tables.py` | Supplementary Material | Assembles the tables of the Supplementary Material (S1, S3–S10, S10b, S13–S17) from the outputs of the other scripts, so that no number is copied by hand |

Time grids enter only through $S_{tt,i}$, $n_i$ and $\mathrm{RSS}_i$; any grid (unequal spacing, different grids per condition, replicates) can be passed to `stage1`.

## Run

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
./run_all.sh        # ~30 s; writes output/*.txt and ../figures/fig{1,2,3}_*.{png,pdf,tiff} (TIFF: 1200 dpi line art, 600 dpi for fig3; fig3 is Figure S1)
```

The `.txt` files in `output/` are the runs used for the paper (seeds fixed in the scripts: 20260919 for Stage A, 1 for Stage B).
Script docstrings and some printed headers are in Japanese (the working language of the project). "paper-A" and "paper-B" in
the comments are earlier studies by the author (github.com/apo-cyber/bayesian-shelf-life-arrhenius and apo-cyber/arrhenius-r2-criterion).

Set `PAPERC_FIG_DIR` to write the figures elsewhere. `./export.sh` builds the standalone copy that is published (scripts,
`output/`, `figures/` as PNG/PDF, `LICENSE`, `CITATION.cff`).

## Data

`tamura2020.py` holds the 60-condition data of Tamura, Ono, Kawabe & Yonemochi (2020) *Chem. Pharm. Bull.* 68(11):1049–1054,
https://doi.org/10.1248/cpb.c20-00443, transcribed from the open-access supplementary material (Tables S1–S3); running the
script checks the transcription against the authors' rate constants. The article itself is not redistributed; the real-time
values in `illustration.py` were digitised from its Figure 7 as documented in `fig7_digitise.py`.

## Licence and citation

MIT licence (see `LICENSE`). Please cite the paper and the Zenodo archive of this repository (`CITATION.cff`).
