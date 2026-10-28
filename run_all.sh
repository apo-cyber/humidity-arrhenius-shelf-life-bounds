#!/usr/bin/env bash
# Reproduce every table and figure of paper-C. ~30 s on a laptop (closed forms + 4000-replicate simulations).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
mkdir -p output
$PY prop3_check.py > output/prop3_check.txt # Supplementary S2: Remark 2 (profile information = Stage 2 information), numerical identity
$PY bias.py      > output/bias.txt       # Section 2.5: Proposition 1 (bias of observed weights), closed form vs simulation, incl. three temperatures
$PY compare.py   > output/compare.txt    # Table 1 (one-sided bounds on the same data: NIG-t with either weight, one-stage delta, FBH, ASAP-type MC)
$PY simulate.py  > output/simulate.txt   # Supplementary S1 (designs x noise), Table 3 and S3/S4/S9 (single-condition study: budget, coverage grid, inflation)
$PY nomogram.py  > output/nomogram.txt   # Table 2 (geometry), variance decomposition, next-condition value, Figure 1
$PY budget.py    > output/budget.txt     # Figure 2 (coverage vs shift; bound after inflating the prior)
$PY robustness.py > output/robustness.txt # Table 5 and Supplementary S5 (batch random effect; kinetic order; isoconversion identity)
$PY pathways.py  > output/pathways.txt   # Section 5.5 and Supplementary S6: two pathways (Proposition 2), sign and size of the extrapolation displacement
$PY illustration.py > output/illustration.txt # Section 6, Table 6, Supplementary S10 and Figure S1 (Tamura et al. 2020 data transcribed in tamura2020.py)
$PY m2_sigma_mismatch.py > output/m2_sigma_mismatch.txt # Stage B when the two laboratories' assay precisions differ (sigma_B/sigma_A = 1/4-4): NIG vs decoupled
$PY sensitivity_truth.py > output/sensitivity_truth.txt # sensitivity grid: 54 truths (Ea 40-120, B, shelf life) x 3 designs x 2 precisions; flat prior, weak prior for comparison
$PY prior_checks.py > output/prior_checks.txt # Supplementary S17: the informative-prior branch (sigma_ref x 1/2, 1, 2; p_eff subtracted or not)
$PY slow_cells.py > output/slow_cells.txt # Supplementary S7: the slow-product cell, bounds near zero and runaway Gauss-Newton under two implementations
$PY worked_example.py > output/worked_example.txt # Supplementary S15: one data set worked through Stage 1, Stage 2 and the bound
$PY supplement_tables.py > output/supplement_tables.md # the tables of the Supplementary Material, assembled from the outputs above
echo "done: output/*.txt and the figures (path printed above)"
