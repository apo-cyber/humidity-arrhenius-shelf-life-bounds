#!/usr/bin/env bash
# Reproduce every table and figure of paper-C. ~30 s on a laptop (closed forms + 4000-replicate simulations).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
mkdir -p output
$PY prop3_check.py > output/prop3_check.txt # Section 2.5: Proposition 3 (profile information = Stage 2 information), numerical identity
$PY bias.py      > output/bias.txt       # Section 2.5: closed-form bias of observed weights, eq. (13'), vs simulation
$PY compare.py   > output/compare.txt    # Table 1 (six one-sided bounds on the same data: paper-A as published, NIG-t with either weight, one-stage delta, FBH, ASAP-type MC)
$PY simulate.py  > output/simulate.txt   # Table 3 (designs x noise), Table 4 (bias budget + coverage grid), Table 5 (price)
$PY nomogram.py  > output/nomogram.txt   # Table 2 (geometry), variance decomposition, next-condition value, Figure 1
$PY budget.py    > output/budget.txt     # Figure 2 (coverage vs shift; price of robustness)
$PY robustness.py > output/robustness.txt # Tables 7-8 (batch random effect; kinetic order; isoconversion identity)
$PY pathways.py  > output/pathways.txt   # Section 5.5, Table 9: two pathways (Proposition 4), sign and size of the extrapolation displacement
$PY illustration.py > output/illustration.txt # Section 6, Table 9, Figure 3 (Tamura et al. 2020 data transcribed in tamura2020.py)
echo "done: output/*.txt and the figures (path printed above)"
