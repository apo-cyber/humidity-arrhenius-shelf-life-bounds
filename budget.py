"""paper-C §4 の閉形式の図。Figure 2:
  (a) 事前平均のずれ(shift、ln t90 単位)に対する片側 95 % 下限の被覆。輸入事前 4 種。× = budget(被覆 = 0.95 の点)
  (b) 頑健化の価格: 備えるずれ(shift)→ 必要な膨張 f → bound/median。Stage B の精度 3 種(0/2/4/6 × 3 バッチ、0/3/6 × 3、0/1/3 × 1)
上軸に dEa 単独 / dB 単独の目盛り。数値は simulate.py の Table 4 / 5 と同じ式(model.py)。

    python3 budget.py > output/budget.txt
"""
import os, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import model as M

HERE = pathlib.Path(__file__).resolve().parent
OUT = M.fig_dir()
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100,
                     "savefig.dpi": 300, "font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "mathtext.fontset": "dejavusans",
                     "pdf.fonttype": 42, "ps.fonttype": 42})


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png"); fig.savefig(OUT / f"{stem}.pdf")
    M.save_tiff(fig, OUT / f"{stem}.tiff", dpi=1200)   # 線画: T&F の規程 1200 dpi


L_T, L_RH = M.levers()
SDATA = M.se_lnk(40, 75, M.MONTHS, 0.02, 3)
PRIORS = [((3, 0.005), 'SD 3, 0.005 (tight)'), ((5, 0.01), 'SD 5, 0.01 (typical Stage A)'), ((10, 0.01), 'SD 10, 0.01'), ((10, 0.02), 'SD 10, 0.02')]
STAGEB = [(M.MONTHS, 3, '0/2/4/6 mo × 3 batches'), (M.MONTHS_ICH, 3, '0/3/6 mo × 3 batches'), (M.MONTHS_SUPAC, 1, '0/1/3 mo × 1 batch')]

print(f"s_data (40/75, 0/2/4/6 x 3 batches, sigma 0.02) = {SDATA:.3f}; levers L_T={L_T:.4f}, L_RH={L_RH:.0f}")
print("\n(a) coverage vs shift, closed form Phi((1.645 s - shift)/s_data)")
shifts = np.linspace(-0.3, 0.8, 221)
for (sdE, sdB), lab in PRIORS:
    s = M.stageB_scale(SDATA, sdE, sdB); b = M.budget(s, SDATA)
    print(f"  {lab:28s} s={s:.3f} budget={b:.3f}: coverage at shift 0 / 0.1 / 0.2 / 0.3 / 0.5 = "
          + " / ".join(f"{M.coverage_cf(s, x, SDATA):.3f}" for x in (0, 0.1, 0.2, 0.3, 0.5)))
print("\n(b) bound/median after inflating the prior (5, 0.01) to cover the shift")
for grid, batches, lab in STAGEB:
    sd = M.se_lnk(40, 75, grid, 0.02, batches); pv = (L_T * 5) ** 2 + (L_RH * 0.01) ** 2
    print(f"  Stage B {lab:24s} s_data={sd:.3f}: bound/med at covered shift 0 / 0.1 / 0.2 / 0.3 / 0.5 = "
          + " / ".join(f"{np.exp(-M.Z95 * M.stageB_scale(sd, 5 * M.inflation_f(x, sd, pv), 0.01 * M.inflation_f(x, sd, pv))):.2f}" for x in (0, 0.1, 0.2, 0.3, 0.5)))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.1))
for (sdE, sdB), lab in PRIORS:
    s = M.stageB_scale(SDATA, sdE, sdB); b = M.budget(s, SDATA)
    ax1.plot(shifts, M.coverage_cf(s, shifts, SDATA), label=lab)
    ax1.plot(b, 0.95, 'x', color='k', ms=5)
ax1.axhline(0.95, color='#999999', lw=0.6, ls='--')
ax1.set_xlabel('shift of the imported mean in $\\ln t_{90}$ (+ = anti-conservative)')
ax1.set_ylabel('coverage of the one-sided 95 % bound'); ax1.set_ylim(0, 1.02); ax1.set_xlim(-0.3, 0.8)
ax1.set_title('(a) coverage; × = bias budget', loc='left'); ax1.legend(frameon=False, fontsize=7, loc='lower left')
sec = ax1.secondary_xaxis('top', functions=(lambda x: x / abs(L_T), lambda e: e * abs(L_T)))
sec.set_xlabel('$\\Delta E_a$ alone (kJ/mol)', fontsize=7.5); sec.tick_params(labelsize=7)

ins = np.linspace(0, 0.8, 161)
for grid, batches, lab in STAGEB:
    sd = M.se_lnk(40, 75, grid, 0.02, batches); pv = (L_T * 5) ** 2 + (L_RH * 0.01) ** 2
    y = [np.exp(-M.Z95 * M.stageB_scale(sd, 5 * M.inflation_f(x, sd, pv), 0.01 * M.inflation_f(x, sd, pv))) for x in ins]
    ax2.plot(ins, y, label=f'Stage B {lab} ($s_{{\\rm data}}$ = {sd:.2f})')
for dE, dB, txt in ((10, 0, '$\\Delta E_a$=10'), (0, 0.02, '$\\Delta B$=0.02'), (10, 0.02, 'both')):
    x = M.shift_of(dE, dB); ax2.axvline(x, color='#bbbbbb', lw=0.6); ax2.text(x, 0.98, txt, fontsize=6.5, ha='center', va='top', color='#555555')
ax2.set_xlabel('displacement covered, shift in $\\ln t_{90}$'); ax2.set_ylabel('one-sided 95 % bound / median')
ax2.set_ylim(0, 1); ax2.set_xlim(0, 0.8); ax2.set_title('(b) bound after inflating the prior (SD 5, 0.01)', loc='left')
ax2.legend(frameon=False, fontsize=6.5, loc='lower left')
sec2 = ax2.secondary_xaxis('top', functions=(lambda x: x / abs(L_RH), lambda e: e * abs(L_RH)))
sec2.set_xlabel('$\\Delta B$ alone (per % RH)', fontsize=7.5); sec2.tick_params(labelsize=7)
fig.tight_layout(); save(fig, 'fig2_bias_budget')
print(f"\nfigure: {os.path.relpath(OUT / 'fig2_bias_budget.png')}")
