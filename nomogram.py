"""paper-C の設計節(§3)の閉形式。Table 2(設計の形だけ: SE(ln k) 一律 0.10)、分散の内訳、次の 1 条件の価値(rank-1)、
Figure 1(bound/median を SE(ln k) の関数として、設計別)。出発点は docs/audit/humidity_identifiability.py(09-18)。

    python3 nomogram.py > output/nomogram.txt        (数秒。図は ../figures/ に PNG + PDF + TIFF)
"""
import os, pathlib
import numpy as np
from scipy import stats
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


ICH3 = [(T, rh) for T in (40, 50, 60) for rh in (60, 75)]
ICH2 = [(40, 60), (40, 75), (60, 60), (60, 75)]
ASAP5 = [(50, 75), (60, 40), (70, 5), (70, 75), (80, 40)]
GEOM = {
    'T-only 40/50/60 (RH 75 fixed)':          [(40, 75), (50, 75), (60, 75)],
    'ICH-type 40/50/60 x 60/75':              ICH3,
    'ICH-type 40/60 x 60/75':                 ICH2,
    'ASAP 50/75, 60/40, 70/5, 70/75, 80/40':  ASAP5,
    'ASAP + 40/75':                           ASAP5 + [(40, 75)],
}

# ================================================================ Table 2: 設計の形だけ(SE(ln k) 一律 0.10、平坦事前)
SIG_U = 0.10
print(f"=== Table 2: geometry only, SE(ln k_i) = {SIG_U} at every condition, flat prior, target 25 C / 60 %RH ===")
print(f"{'design':40s} {'n':>2s} {'SE(Ea)':>7s} {'SE(B)':>7s} {'corr':>6s} {'sd0':>6s} {'bound/med':>9s}")
for name, conds in GEOM.items():
    d = M.design_cov(conds, SIG_U ** 2)
    if d is None:
        print(f"{name:40s} {len(conds):2d} {'6.1':>7s}   ----  B not identifiable (single RH)"); continue
    print(f"{name:40s} {len(conds):2d} {d['se_Ea']:7.1f} {d['se_B']:7.3f} {d['corr']:6.2f} {d['sd0']:6.3f} "
          f"{M.bound_over_median(d['sd0']):9.2f}")
X = np.array([[1.0, M.x1_abs(T)] for T in (40, 50, 60)]); S = SIG_U ** 2 * np.linalg.inv(X.T @ X); x = np.array([1, M.x1_abs(25)])
print(f"(T-only 3-point Arrhenius, RH term absent: SE(Ea)={np.sqrt(S[1, 1]):.1f}, sd0={np.sqrt(x @ S @ x):.3f}, "
      f"bound/med={M.bound_over_median(np.sqrt(x @ S @ x)):.2f})")

# ================================================================ 分散の内訳(§3.4)
print(f"\n=== Variance decomposition of sd0^2 at 25/60 (centred at the precision-weighted centroid; flat prior) ===")
print(f"{'design':40s} {'total':>7s} {'icpt':>7s} {'T':>7s} {'RH':>7s} {'EaxB':>7s} | share T")
for label, v_of in (('uniform SE 0.10', lambda conds, grids: SIG_U ** 2),
                    ('assay sigma=0.02, own grid', lambda conds, grids: np.array([M.se_lnk(T, rh, g, 0.02) ** 2 for (T, rh), g in zip(conds, grids)])),
                    ('degradant sigma=0.0025, own grid', lambda conds, grids: np.array([M.se_lnk(T, rh, g, 0.0025) ** 2 for (T, rh), g in zip(conds, grids)]))):
    print(f"-- {label}")
    ISO = [M.iso_grid(T, rh) for T, rh in ASAP5]
    for name, conds, grids in (('ICH-type 3x2, 0/2/4/6', ICH3, [M.MONTHS] * 6), ('ICH-type 2x2, 0/2/4/6', ICH2, [M.MONTHS] * 4),
                               ('ASAP-5 isoconversion', ASAP5, ISO), ('ASAP-5 iso + 40/75', ASAP5 + [(40, 75)], ISO + [M.MONTHS]),
                               ('ASAP-5 fixed 28 d', ASAP5, [M.DAYS28] * 5)):
        d = M.design_cov(conds, v_of(conds, grids)); t = d['terms']; tot = d['sd0'] ** 2
        print(f"{name:40s} {tot:7.3f} {t['intercept']:7.3f} {t['T']:7.3f} {t['RH']:7.3f} {t['EaB']:7.3f} | {100 * t['T'] / tot:4.0f} %  "
              f"(cross with intercept {t['icpt_T'] + t['icpt_RH']:+.1e}; bound/med {M.bound_over_median(d['sd0']):.2f})")

# ================================================================ 次の 1 条件の価値(rank-1、§3.4)
print("\n=== Time to a 0.4 % degradant limit at the ASAP-5 conditions at the reference truth (0.004 / k, small conversion) ===")
print('   ' + '; '.join(f"{T} C / {rh} %RH: {0.004 / M.k_true(T, rh) * 30.4375:.2f} d" for T, rh in ASAP5))
print(f"\n=== Value of one more condition added to ASAP-5 (rank-1 update); bound/median before -> after ===")
CANDS = [(40, 40), (40, 60), (40, 75), (50, 40), (60, 75), (70, 40), (80, 75), (90, 40), (90, 75)]
for label, v_base, v_new in (('uniform SE 0.10 (new condition also 0.10; = ASAP isoconversion at sigma/alpha* 0.075)', lambda: SIG_U ** 2, lambda T, rh: SIG_U ** 2),
                             ('assay sigma=0.02, ASAP isoconversion base, new condition 0/2/4/6 mo', lambda: np.array([M.se_lnk(T, rh, M.iso_grid(T, rh), 0.02) ** 2 for T, rh in ASAP5]), lambda T, rh: M.se_lnk(T, rh, M.MONTHS, 0.02) ** 2),
                             ('degradant sigma=0.0075, ASAP isoconversion base, new condition 0/2/4/6 mo', lambda: np.array([M.se_lnk(T, rh, M.iso_grid(T, rh), 0.0075) ** 2 for T, rh in ASAP5]), lambda T, rh: M.se_lnk(T, rh, M.MONTHS, 0.0075) ** 2),
                             ('degradant sigma=0.0075, ASAP isoconversion base, new condition also isoconversion', lambda: np.array([M.se_lnk(T, rh, M.iso_grid(T, rh), 0.0075) ** 2 for T, rh in ASAP5]), lambda T, rh: M.se_lnk(T, rh, M.iso_grid(T, rh), 0.0075) ** 2)):
    d = M.design_cov(ASAP5, v_base(), ref='centroid')
    # rank-1 は同じ座標で行う: 重心を固定するため ref を明示
    w = 1 / np.broadcast_to(np.asarray(v_base(), float), (5,)); Xa = M.design_matrix(ASAP5, None)
    c1 = (w * Xa[:, 1]).sum() / w.sum(); c2 = (w * Xa[:, 2]).sum() / w.sum()
    print(f"-- {label}: before {M.bound_over_median(d['sd0']):.2f} (sd0 {d['sd0']:.3f})")
    for T, rh in CANDS:
        xp = np.array([1.0, M.x1_abs(T) - c1, rh - c2]); Sp = M.rank1_add(d['Sigma'], xp, v_new(T, rh))
        sd = np.sqrt(d['x0'] @ Sp @ d['x0'])
        print(f"   + {T}/{rh}: {M.bound_over_median(sd):.2f}")

# ================================================================ Figure 1: bound/median vs SE(ln k)
DESIGNS = [('ICH-type 3×2, 0/2/4/6 mo', ICH3, [M.MONTHS] * 6, '-', 'C0'),
           ('ICH-type 3×2, 0/3/6 mo', ICH3, [M.MONTHS_ICH] * 6, ':', 'C0'),
           ('ICH-type 2×2, 0/2/4/6 mo', ICH2, [M.MONTHS] * 4, '-', 'C1'),
           ('ASAP five conditions, isoconversion', ASAP5, [M.iso_grid(T, rh) for T, rh in ASAP5], '-', 'C3'),
           ('ASAP five conditions, fixed 28 d', ASAP5, [M.DAYS28] * 5, '--', 'C3'),
           ('ASAP isoconversion + 40/75 0/2/4/6 mo', ASAP5 + [(40, 75)], [M.iso_grid(T, rh) for T, rh in ASAP5] + [M.MONTHS], '-', 'C2')]
# Table 3 のシミュレーション値(simulate.py, gn)を点で重ねる: (assay, degradant) の median LB/true
# (assay 0.02, degradant-at-ASAP-precision 0.0075, optimistic degradant 0.0025)
SIM = {'ICH-type 3×2, 0/2/4/6 mo': (0.57, 0.81, 0.93), 'ICH-type 3×2, 0/3/6 mo': (0.55, 0.80, 0.93), 'ICH-type 2×2, 0/2/4/6 mo': (0.40, 0.71, 0.89),
       'ASAP five conditions, isoconversion': (0.38, 0.69, 0.89), 'ASAP five conditions, fixed 28 d': (0.04, 0.32, 0.69),
       'ASAP isoconversion + 40/75 0/2/4/6 mo': (0.50, 0.77, 0.92)}   # 10-27 R2: 平坦な事前・a0 = 0(output/simulate.txt)


def nu_of(grids):
    nu = sum(len(g) - 2 for g in grids) + len(grids)        # 2 a_n with a0 = 0 (flat, 10-27 R2)
    return nu - 3 if M.DF_FIX else nu                            # 平坦な事前: p_eff = 3(10-27、M1)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.9))
se_u = np.logspace(np.log10(0.02), 0, 80)
seen = set()
for name, conds, grids, ls, col in DESIGNS:
    key = tuple(conds)
    if key in seen:
        continue
    seen.add(key)
    y = [M.bound_over_median(M.design_cov(conds, s ** 2)['sd0']) for s in se_u]
    ls_a = ('-', '--', '-', ':')[len(seen) - 1]                                       # 10-27(m10): 重なる 3 本を線種で区別
    ax1.plot(se_u, y, ls_a, color=col, label=name.split(',')[0].replace(' 0/2/4/6 mo', ''))
ax1.set_xscale('log'); ax1.set_xlabel('SE(ln $k_i$), the same at every condition'); ax1.set_ylabel('one-sided 95 % bound / median')
ax1.set_ylim(0, 1); ax1.set_title('(a) geometry only', loc='left', pad=22); ax1.legend(frameon=False, fontsize=7)
sig = np.logspace(np.log10(0.001), np.log10(0.05), 100)
for name, conds, grids, ls, col in DESIGNS:
    y = []
    for s in sig:
        v = np.array([M.se_lnk(T, rh, g, s) ** 2 for (T, rh), g in zip(conds, grids)])
        y.append(M.bound_over_median(M.design_cov(conds, v)['sd0'], nu_of(grids)))
    ax2.plot(sig, y, ls, color=col, label=name)
    ax2.plot([0.02, 0.0075, 0.0025], SIM[name], 'o', color=col, ms=3.5, mfc='white')
    cf3 = []
    for s, sim in zip((0.02, 0.0075, 0.0025), SIM[name]):                            # 閉形式と白丸の差(本文 §3.3)
        v = np.array([M.se_lnk(T, rh, g, s) ** 2 for (T, rh), g in zip(conds, grids)])
        cf3.append(f"sigma {s}: max SE(ln k) {np.sqrt(v.max()):.2f}, closed form {M.bound_over_median(M.design_cov(conds, v)['sd0'], nu_of(grids)):.2f} vs simulated {sim:.2f}")
    print(f"Figure 1b, {name}: " + '; '.join(cf3))
for s, lab in ((0.0025, 'σ/α* = 0.025'), (0.0075, '0.075\n(ASAP-reported)'), (0.02, '0.2 (assay)')):
    ax2.axvline(s, color='#999999', lw=0.6); ax2.text(s, 1.01, lab, ha='center', va='bottom', fontsize=6.5, color='#555555')   # 縦線の上端に(凡例と重ならない)
ax2.set_xscale('log'); ax2.set_xlabel('measurement error $\\sigma$ on $\\ln(C/C_0)$  (= $\\sigma/\\alpha^*$ × 0.10)'); ax2.set_ylim(0, 1)
ax2.set_title('(b) each design with its own grid and rates', loc='left', pad=22)
ax2.legend(frameon=False, fontsize=6.5, loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=2)   # 10-27(m14): 凡例を軸の下へ、曲線と重ねない
fig.tight_layout(); save(fig, 'fig1_design_nomogram')
print(f"\nfigure: {os.path.relpath(OUT / 'fig1_design_nomogram.png')}  (open circles = simulated median LB/true from simulate.py)")
