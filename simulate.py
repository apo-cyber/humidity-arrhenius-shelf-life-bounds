"""paper-C のシミュレーション表。Table 1(重み)、Table 3(設計 × 雑音)、Table 4(Stage B の budget、被覆の格子)、Table 5(価格表)。
出発点は docs/audit/humidity_stageB_import.py(09-19)。元の 5 設計は同じ seed・同じ乱数順なので監査の数字を桁まで再現し、
その後ろに ICH の例示格子 0/3/6 の行(A/B の n_pts = 3)と、3 か月比較 0/1/3 × 1 バッチの Stage B を足す。

    python3 simulate.py > output/simulate.txt      (~10 s)
"""
import numpy as np
from scipy import stats
import model as M

NREP = 4000


def row(o):
    s = M.summarise(o)
    return (f"{s['cover']:6.3f} {s['lb_true']:7.2f} {s['lb_med']:6.2f} {s['med_true']:8.2f} "
            f"{s['z_mean']:6.2f} {s['z_sd']:5.2f} {s['drop']:5.1f}")


print(f"truth Ea={M.EA} B={M.B_TRUE}; t90(25C/60%RH)={M.T90_TRUE:.1f} mo; k(40/75)={M.k_true(40, 75):.4f}/mo "
      f"({100 * (1 - np.exp(-M.k_true(40, 75) * 6)):.1f} % loss/6 mo); k(80/40)={M.k_true(80, 40):.3f}/mo "
      f"({100 * (1 - np.exp(-M.k_true(80, 40) * 14 / 30.4375)):.1f} % loss/14 d); nrep={NREP}")

ICH3 = [(T, rh) for T in (40, 50, 60) for rh in (60, 75)]
ICH2 = [(40, 60), (40, 75), (60, 60), (60, 75)]
ASAP5 = [(50, 75), (60, 40), (70, 5), (70, 75), (80, 40)]
# 元の 5 設計(監査と同じ順)+ 追加 2 設計(0/3/6)
designs = {
    'ICH 3x2 40/50/60 x RH60/75, 0/2/4/6 mo':   (ICH3, [M.MONTHS] * 6),
    'ICH 2x2 40/60 x RH60/75, 0/2/4/6 mo':      (ICH2, [M.MONTHS] * 4),
    'ASAP-5 (50/75,60/40,70/5,70/75,80/40) 14 d': (ASAP5, [M.DAYS14] * 5),
    'ASAP-5 28 d':                              (ASAP5, [M.DAYS28] * 5),
    'ASAP-5 28 d + 40/75 0/2/4/6 mo':           (ASAP5 + [(40, 75)], [M.DAYS28] * 5 + [M.MONTHS]),
    'ICH 3x2, 0/3/6 mo (ICH example grid)':     (ICH3, [M.MONTHS_ICH] * 6),
    'ICH 2x2, 0/3/6 mo':                        (ICH2, [M.MONTHS_ICH] * 4),
    'ASAP-5 isoconversion (each to alpha*)':    (ASAP5, [M.iso_grid(T, rh) for T, rh in ASAP5]),
    'ASAP-5 iso + 40/75 0/2/4/6 mo':            (ASAP5 + [(40, 75)], [M.iso_grid(T, rh) for T, rh in ASAP5] + [M.MONTHS]),
}
hdr = (f"{'design':46s} {'est':4s} {'cover':>6s} {'LB/true':>7s} {'LB/med':>6s} {'med/true':>8s} "
       f"{'z mean':>6s} {'z sd':>5s} {'drop%':>5s}")

# ================================================================ Table 1 / Table 3: Stage A, 推定量 × 設計 × 雑音
for noise, sig in (('assay sigma=0.02', 0.02), ('degradant sigma=0.0025', 0.0025), ('degradant at ASAP-reported precision, sigma=0.0075 (0.03 % on a 0.4 % limit)', 0.0075)):
    print(f"\n=== Stage A, {noise}; weak prior SD(Ea)=30, SD(B)=0.05, 1 batch; LB = one-sided 95 % on t90 ===\n" + hdr)
    rng = np.random.default_rng(20260919)
    for name, (conds, grids) in designs.items():
        s1 = M.stage1(rng, conds, grids, NREP, sig)
        for est in ('obs', 'fit', 'gn'):
            o = M.stage2(s1, [20, M.EA, M.B_TRUE], [100, 30, 0.05], est, sig_ref=sig)
            print(f"{name:46s} {est:4s} {row(o)}")
        print()

# SE(ln k_i) の一覧(本文 §2.2 / §3.3 の数字)
print("=== per-condition SE(ln k_i) = sigma/(k_i sqrt(S_tt,i)), 1 batch ===")
for name, (conds, grids) in designs.items():
    for sig in (0.02, 0.0075, 0.0025):
        se = [M.se_lnk(T, rh, g, sig) for (T, rh), g in zip(conds, grids)]
        loss = [100 * (1 - np.exp(-M.k_true(T, rh) * np.max(g))) for (T, rh), g in zip(conds, grids)]
        print(f"  {name:46s} sigma={sig}: " + ", ".join(f"{T}/{rh}: SE {s:.2f} (loss {l:.1f} %)" for (T, rh), s, l in zip(conds, se, loss)))


# ================================================================ Table 4: Stage B, 単一条件 + 輸入事前
def stageB_block(grid, batches, label, priors, do_grid=True, seed=1, sig=0.02, a0=9.0):
    print(f"\n=== Stage B: 40 C/75 %RH only, {label}, assay sigma={sig}; prior imported from Stage A; est=gn ===")
    L_T, L_RH = M.levers()
    print(f"levers to 25/60: L_T={L_T:.4f}/(kJ/mol), L_RH={L_RH:.0f} %RH.  shift(ln t90) = -(L_T*dEa + L_RH*dB): "
          f"dEa>0 or dB>0 -> longer t90 (anti-conservative)")
    rng = np.random.default_rng(seed)
    s1B = M.stage1(rng, [(40, 75)], [grid], NREP, sig, batches=batches)
    sdata = M.se_lnk(40, 75, grid, sig, batches)
    print(f"stage-1 SE(ln k) at 40/75 ({int(s1B['N'][0])} pts, df {int(s1B['DF'][0])}) = sigma/(k sqrt(Stt)) = {sdata:.3f}")
    print(f"sigma^2 prior for Stage B: IG(a0={a0}, b0=a0 sigma^2) (Stage A's pooled variance, {int(2 * a0)} df) "
          f"-> t with {int(2 * a0 + s1B['DF'][0] + 1)} df")
    print("closed form: s^2 = s_data^2 + (L_T SD_Ea)^2 + (L_RH SD_B)^2;  coverage = Phi((1.645 s - shift)/s_data);  "
          "budget = 1.645 (s - s_data)")
    out = {}
    for sdE, sdB in priors:
        s_cf = M.stageB_scale(sdata, sdE, sdB); bud = M.budget(s_cf, sdata)
        out[(sdE, sdB)] = (s_cf, bud)
        print(f"\n-- prior SD(Ea)={sdE}, SD(B)={sdB}: s={s_cf:.3f}, LB/med={np.exp(-M.Z95 * s_cf):.2f}, budget={bud:.3f} "
              f"-> dEa_max={bud / abs(L_T):.1f} ({bud / abs(L_T) / sdE:.2f} SD) alone | "
              f"dB_max={bud / abs(L_RH):.3f} ({bud / abs(L_RH) / sdB:.2f} SD) alone")
        if not do_grid:
            continue
        print(f"   {'dEa':>4s} {'dB':>6s} {'shift':>6s} | {'cf cov':>6s} | {'gn cov':>6s} {'LB/true':>7s} {'med/true':>8s}")
        for dE in (-10, 0, 5, 10, 15, 20):
            for dB in (-0.02, 0, 0.01, 0.02, 0.03):
                shift = M.shift_of(dE, dB); cf = M.coverage_cf(s_cf, shift, sdata)
                o = M.stage2(s1B, [20, M.EA + dE, M.B_TRUE + dB], [100, sdE, sdB], 'gn', a0=a0, sig_ref=sig)
                c = (o['lb'] <= M.LNT).mean()
                print(f"   {dE:4d} {dB:6.2f} {shift:+6.2f} | {cf:6.3f} | {c:6.3f} {np.exp(np.median(o['lb'])) / M.T90_TRUE:7.2f} "
                      f"{np.exp(np.median(o['med'])) / M.T90_TRUE:8.2f}{'  <90%' if c < 0.90 else ''}")
    return sdata, out


PRIORS = [(5, 0.01), (5, 0.02), (10, 0.01), (10, 0.02), (3, 0.005)]
sdata, _ = stageB_block(M.MONTHS, 3, 'grid 0/2/4/6 mo x 3 batches', PRIORS)

# ---- 共分散込みの輸入(§4.4): rho = corr(Ea, B) の事前(ASAP 型 Stage A で 0.4-0.6)
print("\n=== Stage B with correlated prior (closed form), s_data as above, prior (5, 0.01) ===")
for rho in (0.0, 0.25, 0.5):
    s = M.stageB_scale(sdata, 5, 0.01, rho)
    print(f"   rho={rho:.2f}: s={s:.3f}, LB/med={np.exp(-M.Z95 * s):.2f}, budget={M.budget(s, sdata):.3f}")
s0 = M.stageB_scale(sdata, 5, 0.01, 0.0); s5 = M.stageB_scale(sdata, 5, 0.01, 0.5)
# SD だけ輸入すると(真の Stage A 事後が rho=0.5 なら)幅を過小に見積もる。Stage A の事後で平均した被覆 = Phi(1.645 s_used / s_true)
print(f"   importing SDs only when rho=0.5: s understated by {100 * (1 - s0 / s5):.1f} %, bound {100 * (np.exp(-M.Z95 * s0) / np.exp(-M.Z95 * s5) - 1):.1f} % "
      f"longer than the transferred information supports, coverage averaged over the Stage A posterior "
      f"{stats.norm.cdf(M.Z95 * s0 / s5):.3f} instead of 0.950")

# ---- 他の格子(閉形式のみ + 被覆の 1 点確認)
for grid, batches, label in ((M.MONTHS_ICH, 3, 'grid 0/3/6 mo x 3 batches (ICH example)'),
                             (M.MONTHS_SUPAC, 1, 'grid 0/1/3 mo x 1 batch (3-month comparative, SUPAC-IR level 2)')):
    stageB_block(grid, batches, label, [(5, 0.01)], do_grid=True)

# ================================================================ Table 5: 頑健化の価格(閉形式)
print("\n=== Transfer inflation (closed form): inflate imported SDs by f until budget(f) >= shift; resulting LB/med ===")
L_T, L_RH = M.levers()
for grid, batches, label in ((M.MONTHS, 3, '0/2/4/6 x 3 batches'), (M.MONTHS_ICH, 3, '0/3/6 x 3 batches'),
                             (M.MONTHS_SUPAC, 1, '0/1/3 x 1 batch')):
    sd = M.se_lnk(40, 75, grid, 0.02, batches); pv = (L_T * 5) ** 2 + (L_RH * 0.01) ** 2
    s_base = M.stageB_scale(sd, 5, 0.01)
    print(f"base prior (5, 0.01), Stage B {label}: s_data={sd:.3f}, LB/med={np.exp(-M.Z95 * s_base):.2f}, budget={M.budget(s_base, sd):.3f}")
    for dE, dB in [(5, 0), (10, 0), (0, 0.01), (0, 0.02), (10, 0.02)]:
        shift = M.shift_of(dE, dB); f = M.inflation_f(shift, sd, pv); s = M.stageB_scale(sd, 5 * f, 0.01 * f)
        print(f"   cover dEa={dE:2d} dB={dB:.2f} (shift {shift:+.3f}): f={f:.2f} -> SD(Ea)={5 * f:.1f}, SD(B)={0.01 * f:.3f}; "
              f"LB/med {np.exp(-M.Z95 * s_base):.2f} -> {np.exp(-M.Z95 * s):.2f}")
print("\nrule of thumb: with s_data << prior part, budget ~ 1.645 s_prior and the tolerated drift is ~1 prior SD along the lever.")
