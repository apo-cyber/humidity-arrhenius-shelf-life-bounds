"""paper-C のシミュレーション表。補足 S1(設計 × 雑音)、本文 Table 3 と補足 S3・S4・S9(Stage B の budget、被覆の格子、膨張)、§2.3 の自由度の例。
出発点は docs/audit/humidity_stageB_import.py(09-19)。10-27 R2 で Stage A は平坦な事前・Stage 1 はバッチ別の切片になったので、監査(09-19)の数字とは一致しない。
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
    print(f"\n=== Stage A, {noise}; flat prior (a0 = 0), 1 batch; LB = one-sided 95 % on t90 ===\n" + hdr)
    rng = np.random.default_rng(20260919)
    for name, (conds, grids) in designs.items():
        s1 = M.stage1(rng, conds, grids, NREP, sig)
        for est in ('obs', 'fit', 'gn'):
            o = M.stage2(s1, [20, M.EA, M.B_TRUE], None, est, sig_ref=sig)
            print(f"{name:46s} {est:4s} {row(o)}")
        print()

# SE(ln k_i) の一覧(本文 §2.2 / §3.3 の数字)
print("=== per-condition SE(ln k_i) = sigma/(k_i sqrt(S_tt,i)), 1 batch ===")
for name, (conds, grids) in designs.items():
    for sig in (0.02, 0.0075, 0.0025):
        se = [M.se_lnk(T, rh, g, sig) for (T, rh), g in zip(conds, grids)]
        loss = [100 * (1 - np.exp(-M.k_true(T, rh) * np.max(g))) for (T, rh), g in zip(conds, grids)]
        print(f"  {name:46s} sigma={sig}: " + ", ".join(f"{T}/{rh}: SE {s:.2f} (loss {l:.1f} %)" for (T, rh), s, l in zip(conds, se, loss)))


# ================================================================ Table 4 / 6: Stage B, 単一条件 + 輸入事前
# 10-27: 標準は切り離した Stage B(M.stageB_decoupled: 係数の事前は絶対単位、σ² は Stage B だけ、Satterthwaite の t)。
# 比較のため 10-22 版の NIG(係数の事前が σ² に比例、IG(a0, a0 σ²) で分散を運ぶ)の被覆も並べる。
def stageB_block(grid, batches, label, priors, do_grid=True, seed=1, sig=0.02, a0=M.A0_STAGEB):
    print(f"\n=== Stage B: 40 C/75 %RH only, {label}, assay sigma={sig}; prior imported from Stage A; decoupled (standard) ===")
    L_T, L_RH = M.levers()
    print(f"levers to 25/60: L_T={L_T:.4f}/(kJ/mol), L_RH={L_RH:.0f} %RH.  shift(ln t90) = -(L_T*dEa + L_RH*dB): "
          f"dEa>0 or dB>0 -> longer t90 (anti-conservative)")
    rng = np.random.default_rng(seed)
    s1B = M.stage1(rng, [(40, 75)], [grid], NREP, sig, batches=batches)
    sdata = M.se_lnk(40, 75, grid, sig, batches); df_own = int(s1B['DF'][0])
    var_A = (a0, a0 * sig ** 2)                     # 同じ分析法・分析室を確かめた場合の Stage A の分散(例外でだけ使う)
    df = M.stageB_var_df(df_own, var_A)
    print(f"stage-1 SE(ln k) at 40/75 ({int(s1B['N'][0])} pts, df {df_own}) = sigma/(k sqrt(Stt)) = {sdata:.3f}; "
          f"sigma^2 {'from Stage B alone' if df == df_own else 'pooled with Stage A (exception: df < ' + str(M.DF_MIN_OWN) + ', same method and laboratory)'} ({df} df), Welch-Satterthwaite t; k_hat <= 0 in {100 * (s1B['KH'][:, 0] <= 0).mean():.1f} % of replicates")
    print("closed form: s^2 = s_data^2 + (L_T SD_Ea)^2 + (L_RH SD_B)^2;  coverage = Phi((q s - shift)/s_data) with q = 1.645 (normal) "
          "or the t quantile on the expected Satterthwaite df;  budget = q s - 1.645 s_data")
    out = {}
    for sdE, sdB in priors:
        s_cf = M.stageB_scale(sdata, sdE, sdB); bud = M.budget(s_cf, sdata)
        nu_e = M.stageB_nu_expected(sdata, sdE, sdB, df); q_t = stats.t.ppf(0.95, nu_e); bud_t = q_t * s_cf - M.Z95 * sdata
        out[(sdE, sdB)] = (s_cf, bud)
        print(f"\n-- prior SD(Ea)={sdE}, SD(B)={sdB}: s={s_cf:.3f}, LB/med={np.exp(-M.Z95 * s_cf):.2f} (t: {np.exp(-q_t * s_cf):.2f}, nu~{nu_e:.1f}), "
              f"budget={bud:.3f} (t: {bud_t:.3f}) -> dEa_max={bud / abs(L_T):.1f} ({bud / abs(L_T) / sdE:.2f} SD) alone | "
              f"dB_max={bud / abs(L_RH):.3f} ({bud / abs(L_RH) / sdB:.2f} SD) alone")
        if not do_grid:
            continue
        print(f"   {'dEa':>4s} {'dB':>6s} {'shift':>6s} | {'cf cov':>6s} {'cf t':>6s} | {'dec cov':>7s} {'LB/true':>7s} {'med/true':>8s} | {'NIG cov':>7s} {'LB/true':>7s}")
        for dE in (-10, 0, 5, 10, 15, 20):
            for dB in (-0.02, 0, 0.01, 0.02, 0.03):
                shift = M.shift_of(dE, dB); cf = M.coverage_cf(s_cf, shift, sdata)
                cft = stats.norm.cdf((q_t * s_cf - shift) / sdata)
                od = M.stageB_decoupled(s1B, [M.EA + dE, M.B_TRUE + dB], [sdE, sdB], var_A=var_A); c = (od['lb'] <= M.LNT).mean()
                on = M.stage2(s1B, [20, M.EA + dE, M.B_TRUE + dB], [100, sdE, sdB], 'gn', a0=a0, sig_ref=sig); cn = (on['lb'] <= M.LNT).mean()
                print(f"   {dE:4d} {dB:6.2f} {shift:+6.2f} | {cf:6.3f} {cft:6.3f} | {c:7.3f} {np.exp(np.median(od['lb'])) / M.T90_TRUE:7.2f} "
                      f"{np.exp(np.median(od['med'])) / M.T90_TRUE:8.2f} | {cn:7.3f} {np.exp(np.median(on['lb'])) / M.T90_TRUE:7.2f}{'  <90%' if c < 0.90 else ''}")
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

# ================================================================ §2.3: 厳密な共役の数え方(p_eff を引かない)の例(10-27 R2、平坦な事前)
print("\n=== Strict conjugate count (p_eff not subtracted) vs the count of Section 2.3; ICH 2x2 40/60 x 60/75, 0/3/6 mo, sigma=0.0025, flat prior ===")
for fix in (False, True):
    M.DF_FIX = fix
    rng = np.random.default_rng(20260919)
    s1 = M.stage1(rng, [(40, 60), (40, 75), (60, 60), (60, 75)], [M.MONTHS_ICH] * 4, NREP, 0.0025)
    o = M.stage2(s1, [20, M.EA, M.B_TRUE], None, 'gn', sig_ref=0.0025); s = M.summarise(o)
    print(f"  {'p_eff subtracted (this paper)' if fix else 'strict conjugate count       '}: nu = {np.median(o['nu']):.0f}, cover {s['cover']:.3f}, LB/true {s['lb_true']:.2f}")
M.DF_FIX = True
