"""模擬査読 M2(10-27): 係数の事前が σ² に比例する(β | σ² ~ N(μ0, σ² Σ0*)、Σ0* = 絶対単位の共分散 / σ_ref²)。
開発の分析室(Stage A、σ_A)と商用の QC 分析室(Stage B、σ_B)の精度が違うと、輸入した (Ea, B) の事前は絶対単位で σ̂²/σ_A² 倍に伸び縮みし、
分散の事前 IG(7.5, 7.5 σ_A²) は Stage B のデータの分散を σ_A² の側へ引く。比 r = σ_B/σ_A を 1/4〜4 で振って、片側 95 % 下限の被覆と長さを見る。

Stage B(40/75、0/2/4/6 か月 × 3 バッチ = バッチ別の切片で残差 df 8、σ_B = 0.02 で固定、σ_A = σ_B / r)、輸入する事前 SD(Ea) = 5 kJ/mol、SD(B) = 0.01 /%RH(絶対単位)。
  coupled + IG    本稿: NIG、σ_ref = σ_A、分散の事前 IG(7.5, 7.5 σ_A²)(Stage A の 15 df を運ぶ)
  coupled         NIG、σ_ref = σ_A、分散は運ばない(IG(1/2, σ_A²/2))
  decoupled       係数の事前を σ² から切り離す(絶対単位のまま): s² = σ̂_B² / (k̂² S_tt) + ℓ' Σ_A ℓ、σ̂_B² は Stage B だけ(8 df)、
                  t の df は Welch–Satterthwaite(robustness.py の batch-as-unit と同じ形)。閉形式のまま。
  decoupled + IG  同じく切り離し、ただし σ̂² は IG(7.5, 7.5 σ_A²) と Stage B の残差の合併(23 df)
真値の事前(ずれ 0)と、ずらした事前(ΔEa = +5 kJ/mol、ΔB = +0.01 /%RH)。
(10-27 R2: Stage A の既定が平坦になり σ_ref がどこにも入らないので、Stage A で σ_ref をずらす節は削除した。)

    python3 m2_sigma_mismatch.py > output/m2_sigma_mismatch.txt   (~10 s)
"""
import numpy as np
from scipy import stats
import model as M

NREP = 4000
SIG_B = 0.02
SD_E, SD_B = 5.0, 0.01
RATIOS = (0.25, 0.5, 1 / np.sqrt(2), 1.0, np.sqrt(2), 2.0, 4.0)
L_T, L_RH = M.levers()
P_LEVER = (L_T * SD_E) ** 2 + (L_RH * SD_B) ** 2


def row(o):
    s = M.summarise(o)
    return f"{s['cover']:6.3f} {s['lb_true']:5.2f}"


def decoupled(s1, dE, dB, a_tr=None, b_tr=None):
    """係数の事前は絶対単位、σ² は Stage B の残差(+ 運んだ IG)から。Welch–Satterthwaite の t。"""
    KH = s1['KH'][:, 0]; RSS = s1['RSS'][:, 0]; STT = s1['STT'][0]; DF = s1['DF'][0]
    lnk = np.log(np.maximum(KH, 1e-12))
    if a_tr is None:
        s2 = RSS / DF; nu_d = DF
    else:
        s2 = (2 * b_tr + RSS) / (2 * a_tr + DF); nu_d = 2 * a_tr + DF
    v_data = s2 / (KH ** 2 * STT)
    var = v_data + P_LEVER; nu = var ** 2 / (v_data ** 2 / nu_d)
    med = M.LN_C - (lnk + L_T * (M.EA + dE) + L_RH * (M.B_TRUE + dB))
    return dict(lb=med - stats.t.ppf(0.95, nu) * np.sqrt(var), med=med, scale=np.sqrt(var), drop=0.0)


print(f"=== Stage B: 40/75, 0/2/4/6 mo x 3 batches, sigma_B={SIG_B} (fixed), sigma_A = sigma_B / r; imported prior SD(Ea)={SD_E}, SD(B)={SD_B} (absolute) ===")
print(f"levers L_T={L_T:.4f}, L_RH={L_RH:.0f}; prior term l'Sigma_A l = {P_LEVER:.4f} (sd {np.sqrt(P_LEVER):.3f}); "
      f"s_data = {M.se_lnk(40, 75, M.MONTHS, SIG_B, 3):.3f}")
print("columns: coverage, median LB/true;  'prior x' = median of the factor by which the coupled analysis rescales the imported prior (sigma_hat^2 / sigma_A^2)")
for label, dE, dB in (('prior exact', 0.0, 0.0), ('prior displaced dEa=+5', 5.0, 0.0), ('prior displaced dB=+0.01', 0.0, 0.01)):
    print(f"\n-- {label} (shift {M.shift_of(dE, dB):+.3f})")
    print(f"{'r=sB/sA':>8s} {'var ratio':>9s} | {'coupled+IG (NIG)':>18s} {'prior x':>7s} | {'coupled':>12s} {'prior x':>7s} | "
          f"{'decoupled':>12s} | {'decoupled+IG':>12s}")
    for r in RATIOS:
        sig_A = SIG_B / r
        rng = np.random.default_rng(1)
        s1 = M.stage1(rng, [(40, 75)], [M.MONTHS], NREP, SIG_B, batches=3)
        mu = [20, M.EA + dE, M.B_TRUE + dB]
        o1 = M.stage2(s1, mu, [100, SD_E, SD_B], 'gn', a0=M.A0_STAGEB, sig_ref=sig_A)
        o2 = M.stage2(s1, mu, [100, SD_E, SD_B], 'gn', a0=0.5, sig_ref=sig_A)
        f1 = np.median((o1['cov'][:, 1, 1]) / SD_E ** 2); f2 = np.median((o2['cov'][:, 1, 1]) / SD_E ** 2)
        o3 = decoupled(s1, dE, dB)
        o4 = decoupled(s1, dE, dB, a_tr=M.A0_STAGEB, b_tr=M.A0_STAGEB * sig_A ** 2)
        print(f"{r:8.2f} {r ** 2:9.2f} | {row(o1):>18s} {f1:7.2f} | {row(o2):>12s} {f2:7.2f} | {row(o3):>12s} | {row(o4):>12s}")

# ---- Stage B の残差 df < 5 の例外(10-27): 0/1/3 × 1 バッチ(df 1)で、分散を Stage B だけ / Stage A と合併(同じ分析室)
print("\n=== Stage B with few residual df: 0/1/3 mo x 1 batch (df 1) vs 0/2/4/6 mo x 3 batches (df 8), sigma_A = sigma_B = 0.02; decoupled ===")
print(f"{'design':22s} {'prior':22s} | {'Stage B variance only':>22s} | {'pooled with Stage A IG(7.5, 7.5 s^2)':>32s}")
for grid, batches, lab in ((M.MONTHS_SUPAC, 1, '0/1/3 x 1 batch'), (M.MONTHS, 3, '0/2/4/6 x 3 batches')):
    rng = np.random.default_rng(1)
    s1 = M.stage1(rng, [(40, 75)], [grid], NREP, SIG_B, batches=batches)
    for plab, dE, dB in (('exact', 0.0, 0.0), ('dB = +0.01', 0.0, 0.01), ('dEa = +5', 5.0, 0.0)):
        own = M.summarise(M.stageB_decoupled(s1, [M.EA + dE, M.B_TRUE + dB], [SD_E, SD_B]))
        if s1['DF'][0] < M.DF_MIN_OWN:
            pooled = M.summarise(M.stageB_decoupled(s1, [M.EA + dE, M.B_TRUE + dB], [SD_E, SD_B], var_A=(M.A0_STAGEB, M.A0_STAGEB * SIG_B ** 2)))
            pooled_txt = f"{pooled['cover']:.3f} / {pooled['lb_true']:.2f}"
        else:
            pooled_txt = "(not applied: df >= 5)"
        print(f"{lab:22s} {plab:22s} | {own['cover']:.3f} / {own['lb_true']:.2f}{'':>9s} | {pooled_txt:>32s}")
