"""補足 S15: 計算例(模擬査読 2 回目 M5「実装を手で確かめられる 1 例」)。ICH 型 3×2(40/50/60 °C × 60/75 % RH、0/2/4/6 か月)、
assay σ = 0.02、基準の真値(Ea 80 kJ/mol、B 0.04 /%RH、t90 = 61.6 か月)から 1 組のデータを作り、
Stage 1 の要約 → Stage 2(平坦な事前、k 空間 Gauss–Newton の反復)→ 片側 95 % 下限を、途中の数値つきで出す。一段の当てはめと同じ下限になることも示す。

    python3 worked_example.py > output/worked_example.txt   (<1 s)
"""
import numpy as np
from scipy import stats
import model as M

SIG = 0.02
CONDS = [(T, rh) for T in (40, 50, 60) for rh in (60, 75)]
rng = np.random.default_rng(20261027)

# ---- データ(ln(C/C0)、切片 0)
t = M.MONTHS
Y = np.array([-M.k_true(T, rh) * t + rng.normal(0, SIG, t.size) for T, rh in CONDS])
print("=== Worked example: ICH-type 3x2, 0/2/4/6 months, assay sigma = 0.02, reference truth (Ea 80, B 0.04, t90 61.6 mo) ===")
print("data y = ln(C/C0) at 0 / 2 / 4 / 6 months")
for (T, rh), y in zip(CONDS, Y):
    print(f"  {T} C / {rh} %RH: " + "  ".join(f"{v:+.4f}" for v in y))

# ---- Stage 1: 条件ごとの OLS
tc = t - t.mean(); Stt = (tc ** 2).sum()
KH = np.array([-(y * tc).sum() / Stt for y in Y])
RSS = np.array([((y - (y.mean() + (-k) * tc)) ** 2).sum() for y, k in zip(Y, KH)])
print(f"\nStage 1 (S_tt = {Stt:.0f} month^2, df = 2 per condition)")
print(f"{'condition':>16s} {'k_hat (/mo)':>12s} {'RSS':>10s} {'SE(ln k_hat)':>13s}")
for (T, rh), k, r in zip(CONDS, KH, RSS):
    print(f"{T:>8d} C/{rh:>2d} %RH {k:12.5f} {r:10.2e} {np.sqrt(r / 2 / Stt) / k:13.3f}")
s1 = dict(KH=KH[None], RSS=RSS[None], STT=np.full(6, Stt), DF=np.full(6, 2.0), N=np.full(6, 4.0), conds=CONDS)

# ---- Stage 2: 反復ごとの値(iters = 0 は出発点 = k_hat > 0 の条件の観測重みの当てはめ)
X = M.design_matrix(CONDS); x0 = M.x_target()
print("\nStage 2, flat prior, a0 = b0 = 0; working weights w_i = k_i~^2 S_tt (x1 = -1/(RT) in mol/kJ, x2 = RH)")
print(f"{'iteration':>9s} {'Ea (kJ/mol)':>12s} {'B (/%RH)':>9s} {'ln k(25/60)':>12s}")
for it in range(0, 7):
    o = M.stage2(s1, [0, 80, 0.04], None, 'gn', iters=it)
    b = o['mean'][0]
    print(f"{it:9d} {b[1]:12.3f} {b[2]:9.5f} {b @ x0:12.5f}")
o = M.stage2(s1, [0, 80, 0.04], None, 'gn', iters=6)
nu = float(o['nu'][0]); sc = float(o['scale'][0]); q = stats.t.ppf(0.95, nu)
med = float(o['med'][0]); lb = float(o['lb'][0])
kf = np.exp(o['mean'][0] @ X.T)
print(f"\nresidual sums: sum RSS_i = {RSS.sum():.3e}, sum S_tt (k_hat - k~)^2 = {(Stt * (KH - kf) ** 2).sum():.3e}")
print(f"p_eff = {float(o['p_eff'][0]):.3f}; nu = 2 a_n = sum df_i + n - p_eff = 12 + 6 - 3 = {nu:.0f}")
print(f"scale of ln t90 at 25/60 = {sc:.4f}; t_0.95,{nu:.0f} = {q:.4f}")
print(f"ln t90: median {med:.4f}, one-sided 95 % bound {lb:.4f} = median - {q:.4f} x {sc:.4f}")
print(f"t90 (months): median {np.exp(med):.1f}, bound {np.exp(lb):.1f} (truth {M.T90_TRUE:.1f}); bound / median {np.exp(lb - med):.3f}")
d = M.one_stage_delta(s1)
print(f"one-stage NLS with per-condition intercepts (delta method, t on N - n - 3 = 15 df): median {np.exp(float(d['med'][0])):.1f}, "
      f"bound {np.exp(float(d['lb'][0])):.1f} months; |difference in ln t90| = {abs(float(d['lb'][0]) - lb):.1e}")
