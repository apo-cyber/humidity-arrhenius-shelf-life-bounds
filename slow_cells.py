"""模擬査読 4 回目(10-27 夜)の M4: 遅い製品のセルの被覆 1.000 と、Gauss–Newton が走り去る反復の扱い。既存の出力は変えない。
セル: Ea 40、B 0.02、t_SL 60 か月、ICH 3×2 0/2/4/6、assay sigma 0.02(S7 と同じ乱数 = sensitivity_truth.py の種)。
  (1) 本文のコード(model.stage2: 加法の段 beta + H^-1 g、平坦な事前に 1e-10 の正則化)
  (2) 本文の Algorithm の枠を作業応答 z の形のまま書いたもの(正則化なし、未中心化の X、k~ が 0 に潰れると z が nan)
どちらも非有限の下限は 0(lb = -inf、被覆に含める)。下限がほぼ 0 の反復の割合、被覆、下限の中央値を出す。
    python3 slow_cells.py > output/slow_cells.txt   (~2 s)
"""
import zlib
import numpy as np
from scipy import stats
import model as M

NREP = 4000
Ea, B, t_sl, sig = 40.0, 0.02, 60.0, 0.02
M.EA, M.B_TRUE, M.T_SL = Ea, B, t_sl
M.K25 = -np.log(0.95) / t_sl; M.T90_TRUE = -np.log(0.9) / M.K25; M.LNT = np.log(M.T90_TRUE)
M.LNA = np.log(M.K25) - Ea * M.x1_abs(M.T0) - B * M.RH0
ICH3 = [(T, rh) for T in (40, 50, 60) for rh in (60, 75)]
rng = np.random.default_rng(zlib.crc32(f"{sig}|{Ea}|{B}|{t_sl}|ICH 3x2 0/2/4/6".encode()))
s1 = M.stage1(rng, ICH3, [M.MONTHS] * 6, NREP, sig)
KH, STT, RSS = s1['KH'], s1['STT'], s1['RSS']
nu = s1['DF'].sum() + len(ICH3) - 3


def describe(lb, label):
    ok = np.isfinite(lb)
    print(f"{label:52s} non-finite (bound set to 0) {100 * (~ok).mean():5.2f} % | bound below e^-10 of the truth {100 * (lb < M.LNT - 10).mean():5.1f} % | "
          f"coverage {np.mean(lb <= M.LNT):.3f} | median bound/truth {np.exp(np.median(lb) - M.LNT):.3f}")


print(f"cell: Ea {Ea:g} kJ/mol, B {B:g} /%RH, shelf life {t_sl:g} months; ICH 3x2 on 0/2/4/6 months; assay sigma {sig}; {NREP} replicates; flat prior, nu = {nu:g}")
print(f"conditions with k_hat <= 0: {100 * (KH <= 0).mean():.1f} % of condition fits; replicates with at least one: {100 * (KH <= 0).any(1).mean():.1f} %")
with np.errstate(all='ignore'):
    o = M.stage2(s1, [20, 80.0, 0.04], None, 'gn', sig_ref=sig)
describe(o['lb'], "(1) code: additive step, ridge 1e-10")

X = M.design_matrix(ICH3, None); x0 = M.x_target(None)
lb2 = np.empty(NREP)
for r in range(NREP):
    kh = KH[r]; keep = kh > 0
    with np.errstate(all='ignore'):
        try:
            w = kh[keep] ** 2 * STT[keep]; Xk = X[keep]
            beta = np.linalg.solve(Xk.T @ (w[:, None] * Xk), Xk.T @ (w * np.log(kh[keep])))
            for _ in range(6):
                kt = np.exp(X @ beta); W = kt ** 2 * STT; z = np.log(kt) + (kh - kt) / kt
                beta = np.linalg.inv(X.T @ (W[:, None] * X)) @ (X.T @ (W * z))
            kt = np.exp(X @ beta); Sn = np.linalg.inv(X.T @ ((kt ** 2 * STT)[:, None] * X))
            bn = 0.5 * (RSS[r].sum() + (STT * (kh - kt) ** 2).sum()); an = 0.5 * nu
            lb2[r] = M.LN_C - x0 @ beta - stats.t.ppf(0.95, nu) * np.sqrt(bn / an * x0 @ Sn @ x0)
        except np.linalg.LinAlgError:
            lb2[r] = np.nan
lb2 = np.where(np.isfinite(lb2), lb2, -np.inf)
describe(lb2, "(2) Algorithm as printed: working response, no ridge")
both = (o['lb'] > M.LNT - 10) & (lb2 > M.LNT - 10)
print(f"replicates with both bounds above e^-10 of the truth: {100 * both.mean():.1f} %; max |difference| of the two bounds there {np.abs(o['lb'][both] - lb2[both]).max():.1e} in ln t90")
