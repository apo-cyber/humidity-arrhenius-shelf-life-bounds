"""式 (13′) の照合: 観測重みの偏りの閉形式(2 次)対シミュレーション。§2.5 の本文数値。
10-27: 温度だけの三温度の設計(40/50/60 °C、60 % RH の 1 湿度、0/2/4/6 か月と 0/3/6 か月、ln k = β0 + Ea x1 の 2 列、25 °C へ外挿)を追加。
序論・§7.1 の「三温度の設計で約 5 %」を本稿自身の計算にする(模擬査読 M5 #3)。
   python3 bias.py > output/bias.txt   (~20 s)"""
import numpy as np
import model as M

ICH3 = [(T, rh) for T in (40, 50, 60) for rh in (60, 75)]
ICH2 = [(40, 60), (40, 75), (60, 60), (60, 75)]
ASAP5 = [(50, 75), (60, 40), (70, 5), (70, 75), (80, 40)]
NREP = 20000
print("=== Bias of ln k(25 C / 60 %RH) [= -(bias of ln t90)], flat prior, 20000 replicates ===")
print("Proposition 1: (i) observed weights sigma^2 M^-1 X' (3/2 - 2h) = WLS of c_i = -v_i/2 (Jensen) + 2 v_i (1 - h_ii); (ii) log scale + fitted weights: Jensen only; (iii) k-space GN: Box (1971) -sigma^2/2 M^-1 X' h")
print(f"{'design':14s} {'sigma':>6s} {'max v':>5s} | {'obs: formula':>12s} {'sim mean':>8s} {'sim med':>7s} | {'fit (Jensen): formula':>21s} {'sim mean':>8s} {'sim med':>7s} | {'gn (Box): formula':>17s} {'sim mean':>8s} {'sim med':>7s} | drop")
REP_SHARE = {}
for name, conds, gridf in (('ICH 3Tx2RH', ICH3, lambda c: [M.MONTHS] * 6), ('ICH 2Tx2RH', ICH2, lambda c: [M.MONTHS] * 4),
                           ('ASAP iso', ASAP5, lambda c: [M.iso_grid(T, rh) for T, rh in c]), ('ASAP 28 d', ASAP5, lambda c: [M.DAYS28] * 5)):
    for sig in (0.02, 0.0075, 0.0025):
        grids = gridf(conds); b = M.bias_obs_weights(conds, grids, sig)
        rng = np.random.default_rng(1); s1 = M.stage1(rng, conds, grids, NREP, sig)
        REP_SHARE[(name, sig)] = ((s1['KH'] <= 0).mean(), (s1['KH'] <= 0).any(1).mean())   # 条件の当てはめ単位 / 反復単位
        e = {}
        for meth in ('obs', 'fit', 'gn'):
            o = M.stage2(s1, [0, M.EA, M.B_TRUE], [1e6, 1e4, 1e4], meth, sig_ref=sig)
            d = M.LNT - o['med']; e[meth] = (d.mean(), np.median(d), o['drop'])
        print(f"{name:14s} {sig:6.4f} {b['v'].max():5.2f} | {b['total']:+12.4f} {e['obs'][0]:+8.4f} {e['obs'][1]:+7.4f} | {b['jensen']:+21.4f} {e['fit'][0]:+8.4f} {e['fit'][1]:+7.4f} | {b['box']:+17.4f} {e['gn'][0]:+8.4f} {e['gn'][1]:+7.4f} | {e['obs'][2]:.3f}")
fs, rs = REP_SHARE[('ICH 3Tx2RH', 0.02)]
print(f"ICH 3Tx2RH, assay: k_hat <= 0 in {100 * fs:.1f} % of condition fits, {100 * rs:.1f} % of replicates have at least one (left out of the log-scale fits)")
b = M.bias_obs_weights(ICH3, [M.MONTHS] * 6, 0.02)
print("\nICH 3Tx2RH, assay: v_i =", np.round(b['v'], 3), " h_ii =", np.round(b['h'], 2),
      f"\n  obs-weight term {b['obs']:+.4f} (exp(-.) = {np.exp(-b['obs']):.3f}), Jensen {b['jensen']:+.4f}, total {b['total']:+.4f} (exp(-.) = {np.exp(-b['total']):.3f}; simulated median 0.85, Supplementary S1 and Section 2.5); k-space Box {b['box']:+.4f}")
c = 2 * b['v'] * (1 - b['h']); w = 0.02 ** 2 / b['v']
X = M.design_matrix(ICH3); Minv = np.linalg.inv(X.T @ (w[:, None] * X)); contrib = M.x_target() @ Minv @ X.T * (w * c)
print("  per-condition contribution to the obs-weight term:", np.round(contrib, 4), "-> 40 C share", f"{contrib[:2].sum() / contrib.sum():.2f}")

# ---- 温度だけの三温度の設計(10-27、M5 #3): 1 湿度(60 % RH = 保存湿度)なので B は入らず、X = (1, x1)、外挿先は 25 °C / 60 % RH
T3 = [(40, 60), (50, 60), (60, 60)]
X3 = np.array([[1.0, M.x1_abs(T)] for T, _ in T3]); x03 = np.array([1.0, M.x1_abs(M.T0)])
print("\n=== Temperature only: three temperatures 40/50/60 C at 60 %RH, ln k = b0 + Ea x1, extrapolated to 25 C / 60 %RH, flat prior, 20000 replicates ===")
print(f"{'grid':10s} {'sigma':>6s} {'max v':>5s} | {'obs: formula':>12s} {'sim mean':>8s} {'sim med':>7s} {'median t90 ratio':>16s} | {'gn: sim med':>11s} {'ratio':>6s} | drop")
for gname, grid in (('0/2/4/6', M.MONTHS), ('0/3/6', M.MONTHS_ICH)):
    for sig in (0.02, 0.0075, 0.0025):
        grids = [grid] * 3; b = M.bias_obs_weights(T3, grids, sig, X=X3, x0=x03)
        rng = np.random.default_rng(3); s1 = M.stage1(rng, T3, grids, NREP, sig)
        oo = M.stage2(s1, [0, M.EA], [1e6, 1e4], 'obs', sig_ref=sig, X=X3, x0=x03)
        og = M.stage2(s1, [0, M.EA], [1e6, 1e4], 'gn', sig_ref=sig, X=X3, x0=x03)
        do = M.LNT - oo['med']; dg = M.LNT - og['med']
        print(f"{gname:10s} {sig:6.4f} {b['v'].max():5.2f} | {b['total']:+12.4f} {do.mean():+8.4f} {np.median(do):+7.4f} {np.exp(-np.median(do)):16.3f} | "
              f"{np.median(dg):+11.4f} {np.exp(-np.median(dg)):6.3f} | {oo['drop']:.3f}")
b = M.bias_obs_weights(T3, [M.MONTHS] * 3, 0.02, X=X3, x0=x03)
print(f"  0/2/4/6, assay: v_i = {np.round(b['v'], 3)}, obs-weight term {b['obs']:+.4f}, Jensen {b['jensen']:+.4f}, total {b['total']:+.4f} "
      f"(median t90 ratio exp(-total) = {np.exp(-b['total']):.3f}, i.e. {100 * (1 - np.exp(-b['total'])):.0f} % short)")
