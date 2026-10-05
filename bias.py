"""式 (13′) の照合: 観測重みの偏りの閉形式(2 次)対シミュレーション。§2.5 の本文数値。
   python3 bias.py > output/bias.txt   (~20 s)"""
import numpy as np
import model as M

ICH3 = [(T, rh) for T in (40, 50, 60) for rh in (60, 75)]
ICH2 = [(40, 60), (40, 75), (60, 60), (60, 75)]
ASAP5 = [(50, 75), (60, 40), (70, 5), (70, 75), (80, 40)]
NREP = 20000
print("=== Bias of ln k(25 C / 60 %RH) [= -(bias of ln t90)], flat prior, 20000 replicates ===")
print("Proposition 2: (i) observed weights sigma^2 M^-1 X' (3/2 - 2h) = WLS of c_i = -v_i/2 (Jensen) + 2 v_i (1 - h_ii); (ii) log scale + fitted weights: Jensen only; (iii) k-space GN: Box (1971) -sigma^2/2 M^-1 X' h")
print(f"{'design':14s} {'sigma':>6s} {'max v':>5s} | {'obs: formula':>12s} {'sim mean':>8s} {'sim med':>7s} | {'fit (Jensen): formula':>21s} {'sim mean':>8s} {'sim med':>7s} | {'gn (Box): formula':>17s} {'sim mean':>8s} {'sim med':>7s} | drop")
for name, conds, gridf in (('ICH 3Tx2RH', ICH3, lambda c: [M.MONTHS] * 6), ('ICH 2Tx2RH', ICH2, lambda c: [M.MONTHS] * 4),
                           ('ASAP iso', ASAP5, lambda c: [M.iso_grid(T, rh) for T, rh in c]), ('ASAP 28 d', ASAP5, lambda c: [M.DAYS28] * 5)):
    for sig in (0.02, 0.0075, 0.0025):
        grids = gridf(conds); b = M.bias_obs_weights(conds, grids, sig)
        rng = np.random.default_rng(1); s1 = M.stage1(rng, conds, grids, NREP, sig)
        e = {}
        for meth in ('obs', 'fit', 'gn'):
            o = M.stage2(s1, [0, M.EA, M.B_TRUE], [1e6, 1e4, 1e4], meth, sig_ref=sig)
            d = M.LNT - o['med']; e[meth] = (d.mean(), np.median(d), o['drop'])
        print(f"{name:14s} {sig:6.4f} {b['v'].max():5.2f} | {b['total']:+12.4f} {e['obs'][0]:+8.4f} {e['obs'][1]:+7.4f} | {b['jensen']:+21.4f} {e['fit'][0]:+8.4f} {e['fit'][1]:+7.4f} | {b['box']:+17.4f} {e['gn'][0]:+8.4f} {e['gn'][1]:+7.4f} | {e['obs'][2]:.3f}")
b = M.bias_obs_weights(ICH3, [M.MONTHS] * 6, 0.02)
print("\nICH 3Tx2RH, assay: v_i =", np.round(b['v'], 3), " h_ii =", np.round(b['h'], 2),
      f"\n  obs-weight term {b['obs']:+.4f} (exp(-.) = {np.exp(-b['obs']):.3f}), Jensen {b['jensen']:+.4f}, total {b['total']:+.4f} (exp(-.) = {np.exp(-b['total']):.3f}; Table 1 median 0.86); k-space Box {b['box']:+.4f}")
c = 2 * b['v'] * (1 - b['h']); w = 0.02 ** 2 / b['v']
X = M.design_matrix(ICH3); Minv = np.linalg.inv(X.T @ (w[:, None] * X)); contrib = M.x_target() @ Minv @ X.T * (w * c)
print("  per-condition contribution to the obs-weight term:", np.round(contrib, 4), "-> 40 C share", f"{contrib[:2].sum() / contrib.sum():.2f}")
