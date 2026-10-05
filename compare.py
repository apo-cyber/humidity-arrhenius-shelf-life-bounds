"""paper-C Table 1 の正面比較: 同じ Stage 1 要約から 6 通りの片側 95 % 下限(論文の Table 1 は C / D / F / S の 4 行。A / A2 は記録用、§2.5 の本文の数字)。
  A  paper-A 公開版(観測重み + SE フロア + 正規事後 + 残差フロア + 正規分位)を 2 軸に転記
  A' 二段・観測重み + NIG-t(重みは A、分散は C)
  C  二段・当てはめ重み(k 空間 GN)+ NIG-t   ← 本稿
  D  AccelStab 型: 一段非線形当てはめ + delta 法 + t(N-n-3)、事前なし
  F  AccelStab の FBH: 多変量 t から 4000 回引いて分位(D と同じはず)
  S  ASAP 型(推定仕様): ln t_iso の無重み回帰 + Stage 1 誤差の Monte Carlo(2000 回)の 5 % 点
設計 2 つ(ICH 3×2 0/2/4/6、ASAP-5 isoconversion)× 雑音 2 水準(assay 0.02、degradant 0.0075)。4000 反復。
    python3 compare.py > output/compare.txt   (~1 min)
"""
import numpy as np
import model as M

NREP = 4000
ICH3 = [(T, rh) for T in (40, 50, 60) for rh in (60, 75)]
ASAP5 = [(50, 75), (60, 40), (70, 5), (70, 75), (80, 40)]
DESIGNS = {'ICH 3x2 40/50/60 x 60/75, 0/2/4/6 mo': (ICH3, [M.MONTHS] * 6),
           'ASAP-5 isoconversion (4 points to alpha*)': (ASAP5, [M.iso_grid(T, rh) for T, rh in ASAP5])}
hdr = f"{'method':58s} {'cover':>6s} {'LB/true':>7s} {'LB/med':>6s} {'med/true':>8s} {'z mean':>6s} {'z sd':>5s} {'drop%':>5s}"


def row(o):
    s = M.summarise(o)
    return f"{s['cover']:6.3f} {s['lb_true']:7.2f} {s['lb_med']:6.2f} {s['med_true']:8.2f} {s['z_mean']:6.2f} {s['z_sd']:5.2f} {s['drop']:5.1f}"


for name, (conds, grids) in DESIGNS.items():
    for noise, sig in (('assay sigma=0.02', 0.02), ('degradant sigma=0.0075', 0.0075)):
        print(f"\n=== {name}; {noise}; 1 batch; weak prior where a prior is used ===\n" + hdr)
        rng = np.random.default_rng(20260921)
        s1 = M.stage1(rng, conds, grids, NREP, sig)
        oA = M.stage2_paperA(s1, [20, M.EA, M.B_TRUE], [100, 30, 0.05])
        print(f"{'A  paper-A as published (obs weights, floor, normal, resid var, z)':58s} {row(oA)}")
        oA2 = M.stage2(s1, [20, M.EA, M.B_TRUE], [100, 30, 0.05], 'obs', sig_ref=sig)
        print(f"{'A2 two-stage, observed weights, NIG-t':58s} {row(oA2)}")
        oC = M.stage2(s1, [20, M.EA, M.B_TRUE], [100, 30, 0.05], 'gn', sig_ref=sig)
        print(f"{'C  two-stage, fitted weights (k-space GN), NIG-t  [this paper]':58s} {row(oC)}")
        oD = M.one_stage_delta(s1, fbh_draws=4000, rng=rng)
        print(f"{'D  one-stage NLS + delta method + t (AccelStab-type, no prior)':58s} {row(oD)}")
        oF = dict(oD); oF['lb'] = oD['lb_fbh']
        print(f"{'F  FBH: 4000 draws from the multivariate t, 5 % quantile':58s} {row(oF)}")
        oS = M.asap_type_mc(s1, draws=2000, rng=rng)
        print(f"{'S  ASAP-type: unweighted ln t_iso regression + MC over Stage 1':58s} {row(oS)}")
        print(f"   (D vs F: max |LB difference| = {np.abs(oD['lb'] - oD['lb_fbh']).max():.3f} in ln t90; "
              f"C vs D median: {np.exp(np.median(oC['med'] - oD['med'])):.3f})")
