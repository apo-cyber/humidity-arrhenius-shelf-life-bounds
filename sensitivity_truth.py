"""模擬査読 M4(10-27): 生成モデルの感度格子。これまでの表はすべて 1 点(Ea 80 kJ/mol、B 0.04 /%RH、t_SL(95 %) 30 か月 @ 25/60)。
Ea ∈ {40, 50, 60, 80, 100, 120}、B ∈ {0.02, 0.04, 0.06}、t_SL ∈ {18, 30, 60} か月(t90 = 37 / 62 / 123 か月)の 54 点で、
Stage A の片側 95 % 下限(k 空間 GN、平坦な事前・a0 = 0)の被覆と長さ、設計の閉形式(図 1b)との一致、観測重みの偏り(命題 1)を見る。
10-27 R2(模擬査読 2 回目の M1): Ea を 40・50 に広げ、既定を平坦に。比較のため弱い事前(平均は真値でなく (80, 0.04) に固定、SD 30 / 0.05)の
被覆と長さも出す: 真値が平均より下にあると事前が推定を引き上げ、反保守になる(Ea 40 で 0.86)。
GN が発散する反復(40/75 で 6 か月の減少が測定誤差程度の設計)は「下限なし」(lb = -inf、被覆には含める)として nobound に数える。

設計: ICH 型 3×2(40/50/60 °C × 60/75 % RH、0/2/4/6 か月)、ICH 型 2×2 0/3/6 か月(最小)、ASAP-5 等変換(各条件を α* まで 4 点。
遅い製品では 50 °C の等変換時間が長くなる: 列 t_iso に最長の条件の月数を出す。実際の ASAP は 2–4 週で打ち切る)。
雑音: assay σ = 0.02、不純物 σ = 0.0075(ASAP の報告精度)。4000 反復、Stage 1 の乱数は格子点ごとに固定。

    python3 sensitivity_truth.py > output/sensitivity_truth.txt   (~10 s)
"""
import zlib
import numpy as np
from scipy import stats
import model as M

NREP = 4000
EAS, BS, TSLS = (40.0, 50.0, 60.0, 80.0, 100.0, 120.0), (0.02, 0.04, 0.06), (18.0, 30.0, 60.0)
ICH3 = [(T, rh) for T in (40, 50, 60) for rh in (60, 75)]
ICH2 = [(40, 60), (40, 75), (60, 60), (60, 75)]
ASAP5 = [(50, 75), (60, 40), (70, 5), (70, 75), (80, 40)]
PRIOR_MU, WEAK_MU = [20, 80.0, 0.04], [20, 80.0, 0.04]                     # 平坦では平均は効かない


def set_truth(Ea, B, t_sl):
    M.EA, M.B_TRUE, M.T_SL = Ea, B, t_sl
    M.K25 = -np.log(0.95) / t_sl
    M.T90_TRUE = -np.log(0.9) / M.K25; M.LNT = np.log(M.T90_TRUE)
    M.LNA = np.log(M.K25) - Ea * M.x1_abs(M.T0) - B * M.RH0


def designs():
    return {'ICH 3x2 0/2/4/6': (ICH3, [M.MONTHS] * 6),
            'ICH 2x2 0/3/6': (ICH2, [M.MONTHS_ICH] * 4),
            'ASAP-5 iso': (ASAP5, [M.iso_grid(T, rh) for T, rh in ASAP5])}


def nu_flat(grids):
    return sum(len(g) - 2 for g in grids) + len(grids) - 3


rows = []
print("columns: loss % at 40/75 in 6 mo | max SE(ln k_i) | t_iso max (mo, ASAP only) | gn flat: cover, LB/true, z sd | "
      "closed-form LB/med vs simulated LB/med | obs weights (flat): cover, med/true | gn weak prior at (80, 0.04): cover, LB/true, med/true | no bound %")
for sig, noise in ((0.02, 'assay'), (0.0075, 'degradant')):
    print(f"\n=================== noise: {noise} sigma = {sig} ===================")
    for Ea in EAS:
        for B in BS:
            for t_sl in TSLS:
                set_truth(Ea, B, t_sl)
                loss40 = 100 * (1 - np.exp(-6 * M.k_true(40, 75)))
                for dname, (conds, grids) in designs().items():
                    se = np.array([M.se_lnk(T, rh, g, sig) for (T, rh), g in zip(conds, grids)])
                    tiso = max(g[-1] for g in grids) if dname.startswith('ASAP') else np.nan
                    rng = np.random.default_rng(zlib.crc32(f"{sig}|{Ea}|{B}|{t_sl}|{dname}".encode()))
                    s1 = M.stage1(rng, conds, grids, NREP, sig)
                    o = M.stage2(s1, PRIOR_MU, None, 'gn', sig_ref=sig); s = M.summarise(o)
                    ob = M.stage2(s1, PRIOR_MU, None, 'obs', sig_ref=sig); sb = M.summarise(ob)
                    ow = M.stage2(s1, WEAK_MU, M.WEAK_SD, 'gn', sig_ref=sig); sw = M.summarise(ow)
                    cf = M.bound_over_median(M.design_cov(conds, se ** 2)['sd0'], nu_flat(grids))
                    rows.append(dict(sig=sig, Ea=Ea, B=B, t_sl=t_sl, d=dname, loss=loss40, se=se.max(), cov=s['cover'], lbt=s['lb_true'],
                                     lbm=s['lb_med'], cf=cf, zsd=s['z_sd'], med=s['med_true'], cov_obs=sb['cover'], med_obs=sb['med_true'], drop=s['drop'],
                                     nob=s['nobound'], cov_w=sw['cover'], lbt_w=sw['lb_true'], med_w=sw['med_true']))
                    print(f"Ea {Ea:5.0f} B {B:.2f} tSL {t_sl:4.0f} {dname:16s} loss40 {loss40:5.1f} | SEmax {se.max():5.2f} | "
                          f"t_iso {tiso:6.1f} | {s['cover']:.3f} {s['lb_true']:5.2f} {s['z_sd']:4.2f} | cf {cf:.2f} sim {s['lb_med']:.2f} | "
                          f"obs {sb['cover']:.3f} {sb['med_true']:.2f} | weak {sw['cover']:.3f} {sw['lb_true']:.2f} {sw['med_true']:.2f} | "
                          f"nobound {s['nobound']:.1f}{'  drop%.1f' % s['drop'] if s['drop'] > 0 else ''}")

print("\n=================== summary by the largest per-condition SE(ln k_i) ===================")
print(f"{'SE(ln k) max':>14s} {'cells':>5s} | {'gn cover min–max':>17s} {'< 0.93':>6s} | {'|cf − sim| LB/med max':>21s} {'> 0.05':>6s} | {'obs med/true min–max':>20s}")
for lo, hi in ((0, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, 99)):
    sel = [r for r in rows if lo <= r['se'] < hi]
    if not sel:
        continue
    c = np.array([r['cov'] for r in sel]); d = np.array([abs(r['cf'] - r['lbm']) for r in sel]); mo = np.array([r['med_obs'] for r in sel])
    nb = sum(r['nob'] > 0 for r in sel); cw = np.array([r['cov_w'] for r in sel]); lb = np.array([r['lbt'] for r in sel])
    print(f"{lo:5.1f}–{hi:<5.1f}     {len(sel):5d} | {c.min():.3f}–{c.max():.3f}       {(c < 0.93).sum():6d} | {np.nanmax(d):21.3f} {(d > 0.05).sum():6d} | {mo.min():.2f}–{mo.max():.2f}"
          f" | LB/true {lb.min():.2f}–{lb.max():.2f} | weak prior: cover {cw.min():.3f}–{cw.max():.3f}, < 0.94 in {(cw < 0.94).sum()}"
          f"   (cells with some 'no bound' replicates: {nb})")
print("\n=================== by true Ea: flat (this paper) vs weak prior centred at (80, 0.04) ===================")
print(f"{'Ea':>5s} {'cells':>5s} | {'flat cover':>12s} {'< 0.94':>6s} {'LB/true':>10s} | {'weak cover':>12s} {'< 0.94':>6s} {'LB/true':>10s} {'med/true':>10s}")
for ea in EAS:
    sel = [r for r in rows if r['Ea'] == ea]
    c = np.array([r['cov'] for r in sel]); l = np.array([r['lbt'] for r in sel])
    cw = np.array([r['cov_w'] for r in sel]); lw = np.array([r['lbt_w'] for r in sel]); mw = np.array([r['med_w'] for r in sel])
    print(f"{ea:5.0f} {len(sel):5d} | {c.min():.3f}–{c.max():.3f} {(c < 0.94).sum():6d} {l.min():.2f}–{l.max():.2f} | "
          f"{cw.min():.3f}–{cw.max():.3f} {(cw < 0.94).sum():6d} {lw.min():.2f}–{lw.max():.2f} {mw.min():.2f}–{mw.max():.2f}")
print("\ncells with 'no bound' replicates (flat):")
for r in rows:
    if r['nob'] > 0:
        print(f"  {r['d']:16s} sigma {r['sig']:<6} Ea {r['Ea']:.0f} B {r['B']:.2f} tSL {r['t_sl']:.0f}: no bound in {r['nob']:.1f} %, loss40 {r['loss']:.1f} %, "
              f"SEmax {r['se']:.2f}, cover {r['cov']:.3f}, LB/true {r['lbt']:.2f}; weak prior: cover {r['cov_w']:.3f}, LB/true {r['lbt_w']:.2f}")
low = sorted(rows, key=lambda r: r['cov'])[:8]
print("\nlowest-coverage cells:")
for r in low:
    print(f"  {r['d']:16s} sigma {r['sig']:<6} Ea {r['Ea']:.0f} B {r['B']:.2f} tSL {r['t_sl']:.0f}: cover {r['cov']:.3f}, LB/true {r['lbt']:.2f}, "
          f"SEmax {r['se']:.2f}, loss40 {r['loss']:.1f} %, cf {r['cf']:.2f} vs sim {r['lbm']:.2f}")
