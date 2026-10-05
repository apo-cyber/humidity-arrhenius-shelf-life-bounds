"""paper-C §5 の頑健性(崩し)。
 (A) バッチ間のランダム効果: ln k_batch = ln k + N(0, tau^2)(paper-B と同じ tau 格子)。共通勾配でプールする本稿の Stage 1 は
     tau を s_data に含めない → 被覆がどれだけ落ちるか、「バッチを単位に」(バッチ別の k_hat、間の分散、df = B-1)で戻るか、その価格。
     Stage B(40/75 × 3 バッチ + 輸入事前)と Stage A(ICH 3×2 × 3 バッチ、弱事前)の両方。
 (B) 速度式の次数 n: dalpha/dt = k (1-alpha)^n。真 n in {1, 2, 0} を一次で読む / 真 n で読む / n を格子でプロファイル(plug-in, t 混合)。
     docs/audit/humidity_order_n.py(09-19)の移植。
 (C) isoconversion 入力の構造同一性: k_hat_i = g(alpha*)/t_hat_iso,i は Stage 1 の数値そのもの(一次では恒等)。数値確認 1 行。

    python3 robustness.py > output/robustness.txt      (~1 min)
"""
import numpy as np
from scipy import stats
import model as M

NREP = 4000
ICH3 = [(T, rh) for T in (40, 50, 60) for rh in (60, 75)]


# ================================================================ (A) バッチ間ランダム効果
def stage1_batches(rng, conds, grid, nrep, sig, B, tau):
    """バッチごとに ln k をずらして生成。返り値: プール版 s1(共通勾配)と、バッチ別 k_hat (nrep, n, B)。"""
    n = len(conds); grid = np.asarray(grid, float)
    KHp = np.zeros((nrep, n)); RSSp = np.zeros((nrep, n)); STTp = np.zeros(n); DFp = np.zeros(n); Np = np.zeros(n)
    KHb = np.zeros((nrep, n, B))
    for i, (T, rh) in enumerate(conds):
        lnk = np.log(M.k_true(T, rh)) + rng.normal(0, tau, (nrep, B))          # (nrep, B)
        t = np.tile(grid, B); which = np.repeat(np.arange(B), grid.size)
        y = -np.exp(lnk)[:, which] * t + rng.normal(0, sig, (nrep, t.size))
        tc = t - t.mean(); Stt = (tc ** 2).sum()
        slope = (y * tc).sum(1) / Stt; icpt = y.mean(1) - slope * t.mean()
        RSSp[:, i] = ((y - icpt[:, None] - slope[:, None] * t) ** 2).sum(1)
        KHp[:, i] = -slope; STTp[i] = Stt; DFp[i] = t.size - 2; Np[i] = t.size
        gc = grid - grid.mean(); Sg = (gc ** 2).sum()
        for b in range(B):
            yb = y[:, which == b]; KHb[:, i, b] = -(yb * gc).sum(1) / Sg
    return dict(KH=KHp, RSS=RSSp, STT=STTp, DF=DFp, N=Np, conds=list(conds)), KHb


def batch_as_unit_stageB(KHb, sdE, sdB, dE=0.0, dB=0.0):
    """Stage B、バッチを単位に: ln k_hat_b の平均と間の SE(df = B-1)、事前は正規でてこで運ぶ。Welch–Satterthwaite で t の df。"""
    L_T, L_RH = M.levers()
    lnk = np.log(np.maximum(KHb[:, 0, :], 1e-12)); B = lnk.shape[1]
    m = lnk.mean(1); s_b2 = lnk.var(1, ddof=1) / B
    p = (L_T * sdE) ** 2 + (L_RH * sdB) ** 2
    med = M.LN_C - (m + L_T * (M.EA + dE) + L_RH * (M.B_TRUE + dB))
    var = s_b2 + p; nu = var ** 2 / (s_b2 ** 2 / (B - 1))
    lb = med - stats.t.ppf(0.95, nu) * np.sqrt(var)
    return dict(lb=lb, med=med, scale=np.sqrt(var), drop=0.0)


print("=== (A) batch-to-batch variation: ln k per batch = ln k + N(0, tau^2); assay sigma=0.02; grid 0/2/4/6 mo; B=3 ===")
print("truth for coverage = t90 of the batch-mean ln k (the product's typical batch)")
print("\n-- Stage B: 40/75 only, prior (5, 0.01) exact (shift 0) and displaced (dEa=5); pooled common slope vs batch-as-unit")
print(f"{'tau':>5s} | {'pooled: cover':>13s} {'LB/true':>7s} {'LB/med':>6s} | {'batch-unit: cover':>17s} {'LB/true':>7s} {'LB/med':>6s} | "
      f"{'pooled cover @dEa=5':>19s} {'batch-unit @dEa=5':>17s}")
for tau in (0.0, 0.05, 0.1, 0.2, 0.3):
    rng = np.random.default_rng(11)
    s1, KHb = stage1_batches(rng, [(40, 75)], M.MONTHS, NREP, 0.02, 3, tau)
    o = M.stage2(s1, [20, M.EA, M.B_TRUE], [100, 5, 0.01], 'gn', a0=9.0, sig_ref=0.02); sp = M.summarise(o)
    u = batch_as_unit_stageB(KHb, 5, 0.01); su = M.summarise(u)
    o5 = M.stage2(s1, [20, M.EA + 5, M.B_TRUE], [100, 5, 0.01], 'gn', a0=9.0, sig_ref=0.02)
    u5 = batch_as_unit_stageB(KHb, 5, 0.01, dE=5)
    print(f"{tau:5.2f} | {sp['cover']:13.3f} {sp['lb_true']:7.2f} {sp['lb_med']:6.2f} | {su['cover']:17.3f} {su['lb_true']:7.2f} {su['lb_med']:6.2f} | "
          f"{(o5['lb'] <= M.LNT).mean():19.3f} {(u5['lb'] <= M.LNT).mean():17.3f}")

print("\n-- Stage A: ICH 3x2 x 3 batches, weak prior, pooled common slope (Stage 2 residuals absorb part of tau)")
print(f"{'tau':>5s} {'cover':>6s} {'LB/true':>7s} {'LB/med':>6s} {'med/true':>8s} {'z sd':>5s}")
for tau in (0.0, 0.05, 0.1, 0.2, 0.3):
    rng = np.random.default_rng(12)
    s1, _ = stage1_batches(rng, ICH3, M.MONTHS, NREP, 0.02, 3, tau)
    o = M.stage2(s1, [20, M.EA, M.B_TRUE], [100, 30, 0.05], 'gn', sig_ref=0.02); s = M.summarise(o)
    print(f"{tau:5.2f} {s['cover']:6.3f} {s['lb_true']:7.2f} {s['lb_med']:6.2f} {s['med_true']:8.2f} {s['z_sd']:5.2f}")
print("(paper-B, Section 4.5: between-batch SD of ln k of 0.05-0.1 is typical for a well-controlled product; 0.2-0.3 is large)")


# ================================================================ (B) 速度式の次数 n(監査 humidity_order_n.py の移植)
def g(alpha, n):
    s = 1.0 - np.asarray(alpha, float)
    return -np.log(s) if n == 1 else (s ** (1 - n) - 1.0) / (n - 1)


def ginv(y, n):
    return np.exp(-y) if n == 1 else np.maximum(1.0 + (n - 1) * y, 1e-9) ** (1.0 / (1 - n))


def make_truth(n_true):
    k25 = g(0.05, n_true) / M.T_SL; lnA = np.log(k25) - M.EA * M.x1_abs(M.T0) - M.B_TRUE * M.RH0
    return lnA, g(0.10, n_true) / k25


def simulate_raw(rng, lnA, n_true, conds, grids, nrep, sig):
    out = []
    for (T, rh), t in zip(conds, grids):
        t = np.asarray(t, float); k = np.exp(lnA + M.EA * M.x1_abs(T) + M.B_TRUE * rh)
        out.append((t, ginv(k * t, n_true) * np.exp(rng.normal(0, sig, (nrep, t.size)))))
    return out


def stage1_order(data, n, conds):
    KH, RSS, STT, DF, N = [], [], [], [], []
    for t, frac in data:
        y = g(1.0 - frac, n); tc = t - t.mean(); Stt = (tc ** 2).sum()
        slope = (y * tc).sum(1) / Stt; icpt = y.mean(1) - slope * t.mean()
        RSS.append(((y - icpt[:, None] - slope[:, None] * t) ** 2).sum(1)); KH.append(slope); STT.append(Stt); DF.append(t.size - 2); N.append(t.size)
    return dict(KH=np.array(KH).T, RSS=np.array(RSS).T, STT=np.array(STT), DF=np.array(DF), N=np.array(N), conds=list(conds))


def raw_rss(data, beta, n, conds):
    X = M.design_matrix(conds); kf = np.exp(beta @ X.T); rss = 0
    for i, (t, frac) in enumerate(data):
        rss = rss + ((np.log(frac) - np.log(ginv(kf[:, i:i + 1] * t[None], n))) ** 2).sum(1)
    return rss


def bound_n(o, n_fit):
    lnc = np.log(g(0.10, n_fit)); m = M.LN_C - o['med']          # o['med'] = LN_C - x0'beta  → x0'beta = LN_C - med
    return lnc - m - stats.t.ppf(0.95, o['nu']) * o['scale'], lnc - m


NREP_N = 3000; SIG = 0.02; NGRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
print(f"\n=== (B) kinetic order: dalpha/dt = k (1-alpha)^n; ICH 3x2, 0/2/4/6 mo, 1 batch, assay sigma={SIG}, weak prior; nrep={NREP_N} ===")
print("g_n(0.10): " + ", ".join(f"n={n:g}: {g(0.10, n):.4f}" for n in NGRID))
rng = np.random.default_rng(3); grids = [M.MONTHS] * 6
for n_true in (1.0, 2.0, 0.0):
    lnA, T90 = make_truth(n_true); LNT = np.log(T90)
    loss = [100 * (1 - ginv(np.exp(lnA + M.EA * M.x1_abs(T) + M.B_TRUE * rh) * 6, n_true)) for T, rh in ICH3]
    print(f"\n-- true n={n_true:g}: t90(25/60)={T90:.1f} mo; 6-mo loss per condition = " + " ".join(f"{l:.0f}%" for l in loss))
    data = simulate_raw(rng, lnA, n_true, ICH3, grids, NREP_N, SIG)
    print(f"{'fit':34s} {'cover':>6s} {'LB/true':>7s} {'med/true':>8s} {'LB/med':>6s}")
    fits = {}
    for n_fit in NGRID:
        o = M.stage2(stage1_order(data, n_fit, ICH3), [20, M.EA, M.B_TRUE], [100, 30, 0.05], 'gn', sig_ref=SIG, iters=8)
        fits[n_fit] = (o, raw_rss(data, o['mean'], n_fit, ICH3))
    for n_fit in ((1.0,) if n_true == 1.0 else (1.0, n_true)):
        lb, med = bound_n(fits[n_fit][0], n_fit)
        tag = 'n=1 assumed = true' if n_true == 1.0 else ('first-order (n=1) assumed' if n_fit == 1.0 else f'true order n={n_fit:g} known')
        print(f"{tag:34s} {(lb <= LNT).mean():6.3f} {np.exp(np.median(lb)) / T90:7.2f} {np.exp(np.median(med)) / T90:8.2f} {np.exp(np.median(lb - med)):6.2f}")
    RSSm = np.stack([fits[n][1] for n in NGRID], 1); w = np.exp(-(RSSm - RSSm.min(1, keepdims=True)) / (2 * SIG ** 2)); w /= w.sum(1, keepdims=True)
    ibest = RSSm.argmin(1); nbest = NGRID[ibest]
    LB = np.stack([bound_n(fits[n][0], n)[0] for n in NGRID], 1); MED = np.stack([bound_n(fits[n][0], n)[1] for n in NGRID], 1)
    r = np.arange(NREP_N); lb_plug, med_plug = LB[r, ibest], MED[r, ibest]
    print(f"{'n profiled on grid, plug-in best n':34s} {(lb_plug <= LNT).mean():6.3f} {np.exp(np.median(lb_plug)) / T90:7.2f} {np.exp(np.median(med_plug)) / T90:8.2f} "
          f"{np.exp(np.median(lb_plug - med_plug)):6.2f}   | n_hat: median {np.median(nbest):.1f}, P(|n_hat-n|<=0.5)={np.mean(np.abs(nbest - n_true) <= 0.5):.2f}")
    # t 混合(重み w)の 5 % 分位: 各格子点 400 本の t 乱数から加重分位(監査と同じ近似)
    draws = np.stack([MED[:, j][:, None] - fits[n][0]['scale'][:, None] * stats.t.rvs(np.broadcast_to(fits[n][0]['nu'], (NREP_N, 1)), size=(NREP_N, 400), random_state=rng)
                      for j, n in enumerate(NGRID)], 1)
    lb_mix = np.empty(NREP_N); med_mix = np.empty(NREP_N)
    for i in range(NREP_N):
        d = draws[i].ravel(); ww = np.repeat(w[i] / 400, 400); o_ = np.argsort(d); cw = np.cumsum(ww[o_])
        lb_mix[i] = d[o_][np.searchsorted(cw, 0.05)]; med_mix[i] = d[o_][np.searchsorted(cw, 0.5)]
    print(f"{'n profiled, t-mixture over weights':34s} {(lb_mix <= LNT).mean():6.3f} {np.exp(np.median(lb_mix)) / T90:7.2f} {np.exp(np.median(med_mix)) / T90:8.2f} {np.exp(np.median(lb_mix - med_mix)):6.2f}")

print("\n-- ASAP-like, all conditions below ~20 % conversion: does n matter?")
ASAP5 = [(50, 75), (60, 40), (70, 5), (70, 75), (80, 40)]; grids2 = [M.DAYS28] * 5
for n_true in (1.0, 2.0):
    lnA, T90 = make_truth(n_true); LNT = np.log(T90)
    data = simulate_raw(rng, lnA, n_true, ASAP5, grids2, NREP_N, SIG)
    loss = [100 * (1 - ginv(np.exp(lnA + M.EA * M.x1_abs(T) + M.B_TRUE * rh) * 28 / 30.4375, n_true)) for T, rh in ASAP5]
    print(f"true n={n_true:g}; 28-d loss = " + " ".join(f"{l:.0f}%" for l in loss))
    for n_fit in (1.0, 2.0):
        o = M.stage2(stage1_order(data, n_fit, ASAP5), [20, M.EA, M.B_TRUE], [100, 30, 0.05], 'gn', sig_ref=SIG, iters=8)
        lb, med = bound_n(o, n_fit)
        print(f"   fit n={n_fit:g}: cover {(lb <= LNT).mean():.3f}  med/true {np.exp(np.median(med)) / T90:.2f}  LB/med {np.exp(np.median(lb - med)):.2f}")

# ================================================================ (C) isoconversion 入力の恒等性
rng = np.random.default_rng(5)
s1 = M.stage1(rng, ICH3, [M.MONTHS] * 6, 2000, 0.02)
s1_iso = dict(s1); s1_iso['KH'] = np.log(10 / 9) / (np.log(10 / 9) / s1['KH'])       # k_hat = g(0.1) / t_hat_iso
o1 = M.stage2(s1, [20, M.EA, M.B_TRUE], [100, 30, 0.05], 'gn'); o2 = M.stage2(s1_iso, [20, M.EA, M.B_TRUE], [100, 30, 0.05], 'gn')
print(f"\n=== (C) isoconversion entry k_hat_i = g(alpha*)/t_hat_iso,i: max |LB difference| = {np.abs(o1['lb'] - o2['lb']).max():.2e} (identical by construction) ===")
