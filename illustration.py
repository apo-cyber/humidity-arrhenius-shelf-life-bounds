"""paper-C §6: Tamura et al. (2020) のアセチルサリチル酸 / MgSt 粉体混合物、60 条件(35/40/50 °C × 4 RH × MgSt 0-4 %)、0 次分解。
 1. Stage 1: 条件ごとに総類縁物質 (%) を日数に OLS → k_hat (%/d), S_tt, RSS, df。プールした sigma。
 2. Stage 2: (a) MgSt 水準ごとに 12 条件の 3 パラメータ当てはめ、(b) 60 条件で MgSt 項つき 4 パラメータ(Tamura Eq. 2)。
    著者の無重み重回帰(Ea 227 kJ/mol, B 0.0227, C 0.4465)と比べる。25 °C/60 %RH の k と、限度 alpha* までの isoconversion 時間の片側 95 % 下限。
 3. 設計: 条件ごとの SE(ln k_i)、25/60 の分散の内訳。
 4. 輸入: Stage A = MgSt 0 % の 12 条件 → 事後 (Ea, B, cov, IG) を、MgSt m % の 40 °C/77 %RH 単一条件(Stage B)に事前として輸入し、
    m % の 12 条件当てはめ(真値の代理)と比べる。水準間の (Ea, B) の実際のずれと Stage A の budget を照合。
    python3 illustration.py > output/illustration.txt
"""
import numpy as np
from scipy import stats
import model as M
from tamura2020 import DATA, PAPER

R = M.R
ALPHA0 = 0.56                 # 初期値 (%)
ALPHA_STAR = 2.0              # 例示の限度: 総類縁物質 2.0 %(形成量 1.44 %)
T0, RH0 = 25.0, 60.0


def stage1_raw(keys):
    KH, RSS, STT, DF, N = [], [], [], [], []
    for key in keys:
        t, y, _ = DATA[key]; tc = t - t.mean(); Stt = (tc ** 2).sum()
        slope = (tc * y).sum() / Stt; icpt = y.mean() - slope * t.mean()
        RSS.append(((y - icpt - slope * t) ** 2).sum()); KH.append(slope); STT.append(Stt); DF.append(t.size - 2); N.append(t.size)
    return dict(KH=np.array([KH]), RSS=np.array([RSS]), STT=np.array(STT), DF=np.array(DF), N=np.array(N), conds=[(k[0], k[1]) for k in keys])


# 誤差の大きさは温度(= 分解の水準)で桁が違うので、温度ごとにプールした sigma_T を使い、
# 条件 i の S_tt と RSS を sigma_ref^2 / sigma_T^2 で割って(= 精度を揃えて)共通 sigma^2 の機構に載せる。
_raw_all = stage1_raw(sorted(DATA.keys()))
SIG_T = {T: np.sqrt(sum(r for r, k in zip(_raw_all['RSS'][0], sorted(DATA.keys())) if k[0] == T) /
                    sum(d for d, k in zip(_raw_all['DF'], sorted(DATA.keys())) if k[0] == T)) for T in (35, 40, 50)}
SIG_REF = SIG_T[40]


def stage1_real(keys):
    s = stage1_raw(keys)
    f = np.array([SIG_REF ** 2 / SIG_T[k[0]] ** 2 for k in keys])
    s['STT'] = s['STT'] * f; s['RSS'] = s['RSS'] * f
    return s


def Xmat(keys, mgst=False):
    cols = [[1.0, M.x1_abs(T), rh] + ([m] if mgst else []) for T, rh, m in keys]
    return np.array(cols)


def x0vec(mgst=None):
    return np.array([1.0, M.x1_abs(T0), RH0] + ([mgst] if mgst is not None else []))


def report(o, label, sig2=None):
    lnk0 = M.LN_C - o['med'][0]                      # x0'beta(model は LN_C - x0'beta を med として返す)
    scale = o['scale'][0]; nu = float(np.ravel(o['nu'])[0])
    k0 = np.exp(lnk0) * 30.4375                      # %/month
    tiso = (ALPHA_STAR - ALPHA0) / np.exp(lnk0) / 30.4375      # months to alpha*
    lb = tiso * np.exp(-stats.t.ppf(0.95, nu) * scale)
    ub_k = k0 * np.exp(stats.t.ppf(0.95, nu) * scale)
    b = o['mean'][0]; C = o['cov'][0]
    print(f"{label:34s} Ea={b[1]:6.1f} ± {np.sqrt(C[1, 1]):4.1f} kJ/mol  B={b[2]:.4f} ± {np.sqrt(C[2, 2]):.4f}"
          + (f"  C={b[3]:.4f} ± {np.sqrt(C[3, 3]):.4f}" if b.size > 3 else '')
          + f" | corr(Ea,B)={C[1, 2] / np.sqrt(C[1, 1] * C[2, 2]):+.2f} | k(25/60)={k0:.3f} %/mo (95 % upper {ub_k:.3f}) | "
          f"t to {ALPHA_STAR} %: median {tiso:.1f} mo, one-sided 95 % LB {lb:.1f} mo (LB/med {lb / tiso:.2f}), nu={nu:.0f}")
    return dict(lnk0=lnk0, scale=scale, nu=nu, beta=b, cov=C, tiso=tiso, lb=lb)


keys_all = sorted(DATA.keys())
s1_all = stage1_real(keys_all)
sig2_pool = s1_all['RSS'].sum() / s1_all['DF'].sum()
print(f"=== Stage 1: {len(keys_all)} conditions, {int(s1_all['N'].sum())} points, pooled residual df {int(s1_all['DF'].sum())} ===")
print("residual SD pooled within temperature (absolute % total related substances): " + ", ".join(f"{T} C: {v:.3f}" for T, v in SIG_T.items())
      + f"; conditions rescaled to the 40 C level (sigma_ref = {SIG_REF:.3f}); pooled after rescaling {np.sqrt(sig2_pool):.3f}")
se = np.sqrt(sig2_pool) / (np.abs(s1_all['KH'][0]) * np.sqrt(s1_all['STT']))
print(f"SE(ln k_i) = sigma/(k_i sqrt(S_tt,i)): min {se.min():.2f}, median {np.median(se):.2f}, max {se.max():.2f}")
for key, s_ in sorted(zip(keys_all, se), key=lambda z: -z[1])[:5]:
    print(f"   least precise: {key[0]} C / {key[1]} %RH / MgSt {key[2]} %: k={DATA[key][2]:.4f} %/d, SE(ln k)={s_:.2f}")
print(f"k_hat <= 0: {(s1_all['KH'] <= 0).sum()} conditions")

# ================================================================ Stage 2
SIG = np.sqrt(sig2_pool); A0 = 0.5
print(f"\n=== Stage 2, weakly informative prior (SD Ea 100, B 0.1, C 10; flat intercept), k-space GN, NIG-t; alpha0={ALPHA0} %, limit {ALPHA_STAR} % ===")
print("-- 60 conditions, 4 parameters (Tamura Eq. 2: ln k = b0 + Ea x1 + B RH + C MgSt)")
o4 = M.stage2(s1_all, [90, 200, 0.02, 0.4], [1000, 100, 0.1, 10], 'gn', a0=A0, sig_ref=SIG, iters=12, X=Xmat(keys_all, True), x0=x0vec(2.0))
r4 = report(o4, "this paper, working weights (MgSt 2 %)")
o4o = M.stage2(s1_all, [90, 200, 0.02, 0.4], [1000, 100, 0.1, 10], 'obs', a0=A0, sig_ref=SIG, X=Xmat(keys_all, True), x0=x0vec(2.0))
report(o4o, "  observed weights, for comparison")
# 著者の無重み回帰の再現(ln k_hat の OLS)
X4 = Xmat(keys_all, True); y = np.log(s1_all['KH'][0]); bols, *_ = np.linalg.lstsq(X4, y, rcond=None); res = y - X4 @ bols
print(f"  unweighted OLS of ln k_hat (authors' method): Ea={bols[1]:.1f} kJ/mol ({bols[1] / 4.184:.1f} kcal/mol), B={bols[2]:.4f}, C={bols[3]:.4f}, "
      f"RMSE={np.sqrt((res ** 2).sum() / (60 - 4)):.4f}   [paper: Ea {PAPER['Ea_kJ']:.1f} kJ/mol = {PAPER['Ea_kcal']} kcal/mol, B {PAPER['B']}, C {PAPER['C']}, RMSE {PAPER['RMSE']}]")

print("\n-- per MgSt level, 12 conditions, 3 parameters")
LEVELS = {}
for m in (0, 1, 2, 3, 4):
    keys = [k for k in keys_all if k[2] == m]; s1 = stage1_real(keys)
    o = M.stage2(s1, [90, 200, 0.02], [1000, 100, 0.1], 'gn', a0=A0, sig_ref=SIG, iters=12, X=Xmat(keys), x0=x0vec())
    LEVELS[m] = (s1, report(o, f"MgSt {m} % (12 conditions)"))

# ================================================================ 設計: 内訳(MgSt 0 %、実測 SE(ln k_i))
print("\n=== Design calculus on the real per-condition precisions (MgSt 0 %, flat prior) ===")
s1 = LEVELS[0][0]; v = sig2_pool / (s1['KH'][0] ** 2 * s1['STT'])
d = M.design_cov(s1['conds'], v)
print(f"SE(Ea)={d['se_Ea']:.1f} kJ/mol, SE(B)={d['se_B']:.4f}, corr={d['corr']:+.2f}, sd(ln k0)={d['sd0']:.3f}, bound/median={M.bound_over_median(d['sd0']):.2f}")
t = d['terms']; tot = d['sd0'] ** 2
print(f"decomposition of sd0^2={tot:.3f}: intercept {t['intercept']:.3f}, temperature {t['T']:.3f} ({100 * t['T'] / tot:.0f} %), humidity {t['RH']:.3f}, EaxB cross {t['EaB']:+.3f}")
print("(35 °C is the nearest tested temperature to 25 °C; the humidity axis is interpolated at 60 %RH within 12-77 %RH)")

# ================================================================ 輸入: Stage A = MgSt 0 % → Stage B = MgSt m % at 40 C / 77 %RH
print(f"\n=== Prior transfer: Stage A = MgSt 0 % (12 conditions) -> Stage B = single condition 40 C / 77 %RH of MgSt m % (4 points) ===")
sA = LEVELS[0][1]; muA = sA['beta']; covA = sA['cov']
L_T = M.x1_abs(T0) - M.x1_abs(40); L_RH = RH0 - 77
ell = np.array([L_T, L_RH]); SigA = covA[1:, 1:]
print(f"Stage A posterior: Ea={muA[1]:.1f} ± {np.sqrt(SigA[0, 0]):.1f}, B={muA[2]:.4f} ± {np.sqrt(SigA[1, 1]):.4f}, corr={SigA[0, 1] / np.sqrt(SigA[0, 0] * SigA[1, 1]):+.2f}; "
      f"levers (40/77 -> 25/60): L_T={L_T:.4f}, L_RH={L_RH:.0f}; sqrt(l' Sigma_A l)={np.sqrt(ell @ SigA @ ell):.3f}")
# Stage A の残差分散を IG で運ぶ: a_A = a0 + df_A/2 + n/2, b_A から
sA_s1 = LEVELS[0][0]; aA = A0 + sA_s1['DF'].sum() / 2 + 12 / 2
print(f"{'level':8s} {'Stage B bound (mo)':>18s} {'median':>7s} {'LB/med':>6s} | {'12-cond fit: LB':>15s} {'median':>7s} | {'dEa':>6s} {'dB':>8s} {'shift':>6s} {'budget':>6s} {'cf cov':>6s}")
for m in (1, 2, 3, 4):
    key = (40, 77, m); s1B = stage1_real([key])
    sdata = SIG / (s1B['KH'][0, 0] * np.sqrt(s1B['STT'][0]))
    cov0 = np.zeros((3, 3)); cov0[0, 0] = 1e4; cov0[1:, 1:] = SigA
    oB = M.stage2(s1B, [muA[0], muA[1], muA[2]], None, 'gn', a0=aA, b0=aA * sig2_pool, sig_ref=SIG, iters=12, X=Xmat([key]), x0=x0vec(), cov0=cov0)
    lnk0 = M.LN_C - oB['med'][0]; tiso = (ALPHA_STAR - ALPHA0) / np.exp(lnk0) / 30.4375
    lb = tiso * np.exp(-stats.t.ppf(0.95, oB['nu']) * oB['scale'][0])
    full = LEVELS[m][1]
    bm = full['beta']; dE = muA[1] - bm[1]; dB = muA[2] - bm[2]       # 輸入した平均 − その水準の当てはめ(真値の代理)
    shift = -(L_T * dE + L_RH * dB); s_ = np.sqrt(sdata ** 2 + ell @ SigA @ ell); bud = M.budget(s_, sdata)
    print(f"MgSt {m} % {lb:18.1f} {tiso:7.1f} {lb / tiso:6.2f} | {full['lb']:15.1f} {full['tiso']:7.1f} | {dE:+6.1f} {dB:+8.4f} {shift:+6.2f} {bud:6.3f} {M.coverage_cf(s_, shift, sdata):6.3f}")
print("shift > 0 = imported prior makes the product look more stable than its own 12-condition fit (anti-conservative); budget = 1.645 (s - s_data)")

# 感度: 20 % を超える点を除く(著者の「約 20 % まで 0 次」)
print("\n=== Sensitivity: drop points above 20 % total related substances (authors' linear range) ===")
import copy
DATA20 = {k: (t[y <= 20], y[y <= 20], kp) for k, (t, y, kp) in DATA.items()}
keys20 = [k for k in keys_all if DATA20[k][0].size >= 3]
_D = DATA
DATA.clear(); DATA.update(DATA20)
s1_20 = stage1_raw(keys20)
f20 = np.array([SIG_REF ** 2 / SIG_T[k[0]] ** 2 for k in keys20]); s1_20['STT'] = s1_20['STT'] * f20; s1_20['RSS'] = s1_20['RSS'] * f20
sig2_20 = s1_20['RSS'].sum() / s1_20['DF'].sum()
o20 = M.stage2(s1_20, [90, 200, 0.02, 0.4], [1000, 100, 0.1, 10], 'gn', a0=A0, sig_ref=np.sqrt(sig2_20), iters=12, X=Xmat(keys20, True), x0=x0vec(2.0))
print(f"{len(keys20)} conditions kept (>= 3 points), pooled sigma {np.sqrt(sig2_20):.3f} %")
report(o20, "4-parameter fit, points <= 20 %")
DATA.clear(); DATA.update(_D)

# ================================================================ 長期 25 °C/60 %RH との比較(Tamura Fig. 7 を読み取り、±0.2 % 程度)
REALTIME = {0: [0.71, 1.68, 1.91, 2.85], 1: [0.80, 1.81, 2.54, 3.26], 2: [1.12, 2.20, 3.51, 4.44], 3: [1.49, 3.26, 4.19, 7.76], 4: [2.03, 5.15, 6.64, 11.14]}   # Fig. 7, digitised (fig7_digitise.py)
RT_MONTHS = np.array([1, 2, 3, 6.])
AUTHORS_6MO = {0: 2.60, 1: 3.74, 2: 5.54, 3: 8.36, 4: 12.70}   # 著者のモデル線(Fig. 7 の点線)の 6 か月値、同じ較正で読み取り
print("\n=== Real-time 25 C / 60 %RH (Tamura Fig. 7, digitised with fig7_digitise.py): predicted level at 6 months with one-sided 95 % upper bound ===")
print(f"{'level':8s} {'k(25/60) %/mo':>14s} {'pred 6 mo':>9s} {'95 % upper':>10s} {'observed':>8s} {'authors':>7s} {'auth/obs':>8s} | {'observed k (OLS of 4 pts)':>26s}")
b4 = o4['mean'][0]; C4 = o4['cov'][0]; nu4 = float(np.ravel(o4['nu'])[0])
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, os, pathlib
OUT = M.fig_dir()
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100, "savefig.dpi": 300,
                     "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "mathtext.fontset": "dejavusans",
                     "pdf.fonttype": 42, "ps.fonttype": 42})
fig, ax = plt.subplots(figsize=(4.2, 3.2)); tt = np.linspace(0, 6.5, 50)
for m, col in zip((0, 1, 2, 3, 4), ('C0', 'C1', 'C2', 'C3', 'C4')):
    x0 = x0vec(m); lnk = x0 @ b4; sc = np.sqrt(x0 @ C4 @ x0); k = np.exp(lnk) * 30.4375
    up = np.exp(lnk + stats.t.ppf(0.95, nu4) * sc) * 30.4375
    y = np.array(REALTIME[m]); tc = RT_MONTHS - RT_MONTHS.mean(); kobs = (tc * y).sum() / (tc ** 2).sum()
    resid = y - (y.mean() - kobs * RT_MONTHS.mean()) - kobs * RT_MONTHS; se_k = np.sqrt((resid ** 2).sum() / 2 / (tc ** 2).sum())
    print(f"MgSt {m} % {k:14.3f} {ALPHA0 + 6 * k:9.1f} {ALPHA0 + 6 * up:10.1f} {REALTIME[m][-1]:8.2f} {AUTHORS_6MO[m]:7.2f} {AUTHORS_6MO[m] / REALTIME[m][-1]:8.2f} | {kobs:8.3f} ± {se_k:.3f} %/mo")
    ax.plot(tt, ALPHA0 + k * tt, '-', color=col, lw=1, label=f'MgSt {m} %'); ax.plot(tt, ALPHA0 + up * tt, ':', color=col, lw=0.8)
    ax.plot(RT_MONTHS, y, 'o', color=col, ms=4, mfc='white')
    ax.text(6.55, ALPHA0 + k * 6.5, f'{m} %', color=col, fontsize=7, va='center', ha='left')   # 白黒でも系列が読めるように右端にラベル
ax.set_xlabel('months at 25 °C / 60 % RH'); ax.set_ylabel('total related substances (%)'); ax.set_xlim(0, 6.9); ax.set_ylim(0, 15)
ax.legend(frameon=False, fontsize=7, loc='upper left'); ax.set_title('solid: posterior median; dotted: one-sided 95 % upper bound', loc='left', fontsize=8)
fig.tight_layout()
for ext in ('png', 'pdf'): fig.savefig(OUT / f"fig3_tamura_realtime.{ext}")
M.save_tiff(fig, OUT / "fig3_tamura_realtime.tiff", dpi=600)   # カラー/グレー: 規程 300/600 dpi
print(f"figure: {os.path.relpath(OUT / 'fig3_tamura_realtime.png')}")
