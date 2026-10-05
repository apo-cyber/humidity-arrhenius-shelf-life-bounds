"""paper-C のモデルと推定量(1 か所に集める)。simulate / nomogram / budget はここだけを import する。

  モデル   ln k = beta0 + Ea*x1 + B*x2,  x1 = -(1/T - 1/T_ref)/R [mol/kJ],  x2 = RH - RH_ref [%RH]
           一次分解 ln(C/C0) = -k t + eps,  eps ~ N(0, sigma^2)(paper-A の生成モデル)
  真値     Ea = 80 kJ/mol, B = 0.04 /%RH, t_SL(95 %) = 30 か月 @ 25 °C / 60 %RH → t90 = 61.6 か月(paper-A CORE_BASELINE)
  Stage 1  条件ごとの OLS → k_hat_i, S_tt,i, RSS_i, df_i = n_i - 2   (バッチは共通勾配・共通切片でプール)
  Stage 2  NIG 共役。3 方式:
             'obs'  観測 k_hat の重み k_hat_i^2 S_tt,i(paper-A 流儀。k_hat<=0 は捨てる)
             'fit'  当てはめ k~ の重み(ln k 尺度で反復。k_hat<=0 は捨てる)
             'gn'   k 空間 Gauss-Newton(Var(k_hat_i) = sigma^2/S_tt,i、捨てない)= 当てはめ重みの共役更新の Laplace 近似 ← 採用
           事前 beta|sigma^2 ~ N(mu0, sigma^2 Sigma0*),  Sigma0* = diag(SD^2)/sigma_ref^2,  sigma^2 ~ IG(a0, b0)
           x0'beta の周辺事後 = Student-t(2 a_n)
  評価量   t90(25 °C / 60 %RH)の片側 95 % 下限 LB。固定真値での頻度論的被覆、LB/真値、中央値/真値、z = (中央値 - 真)/scale
  閉形式   設計の共分散(SE(ln k_i) 入力)、分散の内訳、rank-1 更新、Stage B の budget / 被覆 / 価格

時間格子は S_tt,i・n_i・RSS_i としてしか入らない(等間隔・条件間で同一である必要はない)。
"""
from __future__ import annotations
import numpy as np
from scipy import stats

R = 8.314e-3                                   # kJ/mol/K
T0, RH0 = 25.0, 60.0                           # 保存条件
EA, B_TRUE = 80.0, 0.04                        # 真値
T_SL = 30.0                                    # t_SL(95 %) [か月] @ 25/60
K25 = -np.log(0.95) / T_SL
T90_TRUE = -np.log(0.9) / K25                  # 61.62 か月
LNT = np.log(T90_TRUE)
LN_C = np.log(np.log(10 / 9))                  # ln g(0.10), 一次
Z95 = stats.norm.ppf(0.95)                     # 1.645

# 時間格子 [か月]
MONTHS = np.array([0., 2., 4., 6.])                       # paper-A/B の基準(等間隔 4 点)
MONTHS_ICH = np.array([0., 3., 6.])                       # ICH Q1A(R2) の例示 = A/B の n_pts = 3
MONTHS_SUPAC = np.array([0., 1., 3.])                     # 3 か月比較(SUPAC-IR Level 2 相当、時点は慣行)
DAYS14 = np.array([0., 3., 7., 14.]) / 30.4375
DAYS28 = np.array([0., 7., 14., 28.]) / 30.4375
ISO_FRAC = np.array([0., 1 / 3, 2 / 3, 1.])                # isoconversion 格子: 各条件を alpha* 到達時間 t_iso まで、4 点等分


def x1_abs(TC):
    """-1/(R T)。参照条件で中心化する前の温度説明変数。"""
    return -1.0 / (R * (TC + 273.15))


LNA = np.log(K25) - EA * x1_abs(T0) - B_TRUE * RH0


def k_true(T, rh):
    return np.exp(LNA + EA * x1_abs(T) + B_TRUE * rh)


def iso_grid(T, rh, alpha=0.10, frac=ISO_FRAC):
    """ASAP の isoconversion 設計: 条件 (T, rh) を alpha 到達まで走らせる格子(一次、真値の k)。SE(ln k) は sigma/alpha の定数倍になる。"""
    return -np.log(1 - alpha) / k_true(T, rh) * np.asarray(frac, float)


def stt(t):
    t = np.asarray(t, float)
    return float(((t - t.mean()) ** 2).sum())


def se_lnk(T, rh, grid, sig, batches=1):
    """Stage 1 の SE(ln k_i) = sigma / (k_i sqrt(S_tt,i))。バッチは格子を並べる(共通勾配)。"""
    t = np.tile(np.asarray(grid, float), batches)
    return sig / (k_true(T, rh) * np.sqrt(stt(t)))


def design_matrix(conds, ref=None):
    """X (n×3)。ref=(T_ref, RH_ref) で中心化。None なら中心化しない(x1 = -1/RT、RH そのまま)。"""
    X = np.array([[1.0, x1_abs(T), rh] for T, rh in conds])
    if ref is not None:
        X[:, 1] -= x1_abs(ref[0]); X[:, 2] -= ref[1]
    return X


def x_target(ref=None, T=T0, rh=RH0):
    x = np.array([1.0, x1_abs(T), rh])
    if ref is not None:
        x[1] -= x1_abs(ref[0]); x[2] -= ref[1]
    return x


# ------------------------------------------------------------------ Stage 1
def stage1(rng, conds, grids, nrep, sig, batches=1):
    """条件ごとに ln(C/C0) を t に OLS。返り値: KH (nrep×n), RSS (nrep×n), STT (n), DF (n), N (n)。"""
    n = len(conds)
    KH = np.zeros((nrep, n)); RSS = np.zeros((nrep, n)); STT = np.zeros(n); DF = np.zeros(n); N = np.zeros(n)
    for i, ((T, rh), t) in enumerate(zip(conds, grids)):
        t = np.tile(np.asarray(t, float), batches); k = k_true(T, rh)
        y = -k * t + rng.normal(0, sig, (nrep, t.size))
        tc = t - t.mean(); Stt = (tc ** 2).sum()
        slope = (y * tc).sum(1) / Stt; icpt = y.mean(1) - slope * t.mean()
        RSS[:, i] = ((y - icpt[:, None] - slope[:, None] * t) ** 2).sum(1)
        KH[:, i] = -slope; STT[i] = Stt; DF[i] = t.size - 2; N[i] = t.size
    return dict(KH=KH, RSS=RSS, STT=STT, DF=DF, N=N, conds=list(conds))


# ------------------------------------------------------------------ Stage 2
def stage2(s1, mu0, sd, method='gn', a0=0.5, b0=None, sig_ref=0.02, iters=6, ref=None, X=None, x0=None, cov0=None):
    """NIG 共役更新(3 方式)。mu0, sd は (beta0, Ea, B) の事前平均と SD。
    ref で中心化すると beta0 の意味が変わるので、mu0[0] は ref での ln k として与える(平坦なら無関係)。
    返り値: lb (ln t90 の片側 95 % 下限), med (事後中央値), scale, nu, drop (k_hat<=0 の割合), mean (nrep×3), cov (nrep×3×3, sigma^2 単位)。"""
    KH, RSS, STT, DF, conds = s1['KH'], s1['RSS'], s1['STT'], s1['DF'], s1['conds']
    nrep, n = KH.shape
    if X is None:
        X = design_matrix(conds, ref); x0 = x_target(ref)
    X = np.asarray(X, float); x0 = np.asarray(x0, float); p = X.shape[1]
    if b0 is None:
        b0 = a0 * sig_ref ** 2
    mu0 = np.asarray(mu0, float)
    # 事前精度(sigma^2 単位): 対角 sd か、輸入用の共分散 cov0(絶対単位、sigma_ref^2 = b0/a0 でスケール)
    P0m = (b0 / a0) * (np.linalg.inv(np.asarray(cov0, float)) if cov0 is not None else np.diag(1 / np.asarray(sd, float) ** 2))
    P0 = np.diag(P0m)                                                                 # 'obs'/'fit' の b_n 用(対角近似)
    valid = KH > 0; y = np.where(valid, np.log(np.where(valid, KH, 1.0)), 0.0)
    a1 = a0 + DF.sum() / 2; b1 = b0 + RSS.sum(1) / 2
    Wd = np.where(valid, KH ** 2 * STT[None], 0.0)                                   # 観測重み(出発点)

    def lin(Wd):
        Pn = P0m[None] + np.einsum('ri,ij,ik->rjk', Wd, X, X); Sn = np.linalg.inv(Pn)
        mun = np.einsum('rjk,rk->rj', Sn, P0m @ mu0 + np.einsum('ri,ij->rj', Wd * y, X)); return Pn, Sn, mun

    Pn, Sn, beta = lin(Wd)
    if method == 'obs':
        an = a1 + valid.sum(1) / 2
        bn = b1 + 0.5 * ((Wd * (y - beta @ X.T) ** 2).sum(1) + (P0 * (beta - mu0) ** 2).sum(1))
    elif method == 'fit':
        for _ in range(iters):
            Wd = np.where(valid, np.exp(beta @ X.T) ** 2 * STT[None], 0.0); Pn, Sn, beta = lin(Wd)
        an = a1 + valid.sum(1) / 2
        bn = b1 + 0.5 * ((Wd * (y - beta @ X.T) ** 2).sum(1) + (P0 * (beta - mu0) ** 2).sum(1))
    elif method == 'gn':
        for _ in range(iters):
            kf = np.exp(beta @ X.T); J = kf[:, :, None] * X[None]
            H = P0m[None] + np.einsum('i,rij,rik->rjk', STT, J, J)
            g = (mu0 - beta) @ P0m + np.einsum('i,rij,ri->rj', STT, J, KH - kf)
            beta = beta + np.einsum('rjk,rk->rj', np.linalg.inv(H), g)
        kf = np.exp(beta @ X.T); J = kf[:, :, None] * X[None]
        Sn = np.linalg.inv(P0m[None] + np.einsum('i,rij,rik->rjk', STT, J, J))
        an = a1 + n / 2
        d = beta - mu0
        bn = b1 + 0.5 * ((STT[None] * (KH - kf) ** 2).sum(1) + np.einsum('rj,jk,rk->r', d, P0m, d))
    else:
        raise ValueError(method)
    m = beta @ x0; scale = np.sqrt(bn / an * np.einsum('j,rjk,k->r', x0, Sn, x0)); nu = 2 * an
    return dict(lb=LN_C - m - stats.t.ppf(0.95, nu) * scale, med=LN_C - m, scale=scale, nu=nu,
                drop=(~valid).mean(), mean=beta, cov=Sn * (bn / an)[:, None, None])


def summarise(o):
    """被覆, LB/真値, LB/中央値, 中央値/真値, z 平均, z SD, drop %。"""
    z = (o['med'] - LNT) / o['scale']
    return dict(cover=(o['lb'] <= LNT).mean(), lb_true=np.exp(np.median(o['lb'])) / T90_TRUE,
                lb_med=np.exp(np.median(o['lb'] - o['med'])), med_true=np.exp(np.median(o['med'])) / T90_TRUE,
                z_mean=z.mean(), z_sd=z.std(), drop=100 * o['drop'])


# ------------------------------------------------------------------ 閉形式(設計)
def design_cov(conds, v, prior_sd=None, ref='centroid'):
    """データを取る前の共分散。v = 条件ごとの Var(ln k_i)(スカラーか長さ n)。
    prior_sd = (SD_Ea, SD_B) なら事前を足す(切片は平坦)。ref='centroid' は精度重みつき重心で中心化(切片と勾配が直交)。
    返り値 dict: Sigma (3×3, ln k 単位), se_Ea, se_B, corr, sd0 (ln t90 @ 25/60 の SD), terms (内訳), x0, ref。"""
    v = np.broadcast_to(np.asarray(v, float), (len(conds),)); w = 1 / v
    Xa = design_matrix(conds, None)
    if ref == 'centroid':
        c1 = (w * Xa[:, 1]).sum() / w.sum(); c2 = (w * Xa[:, 2]).sum() / w.sum()
        X = Xa.copy(); X[:, 1] -= c1; X[:, 2] -= c2
        x0 = np.array([1.0, x1_abs(T0) - c1, RH0 - c2])
    else:
        X = design_matrix(conds, ref); x0 = x_target(ref)
    P = (X * w[:, None]).T @ X
    if prior_sd is not None:
        P = P + np.diag([0.0, 1 / prior_sd[0] ** 2, 1 / prior_sd[1] ** 2])
    if np.linalg.cond(P) > 1e12:
        return None
    S = np.linalg.inv(P)
    se_Ea, se_B = np.sqrt(S[1, 1]), np.sqrt(S[2, 2])
    terms = dict(intercept=S[0, 0], T=x0[1] ** 2 * S[1, 1], RH=x0[2] ** 2 * S[2, 2],
                 icpt_T=2 * x0[1] * S[0, 1], icpt_RH=2 * x0[2] * S[0, 2], EaB=2 * x0[1] * x0[2] * S[1, 2])
    sd0 = np.sqrt(x0 @ S @ x0)
    return dict(Sigma=S, se_Ea=se_Ea, se_B=se_B, corr=S[1, 2] / (se_Ea * se_B), sd0=sd0, terms=terms, x0=x0)


def rank1_add(S, x_plus, v_plus):
    """条件 x_plus(精度 1/v_plus)を 1 つ足したときの共分散。"""
    Sx = S @ x_plus
    return S - np.outer(Sx, Sx) / (v_plus + x_plus @ Sx)


def bound_over_median(sd0, nu=None):
    """片側 95 % 下限 / 中央値 = exp(-q sd0)。nu=None なら正規分位。"""
    q = Z95 if nu is None else stats.t.ppf(0.95, nu)
    return np.exp(-q * sd0)


# ------------------------------------------------------------------ 閉形式(Stage B: 単一条件への事前輸入)
def levers(T_B=40.0, RH_B=75.0):
    """試験条件 → 保存条件のてこ (L_T [mol/kJ], L_RH [%RH])。ln k0 = ln k_B + L_T Ea + L_RH B。"""
    return x1_abs(T0) - x1_abs(T_B), RH0 - RH_B


def stageB_scale(s_data, sd_Ea, sd_B, rho=0.0, T_B=40.0, RH_B=75.0):
    """s^2 = s_data^2 + l' Sigma_A l。"""
    L_T, L_RH = levers(T_B, RH_B)
    return np.sqrt(s_data ** 2 + (L_T * sd_Ea) ** 2 + (L_RH * sd_B) ** 2 + 2 * L_T * L_RH * rho * sd_Ea * sd_B)


def shift_of(dEa, dB, T_B=40.0, RH_B=75.0):
    """事前平均のずれ (dEa, dB) による ln t90 中央値の変位(> 0 が反保守)。"""
    L_T, L_RH = levers(T_B, RH_B)
    return -(L_T * dEa + L_RH * dB)


def coverage_cf(s, shift, s_data):
    return stats.norm.cdf((Z95 * s - shift) / s_data)


def budget(s, s_data):
    return Z95 * (s - s_data)


def inflation_f(shift, s_data, prior_var_lever):
    """budget(f) = shift を解く f。prior_var_lever = l' Sigma_A l(f=1)。shift <= budget(1) なら 1。"""
    need = max(shift / Z95 + s_data, 0.0) ** 2 - s_data ** 2
    return max(1.0, np.sqrt(need / prior_var_lever))

# ------------------------------------------------------------------ 正面比較用の他の区間法(同じ Stage 1 要約から)
def stage2_paperA(s1, mu0, sd, floor=0.05, ref=None):
    """paper-A 公開版を 2 軸に転記: 観測重み 1/sigma_j^2, sigma_j = max(s_j, floor k_hat_j)/k_hat_j(s_j = Stage 1 の SE(k_hat)、
    残差 df で推定)、beta の正規事後(sigma^2 の事前なし)、区間は v + 残差分散 SS_res/(n-3) を足して正規分位(片側 1.645)。
    k_hat<=0 は捨てる。返り値は stage2 と同じ形(nu=inf)。"""
    KH, RSS, STT, DF, conds = s1['KH'], s1['RSS'], s1['STT'], s1['DF'], s1['conds']
    nrep, n = KH.shape; X = design_matrix(conds, ref); x0 = x_target(ref); p = X.shape[1]
    valid = KH > 0
    s_j = np.sqrt(RSS / DF[None] / STT[None])                                   # SE(k_hat_j) の推定
    sig_j = np.maximum(s_j, floor * np.abs(KH)) / np.where(valid, KH, 1.0)      # delta 法 SE(ln k_hat)、フロア
    W = np.where(valid, 1 / sig_j ** 2, 0.0); y = np.where(valid, np.log(np.where(valid, KH, 1.0)), 0.0)
    P0 = 1 / np.asarray(sd, float) ** 2; mu0 = np.asarray(mu0, float)
    Pn = np.diag(P0)[None] + np.einsum('ri,ij,ik->rjk', W, X, X); Sn = np.linalg.inv(Pn)
    beta = np.einsum('rjk,rk->rj', Sn, P0 * mu0 + np.einsum('ri,ij->rj', W * y, X))
    res = np.where(valid, y - beta @ X.T, 0.0); ssres = (res ** 2).sum(1); dfres = np.maximum(valid.sum(1) - p, 1)
    v = np.einsum('j,rjk,k->r', x0, Sn, x0) + ssres / dfres
    m = beta @ x0; scale = np.sqrt(v)
    return dict(lb=LN_C - m - Z95 * scale, med=LN_C - m, scale=scale, nu=np.inf, drop=(~valid).mean())


def one_stage_delta(s1, ref=None, iters=8, fbh_draws=0, rng=None):
    """AccelStab 型: 全生データの一段非線形当てはめ(条件ごとの切片 + 共通 beta、一次)+ delta 法 + t 分位。
    条件ごとの切片を消去すると、beta の最尤 = k 空間 GN(平坦事前)、総 RSS = sum RSS_i + sum S_tt,i (k_hat_i - k_i(beta))^2 なので
    Stage 1 の要約だけで厳密に計算できる。df = N - n - 3。fbh_draws>0 なら多変量 t から引いて分位(FBH)も返す。"""
    KH, RSS, STT, DF, N, conds = s1['KH'], s1['RSS'], s1['STT'], s1['DF'], s1['N'], s1['conds']
    nrep, n = KH.shape; X = design_matrix(conds, ref); x0 = x_target(ref)
    valid = KH > 0; y = np.where(valid, np.log(np.where(valid, KH, 1.0)), 0.0); Wd = np.where(valid, KH ** 2 * STT[None], 0.0)
    Pn = np.einsum('ri,ij,ik->rjk', Wd, X, X) + 1e-12 * np.eye(3)[None]; beta = np.einsum('rjk,rk->rj', np.linalg.inv(Pn), np.einsum('ri,ij->rj', Wd * y, X))
    for _ in range(iters):
        kf = np.exp(beta @ X.T); J = kf[:, :, None] * X[None]
        H = np.einsum('i,rij,rik->rjk', STT, J, J); g = np.einsum('i,rij,ri->rj', STT, J, KH - kf)
        beta = beta + np.einsum('rjk,rk->rj', np.linalg.inv(H), g)
    kf = np.exp(beta @ X.T); J = kf[:, :, None] * X[None]
    Hinv = np.linalg.inv(np.einsum('i,rij,rik->rjk', STT, J, J))
    df = N.sum() - n - 3; s2 = (RSS.sum(1) + (STT[None] * (KH - kf) ** 2).sum(1)) / df
    m = beta @ x0; scale = np.sqrt(s2 * np.einsum('j,rjk,k->r', x0, Hinv, x0))
    out = dict(lb=LN_C - m - stats.t.ppf(0.95, df) * scale, med=LN_C - m, scale=scale, nu=df, drop=0.0)
    if fbh_draws:
        # theta ~ t_df(beta, s2 Hinv): x0'theta は t_df(m, scale) なので分位を乱数で取る
        tdraw = stats.t.rvs(df, size=(nrep, fbh_draws), random_state=rng)
        out['lb_fbh'] = LN_C - np.quantile(m[:, None] + scale[:, None] * tdraw, 0.95, axis=1)
    return out


def asap_type_mc(s1, draws=2000, rng=None, ref=None):
    """ASAP 型(仕様は公開情報からの推定): ln t_iso,i(= -ln k_hat_i + const)を無重みの最小二乗で (1, x1, RH) に回帰し、
    Stage 1 の誤差 SE(ln k_hat_i)(残差 df で推定)から ln k を draws 回引き直して refit、ln t90 の 5 % 点を下限に。k_hat<=0 は捨てる。"""
    KH, RSS, STT, DF, conds = s1['KH'], s1['RSS'], s1['STT'], s1['DF'], s1['conds']
    nrep, n = KH.shape; X = design_matrix(conds, ref); x0 = x_target(ref)
    valid = KH > 0; y = np.where(valid, np.log(np.where(valid, KH, 1.0)), 0.0)
    se = np.sqrt(RSS / DF[None] / STT[None]) / np.where(valid, KH, 1.0)
    Wv = valid.astype(float)
    Pn = np.einsum('ri,ij,ik->rjk', Wv, X, X) + 1e-12 * np.eye(3)[None]; Pinv = np.linalg.inv(Pn)
    beta = np.einsum('rjk,rk->rj', Pinv, np.einsum('ri,ij->rj', Wv * y, X)); m = beta @ x0
    ystar = y[:, None, :] + se[:, None, :] * rng.normal(size=(nrep, draws, n)) * Wv[:, None, :]
    bstar = np.einsum('rjk,rdk->rdj', Pinv, np.einsum('rdi,ri,ij->rdj', ystar, Wv, X))
    lnk0 = bstar @ x0
    return dict(lb=LN_C - np.quantile(lnk0, 0.95, axis=1), med=LN_C - m, scale=np.std(lnk0, axis=1), nu=np.nan, drop=(~valid).mean())


def bias_obs_weights(conds, grids, sig, batches=1, ref=None):
    """命題 2(式 (13′)): 2 次までの偏り、x0 での ln k(ln t90 の偏りは符号反転)。
    (i) 観測重み: sigma^2 M^{-1} X' (3/2 1 - 2h) = Jensen (-v_i/2) + 観測重み項 (2 v_i (1-h_ii))
    (ii) 対数尺度 + 当てはめ重み: Jensen のみ  (iii) k 空間 GN: Box (1971) の曲率バイアス -sigma^2/2 M^{-1} X' h"""
    X = design_matrix(conds, ref); x0 = x_target(ref)
    v = np.array([se_lnk(T, rh, g, sig, batches) ** 2 for (T, rh), g in zip(conds, grids)])
    w = sig ** 2 / v
    Minv = np.linalg.inv(X.T @ (w[:, None] * X))
    h = w * np.einsum('ij,jk,ik->i', X, Minv, X)
    reg = lambda c: float(x0 @ Minv @ X.T @ (w * c))
    # obs: 観測重み(命題 2 (i))、jensen: 対数尺度 + 当てはめ重み(ii)、box: k 空間 GN の曲率バイアス(iii、Box 1971)
    return dict(v=v, h=h, jensen=reg(-v / 2), obs=reg(2 * v * (1 - h)), total=reg(v * (1.5 - 2 * h)), box=reg(-v * h / 2))


def save_tiff(fig, path, dpi):
    """投稿用 TIFF: RGB(アルファなし)、LZW、dpi をヘッダに記録。"""
    from PIL import Image
    import io
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=dpi, facecolor="white"); buf.seek(0)
    Image.open(buf).convert("RGB").save(path, compression="tiff_lzw", dpi=(dpi, dpi))


def fig_dir():
    """図の出力先: $PAPERC_FIG_DIR、なければ ./figures(公開版の平置き)か ../figures(paper-C/ の配置)。"""
    import os, pathlib
    here = pathlib.Path(__file__).resolve().parent
    out = pathlib.Path(os.environ.get("PAPERC_FIG_DIR", here / "figures" if (here / "figures").is_dir() else here.parent / "figures"))
    out.mkdir(exist_ok=True)
    return out
