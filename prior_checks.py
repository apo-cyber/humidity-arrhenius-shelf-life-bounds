"""模擬査読 4 回目(10-27 夜)の M2: 事前のある枝の入力 2 つ。既存の出力は変えない(新しいファイルだけ)。
  (a) M2a′ sigma_ref: NIG の事前は beta|sigma^2 ~ N(mu0, sigma^2 Sigma0*), Sigma0* = Sigma0 / sigma_ref^2。
      これまでの模擬計算はすべて sigma_ref = 生成の sigma(compare / simulate / sensitivity_truth の sig_ref=sig)。
      弱い事前(SD 30 kJ/mol, 0.05 /%RH)で sigma_ref を sigma/2(事前が絶対単位で 2 倍に広がる)と 2 sigma(半分に狭まる)にずらす。
      セル: 表 1 の Cw(事前の平均は真値、乱数は compare.py と同じ 20260921)と、S7 で弱い事前が最も悪いセル
      (Ea 40, B 0.02, t_SL 30、事前の平均は (80, 0.04) に固定、乱数は sensitivity_truth.py と同じ)。sigma_ref = sigma の行は既存の出力と一致するはず。
  (b) M2b p_eff: 情報のある事前(平均は真値)で、a_n から p_eff を引く本稿の数え方(DF_FIX True)と厳密に共役な数え方(False)の被覆。
      事前 3 つ: 弱い (30, 0.05)、典型的な Stage A の事後 (5, 0.01)、狭い (3, 0.005)。設計 3 つ × 精度 2 つ、sigma_ref = sigma、a0 = 0。
    python3 prior_checks.py > output/prior_checks.txt   (~10 s)
"""
import zlib
import numpy as np
import model as M

NREP = 4000
ICH3 = [(T, rh) for T in (40, 50, 60) for rh in (60, 75)]
ICH2 = [(40, 60), (40, 75), (60, 60), (60, 75)]
ASAP5 = [(50, 75), (60, 40), (70, 5), (70, 75), (80, 40)]


def set_truth(Ea, B, t_sl):
    M.EA, M.B_TRUE, M.T_SL = Ea, B, t_sl
    M.K25 = -np.log(0.95) / t_sl
    M.T90_TRUE = -np.log(0.9) / M.K25; M.LNT = np.log(M.T90_TRUE)
    M.LNA = np.log(M.K25) - Ea * M.x1_abs(M.T0) - B * M.RH0


def designs():
    return {'ICH 3x2 0/2/4/6': (ICH3, [M.MONTHS] * 6),
            'ICH 2x2 0/3/6': (ICH2, [M.MONTHS_ICH] * 4),
            'ASAP-5 iso': (ASAP5, [M.iso_grid(T, rh) for T, rh in ASAP5])}


def cell(o):
    s = M.summarise(o)
    return f"cover {s['cover']:.3f}  LB/true {s['lb_true']:.2f}  med/true {s['med_true']:.2f}  nu {np.median(o['nu']):5.1f}  p_eff {np.median(o['p_eff']):.2f}"


print("=== (a) M2a': weakly informative prior (SD 30 kJ/mol, 0.05 /%RH), sigma_ref = generating sigma x {1/2, 1, 2} ===")
print("    sigma_ref = sigma/2 widens the prior twofold in absolute units, 2 sigma narrows it twofold; the factor-1 rows reproduce Table 1 (Cw) and Table S7")
set_truth(80.0, 0.04, 30.0)
for dname in ('ICH 3x2 0/2/4/6', 'ASAP-5 iso'):
    conds, grids = designs()[dname]
    for noise, sig in (('assay', 0.02), ('degradant', 0.0075)):
        rng = np.random.default_rng(20260921)
        s1 = M.stage1(rng, conds, grids, NREP, sig)
        print(f"-- reference truth (Ea 80, B 0.04, t_SL 30), prior mean at the truth, {dname}, {noise} sigma = {sig}")
        for f in (0.5, 1.0, 2.0):
            o = M.stage2(s1, [20, M.EA, M.B_TRUE], M.WEAK_SD, 'gn', sig_ref=f * sig)
            print(f"   sigma_ref = {f:3.1f} sigma: {cell(o)}")
set_truth(40.0, 0.02, 30.0)
for dname in ('ICH 3x2 0/2/4/6', 'ICH 2x2 0/3/6'):
    conds, grids = designs()[dname]
    sig = 0.02
    rng = np.random.default_rng(zlib.crc32(f"{sig}|{40.0}|{0.02}|{30.0}|{dname}".encode()))
    s1 = M.stage1(rng, conds, grids, NREP, sig)
    print(f"-- slow truth (Ea 40, B 0.02, t_SL 30), prior mean fixed at (80, 0.04), {dname}, assay sigma = {sig}")
    for f in (0.5, 1.0, 2.0):
        o = M.stage2(s1, [20, 80.0, 0.04], M.WEAK_SD, 'gn', sig_ref=f * sig)
        print(f"   sigma_ref = {f:3.1f} sigma: {cell(o)}")

print("\n=== (b) M2b: informative priors centred at the truth; a_n with p_eff subtracted (this paper) vs the strictly conjugate count ===")
print("    reference truth (Ea 80, B 0.04, t_SL 30); sigma_ref = sigma; a0 = b0 = 0; same Stage 1 data for both counts")
set_truth(80.0, 0.04, 30.0)
PRIORS = {'weak (30, 0.05)': [100.0, 30.0, 0.05], 'typical Stage A (5, 0.01)': [100.0, 5.0, 0.01], 'tight (3, 0.005)': [100.0, 3.0, 0.005]}
for dname, (conds, grids) in designs().items():
    for noise, sig in (('assay', 0.02), ('degradant', 0.0075)):
        rng = np.random.default_rng(zlib.crc32(f"M2b|{sig}|{dname}".encode()))
        s1 = M.stage1(rng, conds, grids, NREP, sig)
        print(f"-- {dname}, {noise} sigma = {sig}, residual df of the flat fit {int(s1['DF'].sum() + len(conds) - 3)}")
        for pname, sd in PRIORS.items():
            out = []
            for fix in (True, False):
                M.DF_FIX = fix
                out.append(M.stage2(s1, [20, M.EA, M.B_TRUE], sd, 'gn', sig_ref=sig))
            M.DF_FIX = True
            print(f"   {pname:26s} p_eff subtracted: {cell(out[0])}")
            print(f"   {'':26s} strictly conjugate: {cell(out[1])}")
