"""Proposition 3 の数値検算: 条件ごとの妨害パラメータ(切片 c_i と曲率 d_i)を持つモデル
y_ij = c_i − k_i t_ij + d_i t_ij² + ε、k_i = exp(x_iᵀβ)、ε ~ N(0, σ²) で、
全モデルの Fisher 情報から (c, d) をプロファイルした β の情報(Schur 補元)が、
条件ごとの単独当てはめの SE²(ln k̂_i) を重みにした段階 2 の情報 Σ x_i x_iᵀ / SE² に一致することを確認する。"""
import numpy as np
R=8.314e-3; Tref=323.15; RHref=67.5
conds=[(40,60),(40,75),(50,60),(50,75),(60,60),(60,75)]          # 参照構成
X=np.array([[1.0, -(1/R)*(1/(T+273.15)-1/Tref), rh-RHref] for T,rh in conds])
beta=np.array([np.log(0.02), 80.0, 0.04]); sigma=0.02; t=np.array([0.,2.,4.,6.])
k=np.exp(X@beta); n=len(conds); p=3
# 全モデルのヤコビアン: パラメータ (β[3], c[6], d[6])
rows=[]
for i in range(n):
    for tj in t:
        r=np.zeros(p+2*n); r[:p]=-k[i]*tj*X[i]; r[p+i]=1.0; r[p+n+i]=tj**2; rows.append(r)
J=np.array(rows); I_full=J.T@J/sigma**2
Ibb=I_full[:p,:p]; Ibn=I_full[:p,p:]; Inn=I_full[p:,p:]
I_prof=Ibb-Ibn@np.linalg.solve(Inn,Ibn.T)
# 段階 1: 条件ごとに y を (1, t, t²) に回帰 → 傾き(−k̂)の分散 → SE²(ln k̂)
Z=np.column_stack([np.ones_like(t),t,t**2]); var_slope=sigma**2*np.linalg.inv(Z.T@Z)[1,1]
se2=var_slope/k**2
I_two=sum(np.outer(X[i],X[i])/se2[i] for i in range(n))
print("max |I_profile − I_two-stage| / |I_profile| =", np.linalg.norm(I_prof-I_two)/np.linalg.norm(I_prof))
print("SE(ln k̂_i) per condition:", np.round(np.sqrt(se2),3))
# 対照: 妨害を切片だけにした一次モデル(本文 Table 1 の消去)でも同じ
rows=[]
for i in range(n):
    for tj in t:
        r=np.zeros(p+n); r[:p]=-k[i]*tj*X[i]; r[p+i]=1.0; rows.append(r)
J=np.array(rows); I=J.T@J/sigma**2; Iprof1=I[:p,:p]-I[:p,p:]@np.linalg.solve(I[p:,p:],I[:p,p:].T)
Stt=np.sum((t-t.mean())**2); I_two1=sum(np.outer(X[i],X[i])*k[i]**2*Stt for i in range(n))/sigma**2
print("(対照: 切片のみ) max rel diff =", np.linalg.norm(Iprof1-I_two1)/np.linalg.norm(Iprof1))
