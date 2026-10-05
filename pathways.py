"""Section 5.5 / Proposition 4: two pathways. Generating model = first-order loss whose rate is the SUM of two
Arrhenius pathways (parallel; ln k convex in 1/T) or the reciprocal sum (consecutive; concave), both sharing B.
Calibrated so that k(25 °C/60 % RH) = K25 (t90 = 61.6 mo unchanged) with equal pathway rates at 25 °C.
The estimator of the paper (single Arrhenius line, working weights, NIG-t, weak prior) is applied unchanged.
Reported: coverage of the one-sided 95 % bound, LB/true, median/true, the simulated displacement of ln k at 25 °C
(= -ln(median/true)), the noise-free (pseudo-true) displacement of the weighted least-squares line, and the
second-order closed form 1/2 Var_w(E) (l_T^2 - s_T^2) of Proposition 4(iii)."""
import numpy as np
import model as M

NREP = 4000; SIG = 0.02; SEED = 11
ICH3 = [(T, rh) for T in (40, 50, 60) for rh in (60, 75)]
ASAP5 = [(50, 75), (60, 40), (70, 5), (70, 75), (80, 40)]


def two_path(delta, kind):
    """returns k(T, rh) for two pathways with E = EA ± delta/2, equal rates at 25 °C, common B."""
    E1, E2 = M.EA - delta / 2, M.EA + delta / 2
    x25 = M.x1_abs(M.T0)
    if kind == 'parallel':
        lnA1 = np.log(M.K25 / 2) - E1 * x25 - M.B_TRUE * M.RH0; lnA2 = np.log(M.K25 / 2) - E2 * x25 - M.B_TRUE * M.RH0
        return lambda T, rh: np.exp(lnA1 + E1 * M.x1_abs(T) + M.B_TRUE * rh) + np.exp(lnA2 + E2 * M.x1_abs(T) + M.B_TRUE * rh)
    else:  # consecutive: 1/k = 1/k1 + 1/k2, each = 2 K25 at 25 °C
        lnA1 = np.log(2 * M.K25) - E1 * x25 - M.B_TRUE * M.RH0; lnA2 = np.log(2 * M.K25) - E2 * x25 - M.B_TRUE * M.RH0
        return lambda T, rh: 1.0 / (np.exp(-(lnA1 + E1 * M.x1_abs(T) + M.B_TRUE * rh)) + np.exp(-(lnA2 + E2 * M.x1_abs(T) + M.B_TRUE * rh)))


def stage1_k(rng, conds, grids, nrep, sig, kfun):
    n = len(conds); KH = np.zeros((nrep, n)); RSS = np.zeros((nrep, n)); STT = np.zeros(n); DF = np.zeros(n); N = np.zeros(n)
    for i, ((T, rh), t) in enumerate(zip(conds, grids)):
        t = np.asarray(t, float); k = kfun(T, rh); y = -k * t + rng.normal(0, sig, (nrep, t.size))
        tc = t - t.mean(); Stt = (tc ** 2).sum(); slope = (y * tc).sum(1) / Stt; icpt = y.mean(1) - slope * t.mean()
        RSS[:, i] = ((y - icpt[:, None] - slope[:, None] * t) ** 2).sum(1); KH[:, i] = -slope; STT[i] = Stt; DF[i] = t.size - 2; N[i] = t.size
    return dict(KH=KH, RSS=RSS, STT=STT, DF=DF, N=N, conds=list(conds))


def wls_line_displacement(conds, grids, kfun, sig):
    """noise-free weighted LS line of ln k on (1, x1, rh) with the working weights k^2 S_tt, evaluated at 25/60, minus true ln k."""
    X = M.design_matrix(conds); y = np.array([np.log(kfun(T, rh)) for T, rh in conds])
    w = np.array([kfun(T, rh) ** 2 * M.stt(np.asarray(g, float)) for (T, rh), g in zip(conds, grids)])
    W = np.diag(w); beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
    return float(M.x_target() @ beta - np.log(kfun(M.T0, M.RH0))), w


def closed_form(conds, w, delta, kind):
    """1/2 Var_w(E) (l_T^2 - s_T^2), Var_w(E) at the weighted design centre (temperature), sign by convexity."""
    x = np.array([M.x1_abs(T) for T, _ in conds]); Wn = w / w.sum(); xb = (Wn * x).sum(); l = M.x1_abs(M.T0) - xb; s2 = (Wn * (x - xb) ** 2).sum()
    E1, E2 = M.EA - delta / 2, M.EA + delta / 2
    # pathway weights at the centre: equal at 25 °C, then ∝ exp(E (x - x25)) (parallel) or ∝ exp(-E (x - x25)) (consecutive, in 1/k)
    dx = xb - M.x1_abs(M.T0); sgn = 1.0 if kind == 'parallel' else -1.0
    p1 = np.exp(sgn * E1 * dx); p2 = np.exp(sgn * E2 * dx); p1, p2 = p1 / (p1 + p2), p2 / (p1 + p2)
    var = p1 * p2 * delta ** 2
    return sgn * (-0.5) * var * (l * l - s2), var, l, s2   # convex (parallel): line below truth → displacement negative


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    print(f"=== Proposition 4: two pathways, E = 80 ± delta/2, equal rates at 25 °C, B = 0.04 shared; first-order loss; sigma={SIG}; weak prior; nrep={NREP} ===")
    print("displacement = ln k_hat(25/60) - ln k(25/60) of the fitted line (negative = rate too low = shelf life too long = anti-conservative)")
    for name, conds, grids in (("ICH 3x2, 0/2/4/6 mo", ICH3, [M.MONTHS] * 6), ("ASAP-5, isoconversion", ASAP5, None)):
        print(f"\n-- {name}")
        print(f"{'kind':12s} {'delta':>5s} {'Var_w(E)':>8s} {'cover':>6s} {'LB/true':>7s} {'med/true':>8s} {'disp sim':>9s} {'disp WLS':>9s} {'closed':>7s}")
        for kind in ('parallel', 'consecutive'):
            for delta in (0, 20, 40, 60):
                kf = two_path(delta, kind)
                g = grids if grids is not None else [(-np.log(0.9) / kf(T, rh)) * M.ISO_FRAC for T, rh in conds]
                s1 = stage1_k(rng, conds, g, NREP, SIG, kf)
                o = M.stage2(s1, [20, M.EA, M.B_TRUE], [100, 30, 0.05], 'gn', sig_ref=SIG, iters=8)
                sm = M.summarise(o)
                disp_sim = -np.log(sm['med_true'])
                disp_wls, w = wls_line_displacement(conds, g, kf, SIG)
                cf, var, l, s2 = closed_form(conds, w, delta, kind)
                print(f"{kind:12s} {delta:5d} {var:8.0f} {sm['cover']:6.3f} {sm['lb_true']:7.2f} {sm['med_true']:8.2f} {disp_sim:+9.3f} {disp_wls:+9.3f} {cf:+7.3f}")
        print(f"   (lever l_T = {l:+.4f} mol/kJ from the weighted centre, s_T^2 = {s2:.1e})")
