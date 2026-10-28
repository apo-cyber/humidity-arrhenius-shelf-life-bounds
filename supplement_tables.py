"""補足資料(Supplementary Material)の表を output/*.txt から組み立てる(10-27)。表の数値は手で写さない。
    python3 supplement_tables.py > output/supplement_tables.md
出力は表ごとに <!-- Sn --> の印をつけた Markdown のパイプ表。paper-C-SI.md の表はこの出力をそのまま貼ったもの。
"""
import re
import pathlib

OUT = pathlib.Path(__file__).resolve().parent / 'output'


def read(name):
    return (OUT / name).read_text()


# ------------------------------------------------------------------ S1: 設計 × 雑音(simulate.txt、Stage A、gn)
def s1():
    names = [('ICH 3x2 40/50/60 x RH60/75, 0/2/4/6 mo', 'ICH-type 40/50/60 × 60/75', '0/2/4/6 mo'),
             ('ICH 3x2, 0/3/6 mo (ICH example grid)', 'ICH-type 40/50/60 × 60/75', '0/3/6 mo'),
             ('ICH 2x2 40/60 x RH60/75, 0/2/4/6 mo', 'ICH-type 40/60 × 60/75', '0/2/4/6 mo'),
             ('ICH 2x2, 0/3/6 mo', 'ICH-type 40/60 × 60/75', '0/3/6 mo'),
             ('ASAP-5 isoconversion (each to alpha*)', 'ASAP five conditions', 'isoconversion'),
             ('ASAP-5 iso + 40/75 0/2/4/6 mo', 'ASAP isoconversion + 40/75', 'iso + 0/2/4/6 mo'),
             ('ASAP-5 28 d', 'ASAP five conditions', 'fixed 0/7/14/28 d'),
             ('ASAP-5 (50/75,60/40,70/5,70/75,80/40) 14 d', 'ASAP five conditions', 'fixed 0/3/7/14 d'),
             ('ASAP-5 28 d + 40/75 0/2/4/6 mo', 'ASAP fixed 28 d + 40/75', '28 d + 0/2/4/6 mo')]
    vals = {}; sig = None
    for line in read('simulate.txt').splitlines():
        m = re.search(r'sigma=([\d.]+)', line)
        if line.startswith('=== Stage A') and m:
            sig = float(m.group(1))
        if ' gn ' in line and sig is not None:
            f = line[46:].split()
            vals[(line[:46].strip(), sig)] = (float(f[1]), float(f[2]), float(f[6]), float(f[7]))   # cover, LB/true, z sd, drop%
    rows = ['| Design | Grid | 0.2 (assay): bound / cov. | $\\hat k \\le 0$ (assay) | 0.075 (degradant): bound / cov. | 0.025 (optimistic): bound / cov. |',
            '|---|---|---|---|---|---|']
    for key, lab, grid in names:
        a, d, o = vals[(key, 0.02)], vals[(key, 0.0075)], vals[(key, 0.0025)]
        rows.append(f"| {lab} | {grid} | {a[1]:.2f} / {a[0]:.3f} | {a[3]:.1f} % | {d[1]:.2f} / {d[0]:.3f} | {o[1]:.2f} / {o[0]:.3f} |")
    return '\n'.join(rows)


# ------------------------------------------------------------------ Stage B の格子(simulate.txt)
def stageB_grid(block_key, prior_key):
    txt = read('simulate.txt'); i = txt.index(block_key); j = txt.index(prior_key, i)
    k = txt.find('\n-- prior', j + 5); e = txt.find('\n===', j + 5)
    end = min(x for x in (k, e) if x > 0)
    grid = {}
    for line in txt[j:end].splitlines():
        m = re.match(r'\s+(-?\d+)\s+(-?[\d.]+)\s+([+-][\d.]+) \|\s+([\d.]+)\s+([\d.]+) \|\s+([\d.]+)\s+([\d.]+)\s+([\d.]+) \|\s+([\d.]+)\s+([\d.]+)', line)
        if m:
            grid[(int(m.group(1)), float(m.group(2)))] = dict(shift=float(m.group(3)), cf=float(m.group(4)), cft=float(m.group(5)),
                                                              sim=float(m.group(6)), lb=float(m.group(7)), nig=float(m.group(9)))
    return grid


def s4():
    g = stageB_grid('grid 0/2/4/6 mo x 3 batches', 'prior SD(Ea)=5, SD(B)=0.01')
    dBs = (-0.02, 0.0, 0.01, 0.02, 0.03)
    rows = ['| $\\Delta E_a$ \\\\ $\\Delta B$ | ' + ' | '.join(f'{d:+.2f}' for d in dBs) + ' |', '|---|' + '---|' * len(dBs)]
    for dE in (-10, 0, 5, 10, 15, 20):
        rows.append(f'| {dE:+d} | ' + ' | '.join(f"{g[(dE, d)]['cf']:.3f} / {g[(dE, d)]['sim']:.3f}" for d in dBs) + ' |')
    return '\n'.join(rows)


def s9():
    out = ['| Stage B design | $\\Delta E_a$, $\\Delta B$ | shift | closed form | closed form, $t$ | simulated | bound / truth |', '|---|---|---|---|---|---|---|']
    for key, lab in (('grid 0/3/6 mo x 3 batches', '0/3/6 mo × 3 batches'), ('grid 0/1/3 mo x 1 batch', '0/1/3 mo × 1 batch')):
        g = stageB_grid(key, 'prior SD(Ea)=5, SD(B)=0.01')
        for dE, dB in ((0, 0.0), (5, 0.0), (10, 0.0), (0, 0.01), (0, 0.02), (10, 0.02)):
            c = g[(dE, dB)]
            out.append(f"| {lab} | {dE:+d}, {dB:+.2f} | {c['shift']:+.2f} | {c['cf']:.3f} | {c['cft']:.3f} | {c['sim']:.3f} | {c['lb']:.2f} |")
    return '\n'.join(out)


# ------------------------------------------------------------------ S3: 膨張(simulate.txt「Transfer inflation」0/2/4/6)
def s3():
    txt = read('simulate.txt'); i = txt.index('base prior (5, 0.01), Stage B 0/2/4/6'); j = txt.index('base prior', i + 10)
    rows = ['| Displacement covered | shift | $f$ | Inflated $\\mathrm{SD}(E_a)$, $\\mathrm{SD}(B)$ | Bound / median |', '|---|---|---|---|---|',
            '| none | 0 | 1.00 | 5, 0.010 | ' + re.search(r'LB/med=([\d.]+)', txt[i:j]).group(1) + ' |']
    for line in txt[i:j].splitlines():
        m = re.search(r'cover dEa=\s*(\d+) dB=([\d.]+) \(shift ([+-][\d.]+)\): f=([\d.]+) -> SD\(Ea\)=([\d.]+), SD\(B\)=([\d.]+); LB/med [\d.]+ -> ([\d.]+)', line)
        if m:
            dE, dB = int(m.group(1)), float(m.group(2))
            lab = ' and '.join(x for x in ((f'$\\Delta E_a = {dE}$' if dE else ''), (f'$\\Delta B = {dB:g}$' if dB else '')) if x)
            rows.append(f"| {lab} | {float(m.group(3)):.2f} | {m.group(4)} | {m.group(5)}, {m.group(6)} | {m.group(7)} |")
    return '\n'.join(rows)


# ------------------------------------------------------------------ S5: 反応次数(robustness.txt (B))
def s5():
    txt = read('robustness.txt'); i = txt.index('=== (B)'); j = txt.index('-- ASAP-like', i)
    rows = ['| True $n$ | Fit | Coverage | Median $t_{90}^{\\,L}/t_{90}$ | Median $\\hat t_{90}/t_{90}$ |', '|---|---|---|---|---|']
    true_n = None
    labels = {'n=1 assumed = true': '$n = 1$ (correct)', 'first-order (n=1) assumed': '$n = 1$ assumed',
              'n profiled on grid, plug-in best n': '$n$ profiled, plug-in', 'n profiled, t-mixture over weights': '$n$ profiled, $t$-mixture'}
    for line in txt[i:j].splitlines():
        m = re.match(r'-- true n=(\d)', line)
        if m:
            true_n = m.group(1); continue
        m = re.match(r'(.{34}) +([\d.]+) +([\d.]+) +([\d.]+)', line)
        if m and true_n is not None:
            key = m.group(1).strip()
            lab = labels.get(key) or (f'$n = {true_n}$ known' if key.startswith('true order') else None)
            if lab:
                rows.append(f"| {true_n} | {lab} | {m.group(2)} | {m.group(3)} | {m.group(4)} |")
    return '\n'.join(rows)


# ------------------------------------------------------------------ S6: 2 経路(pathways.txt)
def s6():
    txt = read('pathways.txt')
    rows = ['| Design | Pathways | $\\Delta$ | $\\mathrm{Var}_w(E)$ | Coverage | Median $\\hat t_{90}/t_{90}$ | Displ., simulated | Displ., line | Displ., closed form |',
            '|---|---|---|---|---|---|---|---|---|']
    for key, lab in (('-- ICH 3x2', 'ICH-type 3 × 2'), ('-- ASAP-5, isoconversion', 'ASAP isoconversion')):
        i = txt.index(key); j = txt.find('(lever', i)
        for line in txt[i:j].splitlines():
            m = re.match(r'(parallel|consecutive)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+[\d.]+\s+([\d.]+)\s+([+-][\d.]+)\s+([+-][\d.]+)\s+([+-][\d.]+)', line)
            if m and m.group(2) != '0':
                rows.append(f"| {lab} | {'par.' if m.group(1) == 'parallel' else 'cons.'} | {m.group(2)} | {m.group(3)} | {m.group(4)} | {m.group(5)} | {float(m.group(6)):+.2f} | {float(m.group(7)):+.2f} | {float(m.group(8)):+.2f} |")
    levers = re.findall(r'lever l_T = ([-\d.]+ to [-\d.]+) mol/kJ from the weighted centre, s_T\^2 = ([\d.e+-]+ to [\d.e+-]+)', txt)
    return '\n'.join(rows), levers


# ------------------------------------------------------------------ S7: 感度格子(sensitivity_truth.txt)。セルは 平坦: 被覆 / LB/真値(弱い事前の被覆)
def s7():
    cells = {}; noise = None
    for line in read('sensitivity_truth.txt').splitlines():
        m = re.search(r'noise: (\w+) sigma', line)
        if m:
            noise = m.group(1)
        m = re.match(r'Ea +(\d+) B ([\d.]+) tSL +(\d+) (.{16}) loss40 +([\d.]+) \| SEmax +([\d.]+) \| t_iso +\S+ \| ([\d.]+) +([\d.]+).*\| weak ([\d.]+) ', line)
        if m:
            cells[(m.group(1), m.group(2), m.group(3), m.group(4).strip(), noise)] = (m.group(5), m.group(7), m.group(8), m.group(9))
    designs = ('ICH 3x2 0/2/4/6', 'ICH 2x2 0/3/6', 'ASAP-5 iso')
    truths = sorted({k[:3] for k in cells}, key=lambda t: (int(t[0]), float(t[1]), int(t[2])))
    out = []
    for n in ('assay', 'degradant'):
        rows = ['| $E_a$ | $B$ | Shelf life (mo) | Loss at 40/75 in 6 mo (%) | ICH-type 3 × 2 | ICH-type 2 × 2, 0/3/6 | ASAP isoconversion |',
                '|---|---|---|---|---|---|---|']
        for Ea, B, tsl in truths:
            loss = cells[(Ea, B, tsl, designs[0], 'assay')][0]
            rows.append(f'| {Ea} | {B} | {tsl} | {loss} | ' + ' | '.join(f"{cells[(Ea, B, tsl, d, n)][1]} / {cells[(Ea, B, tsl, d, n)][2]} ({cells[(Ea, B, tsl, d, n)][3]})" for d in designs) + ' |')
        out.append('\n'.join(rows))
    return out


# ------------------------------------------------------------------ S8: 分析室の精度の違い(m2_sigma_mismatch.txt)
def s8():
    txt = read('m2_sigma_mismatch.txt')
    i = txt.index('-- prior displaced dB=+0.01'); j = txt.index('=== Stage B with few', i)
    rows = ['| $\\sigma_B/\\sigma_A$ | NIG, variance carried | factor on prior | NIG, variance not carried | decoupled (Stage B variance) | decoupled, variance carried |',
            '|---|---|---|---|---|---|']
    for line in txt[i:j].splitlines():
        m = re.match(r'\s+([\d.]+)\s+[\d.]+ \|\s+([\d.]+)\s+([\d.]+)\s+([\d.]+) \|\s+([\d.]+)\s+([\d.]+)\s+[\d.]+ \|\s+([\d.]+)\s+([\d.]+) \|\s+([\d.]+)\s+([\d.]+)', line)
        if m:
            rows.append(f"| {m.group(1)} | {m.group(2)} / {m.group(3)} | {m.group(4)} | {m.group(5)} / {m.group(6)} | {m.group(7)} / {m.group(8)} | {m.group(9)} / {m.group(10)} |")
    e = j                                                                            # 10-27 R2: Stage A の σ_ref の表(旧 S8b)は削除
    d = ['| Stage B design | Prior | Stage B variance only | Pooled with Stage A (same laboratory) |', '|---|---|---|---|']
    for line in txt[e:].splitlines():
        m = re.match(r'(0/\S+ x \d+ batch(?:es)?)\s+(.{22}) \| ([\d.]+ / [\d.]+)\s+\|\s+(.*)$', line)
        if m:
            d.append(f"| {m.group(1).replace(' x ', ' × ')} | {m.group(2).strip()} | {m.group(3)} | {m.group(4).strip()} |")
    return '\n'.join(rows), '\n'.join(d)


# ------------------------------------------------------------------ S10: 実時間の全時点(illustration.txt)
def s10():
    txt = read('illustration.txt'); i = txt.index('all time points'); j = txt.find('=== Prediction bound')   # 10-28: 予測の上限の段の手前まで
    rows = ['| MgSt (%) | 1 month | 2 months | 3 months | 6 months |', '|---|---|---|---|---|']
    for line in txt[i:j if j > 0 else None].splitlines():
        m = re.match(r'MgSt (\d) %  (.*)', line)
        if m:
            cells = []
            for c in m.group(2).split('|'):
                mm = re.search(r'([\d.]+) / +([\d.]+) / +([\d.]+)(\*?)', c)
                cells.append(f"{mm.group(1)} / {mm.group(2)} / {mm.group(3)}{' †' if mm.group(4) else ''}")
            rows.append(f"| {m.group(1)} | " + ' | '.join(cells) + ' |')
    return '\n'.join(rows)


# ------------------------------------------------------------------ S13: 推定量 3 つの偏り(bias.txt、命題 1)
def s13():
    txt = read('bias.txt'); i = txt.index('design          sigma'); j = txt.index('ICH 3Tx2RH, assay: v_i')
    names = {'ICH 3Tx2RH': 'ICH-type 3 × 2', 'ICH 2Tx2RH': 'ICH-type 2 × 2', 'ASAP iso': 'ASAP isoconversion', 'ASAP 28 d': 'ASAP fixed 28 d'}
    a = ['| Design | $\\sigma$ | largest $v_i$ | (i) observed weights | (ii) fitted weights, log scale | (iii) $k$-space Gauss–Newton | $\\hat k \\le 0$ |',
         '|---|---|---|---|---|---|---|']
    for line in txt[i:j].splitlines():
        m = re.match(r'(ICH 3Tx2RH|ICH 2Tx2RH|ASAP iso|ASAP 28 d)\s+([\d.]+)\s+([\d.]+) \|\s+(\S+)\s+(\S+)\s+\S+ \|\s+(\S+)\s+(\S+)\s+\S+ \|\s+(\S+)\s+(\S+)\s+\S+ \| ([\d.]+)', line)
        if m:
            a.append(f"| {names[m.group(1)]} | {float(m.group(2)):g} | {m.group(3)} | {m.group(4)} / {m.group(5)} | {m.group(6)} / {m.group(7)} | "
                     f"{m.group(8)} / {m.group(9)} | {100 * float(m.group(10)):.1f} % |")
    k = txt.index('=== Temperature only'); e = txt.index('0/2/4/6, assay: v_i', k)
    b = ['| Grid | $\\sigma$ | largest $v_i$ | (i) closed form / simulated mean | median $\\hat t_{90}/t_{90}$, (i) | median $\\hat t_{90}/t_{90}$, (iii) |', '|---|---|---|---|---|---|']
    for line in txt[k:e].splitlines():
        m = re.match(r'(0/2/4/6|0/3/6)\s+([\d.]+)\s+([\d.]+) \|\s+(\S+)\s+(\S+)\s+\S+\s+([\d.]+) \|\s+\S+\s+([\d.]+)', line)
        if m:
            b.append(f"| {m.group(1)} mo | {float(m.group(2)):g} | {m.group(3)} | {m.group(4)} / {m.group(5)} | {m.group(6)} | {m.group(7)} |")
    return '\n'.join(a), '\n'.join(b)


# ------------------------------------------------------------------ S14: バッチ間の変動(robustness.txt (A)、旧本文 Table 5)
def s14():
    txt = read('robustness.txt'); i = txt.index('-- Stage B: 40/75 only'); j = txt.index('-- Stage A: ICH 3x2 x 3 batches')
    sb = {}
    for line in txt[i:j].splitlines():
        m = re.match(r'\s*([\d.]+) \|\s+([\d.]+)\s+([\d.]+)\s+[\d.]+ \|\s+([\d.]+)\s+([\d.]+)\s+[\d.]+ \|\s+([\d.]+)\s+([\d.]+)', line)
        if m:
            sb[m.group(1)] = m.groups()[1:]
    e = txt.index('=== (B)')
    sa = {}; common = {}
    for line in txt[j:e].splitlines():
        m = re.match(r'\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+[\d.]+\s+[\d.]+\s+[\d.]+$', line)
        if m:
            sa[m.group(1)] = (m.group(2), m.group(3))
        for mm in re.finditer(r'tau ([\d.]+): ([\d.]+)', line):
            common[mm.group(1)] = mm.group(2)
    rows = ['| $\\tau$ | Stage B pooled, prior exact | Stage B pooled, $\\Delta E_a = 5$ | Stage B batch as unit, prior exact | Stage B batch as unit, $\\Delta E_a = 5$ | Stage A: coverage | Stage A: $t_{90}^{\\,L}/t_{90}$ | Stage A, common intercept: coverage |',
            '|---|---|---|---|---|---|---|---|']
    for tau, (pc, pl, uc, ul, p5, u5) in sb.items():
        rows.append(f"| {float(tau):g} | {pc} ({pl}) | {p5} | {uc} ({ul}) | {u5} | {sa[tau][0]} | {sa[tau][1]} | {common[tau]} |")
    return '\n'.join(rows)


# ------------------------------------------------------------------ S15: 計算例(worked_example.txt)
def s15():
    txt = read('worked_example.txt')
    data = ['| Condition | 0 mo | 2 mo | 4 mo | 6 mo |', '|---|---|---|---|---|']
    st1 = ['| Condition | $\\hat k_i$ (per month) | $\\mathrm{RSS}_i$ | $\\mathrm{SE}(\\ln\\hat k_i)$ |', '|---|---|---|---|']
    it = ['| Iteration | $E_a$ (kJ/mol) | $B$ (per % RH) | $\\ln k$ at 25 °C / 60 % RH |', '|---|---|---|---|']
    tail = []
    for line in txt.splitlines():
        m = re.match(r'\s+(\d+) C / (\d+) %RH: (.*)', line)
        if m:
            data.append(f"| {m.group(1)} °C / {m.group(2)} % RH | " + ' | '.join(m.group(3).split()) + ' |')
        m = re.match(r'\s+(\d+) C/(\d+) %RH\s+([\d.]+)\s+(\S+)\s+([\d.]+)', line)
        if m:
            st1.append(f"| {m.group(1)} °C / {m.group(2)} % RH | {m.group(3)} | {m.group(4)} | {m.group(5)} |")
        m = re.match(r'\s+(\d)\s+([\d.]+)\s+([\d.]+)\s+(-[\d.]+)$', line)
        if m:
            it.append(f"| {m.group(1)} | {m.group(2)} | {m.group(3)} | {m.group(4)} |")
    g = lambda pat: re.search(pat, txt).group(1)
    tail = ['| Quantity | Value |', '|---|---|',
            f"| $\\sum_i \\mathrm{{RSS}}_i$, $\\sum_i S_{{tt,i}}(\\hat k_i - \\tilde k_i)^2$ | {g(r'sum RSS_i = (\S+),')}, {g(r'k~\)\^2 = (\S+)')} |",
            f"| $p_{{\\mathrm{{eff}}}}$; $\\nu = 2a_n$ | {g(r'p_eff = ([\d.]+)')}; {g(r'= 12 \+ 6 - 3 = (\d+)')} |",
            f"| scale of $\\ln t_{{90}}$ at 25 °C / 60 % RH; $t_{{0.95,\\nu}}$ | {g(r'25/60 = ([\d.]+);')}; {g(r't_0.95,\d+ = ([\d.]+)')} |",
            f"| $\\ln t_{{90}}$: posterior median; one-sided 95 % bound | {g(r'ln t90: median ([\d.]+)')}; {g(r'one-sided 95 % bound ([\d.]+)')} |",
            f"| $t_{{90}}$ (months): median; bound; bound / median (truth {g(r'truth ([\d.]+)')}) | {g(r't90 \(months\): median ([\d.]+)')}; {g(r', bound ([\d.]+) \(truth')}; {g(r'bound / median ([\d.]+)')} |",
            f"| one-stage fit (delta method, $t$ on 15 df): median; bound (months); difference in $\\ln t_{{90}}$ | {g(r'NLS.*?median ([\d.]+)')}; {g(r'NLS.*?bound ([\d.]+) months')}; {g(r'difference in ln t90\| = (\S+)')} |"]
    return '\n'.join(data), '\n'.join(st1), '\n'.join(it), '\n'.join(tail)


# ------------------------------------------------------------------ S16: ASAP 型の再構成の 3 変種(compare.txt、模擬査読 3 回目 M5)
def s16():
    txt = read('compare.txt'); i = txt.index('=== ASAP-5 isoconversion')
    blocks = txt[i:].split('\n=== ')
    lab = {'S ': 'per-condition standard error (2 df)', 'Sp': '$\\sigma$ pooled over conditions', 'Sk': '$\\sigma$ supplied as known'}
    vals = {}
    for b in blocks:
        prec = 'assay' if 'assay sigma=0.02' in b else 'degradant'
        for line in b.splitlines():
            m = re.match(r'(S |Sp|Sk) .*?\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(-?[\d.]+)\s+([\d.]+)\s+([\d.]+)$', line)
            if m:
                vals[(m.group(1), prec)] = (m.group(2), m.group(3))
    rows = ['| Stage 1 standard error behind the Monte Carlo draws | $\\sigma/\\alpha^\\ast$ = 0.2: coverage / bound | 0.075: coverage / bound |', '|---|---|---|']
    for k in ('S ', 'Sp', 'Sk'):
        rows.append(f"| {lab[k]} | {vals[(k, 'assay')][0]} / {vals[(k, 'assay')][1]} | {vals[(k, 'degradant')][0]} / {vals[(k, 'degradant')][1]} |")
    return '\n'.join(rows)


# ------------------------------------------------------------------ S10b: 個々の測定値の片側 95 % 予測上限(illustration.txt、模擬査読 4 回目 M6)
def s10b():
    txt = read('illustration.txt'); txt = txt[txt.index('=== Prediction bound'):]
    blocks = re.split(r'-- sigma_e = ', txt)[1:]
    out = {}
    for b in blocks:
        se = re.match(r'([\d.]+)', b).group(1)
        for line in b.splitlines():
            m = re.match(r'MgSt (\d) %  (.*)', line)
            if m:
                cells = [re.search(r'([\d.]+) / +([\d.]+)(\*?)', c) for c in m.group(2).split('|')]
                out.setdefault(m.group(1), []).append([(c.group(1), c.group(2), c.group(3)) for c in cells])
    ses = [re.match(r'([\d.]+)', b).group(1) for b in blocks]
    rows = ['| MgSt (%) | 1 month | 2 months | 3 months | 6 months |', '|---|---|---|---|---|']
    for lev, (a, b) in out.items():
        rows.append(f"| {lev} | " + ' | '.join(f"{x[0]} / {x[1]}{' †' if x[2] else ''} / {y[1]}{' †' if y[2] else ''}" for x, y in zip(a, b)) + ' |')
    return '\n'.join(rows), ses


# ------------------------------------------------------------------ S17: 事前のある枝(prior_checks.txt、模擬査読 4 回目 M2)
def s17():
    lab = lambda d: {'ICH 3x2 0/2/4/6': 'ICH 3 × 2', 'ICH 2x2 0/3/6': 'ICH 2 × 2', 'ASAP-5 iso': 'ASAP iso'}[d]
    txt = read('prior_checks.txt'); ia = txt.index('=== (a)'); ib = txt.index('=== (b)')
    a = ['| Truth, prior mean | Design, precision | $\\sigma_{\\mathrm{ref}} = \\sigma/2$ | $\\sigma_{\\mathrm{ref}} = \\sigma$ | $\\sigma_{\\mathrm{ref}} = 2\\sigma$ |', '|---|---|---|---|---|']
    head = None; vals = []
    def flush():
        if head:
            a.append(f"| {head[0]} | {head[1]} | " + ' | '.join(vals) + ' |')
    for line in txt[ia:ib].splitlines():
        m = re.match(r'-- (reference|slow) truth .*?, prior mean (at the truth|fixed at \(80, 0.04\)), (.*?), (assay|degradant) sigma', line)
        if m:
            flush(); vals = []
            head = ('reference, at the truth' if m.group(1) == 'reference' else 'slow, at (80, 0.04)', f"{lab(m.group(3))}, {m.group(4)}")
        m = re.search(r'cover ([\d.]+)\s+LB/true ([\d.]+)', line)
        if m and 'sigma_ref' in line:
            vals.append(f"{m.group(1)} / {m.group(2)}")
    flush()
    b = ['| Design, precision (df) | Prior SD | $p_{\\mathrm{eff}}$ | Section 2.3: cov. / bound ($\\nu$) | Conjugate: cov. / bound ($\\nu$) |', '|--------------------|---------|-----|----------------------|----------------------|']   # 幅の比(pandoc は区切りの - の数で列幅を決める)
    des = None; pri = None; sub = None
    for line in txt[ib:].splitlines():
        m = re.match(r'-- (.*?), (assay|degradant) sigma = [\d.]+, residual df of the flat fit (\d+)', line)
        if m:
            des = f"{lab(m.group(1))}, {m.group(2)} ({m.group(3)})"
        m = re.match(r'\s+(weak|typical Stage A|tight) \(([\d.]+), ([\d.]+)\)\s+p_eff subtracted: cover ([\d.]+)\s+LB/true ([\d.]+).*?nu\s+([\d.]+)\s+p_eff ([\d.]+)', line)
        if m:
            pri = f"{m.group(2)}, {m.group(3)}"; sub = (m.group(4), m.group(5), m.group(6), m.group(7))
        m = re.match(r'\s+strictly conjugate: cover ([\d.]+)\s+LB/true ([\d.]+).*?nu\s+([\d.]+)', line)
        if m:
            b.append(f"| {des} | {pri} | {sub[3]} | {sub[0]} / {sub[1]} ({sub[2]}) | {m.group(1)} / {m.group(2)} ({m.group(3)}) |")
    return '\n'.join(a), '\n'.join(b)


def minus(text):
    """表の負号を活字の負号(U+2212)に。"""
    return re.sub(r'(?<![\w$])-(?=\d)', '\u2212', text)


if __name__ == '__main__':
    import builtins
    _print = builtins.print
    print = lambda *a, **k: _print(*(minus(x) if isinstance(x, str) else x for x in a), **k)
    print('<!-- S1 -->\n' + s1() + '\n')
    print('<!-- S3 -->\n' + s3() + '\n')
    print('<!-- S4 -->\n' + s4() + '\n')
    print('<!-- S5 -->\n' + s5() + '\n')
    t6, lev = s6()
    print('<!-- S6 -->\n' + t6 + f'\n\n<!-- S6 levers (factorial, ASAP): {lev} -->\n')
    a7, b7 = s7()
    print('<!-- S7a -->\n' + a7 + '\n\n<!-- S7b -->\n' + b7 + '\n')
    a, b = s8()
    print('<!-- S8a -->\n' + a + '\n\n<!-- S8b -->\n' + b + '\n')
    print('<!-- S9 -->\n' + s9() + '\n')
    print('<!-- S10 -->\n' + s10() + '\n')
    t10b, ses10b = s10b()
    print(f'<!-- S10b (sigma_e = {ses10b[0]} / {ses10b[1]}) -->\n' + t10b + '\n')
    a13, b13 = s13()
    print('<!-- S13a -->\n' + a13 + '\n\n<!-- S13b -->\n' + b13 + '\n')
    print('<!-- S14 -->\n' + s14() + '\n')
    d15, k15, i15, t15 = s15()
    print('<!-- S16 -->\n' + s16() + '\n')
    a17, b17 = s17()
    print('<!-- S17a -->\n' + a17 + '\n\n<!-- S17b -->\n' + b17 + '\n')
    print('<!-- S15a -->\n' + d15 + '\n\n<!-- S15b -->\n' + k15 + '\n\n<!-- S15c -->\n' + i15 + '\n\n<!-- S15d -->\n' + t15 + '\n')
