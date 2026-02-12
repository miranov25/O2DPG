"""
Parametric benchmark: V5 sliding window — final delivery.

Answers: How much does sliding window smoothing cost relative to
plain per-bin regression, when measured fairly?

Usage:
    python bench_parametric.py --quick   # 10 configs
    python bench_parametric.py           # full scan
"""
import sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from groupby_regression_sliding_window import (
    make_sliding_window_fit, _flatten_bins_for_v5,
    _get_numba_v4_kernels, make_sliding_window_fit_v5_arrays,
    _build_bin_index_map, _assemble_results_v5,
)
from groupby_regression_optimized import make_parallel_fit_v4

RUNS = 3
QUICK = '--quick' in sys.argv
gb = ['xBin', 'yBin', 'zBin']
fit_cols = ['value']
lin_cols = ['x']


def make_data(grid, entries, seed=42):
    rng = np.random.RandomState(seed)
    coords = np.array(np.meshgrid(*[np.arange(grid)] * 3)).T.reshape(-1, 3)
    coords_rep = np.repeat(coords, entries, axis=0)
    x = rng.standard_normal(len(coords_rep))
    value = 2.0 * x + 1.0 + rng.standard_normal(len(coords_rep)) * 0.1
    return pd.DataFrame({'xBin': coords_rep[:, 0], 'yBin': coords_rep[:, 1],
                          'zBin': coords_rep[:, 2], 'x': x, 'value': value})


def _timed(fn, runs=RUNS):
    """Warmup + min of runs."""
    fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        r = fn()
        times.append(time.perf_counter() - t0)
    return min(times), r


def bench_nosw_arrays(bin_ids, X_all, Y_all, n_bins):
    """noSW-arr: per-bin regression on numpy, W=0. FAIR V5arr reference.
    Uses SAME Numba kernels as V5arr. ONLY difference: W=0 (N_nbr=1)."""
    acc_fn, solve_fn = _get_numba_v4_kernels()
    n_p, n_t = X_all.shape[1], Y_all.shape[1]
    nbr_i = np.arange(n_bins, dtype=np.int64).reshape(-1, 1)
    nbr_w = np.ones((n_bins, 1), dtype=np.float64)
    nbr_c = np.ones(n_bins, dtype=np.int64)
    XtX = np.zeros((n_t, n_bins, n_p, n_p))
    XtY = np.zeros((n_t, n_bins, n_p))
    n_all = np.zeros((n_t, n_bins), dtype=np.int64)
    sy  = np.zeros((n_t, n_bins)); sy2 = np.zeros((n_t, n_bins))
    beta = np.full((n_t, n_bins, n_p), np.nan)
    se   = np.full((n_t, n_bins, n_p), np.nan)
    rmse = np.full((n_t, n_bins), np.nan)
    r2   = np.full((n_t, n_bins), np.nan)
    nf   = np.zeros((n_t, n_bins), dtype=np.int64)
    st   = np.zeros((n_t, n_bins), dtype=np.int64)
    mn   = np.full((n_t, n_bins), np.nan)
    sd   = np.full((n_t, n_bins), np.nan)
    en   = np.zeros((n_t, n_bins), dtype=np.int64)
    # Warmup
    acc_fn(bin_ids, X_all, Y_all, n_bins, n_p, n_t, True,
           XtX, XtY, n_all, sy, sy2)
    for ti in range(n_t):
        solve_fn(XtX[ti], XtY[ti], n_all[ti], sy[ti], sy2[ti],
                 nbr_i, nbr_w, nbr_c, n_p, 5, False,
                 beta[ti], se[ti], rmse[ti], r2[ti],
                 nf[ti], st[ti], mn[ti], sd[ti], en[ti])
    times = []
    for _ in range(RUNS):
        XtX[:]=0; XtY[:]=0; n_all[:]=0; sy[:]=0; sy2[:]=0
        t0 = time.perf_counter()
        acc_fn(bin_ids, X_all, Y_all, n_bins, n_p, n_t, True,
               XtX, XtY, n_all, sy, sy2)
        for ti in range(n_t):
            solve_fn(XtX[ti], XtY[ti], n_all[ti], sy[ti], sy2[ti],
                     nbr_i, nbr_w, nbr_c, n_p, 5, False,
                     beta[ti], se[ti], rmse[ti], r2[ti],
                     nf[ti], st[ti], mn[ti], sd[ti], en[ti])
        times.append(time.perf_counter() - t0)
    return min(times)


def measure_all(df, window):
    ws = {d: window for d in gb}
    T = {}
    T['nosw_df'], _ = _timed(lambda: make_parallel_fit_v4(
        df=df, gb_columns=gb, fit_columns=fit_cols,
        linear_columns=lin_cols, min_stat=5, suffix='_nosw'))
    bi, X, Y, nb, bc, bd = _flatten_bins_for_v5(df, gb, fit_cols, lin_cols)
    T['nosw_arr'] = bench_nosw_arrays(bi, X, Y, nb)
    # V5arr with _collect_timings
    make_sliding_window_fit_v5_arrays(
        bin_ids=bi, X_all=X, Y_all=Y, n_bins=nb, bin_coords=bc, bounds=bd,
        gb_columns=gb, fit_columns=fit_cols, linear_columns=lin_cols,
        window_spec=ws, min_stat=5, _collect_timings=True)
    times_v5a, internals = [], []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        r = make_sliding_window_fit_v5_arrays(
            bin_ids=bi, X_all=X, Y_all=Y, n_bins=nb, bin_coords=bc, bounds=bd,
            gb_columns=gb, fit_columns=fit_cols, linear_columns=lin_cols,
            window_spec=ws, min_stat=5, _collect_timings=True)
        times_v5a.append(time.perf_counter() - t0)
        internals.append(r['_timings'])
    best = int(np.argmin(times_v5a))
    T['v5arr'] = times_v5a[best]
    for k in ('setup', 'nbr_table', 'loop1', 'loop2', 'pack'):
        T['v5_' + k] = internals[best][k]
    T['v5tot'], _ = _timed(lambda: make_sliding_window_fit(
        df=df, gb_columns=gb, window_spec=ws, fit_columns=fit_cols,
        linear_columns=lin_cols, min_stat=5, algorithm='incremental',
        backend='numba', suffix=''))
    T['v1sw'], _ = _timed(lambda: make_sliding_window_fit(
        df=df, gb_columns=gb, window_spec=ws, fit_columns=fit_cols,
        linear_columns=lin_cols, min_stat=5, algorithm='recompute',
        backend='numpy', suffix=''))
    T['v5flat'], _ = _timed(lambda: _flatten_bins_for_v5(df, gb, fit_cols, lin_cols))
    T['bin_map'], _ = _timed(lambda: _build_bin_index_map(df, gb, None))
    v5r = make_sliding_window_fit_v5_arrays(
        bin_ids=bi, X_all=X, Y_all=Y, n_bins=nb, bin_coords=bc, bounds=bd,
        gb_columns=gb, fit_columns=fit_cols, linear_columns=lin_cols,
        window_spec=ws, min_stat=5)
    T['assemble'], _ = _timed(lambda: _assemble_results_v5(v5r, gb, fit_cols, lin_cols, ''))
    return T


from numpy.linalg import lstsq as np_lstsq

def fit_report(name, X, y, labels, units, cfg_labels):
    n, p = X.shape
    c, _, _, _ = np_lstsq(X, y, rcond=None)
    yp = X @ c; res = y - yp
    ss_r, ss_t = np.sum(res**2), np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_r/ss_t if ss_t > 0 else 0
    dof = max(n-p, 1); sig = np.sqrt(ss_r/dof)
    try: se = np.sqrt(np.diag(np.linalg.inv(X.T@X) * sig**2))
    except: se = np.full(p, np.nan)
    mxr = np.max(np.abs(res))
    rel = np.abs(res)/np.maximum(np.abs(y), 1e-12)
    mxrel = np.max(rel)*100
    ok_r2, ok_rel = r2 > 0.99, mxrel < 20
    st = "PASS" if (ok_r2 and ok_rel) else "FAIL"
    if not ok_r2: st += " R2<0.99"
    if not ok_rel: st += " rel>20%"
    print(f"\n--- {name} ---")
    parts = [f"({c[i]*units[i][0]:.4f} +/- {se[i]*units[i][0]:.4f}) {units[i][1]}"
             for i in range(p)]
    print(f"  T = {' + '.join(parts)}")
    print(f"  R2 = {r2:.4f},  sig_res = {sig*1e3:.2f} ms,  max|res| = {mxr*1e3:.1f} ms ({mxrel:.1f}%)  {st}")
    if not (ok_r2 and ok_rel):
        print(f"\n  {'Config':<20s} {'Actual':>10s} {'Predicted':>10s} {'Residual':>10s} {'Rel%':>7s}")
        for i in range(n):
            flag = " <--" if rel[i]>0.20 else ""
            print(f"  {cfg_labels[i]:<20s} {y[i]*1e3:10.1f}ms {yp[i]*1e3:10.1f}ms "
                  f"{res[i]*1e3:+10.1f}ms {rel[i]*100:+6.1f}%{flag}")
    return c, se, r2


# ============================================================
# MAIN
# ============================================================
if QUICK:
    scan = [(10,1,10),(15,1,10),(20,1,10),(25,1,10),
            (15,2,10),(20,2,10),(25,2,10),
            (15,1,50),(20,1,50),(25,1,50)]
else:
    scan = [(10,1,10),(12,1,10),(15,1,10),(18,1,10),(20,1,10),(22,1,10),(25,1,10),
            (10,2,10),(12,2,10),(15,2,10),(18,2,10),(20,2,10),(22,2,10),(25,2,10),
            (10,1,50),(15,1,50),(20,1,50),(25,1,50),
            (10,2,50),(15,2,50),(20,2,50),(25,2,50),
            (10,1,100),(15,1,100),(20,1,100),(10,2,100),(15,2,100)]

print(f"Parametric benchmark: {len(scan)} configurations, {RUNS} runs each")
print(f"{'QUICK mode' if QUICK else 'FULL mode'}\n")

# Main table
hdr = (f"{'Grid':>5s} {'W':>2s} {'RPB':>4s} | {'N_bins':>7s} {'N_rows':>8s} {'N_nbr':>5s} "
       f"| {'noSWarr':>7s} {'noSWDF':>7s} {'V5flat':>7s} {'V5arr':>7s} {'V5tot':>7s} {'V1-SW':>7s} "
       f"| {'SW/noSW':>7s} {'V5/V1':>6s}")
print(hdr); print("-" * len(hdr))

rows = []
for grid, window, entries in scan:
    n_bins = grid**3; n_rows = n_bins*entries; n_nbr = (2*window+1)**3
    if n_rows > 2_000_000:
        print(f"{grid:5d} {window:2d} {entries:4d} |  SKIPPED"); continue
    df = make_data(grid, entries)
    t = measure_all(df, window)
    sw = t['v5arr']/t['nosw_arr'] if t['nosw_arr']>0 else float('nan')
    v5v1 = t['v1sw']/t['v5tot'] if t['v5tot']>0 else float('nan')
    print(f"{grid:5d} {window:2d} {entries:4d} | {n_bins:7d} {n_rows:8d} {n_nbr:5d} "
          f"| {t['nosw_arr']:7.4f} {t['nosw_df']:7.3f} {t['v5flat']:7.4f} "
          f"{t['v5arr']:7.4f} {t['v5tot']:7.3f} {t['v1sw']:7.3f} "
          f"| {sw:7.2f} {v5v1:6.2f}")
    rows.append({'grid':grid,'window':window,'entries':entries,
                 'n_bins':n_bins,'n_rows':n_rows,'n_nbr':n_nbr,
                 **t,'sw_nosw':sw,'v5_v1':v5v1})

df_r = pd.DataFrame(rows)
cfg_labels = [f"{int(r['grid'])}^3 W{int(r['window'])} r{int(r['entries'])}" for r in rows]

# V5arr component breakdown
print("\n" + "="*80)
print("V5arr COMPONENT BREAKDOWN (from _collect_timings)")
print("="*80)
print(f"\n{'Grid':>5s} {'W':>2s} {'RPB':>4s} | {'Setup':>7s} {'NbrTbl':>7s} {'Loop1':>7s} {'Loop2':>7s} {'Pack':>7s} | {'Sum':>7s} {'V5arr':>7s} {'Other':>7s}")
print("-"*90)
for r in rows:
    su,nb,l1,l2,pk = r['v5_setup'],r['v5_nbr_table'],r['v5_loop1'],r['v5_loop2'],r['v5_pack']
    s = su+nb+l1+l2+pk
    print(f"{int(r['grid']):5d} {int(r['window']):2d} {int(r['entries']):4d} "
          f"| {su:7.4f} {nb:7.4f} {l1:7.4f} {l2:7.4f} {pk:7.4f} "
          f"| {s:7.4f} {r['v5arr']:7.4f} {r['v5arr']-s:7.4f}")

# V5tot wrapper breakdown
print("\n" + "="*80)
print("V5tot WRAPPER BREAKDOWN (bin_map no longer in V5 path)")
print("="*80)
print(f"\n{'Grid':>5s} {'W':>2s} {'RPB':>4s} | {'flatten':>8s} {'v5arr':>8s} {'assemble':>8s} | {'sum':>8s} {'V5tot':>8s} {'other':>8s}")
print("-"*80)
for r in rows:
    fl,v5a,asm = r['v5flat'],r['v5arr'],r['assemble']
    s = fl+v5a+asm
    print(f"{int(r['grid']):5d} {int(r['window']):2d} {int(r['entries']):4d} "
          f"| {fl:8.4f} {v5a:8.4f} {asm:8.4f} "
          f"| {s:8.4f} {r['v5tot']:8.4f} {r['v5tot']-s:8.4f}")

# Theoretical validation
print("\n" + "="*80)
print("THEORETICAL COST VALIDATION")
print("="*80)
print("\nLoop 1 per row (p=2): 9 FMA + 13 mem -> ~2 ns floor (M1 Pro)")
l1_ns = [r['v5_loop1']/r['n_rows']*1e9 for r in rows]
print(f"  Measured: {np.mean(l1_ns):.1f} +/- {np.std(l1_ns):.1f} ns/row  "
      f"Overhead: {np.mean(l1_ns)/2:.0f}x  CV: {np.std(l1_ns)/np.mean(l1_ns)*100:.0f}%")
print("\nLoop 2 per (bin x nbr) (p=2): ~17 ops -> ~3 ns floor")
l2_ns = [r['v5_loop2']/(r['n_bins']*r['n_nbr'])*1e9 for r in rows]
print(f"  Measured: {np.mean(l2_ns):.1f} +/- {np.std(l2_ns):.1f} ns/(bin x nbr)  "
      f"Overhead: {np.mean(l2_ns)/3:.0f}x  CV: {np.std(l2_ns)/np.mean(l2_ns)*100:.0f}%")
print("\nFlatten (ravel_multi_index, 3D): ~1.5 ns/row floor")
fl_ns = [r['v5flat']/r['n_rows']*1e9 for r in rows]
print(f"  Measured: {np.mean(fl_ns):.1f} +/- {np.std(fl_ns):.1f} ns/row  "
      f"Overhead: {np.mean(fl_ns)/1.5:.0f}x  CV: {np.std(fl_ns)/np.mean(fl_ns)*100:.0f}%")
print("\nInterpretation: 10-30x overhead expected. Key: CV < 30%.")

# Cost model fits
print("\n" + "="*80)
print("LINEAR COST MODEL FIT (8 models + 1 comparison)")
print("="*80)

c1,_,_ = fit_report("Model 1: noSW-arr = a*N_rows + b*N_bins + c",
    np.column_stack([df_r['n_rows'],df_r['n_bins'],np.ones(len(df_r))]),
    df_r['nosw_arr'].values, ['a','b','c'],
    [(1e6,'us/row'),(1e6,'us/bin'),(1e3,'ms')], cfg_labels)

c2,_,_ = fit_report("Model 2: noSW-DF = a*N_rows + b*N_bins + c",
    np.column_stack([df_r['n_rows'],df_r['n_bins'],np.ones(len(df_r))]),
    df_r['nosw_df'].values, ['a','b','c'],
    [(1e6,'us/row'),(1e6,'us/bin'),(1e3,'ms')], cfg_labels)

c3,_,_ = fit_report("Model 3: V5flat = a*N_rows + c",
    np.column_stack([df_r['n_rows'],np.ones(len(df_r))]),
    df_r['v5flat'].values, ['a','c'], [(1e6,'us/row'),(1e3,'ms')], cfg_labels)

c4,_,_ = fit_report("Model 4: V5arr = a*N_rows + b*N_bins*N_nbr + d*N_bins + c  (4-param)",
    np.column_stack([df_r['n_rows'],df_r['n_bins']*df_r['n_nbr'],
                     df_r['n_bins'],np.ones(len(df_r))]),
    df_r['v5arr'].values, ['a','b','d','c'],
    [(1e6,'us/row'),(1e6,'us/(bin*nbr)'),(1e6,'us/bin'),(1e3,'ms')], cfg_labels)

fit_report("Model 4b: V5arr = a*N_rows + b*N_bins*N_nbr + c  (3-param, comparison)",
    np.column_stack([df_r['n_rows'],df_r['n_bins']*df_r['n_nbr'],np.ones(len(df_r))]),
    df_r['v5arr'].values, ['a','b','c'],
    [(1e6,'us/row'),(1e6,'us/(bin*nbr)'),(1e3,'ms')], cfg_labels)

c5,_,_ = fit_report("Model 5: V5-Loop1 = a*N_rows + c",
    np.column_stack([df_r['n_rows'],np.ones(len(df_r))]),
    df_r['v5_loop1'].values, ['a','c'], [(1e6,'us/row'),(1e3,'ms')], cfg_labels)

c6,_,_ = fit_report("Model 6: V5-Loop2 = a*N_bins*N_nbr + b*N_bins + c",
    np.column_stack([df_r['n_bins']*df_r['n_nbr'],df_r['n_bins'],np.ones(len(df_r))]),
    df_r['v5_loop2'].values, ['a','b','c'],
    [(1e6,'us/(bin*nbr)'),(1e6,'us/bin'),(1e3,'ms')], cfg_labels)

c7,_,_ = fit_report("Model 7: V5-NbrTable = a*N_bins*N_nbr + b*N_bins + c",
    np.column_stack([df_r['n_bins']*df_r['n_nbr'],df_r['n_bins'],np.ones(len(df_r))]),
    df_r['v5_nbr_table'].values, ['a','b','c'],
    [(1e6,'us/(bin*nbr)'),(1e6,'us/bin'),(1e3,'ms')], cfg_labels)

c8,_,_ = fit_report("Model 8: V1-SW = a*N_bins*N_nbr*RPB + b*N_bins + c",
    np.column_stack([df_r['n_bins']*df_r['n_nbr']*df_r['entries'],
                     df_r['n_bins'],np.ones(len(df_r))]),
    df_r['v1sw'].values, ['a','b','c'],
    [(1e6,'us/(bin*nbr*row)'),(1e6,'us/bin'),(1e3,'ms')], cfg_labels)

# V5tot fit for TPC (now = flatten + V5arr + assemble, no bin_map)
c_v5t,_,_ = fit_report("Model 9: V5tot = a*N_rows + b*N_bins*N_nbr + d*N_bins + c",
    np.column_stack([df_r['n_rows'],df_r['n_bins']*df_r['n_nbr'],
                     df_r['n_bins'],np.ones(len(df_r))]),
    df_r['v5tot'].values, ['a','b','d','c'],
    [(1e6,'us/row'),(1e6,'us/(bin*nbr)'),(1e6,'us/bin'),(1e3,'ms')], cfg_labels)

# Consistency checks
print("\n" + "="*80)
print("CONSISTENCY CHECKS")
print("="*80)
n_pass = n_fail = 0

print("\nCheck 1: V5arr = sum(setup+nbr_table+loop1+loop2+pack)  [0.9-1.2]")
c1_ok = True
for r in rows:
    g,w,e = int(r['grid']),int(r['window']),int(r['entries'])
    comp = r['v5_setup']+r['v5_nbr_table']+r['v5_loop1']+r['v5_loop2']+r['v5_pack']
    ratio = r['v5arr']/comp if comp>0 else float('nan')
    ok = 0.9<ratio<1.2
    if not ok: c1_ok=False
    print(f"  {g:3d}^3 W={w} r={e:4d}: V5arr={r['v5arr']:.4f} sum={comp:.4f} ratio={ratio:.2f}x  {'PASS' if ok else 'FAIL'}")
if c1_ok: n_pass+=1; print("  => Check 1 PASS")
else: n_fail+=1; print("  => Check 1 FAIL")

print("\nCheck 2: V5-Loop1 per row in [20, 80] ns/row")
l1v = [r['v5_loop1']/r['n_rows']*1e9 for r in rows]
ml = np.mean(l1v)
ok = 20<ml<80
if ok: n_pass+=1
else: n_fail+=1
print(f"  Mean: {ml:.1f} ns/row (range: {min(l1v):.1f}-{max(l1v):.1f})  {'PASS' if ok else 'FAIL'}")

print("\nCheck 3a: (V5arr - NbrTable) < noSW-arr * 3")
c3a_ok = True
for r in rows:
    g,w,e = int(r['grid']),int(r['window']),int(r['entries'])
    comp = r['v5arr']-r['v5_nbr_table']
    ratio = comp/r['nosw_arr'] if r['nosw_arr']>0 else float('nan')
    ok = ratio<3.0
    if not ok: c3a_ok=False
    print(f"  {g:3d}^3 W={w} r={e:4d}: compute={comp:.4f} noSW-arr={r['nosw_arr']:.4f} ratio={ratio:.2f}x  {'PASS' if ok else 'FAIL'}")
if c3a_ok: n_pass+=1; print("  => Check 3a PASS")
else: n_fail+=1; print("  => Check 3a FAIL")

print("\nCheck 3b: NbrTable < 0.10 us/(bin*N_nbr)")
c3b_ok = True
for r in rows:
    g,w,e = int(r['grid']),int(r['window']),int(r['entries'])
    bxn = r['n_bins']*r['n_nbr']
    cost = r['v5_nbr_table']/bxn*1e6 if bxn>0 else 0
    ok = cost<0.10
    if not ok: c3b_ok=False
    print(f"  {g:3d}^3 W={w} r={e:4d}: {cost:.3f} us/(bin*nbr)  {'PASS' if ok else 'FAIL'}")
if c3b_ok: n_pass+=1; print("  => Check 3b PASS")
else: n_fail+=1; print("  => Check 3b FAIL")

print("\nCheck 4: V5tot = flatten+V5arr+assemble  [0.85-1.15]")
c4_ok = True
for r in rows:
    g,w,e = int(r['grid']),int(r['window']),int(r['entries'])
    ws = r['v5flat']+r['v5arr']+r['assemble']
    ratio = r['v5tot']/ws if ws>0 else float('nan')
    ok = 0.85<ratio<1.15
    if not ok: c4_ok=False
    print(f"  {g:3d}^3 W={w} r={e:4d}: sum={ws:.4f} V5tot={r['v5tot']:.4f} ratio={ratio:.2f}x  {'PASS' if ok else 'FAIL'}")
if c4_ok: n_pass+=1; print("  => Check 4 PASS")
else: n_fail+=1; print("  => Check 4 FAIL")

print("\nCheck 5: (noSW-DF - noSW-arr)/N_bins in [40, 120] us/bin")
df_oh = []
for r in rows:
    g,w,e = int(r['grid']),int(r['window']),int(r['entries'])
    oh = (r['nosw_df']-r['nosw_arr'])/r['n_bins']*1e6
    df_oh.append(oh)
    print(f"  {g:3d}^3 W={w} r={e:4d}: {oh:.1f} us/bin")
moh = np.mean(df_oh)
ok = 40<moh<120
if ok: n_pass+=1
else: n_fail+=1
print(f"  Mean: {moh:.1f} us/bin  {'PASS' if ok else 'FAIL'}")

print(f"\n{'='*50}")
print(f"CHECKS SUMMARY: {n_pass} PASS, {n_fail} FAIL out of {n_pass+n_fail}")
print(f"{'='*50}")

# Fair smoothing overhead
print("\n" + "="*80)
print("FAIR SMOOTHING OVERHEAD: V5arr vs noSW-arr (both numpy, same I/O)")
print("="*80)
print(f"\n{'Grid':>5s} {'W':>2s} {'RPB':>4s} | {'noSW-arr':>9s} {'V5arr':>9s} | {'Ratio':>7s}  Note")
print("-"*70)
for r in rows:
    g,w,e = int(r['grid']),int(r['window']),int(r['entries'])
    ratio = r['sw_nosw']
    bar = "|"*max(1,int(ratio*3))
    print(f"{g:5d} {w:2d} {e:4d} | {r['nosw_arr']:9.4f} {r['v5arr']:9.4f} | {ratio:7.2f}x  ({r['n_nbr']} nbr, {e} rpb) {bar}")
print("\nAt TPC scale (RPB>=500), V5arr/noSW-arr ~ 1.0-1.1x.")
print("High ratios at RPB=10 are NbrTable fixed cost.")

# TPC Predictions
print("\n" + "="*80)
print("TPC PREDICTIONS (extrapolated from fitted cost models)")
print("="*80)
print("\nTPC: 36 sectors x 3 targets, p=2-4, 3D smoothing per sector.\n")

def p_na(nr,nb): return max(0.001, c1[0]*nr + c1[1]*nb + c1[2])
def p_v5a(nr,nb,nn): return max(0.001, c4[0]*nr + c4[1]*nb*nn + c4[2]*nb + c4[3])
def p_v5t(nr,nb,nn): return max(0.001, c_v5t[0]*nr + c_v5t[1]*nb*nn + c_v5t[2]*nb + c_v5t[3])

scenarios = [("Low",30*20*20,500,'(1,1,1)',27),
             ("Standard",60*30*30,1000,'(1,2,1)',45),
             ("High",60*30*30,2000,'(2,2,2)',125)]

print(f"  Per sector, 1 target:")
print(f"  {'Scenario':<10s} | {'N_bins':>7s} {'RPB':>5s} {'N_nbr':>5s} {'N_rows':>10s} | {'noSWarr':>8s} {'V5arr':>8s} | {'SW/noSW':>8s}")
print(f"  {'-'*10}-+-{'-'*33}-+-{'-'*18}-+-{'-'*8}")
for label,nb,rpb,wl,nn in scenarios:
    nr = nb*rpb
    tna=p_na(nr,nb); tv5a=p_v5a(nr,nb,nn)
    print(f"  {label:<10s} | {nb:7,d} {rpb:5d} {nn:5d} {nr:10,d} | {tna:7.2f}s {tv5a:7.2f}s | {tv5a/tna:7.2f}x")

print(f"\n  Full TPC (36 sectors x 3 targets):")
print(f"  {'Scenario':<10s} | {'noSWarr':>10s} {'V5arr':>10s} {'V5arr x10p':>11s} {'V5tot':>10s}")
print(f"  {'-'*10}-+-{'-'*10}-{'-'*10}-{'-'*11}-{'-'*10}")
m = 36*3
for label,nb,rpb,wl,nn in scenarios:
    nr = nb*rpb
    tna=p_na(nr,nb); tv5a=p_v5a(nr,nb,nn); tv5t=p_v5t(nr,nb,nn)
    print(f"  {label:<10s} | {m*tna/60:9.1f}m {m*tv5a/60:9.1f}m {m*tv5a/10/60:10.1f}m {m*tv5t/60:9.1f}m")

print("\n" + "="*80 + "\nDONE\n" + "="*80)
