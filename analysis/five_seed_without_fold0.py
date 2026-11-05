# Five-seed contrasts without fold 0 (the development-informed fold): CBIS-DDSM test folds 1-4 only; DMID with the
# fold-0 models removed (mean of the four remaining fold models per crop). Same crossed bootstrap, B = 10,000, rng(0).
import sys, numpy as np, pandas as pd
IN, OUT = sys.argv[1], sys.argv[2]
B = 10_000; EQ = 0.02
NINE = [('g_q', 'g_cl'), ('g_q', 'g_id'), ('g_cl', 'g_id'), ('p_q', 'p_cl'), ('p_q', 'p_id'), ('p_cl', 'p_id'), ('p_q', 'g_q'), ('p_cl', 'g_cl'), ('p_id', 'g_id')]
TRIG = [('g_q', 'g_tr'), ('p_q', 'p_tr')]
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj
def matrix(df):
    g = df.groupby(['model', 'seed', 'roi_id', 'cluster'], as_index=False)['dice'].mean()
    items = g[['roi_id', 'cluster']].drop_duplicates('roi_id').sort_values('roi_id').reset_index(drop=True)
    pos = pd.Series(np.arange(len(items)), index=items.roi_id); codes = pd.factorize(items.cluster)[0]; M = {}
    for (m, s), t in g.groupby(['model', 'seed']):
        a = np.full(len(items), np.nan); a[pos[t.roi_id].to_numpy()] = t['dice'].to_numpy(); M.setdefault(m, {})[s] = a
    return {m: np.vstack([v[s] for s in sorted(v)]) for m, v in M.items()}, codes
def boot(a, b, codes, rng):
    c = pd.factorize(codes)[0]; n = c.max() + 1; cnt = np.bincount(c, minlength=n)
    A = np.vstack([np.bincount(c, r, n) for r in a]); Bm = np.vstack([np.bincount(c, r, n) for r in b]); out = np.empty(B)
    for t in range(B):
        w = np.bincount(rng.integers(0, n, n), minlength=n); s = rng.integers(0, len(A), len(A))
        out[t] = (w @ A[s].mean(0) - w @ Bm[s].mean(0)) / (w @ cnt)
    p = max(min(1.0, 2 * min((out <= 0).mean(), (out >= 0).mean())), 1 / B)
    return dict(diff=a.mean() - b.mean(), ci_lo=np.percentile(out, 2.5), ci_hi=np.percentile(out, 97.5), p=p, n_items=a.shape[1], n_clusters=n)
rng = np.random.default_rng(0); rows = []
for key in ('cbis', 'dmid'):
    d = pd.read_csv(f'{IN}/rev5_{key}_per_image.csv.gz'); d['cluster'] = d['patient'].astype(str)
    d = d[d.fold != 0]
    for sch in ('train', 'val'):
        M, codes = matrix(d[d.schedule == sch])
        for fam, pairs in (('nine', NINE), ('circuit-trig', TRIG)):
            rr = [dict(set=key, schedule=sch, family=fam, contrast=f'{a} - {b}', **boot(M[a], M[b], codes, rng)) for a, b in pairs]
            if fam == 'nine':
                for r, q in zip(rr, holm([r['p'] for r in rr])): r['p_holm'] = q
            rows += rr
R = pd.DataFrame(rows)
for _, g in R[R.family == 'circuit-trig'].groupby('set'): R.loc[g.index, 'p_holm'] = holm(g.p.values)
R.to_csv(f'{OUT}/rev5_contrasts_without_fold0.csv', index=False, float_format='%.6g'); print(R.round(4).to_string())
