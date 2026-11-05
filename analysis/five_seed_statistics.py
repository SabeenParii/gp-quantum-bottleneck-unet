# Five-seed extension: prespecified analysis (protocols/PROTOCOL_five_seed_extension.txt, section C) of the notebook-31 per-image tables.
# Crossed hierarchical bootstrap over seeds and clusters (patients; DMID: source images), B = 10,000, rng(0); Holm within
# families; equivalence when the 95% CI lies within +-0.02 Dice. Plus seed sensitivity and the additional
# independent-seed resampling analysis for contrasts whose models do not share the decoder initialization (id, tr).
# Usage: python analysis/five_seed_statistics.py results/five_seed/per_image results/five_seed/stats
# (CSAW-S per-image scores are not in the public release; with them absent, the CSAW-S rows are skipped.)
import sys, os, glob, numpy as np, pandas as pd
IN, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
B = 10_000; EQ = 0.02
BLOCK = [('g_q', 'g_cl'), ('g_q', 'g_id'), ('g_cl', 'g_id'), ('p_q', 'p_cl'), ('p_q', 'p_id'), ('p_cl', 'p_id')]
ENC = [('p_q', 'g_q'), ('p_cl', 'g_cl'), ('p_id', 'g_id')]
NINE = BLOCK + ENC
TRIG = [('g_q', 'g_tr'), ('p_q', 'p_tr')]
REF = [('full', 'g_id'), ('full', 'p_id'), ('unet', 'full'), ('unet', 'g_q'), ('unet', 'p_q'), ('full', 'g_q'), ('full', 'p_q')]
DECODER_DIFFERS = {'id', 'tr'}          # init audit: id and tr controls start from a different decoder initialization


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def matrix(df, col='dice'):
    """{model: (seeds, items)} with items aligned; cluster codes. One value per model x seed x crop (mean over fold models)."""
    g = df.groupby(['model', 'seed', 'roi_id', 'cluster'], as_index=False)[col].mean()
    items = g[['roi_id', 'cluster']].drop_duplicates('roi_id').sort_values('roi_id').reset_index(drop=True)
    pos = pd.Series(np.arange(len(items)), index=items.roi_id); codes = pd.factorize(items.cluster)[0]
    M = {}
    for (m, s), t in g.groupby(['model', 'seed']):
        a = np.full(len(items), np.nan); a[pos[t.roi_id].to_numpy()] = t[col].to_numpy(); M.setdefault(m, {})[s] = a
    return {m: (sorted(v), np.vstack([v[s] for s in sorted(v)])) for m, v in M.items()}, codes


def boot(a, b, codes, rng, joint=True, seeds_a=None, seeds_b=None):
    ok = ~(np.isnan(a).any(0) | np.isnan(b).any(0)); a, b, c = a[:, ok], b[:, ok], codes[ok]
    c = pd.factorize(c)[0]; n = c.max() + 1; cnt = np.bincount(c, minlength=n)
    A = np.vstack([np.bincount(c, r, n) for r in a]); Bm = np.vstack([np.bincount(c, r, n) for r in b])
    est = a.mean() - b.mean(); out = np.empty(B)
    for t in range(B):
        w = np.bincount(rng.integers(0, n, n), minlength=n); den = w @ cnt
        sa = rng.integers(0, len(A), len(A))
        sb = sa if (joint and len(A) == len(Bm)) else rng.integers(0, len(Bm), len(Bm))
        out[t] = (w @ A[sa].mean(0) - w @ Bm[sb].mean(0)) / den
    p = max(min(1.0, 2 * min((out <= 0).mean(), (out >= 0).mean())), 1 / B)
    lo, hi = np.percentile(out, [2.5, 97.5]); l9, h9 = np.percentile(out, [5, 95])
    return dict(diff=est, ci_lo=lo, ci_hi=hi, ci90_lo=l9, ci90_hi=h9, p=p, n_items=int(ok.sum()), n_clusters=int(n),
                n_seeds=len(A), boot=out)


def verdict(r):
    if r['ci_lo'] >= -EQ and r['ci_hi'] <= EQ: v = 'equivalent'
    else: v = 'inconclusive'
    if r['ci_lo'] > 0 or r['ci_hi'] < 0: v = ('difference, ' + v) if v == 'equivalent' else 'difference'
    return v


def family(M, codes, pairs, rng, label, seeds=None, joint=True):
    rows = []
    for a, b in pairs:
        if a not in M or b not in M: continue
        sa, Ma = M[a]; sb, Mb = M[b]
        common = sorted(set(sa) & set(sb)) if seeds is None else [s for s in seeds if s in sa and s in sb]
        if joint:
            Ma = Ma[[sa.index(s) for s in common]]; Mb = Mb[[sb.index(s) for s in common]]
        r = boot(Ma, Mb, codes, rng, joint=joint); r.pop('boot')
        rows.append(dict(family=label, contrast=f'{a} - {b}', A=a, B=b, **r))
    if rows:
        adj = holm([r['p'] for r in rows])
        for r, q in zip(rows, adj): r['p_holm'] = q; r['verdict'] = verdict(r)
    return rows


def load(name):
    f = f'{IN}/rev5_{name}_per_image.csv.gz'; parts = sorted(glob.glob(f'{IN}/rev5_{name}_per_image.part*.csv.gz'))
    if os.path.exists(f): d = pd.read_csv(f)
    elif parts: d = pd.concat([pd.read_csv(x) for x in parts], ignore_index=True)
    else: print('missing', f); return None
    d['cluster'] = d['patient'].astype(str) if 'patient' in d and d['patient'].notna().all() else d.roi_id.str.split('_').str[0]
    return d


rng = np.random.default_rng(0)
ALL = []
SETS = {'cbis': 'CBIS-DDSM test folds', 'frozen': 'Frozen CBIS-DDSM cohort', 'dmid': 'DMID', 'csaws': 'CSAW-S', 'cddcesm': 'CDD-CESM'}
LEVELS = []
for key, title in SETS.items():
    d = load(key)
    if d is None: continue
    refs = [('dice', 'consensus')] + ([(c, c.replace('dice_', '')) for c in d.columns if c.startswith('dice_annotator')] if key == 'csaws' else [])
    for col, refname in refs:
        for sch in ('train', 'val'):
            ds = d[d.schedule == sch]
            M, codes = matrix(ds, col)
            tag = f'{key}|{refname}|{sch}'
            ALL += [dict(set=key, reference=refname, schedule=sch, **r) for r in family(M, codes, NINE, rng, 'nine')]
            ALL += [dict(set=key, reference=refname, schedule=sch, **r) for r in family(M, codes, TRIG, rng, 'circuit-trig')]
            if sch == 'train':
                ALL += [dict(set=key, reference=refname, schedule=sch, **r) for r in family(M, codes, REF, rng, 'reference')]
            if refname != 'consensus': continue
            # model levels with CI (seeds x clusters)
            for m, (ss, X) in M.items():
                zero = np.zeros_like(X); r = boot(X, zero, codes, rng); r.pop('boot')
                lv = dict(set=key, schedule=sch, model=m, dice=r['diff'], ci_lo=r['ci_lo'], ci_hi=r['ci_hi'], n_seeds=len(ss))
                sub = ds[ds.model == m]
                if 'empty' in sub: lv['empty_rate'] = float(sub.groupby(['seed', 'roi_id']).empty.mean().mean())
                if 'hd95' in sub: lv['hd95_median'] = float(sub.groupby(['seed', 'roi_id']).hd95.mean().median())
                LEVELS.append(lv)
            if key == 'cbis':
                # seed sensitivity: seeds 0-2 only, 3-4 only, block difference, leave one seed out
                for fam, pairs in (('nine', NINE), ('circuit-trig', TRIG)):
                    for blk, ss in (('seeds0-2', [0, 1, 2]), ('seeds3-4', [3, 4])):
                        ALL += [dict(set=key, reference=refname, schedule=sch, analysis=blk, **r) for r in family(M, codes, pairs, rng, fam, seeds=ss)]
                    for drop in range(5):
                        ss = [s for s in range(5) if s != drop]
                        ALL += [dict(set=key, reference=refname, schedule=sch, analysis=f'leave-out-seed{drop}', **r)
                                for r in family(M, codes, pairs, rng, fam, seeds=ss)]
                # block difference of each contrast (seeds 0-2 minus 3-4), clusters resampled jointly
                for a, b in NINE + TRIG:
                    sa, Ma = M[a]; sb, Mb = M[b]; D = Ma - Mb
                    x, y = D[:3], D[3:]
                    c = pd.factorize(codes)[0]; n = c.max() + 1; cnt = np.bincount(c, minlength=n)
                    X = np.vstack([np.bincount(c, r, n) for r in x]); Y = np.vstack([np.bincount(c, r, n) for r in y]); out = np.empty(B)
                    for t in range(B):
                        w = np.bincount(rng.integers(0, n, n), minlength=n)
                        out[t] = (w @ X[rng.integers(0, 3, 3)].mean(0) - w @ Y[rng.integers(0, 2, 2)].mean(0)) / (w @ cnt)
                    ALL.append(dict(set=key, reference=refname, schedule=sch, analysis='block 0-2 minus 3-4', family='seed-block',
                                    contrast=f'{a} - {b}', diff=x.mean() - y.mean(), ci_lo=np.percentile(out, 2.5), ci_hi=np.percentile(out, 97.5)))
                # additional analysis: independent seed resampling where the decoder initialization differs
                pairs = [(a, b) for a, b in NINE + TRIG if a.split('_')[1] in DECODER_DIFFERS or b.split('_')[1] in DECODER_DIFFERS]
                ALL += [dict(set=key, reference=refname, schedule=sch, analysis='independent seeds (additional)', **r)
                        for r in family(M, codes, pairs, rng, 'indep', joint=False)]
            print('done', tag, flush=True)

R = pd.DataFrame(ALL); R['analysis'] = R.get('analysis', pd.Series(dtype=str)).fillna('main')
# protocol: Holm within the four circuit - trigonometric contrasts (both schedules) of each test set and reference
_m = (R.family == 'circuit-trig') & (R.analysis == 'main')
for _, _g in R[_m].groupby(['set', 'reference']):
    R.loc[_g.index, 'p_holm'] = holm(_g.p.values)
R.to_csv(f'{OUT}/rev5_contrasts.csv', index=False, float_format='%.6g')
L = pd.DataFrame(LEVELS); L.to_csv(f'{OUT}/rev5_levels.csv', index=False, float_format='%.6g')
pd.set_option('display.width', 250)
show = ['set', 'reference', 'schedule', 'family', 'contrast', 'diff', 'ci_lo', 'ci_hi', 'p_holm', 'verdict']
print(R[(R.analysis == 'main') & (R.reference == 'consensus')][show].round(4).to_string(index=False))
print(L.round(4).to_string(index=False))
