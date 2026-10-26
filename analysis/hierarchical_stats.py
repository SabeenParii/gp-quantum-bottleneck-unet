# Randomized-crop experiment: main statistics from the per-image results
# Crossed hierarchical bootstrap over clusters (CBIS patients / DMID images) and seeds, 10,000 replicates, rng(0).
# Usage: python analysis/hierarchical_stats.py [--results FILE] [--out DIR] [--exclude-uncertain]
import numpy as np, pandas as pd, os, sys, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--results', default=os.path.join(ROOT, 'results', 'per_image_results.csv.gz'))
ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'stats'))
ap.add_argument('--exclude-uncertain', action='store_true', help='drop the 24 uncertain DMID records')
args, _ = ap.parse_known_args()
OUT = args.out; os.makedirs(OUT, exist_ok=True)
B = 10000
df = pd.read_csv(args.results)
if args.exclude_uncertain:
    df = df[~df.roi_id.isin(pd.read_csv(os.path.join(ROOT, 'data', 'dmid_uncertain_records.csv')).roi_id)]
df = df[df.keep == True].copy()
df['cluster'] = np.where(df.dataset == 'CBIS', df.roi_id.str.extract(r'(P_\d{5})')[0], df.roi_id.str.split('_').str[0])
SW = [f'dice_swap{k}' for k in range(10)]
df['dice_swapavg'] = df[SW].mean(1)
# The collapsed run (primary schedule, global identity, fold 0, seed 1; CBIS Dice 0.099) is kept here.
# Its sensitivity analysis (removed from BOTH models of every contrast) is in additional_analyses.py.


def unit_table(t, col='dice'):
    """Per image x seed value; DMID: mean over the five fold models."""
    return t.groupby(['roi_id', 'cluster', 'seed'], as_index=False)[col].mean()


def boot_stat(u, rng):
    """u: DataFrame(roi_id, cluster, seed, v). Crossed bootstrap: clusters and seeds resampled independently."""
    seeds = np.sort(u.seed.unique()); cl = np.sort(u.cluster.unique())
    ci = {c: i for i, c in enumerate(cl)}; si = {s: i for i, s in enumerate(seeds)}
    S = np.zeros((len(cl), len(seeds))); N = np.zeros((len(cl), len(seeds)))
    for c, s, v in zip(u.cluster, u.seed, u.v):
        S[ci[c], si[s]] += v; N[ci[c], si[s]] += 1
    est = S.sum() / N.sum()
    wc = rng.multinomial(len(cl), np.ones(len(cl)) / len(cl), size=B)
    ws = rng.multinomial(len(seeds), np.ones(len(seeds)) / len(seeds), size=B) if len(seeds) > 1 else np.ones((B, 1))
    num = np.einsum('bc,cs,bs->b', wc, S, ws); den = np.einsum('bc,cs,bs->b', wc, N, ws)
    bs = num / den
    p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    p = max(p, 1.0 / B)
    return est, np.percentile(bs, 2.5), np.percentile(bs, 97.5), p, bs


def contrast(t, a, b, col='dice', rng=None):
    x = unit_table(t[t.model == a], col).rename(columns={col: 'a'})
    y = unit_table(t[t.model == b], col).rename(columns={col: 'b'})
    m = x.merge(y, on=['roi_id', 'cluster', 'seed'])
    m['v'] = m.a - m.b
    return boot_stat(m[['roi_id', 'cluster', 'seed', 'v']], rng)


def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty_like(p); run = 0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1, run)
    return adj


CONTRASTS = [('p_q', 'g_q'), ('p_cl', 'g_cl'), ('p_id', 'g_id'),                    # placement, per block
             ('g_q', 'g_cl'), ('g_q', 'g_id'), ('g_cl', 'g_id'),                    # block, global
             ('p_q', 'p_cl'), ('p_q', 'p_id'), ('p_cl', 'p_id')]                    # block, patch-wise
rows, summ = [], []
for sched in ('train', 'val'):
    for ds in ('CBIS', 'DMID'):
        t = df[(df.set == sched) & (df.dataset == ds)]
        rng = np.random.default_rng(0)
        res = []
        for a, b in CONTRASTS:
            e, lo, hi, p, bs = contrast(t, a, b, rng=rng)
            res.append(dict(schedule=sched, dataset=ds, contrast=f'{a} - {b}', diff=e, ci_low=lo, ci_high=hi, p=p,
                            bound90=max(abs(np.percentile(bs, 5)), abs(np.percentile(bs, 95)))))
        # interaction: placement effect of circuit vs Dense(4): (p_q - g_q) - (p_cl - g_cl)
        x = unit_table(t[t.model == 'p_q']).merge(unit_table(t[t.model == 'g_q']), on=['roi_id', 'cluster', 'seed'], suffixes=('_pq', '_gq'))
        y = unit_table(t[t.model == 'p_cl']).merge(unit_table(t[t.model == 'g_cl']), on=['roi_id', 'cluster', 'seed'], suffixes=('_pc', '_gc'))
        m = x.merge(y, on=['roi_id', 'cluster', 'seed']); m['v'] = (m.dice_pq - m.dice_gq) - (m.dice_pc - m.dice_gc)
        e, lo, hi, p, _ = boot_stat(m[['roi_id', 'cluster', 'seed', 'v']], rng)
        res.append(dict(schedule=sched, dataset=ds, contrast='(p_q - g_q) - (p_cl - g_cl)', diff=e, ci_low=lo, ci_high=hi, p=p))
        r = pd.DataFrame(res); fam = ~r.contrast.str.startswith('(')
        r['p_holm'] = np.nan; r.loc[fam, 'p_holm'] = holm(r.loc[fam, 'p'].values); rows.append(r)
        for mdl in sorted(t.model.unique()):
            tt = t[t.model == mdl]
            u = unit_table(tt).rename(columns={'dice': 'v'})
            e, lo, hi, _, _ = boot_stat(u, np.random.default_rng(0))
            d = dict(schedule=sched, dataset=ds, model=mdl, n_runs=tt[['fold', 'seed']].drop_duplicates().shape[0],
                     dice=e, ci_low=lo, ci_high=hi, iou=tt.iou.mean(), hd95_median=tt.hd95.median())
            if tt.dice_const.notna().any():
                for col, nm in (('dice_const', 'const'), ('dice_swap0', 'swap0'), ('dice_swapavg', 'swapavg')):
                    tt2 = tt.assign(drop=tt.dice - tt[col])
                    u = unit_table(tt2, 'drop').rename(columns={'drop': 'v'})
                    e2, lo2, hi2, _, _ = boot_stat(u, np.random.default_rng(0))
                    d.update({f'drop_{nm}': e2, f'drop_{nm}_lo': lo2, f'drop_{nm}_hi': hi2})
                d['swap_draw_sd'] = tt.groupby(['fold', 'seed'])[SW].mean().std(axis=1).mean()   # spread over the 10 swap draws (fixed permutations of one bank)
                d['flip_const'] = tt.flip_const.mean(); d['flip_swap'] = tt.flip_swap.mean()
            summ.append(d)
con = pd.concat(rows, ignore_index=True); con.to_csv(f'{OUT}/contrasts.csv', index=False)
sm = pd.DataFrame(summ); sm.to_csv(f'{OUT}/summary_models.csv', index=False)

# seed-0 comparisons with the classical references (train schedule only; 5 runs each)
ref = []
for ds in ('CBIS', 'DMID'):
    t = df[(df.set == 'train') & (df.dataset == ds) & (df.seed == 0)]
    rng = np.random.default_rng(0)
    for a in ('p_q', 'p_cl', 'g_q', 'g_cl'):
        for b in ('unet', 'full'):
            e, lo, hi, p, _ = contrast(t, a, b, rng=rng)
            ref.append(dict(dataset=ds, contrast=f'{a} - {b}', diff=e, ci_low=lo, ci_high=hi, p=p))
ref = pd.DataFrame(ref); ref['p_holm'] = ref.groupby('dataset').p.transform(lambda v: holm(v.values))
ref.to_csv(f'{OUT}/contrasts_vs_reference_seed0.csv', index=False)

# per-run table (convergence)
run = df[df.dataset == 'CBIS'].groupby(['set', 'model', 'fold', 'seed']).agg(dice=('dice', 'mean'), epochs=('epochs_run', 'first'), best=('best_epoch', 'first')).reset_index()
run.to_csv(f'{OUT}/per_run.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
print(con.round(4).to_string()); print(ref.round(4).to_string())
print(run.groupby(['set', 'model'])[['dice', 'epochs', 'best']].agg(['mean', 'min', 'max']).round(3).to_string())

# ---- per-run paired contrasts (one value per fold x seed; CBIS only; descriptive)
pr = []
for sched in ('train', 'val'):
    t = df[(df.set == sched) & (df.dataset == 'CBIS')].groupby(['model', 'fold', 'seed']).dice.mean().unstack('model')
    for a, b in CONTRASTS:
        d = (t[a] - t[b]).dropna()
        pr.append(dict(schedule=sched, contrast=f'{a} - {b}', n_runs=len(d), mean=d.mean(), min=d.min(), max=d.max(), n_positive=int((d > 0).sum())))
pr = pd.DataFrame(pr); pr.to_csv(f'{OUT}/per_run_contrasts.csv', index=False); print(pr.round(4).to_string())

# ---- external drop (CBIS - DMID, mean Dice) and comparison with no-network baselines
sm2 = sm.pivot_table(index=['schedule', 'model'], columns='dataset', values='dice')
sm2['drop'] = sm2.CBIS - sm2.DMID; sm2.to_csv(f'{OUT}/external_drop.csv'); print(sm2.round(4).to_string())

# ---- severing: per-image distribution (share of images with drop > 0.05 / < -0.05) and prediction-map change
sv = []
for (sched, ds, mdl), tt in df[df.dice_const.notna()].groupby(['set', 'dataset', 'model']):
    dc = tt.dice - tt.dice_const; dsw = tt.dice - tt.dice_swapavg
    sv.append(dict(schedule=sched, dataset=ds, model=mdl, const_gt05=(dc > 0.05).mean(), const_lt_m05=(dc < -0.05).mean(),
                   swap_gt05=(dsw > 0.05).mean(), swap_lt_m05=(dsw < -0.05).mean(),
                   flip_const=tt.flip_const.mean(), flip_swap=tt.flip_swap.mean(), absdp_const=tt.absdp_const.mean(), absdp_swap=tt.absdp_swap.mean()))
sv = pd.DataFrame(sv); sv.to_csv(f'{OUT}/severing_distribution.csv', index=False); print(sv.round(3).to_string())
