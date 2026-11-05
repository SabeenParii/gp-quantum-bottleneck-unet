# Bootstrap core used by stage2_analysis.py (functions build, hier, holm and the contrast list); the script slices it, it is not run on its own.
# 06 · Hierarchical statistics (CPU only). Reads the per-image tables written by notebook 05
# (results/p2_analysis_v2/) and writes results/p2_analysis_v3/. Retrains nothing, overwrites nothing.
#
# Primary analysis (as in Paper 1): hierarchical bootstrap that resamples BOTH the training seeds and the
# clusters (CBIS: patients; DMID: images), so the CIs include seed-to-seed variance.
#   * CBIS: each ROI is scored by the model of its own test fold, once per seed.
#   * DMID: each de-duplicated crop is scored by all 5 fold models of a seed; these are averaged per seed.
#   * A bootstrap p-value (two-sided) is computed from the same replicates, so p and CI always agree; Holm
#     correction within each family (paired comparisons, severing) and dataset.
# Secondary: cluster-only bootstrap with seeds fixed (old 05 analysis), per-seed differences, median cluster
# difference, share of clusters where A > B, and the Wilcoxon signed-rank test on cluster means.
import os
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

PROJECT = globals().get('PROJECT', '/content/drive/MyDrive/quantum-placement')
IN = f'{PROJECT}/results/p2_analysis_v2'
OUT = f'{PROJECT}/results/p2_analysis_v3'
os.makedirs(OUT, exist_ok=True)
B = 10000
SEEDS = (0, 1, 2)
CONTRASTS = [('g_q', 'g_cl'), ('g_q', 'g_id'), ('g_cl', 'g_id'), ('p_q', 'p_cl'), ('p_q', 'p_id'), ('p_cl', 'p_id'),
             ('p_q', 'g_q'), ('p_cl', 'g_cl'), ('p_id', 'g_id'), ('full', 'g_id'), ('full', 'p_id'), ('unet', 'full')]
SEVERABLE = ['g_q', 'g_cl', 'g_id', 'p_q', 'p_cl', 'p_id']

cb = pd.read_csv(f'{IN}/cbis_per_image_all.csv')
dm = pd.read_csv(f'{IN}/dmid_per_image_all.csv')
dm = dm[dm['keep'] == True]
print('CBIS rows:', len(cb), '| DMID rows (dedup):', len(dm))


def build(df, key, cluster, cols):
    """{(model, col): array (n_seeds_of_model, n_items)} with items aligned on `key`; plus cluster codes."""
    # one value per model x seed x item (DMID: mean over the 5 fold models of that seed)
    g = df.groupby(['model', 'seed', key, cluster], as_index=False)[cols].mean()
    items = g[[key, cluster]].drop_duplicates(key).sort_values(key).reset_index(drop=True)
    idx = pd.Series(np.arange(len(items)), index=items[key])
    codes, uniq = pd.factorize(items[cluster])
    data = {}
    for (m, s), t in g.groupby(['model', 'seed']):
        for c in cols:
            arr = np.full(len(items), np.nan)
            arr[idx[t[key]].to_numpy()] = t[c].to_numpy()
            data.setdefault((m, c), {})[s] = arr
    data = {k: np.vstack([v[s] for s in sorted(v)]) for k, v in data.items()}
    return data, codes, len(uniq)


def hier(a, b, codes, n_cl, rng, joint=True):
    """a, b: (seeds, items). Returns dict with point estimate, hierarchical and cluster-only bootstrap results."""
    ok = ~(np.isnan(a).any(0) | np.isnan(b).any(0))
    a, b, codes = a[:, ok], b[:, ok], codes[ok]
    n_cl = codes.max() + 1
    ones = np.ones(len(codes))
    cl_n = np.bincount(codes, ones, n_cl)
    A = np.vstack([np.bincount(codes, r, n_cl) for r in a])          # cluster sums per seed
    Bm = np.vstack([np.bincount(codes, r, n_cl) for r in b])
    est = a.mean() - b.mean()
    per_seed = (a.mean(1) - b.mean(1)) if a.shape[0] == b.shape[0] else np.array([])
    hb, cb_ = np.empty(B), np.empty(B)
    sa_all, sb_all = A.mean(0), Bm.mean(0)
    for t in range(B):
        w = np.bincount(rng.integers(0, n_cl, n_cl), minlength=n_cl)
        den = w @ cl_n
        sa = rng.integers(0, A.shape[0], A.shape[0])
        sb = sa if (joint and A.shape[0] == Bm.shape[0]) else rng.integers(0, Bm.shape[0], Bm.shape[0])
        hb[t] = (w @ A[sa].mean(0) - w @ Bm[sb].mean(0)) / den
        cb_[t] = (w @ sa_all - w @ sb_all) / den
    p = min(1.0, 2 * min((hb <= 0).mean(), (hb >= 0).mean()))
    p = max(p, 1.0 / B)
    nz = cl_n > 0
    clmean = (A.mean(0)[nz] - Bm.mean(0)[nz]) / cl_n[nz]               # seed-averaged cluster mean differences
    pw = wilcoxon(clmean).pvalue if np.any(clmean != 0) else 1.0
    lo90, hi90 = np.percentile(hb, [5, 95])
    return dict(n_items=int(ok.sum()), n_clusters=int(nz.sum()), n_seeds_A=a.shape[0], n_seeds_B=b.shape[0],
                mean_diff=est, ci95_low=np.percentile(hb, 2.5), ci95_high=np.percentile(hb, 97.5),
                ci90_low=lo90, ci90_high=hi90, smallest_equiv_bound=max(abs(lo90), abs(hi90)), p_boot=p,
                cluster_only_ci95_low=np.percentile(cb_, 2.5), cluster_only_ci95_high=np.percentile(cb_, 97.5),
                **{f'diff_seed{i}': v for i, v in enumerate(per_seed)},
                median_cluster_diff=float(np.median(clmean)), frac_clusters_A_better=float((clmean > 0).mean()),
                p_wilcoxon_cluster=pw)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); k = len(p); adj = np.empty(k); run = 0.0
    for i, j in enumerate(o):
        run = max(run, (k - i) * p[j]); adj[j] = min(1.0, run)
    return adj


rng = np.random.default_rng(0)
cols = ['dice', 'dice_const', 'dice_swap']
sets = {'CBIS': build(cb, 'roi_id', 'patient', [c for c in cols if c in cb.columns]),
        'DMID': build(dm, 'roi_id', 'img', [c for c in cols if c in dm.columns])}

pair, sev = [], []
for ds, (data, codes, n_cl) in sets.items():
    for a, b in CONTRASTS:
        pair.append(dict(dataset=ds, A=a, B=b, **hier(data[(a, 'dice')], data[(b, 'dice')], codes, n_cl, rng)))
    for m in SEVERABLE:
        for mode in ('const', 'swap'):
            if (m, f'dice_{mode}') in data:
                r = hier(data[(m, 'dice')], data[(m, f'dice_{mode}')], codes, n_cl, rng)
                r = {('drop' if k == 'mean_diff' else k): v for k, v in r.items()}
                sev.append(dict(dataset=ds, model=m, mode=mode, **r))
pair, sev = pd.DataFrame(pair), pd.DataFrame(sev)
for df in (pair, sev):
    df['p_holm'] = np.nan
    for ds in df.dataset.unique():
        m = df.dataset == ds
        df.loc[m, 'p_holm'] = holm(df.loc[m, 'p_boot'])
    df['ci_excludes_0'] = (df.ci95_low > 0) | (df.ci95_high < 0)
pair.round(5).to_csv(f'{OUT}/paired_hierarchical.csv', index=False)
sev.round(5).to_csv(f'{OUT}/severing_hierarchical.csv', index=False)
show = ['mean_diff', 'ci95_low', 'ci95_high', 'p_boot', 'p_holm', 'cluster_only_ci95_low', 'cluster_only_ci95_high',
        'diff_seed0', 'diff_seed1', 'diff_seed2', 'median_cluster_diff', 'frac_clusters_A_better', 'p_wilcoxon_cluster',
        'smallest_equiv_bound']
pd.set_option('display.width', 250)
print('\n## Paired comparisons (A - B), hierarchical bootstrap over seeds and clusters, B =', B)
print(pair[['dataset', 'A', 'B'] + [c for c in show if c in pair.columns]].round(4).to_string())
print('\n## Severing (drop = normal - severed), hierarchical bootstrap')
print(sev[['dataset', 'model', 'mode', 'drop'] + [c for c in show[1:] if c in sev.columns]].round(4).to_string())
# consistency check: bootstrap p < 0.05 must coincide with a 95% CI that excludes 0
for name, df in (('paired', pair), ('severing', sev)):
    bad = df[(df.p_boot < 0.05) != df.ci_excludes_0]
    print(f'\n{name}: rows where p_boot<0.05 and CI-excludes-0 disagree:', len(bad))
