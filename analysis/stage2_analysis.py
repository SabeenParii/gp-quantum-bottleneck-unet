# Stage-2 analyses (prespecified in protocols/PROTOCOL_stage2_trigonometric_and_frozen.txt): A frozen cohort, C trigonometric control.
# Usage: python analysis/stage2_analysis.py <folder with the full-precision exports of notebook 22: full_cbis_<schedule>.csv.gz,
#        full_dmid_corrected_<schedule>.csv.gz>
import os, sys, numpy as np, pandas as pd
from scipy.stats import wilcoxon
HERE = os.path.dirname(os.path.abspath(__file__)); D = sys.argv[1] if len(sys.argv) > 1 else 'full_precision_exports'
sys.argv = ['x']; PROJECT = '/nonexistent'
src = open(os.path.join(HERE, 'stage2_bootstrap_core.py')).read()
exec(src[src.index('B = 10000'):src.index('cb = pd.read_csv')]); exec(src[src.index('def build'):src.index('rng = np.random.default_rng(0)')])   # B, CONTRASTS, hier, holm
R = os.path.join(HERE, '..', 'results', 'stage2')
pid = lambda s: s.str.extract(r'(P_\d+)')[0]
out = []
# ---- A. frozen cohort: nine contrasts per schedule, DMID-style averaging over fold models
fz = pd.read_csv(f'{R}/rev4_frozen_per_image.csv.gz')
for sch, sets in (('train', ('train', 'train4')), ('val', ('val', 'val4'))):
    t = fz[fz.set.isin(sets)]
    data, codes, ncl = build(t, 'roi_id', 'patient', ['dice'])
    fam = []
    for a, b in CONTRASTS[:9]:
        fam.append(dict(part='A_frozen', schedule=sch, dataset='frozen', A=a, B=b, **hier(data[(a, 'dice')], data[(b, 'dice')], codes, ncl, np.random.default_rng(0))))
    for a, b in (('g_q', 'g_tr'), ('p_q', 'p_tr'), ('g_tr', 'g_cl'), ('p_tr', 'p_cl'), ('g_tr', 'g_id'), ('p_tr', 'p_id')):
        fam.append(dict(part='A_frozen_trig(secondary)', schedule=sch, dataset='frozen', A=a, B=b, **hier(data[(a, 'dice')], data[(b, 'dice')], codes, ncl, np.random.default_rng(0))))
    f = pd.DataFrame(fam); m = f.part == 'A_frozen'; f['p_holm'] = np.nan; f.loc[m, 'p_holm'] = holm(f.loc[m, 'p_boot'])
    out.append(f)
    lv = t.groupby(['model', 'seed', 'roi_id']).dice.mean().groupby('model').mean()
    print(sch, 'frozen levels', lv.round(4).to_dict())
bl = pd.read_csv('/dev/null') if False else None
# ---- C. trigonometric control on CBIS test folds and DMID
rp = pd.read_csv(f'{R}/rev4_replace_per_image.csv.gz')
prim = []
for sch in ('train', 'val'):
    st = pd.read_csv(f'{D}/full_cbis_{sch}.csv.gz'); st = st[st.model.isin(['g_q', 'p_q', 'g_cl', 'p_cl', 'g_id', 'p_id'])]
    nw = rp[(rp.schedule == sch) & (rp.dataset == 'CBIS')]
    cb = pd.concat([st[['model', 'seed', 'fold', 'roi_id', 'dice', 'dice_const', 'dice_swap0']], nw[['model', 'seed', 'fold', 'roi_id', 'dice', 'dice_const', 'dice_swap0']]])
    cb['patient'] = pid(cb.roi_id)
    sd = pd.read_csv(f'{D}/full_dmid_corrected_{sch}.csv.gz'); sd = sd[sd.keep == True]; keep = set(sd.roi_id)
    sd = sd[sd.model.isin(['g_q', 'p_q', 'g_cl', 'p_cl', 'g_id', 'p_id'])]
    nd = rp[(rp.schedule == sch) & (rp.dataset == 'DMID') & rp.roi_id.isin(keep)]
    dm = pd.concat([sd[['model', 'seed', 'fold', 'roi_id', 'dice']], nd[['model', 'seed', 'fold', 'roi_id', 'dice']]]); dm['img'] = dm.roi_id.str.split('_').str[0]
    for ds, df, cl in (('CBIS', cb, 'patient'), ('DMID', dm, 'img')):
        data, codes, ncl = build(df, 'roi_id', cl, ['dice'])
        for a, b, part in (('g_q', 'g_tr', 'C_primary'), ('p_q', 'p_tr', 'C_primary'), ('g_tr', 'g_cl', 'C_secondary'), ('p_tr', 'p_cl', 'C_secondary'),
                           ('g_tr', 'g_id', 'C_secondary'), ('p_tr', 'p_id', 'C_secondary')):
            prim.append(dict(part=part if ds == 'CBIS' else part + '_DMID(exploratory)', schedule=sch, dataset=ds, A=a, B=b,
                             **hier(data[(a, 'dice')], data[(b, 'dice')], codes, ncl, np.random.default_rng(0))))
    # code-replacement drops of trig models (CBIS)
    for m in ('g_tr', 'p_tr'):
        u = nw[nw.model == m].copy(); u['patient'] = pid(u.roi_id); u['swapavg'] = u[[f'dice_swap{i}' for i in range(10)]].mean(1)
        data, codes, ncl = build(u, 'roi_id', 'patient', ['dice', 'dice_const', 'swapavg'])
        for mode in ('dice_const', 'swapavg'):
            prim.append(dict(part='C_drop', schedule=sch, dataset='CBIS', A=m, B=mode, **hier(data[(m, 'dice')], data[(m, mode)], codes, ncl, np.random.default_rng(0))))
P = pd.DataFrame(prim); P['p_holm'] = np.nan
m = P.part == 'C_primary'; P.loc[m, 'p_holm'] = holm(P.loc[m, 'p_boot'])
ALL = pd.concat(out + [P], ignore_index=True); ALL.to_csv(os.path.join(R, 'rev4_contrasts.csv'), index=False)
pd.set_option('display.width', 250)
cols = ['part', 'schedule', 'dataset', 'A', 'B', 'n_clusters', 'n_seeds_A', 'mean_diff', 'ci95_low', 'ci95_high', 'p_boot', 'p_holm', 'ci90_low', 'ci90_high']
print(ALL[cols].round(4).to_string())
