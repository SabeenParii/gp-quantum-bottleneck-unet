# Code-replacement drops with five seeds (protocols/PROTOCOL_five_seed_extension.txt, secondary): Dice(normal) - Dice(replaced),
# crossed bootstrap over seeds and clusters (CBIS-DDSM patients, DMID source images), B = 10,000, rng(0); Holm within the 16 tests
# of each schedule and dataset. Sources: CBIS-DDSM from rev5_cbis_per_image (all seeds); DMID seeds 0-2 of the circuit, Dense(4)
# and identity models from results/replacement_v2_per_image.csv.gz, seeds 0-2 of the trigonometric models from
# results/stage2/rev4_replace_per_image.csv.gz, seeds 3-4 of all models from results/five_seed/per_image/rev5_replace_per_image.part*.
# Usage: python analysis/five_seed_code_replacement.py
import os, sys, glob, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, '..', 'results'); FS = os.path.join(RES, 'five_seed')
core = open(os.path.join(HERE, 'five_seed_statistics.py')).read()
sys.argv = ['x', FS, FS]; exec(core.split("rng = np.random.default_rng(0)")[0])   # holm, matrix, boot
SW = [f'dice_swap{i}' for i in range(10)]
def drops(df, cluster):
    df = df.copy(); df['drop_const'] = df.dice - df.dice_const; df['drop_swap'] = df.dice - df[[c for c in SW if c in df]].mean(axis=1)
    df['cluster'] = cluster(df); return df
cb = pd.read_csv(os.path.join(FS, 'per_image', 'rev5_cbis_per_image.csv.gz')); cb = cb[cb.dice_const.notna()]
cb['drop_const'] = cb.dice - cb.dice_const; cb['drop_swap'] = cb.dice - cb.dice_swap; cb['cluster'] = cb.patient.astype(str)
v2 = pd.read_csv(os.path.join(RES, 'replacement_v2_per_image.csv.gz')).rename(columns={'set': 'schedule'})
r4 = pd.read_csv(os.path.join(RES, 'stage2', 'rev4_replace_per_image.csv.gz'))
r5 = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(FS, 'per_image', 'rev5_replace_per_image.part*.csv.gz')))], ignore_index=True)
img = lambda d: d.roi_id.str.split('_').str[0]
dm = pd.concat([drops(t[t.dataset == 'DMID'], img) for t in (v2, r4, r5)], ignore_index=True).drop_duplicates(['roi_id', 'schedule', 'model', 'fold', 'seed'])
print(dm.groupby(['schedule', 'model']).seed.unique().apply(sorted).to_string())
for name, d in (('cbis', cb), ('dmid', dm)):
    rng = np.random.default_rng(0); rows = []
    for sch in ('train', 'val'):
        fam = []
        for kind in ('const', 'swap'):
            M, codes = matrix(d[d.schedule == sch], f'drop_{kind}')
            for m, (ss, X) in sorted(M.items()):
                r = boot(X, np.zeros_like(X), codes, rng); r.pop('boot'); fam.append(dict(schedule=sch, model=m, replacement=kind, seeds=len(ss), **r))
        for f, q in zip(fam, holm([f['p'] for f in fam])): f['p_holm'] = q
        rows += fam
    R = pd.DataFrame(rows); R.to_csv(os.path.join(FS, 'stats', f'rev5_{name}_drops.csv'), index=False, float_format='%.6g')
    print(name); print(R[['schedule', 'model', 'replacement', 'seeds', 'diff', 'ci_lo', 'ci_hi', 'p_holm']].round(4).to_string(index=False))
