"""What the trained global circuits compute (main text: Table 4, rows after training; additional analysis).

Input: one .npz file per run of A1 (g_q_fast, stage 3) and A4 (g_q_wide, stage 4) in
results/additional_experiments/trained_circuit_codes/, each holding the codes entering the trained circuit for all
400 CBIS-DDSM crops (cin), the circuit outputs (cout), the angle scale (enc) and the run identifiers.
Output: results/additional_experiments/trained_circuit_function.csv with the share of output variance explained by
all 81 terms of Eq. (2) (r81), by the single-input terms (r1), by terms with at most two inputs (r2), and by the best
fitting trigonometric (rtr) and Dense(4) (rd) blocks.
Usage: python analysis/trained_circuit_function.py
"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.join(HERE, '..', 'results', 'additional_experiments', 'trained_circuit_codes')
OUT = os.path.join(HERE, '..', 'results', 'additional_experiments', 'trained_circuit_function.csv')
import numpy as np, itertools, glob, sys
from scipy.optimize import least_squares
N=4
def basis(A, maxorder):
    f=[np.ones_like(A),np.cos(A),np.sin(A)]; cols=[]
    for k in itertools.product(range(3),repeat=N):
        if sum(1 for x in k if x)<=maxorder: cols.append(np.prod([f[k[j]][:,j] for j in range(N)],0))
    return np.stack(cols,1)
def r2_lin(Y,F):
    c=np.linalg.lstsq(F,Y,rcond=None)[0]; return 1-((Y-F@c)**2).sum()/((Y-Y.mean(0))**2).sum()
def r2_fit(Y,C,kind,scale,restarts=5,seed=0):
    rng=np.random.default_rng(seed); sst=((Y-Y.mean(0))**2).sum(); best=-np.inf
    def f(p):
        if kind=='tr': A,ph=p[:16].reshape(4,4),p[16:]; return np.tanh(np.cos(C*scale+ph)@A.T)
        W,b=p[:16].reshape(4,4),p[16:]; return np.tanh(C@W.T+b)
    for r in range(restarts):
        p0=np.concatenate([rng.normal(0,1,16), rng.uniform(0,2*np.pi,4) if kind=='tr' else rng.normal(0,0.1,4)])
        res=least_squares(lambda p:(f(p)-Y).ravel(),p0,max_nfev=4000); best=max(best,1-(res.fun**2).sum()/sst)
    return best
rows=[]
for fn in sorted(glob.glob(os.path.join(CODES, '*.npz'))):
    d=np.load(fn,allow_pickle=False); C=d['cin'].astype(np.float64); Y=d['cout'].astype(np.float64); enc=float(d['enc'])
    s=enc*np.pi/2; A=C*s
    rows.append(dict(model=str(d['model']),fold=int(d['fold']),seed=int(d['seed']),enc=enc,code_sd=C.std(0).mean(),
        r81=r2_lin(Y,basis(A,4)),r1=r2_lin(Y,basis(A,1)),r2=r2_lin(Y,basis(A,2)),rtr=r2_fit(Y,C,'tr',s),rd=r2_fit(Y,C,'d',s)))
import pandas as pd
df=pd.DataFrame(rows); df.to_csv(OUT,index=False)
print(df.round(3).to_string())
print(df.groupby('model')[['r81','r1','r2','rtr','rd','code_sd']].median().round(3))
print(df.groupby('model')[['r1','rtr','rd']].quantile([0.25,0.75]).round(3))
