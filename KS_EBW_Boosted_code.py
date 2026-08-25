"""
Simulation study 2: Kang--Schafer-style missing-data design.

This version uses B_max=100 weak CART learners, learning rate nu=0.10,
interaction depth 3, early stopping, and an ESS safeguard. The hybrid
variants use estimated classical EB dual parameters as offsets: the main-offset
hybrid preserves main moments, and the pairwise-offset hybrid preserves both
main and pairwise moments after each boosted tree correction.
"""
import os, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

from entropy_common import (
    normalize_weights, effective_sample_size, sigmoid, eb_fit, eb_weights,
    pairwise_products, compact_leaf_ids_for_two, props, fit_balance_tree, hybrid
)

OUT_DIR=os.path.dirname(os.path.abspath(__file__))
SEED=20260625

def generate_ks(n=1000,rng=None):
    if rng is None: rng=np.random.default_rng()
    Z=rng.normal(size=(n,4)); z1,z2,z3,z4=Z.T
    X=np.column_stack([np.exp(z1/2.0), z2/(1+np.exp(z1))+10.0, (z1*z3/25.0+0.6)**3, (z2+z4+20.0)**2])
    y=210+27.4*z1+13.7*z2+13.7*z3+13.7*z4+rng.normal(scale=1.0,size=n)
    p=sigmoid(-z1+0.5*z2-0.25*z3-0.1*z4); r=rng.binomial(1,p,size=n).astype(bool)
    return Z,X,y,r

def one_rep(rng,n=1000,B=100,nu=0.10):
    Z,X,y,resp=generate_ks(n,rng); mu=X.mean(0); sd=X.std(0); sd[sd<1e-12]=1
    Xs_all=(X-mu)/sd; Xt=Xs_all; Xr=Xs_all[resp]; yr=y[resp]; Zr=Z[resp]
    if Xr.shape[0]<50: raise RuntimeError('too few respondents')
    target=float(y.mean()); mu_main=Xt.mean(0); Pr=pairwise_products(Xr); Pt=pairwise_products(Xt)
    Xpair=np.hstack([Xr,Pr]); mu_pair=np.r_[mu_main,Pt.mean(0)]; q=np.ones(Xr.shape[0])/Xr.shape[0]
    out=[]
    def add(method,est,w=None,trees=np.nan):
        row={'method':method,'estimate':est,'error':est-target,'ess':np.nan,'main_l2':np.nan,'pair_l2':np.nan,'latent_z_l2':np.nan,'num_trees':trees}
        if w is not None:
            row['ess']=effective_sample_size(w); row['main_l2']=float(np.linalg.norm(Xr.T@w-mu_main)); row['pair_l2']=float(np.linalg.norm(Pr.T@w-Pt.mean(0))); row['latent_z_l2']=float(np.linalg.norm(Zr.T@w-Z.mean(0)))
        out.append(row)
    add('Naive respondents',float(np.mean(yr)),q,0)
    A=np.column_stack([np.ones(Xr.shape[0]),Xr]); beta=np.linalg.lstsq(A,yr,rcond=None)[0]; pred=np.column_stack([np.ones(n),Xt])@beta
    add('OLS prediction',float(np.mean(pred)),None,np.nan)
    w_main,lam_main=eb_fit(Xr,mu_main,q=q,max_iter=80,tol=1e-8); add('EB main',float(w_main@yr),w_main,0)
    w_pair,lam_pair=eb_fit(Xpair,mu_pair,q=q,max_iter=100,tol=1e-8); add('EB pairwise',float(w_pair@yr),w_pair,0)
    rs=int(rng.integers(0,2**31-1))
    # EB-offset main hybrid: start from the fitted main-effect EB offset and
    # project back to the same main-effect hard constraints after each tree.
    w_hyb,nt=hybrid(Xr,Xt,Xr,mu_main,q0=w_main,B=B,nu=nu,random_state=rs); add('EB-offset hybrid: main',float(w_hyb@yr),w_hyb,nt)
    # EB-offset pairwise hybrid: start from the fitted pairwise EB offset and
    # project back to the same main+pairwise constraints after each tree.
    w_hyb_p,ntp=hybrid(Xr,Xt,Xpair,mu_pair,q0=w_pair,B=B,nu=nu,random_state=rs+10000); add('EB-offset hybrid: pairwise',float(w_hyb_p@yr),w_hyb_p,ntp)
    return out

def one_rep_seed(args):
    seed,n,B,nu=args; rng=np.random.default_rng(seed); return one_rep(rng,n=n,B=B,nu=nu)

def run(R=1000,n=1000,B=100,nu=0.10,n_jobs=None):
    import multiprocessing as mp
    seeds=[SEED+r for r in range(R)]; args=[(s,n,B,nu) for s in seeds]
    if n_jobs is None: n_jobs=min(8,max(1,(os.cpu_count() or 2)-1))
    rows=[]
    if n_jobs<=1:
        for r,arg in enumerate(args):
            for row in one_rep_seed(arg): row['rep']=r; rows.append(row)
    else:
        with mp.Pool(processes=n_jobs) as pool:
            for r,rep_rows in enumerate(pool.imap(one_rep_seed,args,chunksize=5)):
                for row in rep_rows: row['rep']=r; rows.append(row)
    raw=pd.DataFrame(rows); summ=[]
    for method,g in raw.groupby('method'):
        e=g['error'].to_numpy(); summ.append({'method':method,'bias':e.mean(),'rmse':math.sqrt(np.mean(e**2)),'mae':np.mean(np.abs(e)),'ess_mean':g['ess'].mean(),'main_l2_mean':g['main_l2'].mean(),'pair_l2_mean':g['pair_l2'].mean(),'latent_z_l2_mean':g['latent_z_l2'].mean(),'num_trees_mean':g['num_trees'].mean()})
    return raw,pd.DataFrame(summ)

def main():
    raw,summary=run(R=1000,n=1000,B=100,nu=0.10,n_jobs=8)
    raw.to_csv(os.path.join(OUT_DIR,'ks_sim_raw_rows.csv'),index=False); summary.to_csv(os.path.join(OUT_DIR,'ks_sim_summary.csv'),index=False)
    order=['Naive respondents','OLS prediction','EB main','EB pairwise','EB-offset hybrid: main','EB-offset hybrid: pairwise']; labels=['Naive','OLS','EB main','EB pairwise','Hybrid EB-O','Hybrid EB-O (pair)']
    sm=summary.set_index('method').loc[order].reset_index()
    plt.figure(figsize=(7.8,4.2)); plt.bar(labels,sm['rmse']); plt.ylabel('RMSE'); plt.xticks(rotation=25,ha='right'); plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,'ks_fig_rmse_R1000.pdf')); plt.close()
    z=sm[sm['latent_z_l2_mean'].notna()]
    plt.figure(figsize=(7.8,4.2)); plt.bar([labels[order.index(m)] for m in z['method']],z['latent_z_l2_mean']); plt.ylabel('Mean latent Z L2'); plt.xticks(rotation=25,ha='right'); plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,'ks_fig_z_l2_R1000.pdf')); plt.close()
    print(sm.to_string(index=False)); plt.close('all')
if __name__=='__main__': main()
