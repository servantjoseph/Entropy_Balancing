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

OUT_DIR=os.path.dirname(os.path.abspath(__file__))
SEED=20260625

def normalize_weights(w):
    w=np.maximum(np.asarray(w,float),1e-300); return w/w.sum()
def effective_sample_size(w):
    w=normalize_weights(w); return 1.0/np.sum(w**2)
def sigmoid(z):
    z=np.asarray(z,float); out=np.empty_like(z); pos=z>=0; out[pos]=1/(1+np.exp(-z[pos])); ez=np.exp(z[~pos]); out[~pos]=ez/(1+ez); return out

def eb_fit(X, mu, q=None, max_iter=80, tol=1e-8, ridge=1e-7):
    """Classical entropy-balancing fit returning weights and dual parameters."""
    X=np.asarray(X,float); mu=np.asarray(mu,float); n,p=X.shape
    q=np.ones(n)/n if q is None else normalize_weights(q)
    lam=np.zeros(p)
    for _ in range(max_iter):
        eta=X@lam; a=q*np.exp(eta-np.max(eta)); w=a/a.sum(); g=X.T@w-mu
        if np.linalg.norm(g)<tol: return w,lam
        m=w@X; Xc=X-m; H=(Xc.T*w)@Xc + ridge*np.eye(p)
        try: step=np.linalg.solve(H,g)
        except np.linalg.LinAlgError: step=np.linalg.lstsq(H,g,rcond=None)[0]
        alpha=1.0; ng=np.linalg.norm(g)
        for _ in range(25):
            lam2=lam-alpha*step; a2=q*np.exp(X@lam2-np.max(X@lam2)); w2=a2/a2.sum()
            if np.linalg.norm(X.T@w2-mu)<=ng+1e-12:
                lam=lam2; break
            alpha*=0.5
        else: lam=lam-0.05*step
    eta=X@lam; a=q*np.exp(eta-np.max(eta)); return a/a.sum(),lam


def eb_weights(X, mu, q=None, max_iter=80, tol=1e-8, ridge=1e-7):
    w,_=eb_fit(X,mu,q=q,max_iter=max_iter,tol=tol,ridge=ridge)
    return w

def pairwise_products(X):
    cols=[]
    for j in range(X.shape[1]):
        for k in range(j+1,X.shape[1]): cols.append((X[:,j]*X[:,k])[:,None])
    return np.hstack(cols) if cols else np.zeros((X.shape[0],0))

def compact_leaf_ids_for_two(clf,Xs,Xt):
    raw_s=clf.apply(Xs); raw_t=clf.apply(Xt); vals=np.unique(np.r_[raw_s,raw_t]); mp={old:i for i,old in enumerate(vals)}
    return np.array([mp[z] for z in raw_s],int), np.array([mp[z] for z in raw_t],int)

def props(ids,w,J=None):
    if J is None: J=int(ids.max())+1
    p=np.bincount(ids,weights=normalize_weights(w),minlength=J).astype(float); return p/p.sum()

def fit_balance_tree(Xs,Xt,ws,wt,interaction_depth=3,min_mass=0.001,random_state=0):
    ws=normalize_weights(ws); wt=normalize_weights(wt); ns=Xs.shape[0]; nt=Xt.shape[0]
    X_aug=np.vstack([Xs,Xt]); y=np.r_[np.zeros(ns,dtype=int),np.ones(nt,dtype=int)]; sw=np.r_[0.5*ws,0.5*wt]
    best=None
    # Try a small grid of CART leaf-size controls and keep the highest valid leaf discrepancy.
    for min_frac in (0.01,0.02,0.05,0.10):
        clf=DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=interaction_depth,min_weight_fraction_leaf=min_frac,random_state=random_state)
        clf.fit(X_aug,y,sample_weight=sw)
        ids_s,ids_t=compact_leaf_ids_for_two(clf,Xs,Xt); J=max(ids_s.max(),ids_t.max())+1
        ps=props(ids_s,ws,J); pt=props(ids_t,wt,J)
        if np.min(ps)<min_mass or np.min(pt)<min_mass or J<2: continue
        score=float(np.sum((ps-pt)**2/pt))
        if best is None or score>best[0]: best=(score,clf,ids_s,ids_t,ps,pt)
    if best is None: return None,None,None,-np.inf,None,None
    score,clf,ids_s,ids_t,ps,pt=best
    return clf,ids_s,ids_t,score,ps,pt

def hybrid(Xs,Xt,Xhard,mu_hard,q0=None,B=100,nu=0.10,min_mass=0.001,interaction_depth=3,random_state=0,min_ess_frac=0.10,score_tol=0.05):
    wt=np.ones(Xt.shape[0])/Xt.shape[0]
    w=eb_weights(Xhard,mu_hard,q=q0,tol=1e-8)
    nt=0
    for b in range(B):
        clf,ids_s,ids_t,score,ps,pt=fit_balance_tree(Xs,Xt,w,wt,interaction_depth=interaction_depth,min_mass=min_mass,random_state=random_state+b)
        if clf is None or not np.isfinite(score) or score<score_tol: break
        ratio=np.power(np.maximum(pt,1e-12)/np.maximum(ps,1e-12),nu)
        qtemp=normalize_weights(w*ratio[ids_s])
        w_new=eb_weights(Xhard,mu_hard,q=qtemp,tol=1e-8,max_iter=60)
        if effective_sample_size(w_new)<min_ess_frac*Xs.shape[0]: break
        w=w_new; nt+=1
    return w,nt

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
    order=['Naive respondents','OLS prediction','EB main','EB pairwise','EB-offset hybrid: main','EB-offset hybrid: pairwise']; labels=['Naive','OLS','EB main','EB pairwise','Hybrid EB-O','Hybrid EB-OP']
    sm=summary.set_index('method').loc[order].reset_index()
    plt.figure(figsize=(7.8,4.2)); plt.bar(labels,sm['rmse']); plt.ylabel('RMSE'); plt.xticks(rotation=25,ha='right'); plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,'ks_fig_rmse_R1000.pdf')); plt.close()
    z=sm[sm['latent_z_l2_mean'].notna()]
    plt.figure(figsize=(7.8,4.2)); plt.bar([labels[order.index(m)] for m in z['method']],z['latent_z_l2_mean']); plt.ylabel('Mean latent Z L2'); plt.xticks(rotation=25,ha='right'); plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,'ks_fig_latentz_R1000.pdf')); plt.close()
    print(sm.to_string(index=False)); plt.close('all')
if __name__=='__main__': main()
