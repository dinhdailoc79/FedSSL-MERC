import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, fedsim, fedtrain
o=fedsim.make_class_centers
def hc(C,d,rng,spread=2.0): return o(C,d,rng,spread=spread)
fedsim.make_class_centers=hc; fedtrain.make_class_centers=hc
from fedsim import _flat, _unflat, _weighted
from fedtrain import build_clients, fed_train, f1_macro, AGGS

# --- improved guard variants registered into AGGS for testing ---
def guard_v2(deltas, ns, us, server_delta=None, beta=8.0, **kw):
    """FLTrust-normalized trust as PRIMARY (not spoofable evidential weight),
    evidential weight only modulates among trusted; norm-clip to root."""
    sv=_flat(server_delta); svn=np.linalg.norm(sv)+1e-12
    ev=np.exp(-beta*np.array(us))           # evidential modulation
    trust=[]; scaled=[]
    for d in deltas:
        dv=_flat(d); dn=np.linalg.norm(dv)+1e-12
        cs=(dv@sv)/dn/svn
        t=max(0.0,cs)
        trust.append(t)
        scaled.append(_unflat(deltas[0], dv*(svn/dn)))   # normalize to root norm
    trust=np.array(trust)
    w=trust*ev*np.array(ns,float)
    if w.sum()<=0: w=np.array(ns,float)
    w=w/w.sum()
    return _weighted(scaled,w), w

def guard_v3(deltas, ns, us, server_delta=None, beta=8.0, **kw):
    """v2 + relative cosine threshold (drop bottom by trust) + multi-krum core
    among survivors."""
    sv=_flat(server_delta); svn=np.linalg.norm(sv)+1e-12
    V=np.stack([_flat(d) for d in deltas]); K=len(deltas)
    norms=np.linalg.norm(V,axis=1,keepdims=True)+1e-12
    cs=(V@sv)/norms[:,0]/svn
    trust=np.maximum(0,cs)
    # keep clients with trust above half the max trust (adaptive threshold)
    thr=0.3*trust.max()
    keep=np.where(trust>=thr)[0]
    if len(keep)<2: keep=np.argsort(-trust)[:max(2,K//2)]
    # normalize survivors to root norm
    Vs=V[keep]*(svn/norms[keep,0:1])
    # multi-krum among survivors
    f=max(1,len(keep)//5); m=max(1,len(keep)-f-2)
    sc=[]
    for i in range(len(keep)):
        dd=np.sum((Vs-Vs[i])**2,1); dd[i]=np.inf
        sc.append(np.sort(dd)[:m].sum())
    sel=np.argsort(sc)[:max(1,len(keep)-f)]
    ev=np.exp(-beta*np.array(us)[keep][sel])
    w=ev/ev.sum()
    agg=np.zeros_like(Vs[0])
    for j,i in enumerate(sel): agg+=w[j]*Vs[i]
    return _unflat(deltas[0],agg)

AGGS['Guard-v2']=guard_v2
AGGS['Guard-v3']=guard_v3

C,dt,da,K=6,16,16,20
atk=set(range(16,20))
rng=np.random.default_rng(200)
clients,test,_=build_clients(K,C,dt,da,rng,alpha=0.5,attack_clients=atk,n_per=200)
for attack in ['label-flip','sign-flip','adaptive']:
    line=[attack]
    for agg in ['Krum','EAFA-Guard','Guard-v2','Guard-v3']:
        f1=[f1_macro(fed_train(clients,test,dt,da,C,agg,np.random.default_rng(s),rounds=35,lr=0.015,lam=0.05,anneal=15,beta=8.0,attack=attack,attack_clients=atk)['model'],test,C) for s in range(4)]
        line.append(f'{agg}=%.1f'%np.mean(f1))
    print('  '.join(line))
