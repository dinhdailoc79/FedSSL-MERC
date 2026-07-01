import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, fedsim, fedtrain
o=fedsim.make_class_centers
def hc(C,d,rng,spread=2.0): return o(C,d,rng,spread=spread)
fedsim.make_class_centers=hc; fedtrain.make_class_centers=hc
from fedsim import _flat, _unflat, _weighted
from fedtrain import build_clients, fed_train, f1_macro, AGGS

def guard_v5(deltas, ns, us, server_delta=None, beta=8.0, **kw):
    """Hard direction filter (drop cos below median), keep magnitudes,
    weight survivors by size*ReLU(cos)*evidential. No norm rescaling."""
    sv=_flat(server_delta); svn=np.linalg.norm(sv)+1e-12
    V=np.stack([_flat(d) for d in deltas]); K=len(deltas)
    norms=np.linalg.norm(V,axis=1,keepdims=True)+1e-12
    cs=(V@sv)/norms[:,0]/svn
    med=np.median(cs)
    ev=np.exp(-beta*np.array(us))
    keep=cs>=med            # drop the lower half by direction
    w=np.where(keep, np.maximum(0,cs)*ev*np.array(ns,float), 0.0)
    # clip magnitudes of survivors to median survivor norm (bound leverage)
    sn=norms[:,0].copy()
    if keep.sum()>0:
        cap=np.median(sn[keep])
        scale=np.minimum(1.0, cap/sn)
    else:
        scale=np.ones(K)
    if w.sum()<=0: w=np.array(ns,float)
    w=w/w.sum()
    agg=((w*scale)[:,None]*V).sum(0)
    return _unflat(deltas[0],agg), w

def guard_v6(deltas, ns, us, server_delta=None, beta=8.0, **kw):
    """v5 but trust^2 sharpening + size weighting only (ignore spoofable ev)."""
    sv=_flat(server_delta); svn=np.linalg.norm(sv)+1e-12
    V=np.stack([_flat(d) for d in deltas]); K=len(deltas)
    norms=np.linalg.norm(V,axis=1,keepdims=True)+1e-12
    cs=(V@sv)/norms[:,0]/svn
    med=np.median(cs)
    keep=cs>=med
    t=np.maximum(0,cs)**2
    w=np.where(keep, t*np.array(ns,float), 0.0)
    cap=np.median(norms[keep,0]) if keep.sum()>0 else norms[:,0].max()
    scale=np.minimum(1.0, cap/norms[:,0])
    if w.sum()<=0: w=np.array(ns,float)
    w=w/w.sum()
    agg=((w*scale)[:,None]*V).sum(0)
    return _unflat(deltas[0],agg), w

AGGS['Guard-v5']=guard_v5; AGGS['Guard-v6']=guard_v6
C,dt,da,K=6,16,16,20
atk=set(range(16,20))
rng=np.random.default_rng(200)
clients,test,_=build_clients(K,C,dt,da,rng,alpha=0.5,attack_clients=atk,n_per=200)
for attack in ['label-flip','sign-flip','adaptive']:
    line=[attack]
    for agg in ['Krum','EAFA-Guard','Guard-v5','Guard-v6']:
        f1=[f1_macro(fed_train(clients,test,dt,da,C,agg,np.random.default_rng(s),rounds=35,lr=0.015,lam=0.05,anneal=15,beta=8.0,attack=attack,attack_clients=atk)['model'],test,C) for s in range(4)]
        line.append(f'{agg}=%.1f'%np.mean(f1))
    print('  '.join(line))
