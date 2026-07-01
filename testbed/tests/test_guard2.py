import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, fedsim, fedtrain
o=fedsim.make_class_centers
def hc(C,d,rng,spread=2.0): return o(C,d,rng,spread=spread)
fedsim.make_class_centers=hc; fedtrain.make_class_centers=hc
from fedsim import _flat, _unflat, _weighted
from fedtrain import build_clients, fed_train, f1_macro, AGGS

def guard_v4(deltas, ns, us, server_delta=None, beta=8.0, **kw):
    """evidential weight * ReLU(cos), but zero out clients below MEDIAN cosine
    (attackers are minority so median is honest); root-norm normalize."""
    sv=_flat(server_delta); svn=np.linalg.norm(sv)+1e-12
    V=np.stack([_flat(d) for d in deltas]); K=len(deltas)
    norms=np.linalg.norm(V,axis=1,keepdims=True)+1e-12
    cs=(V@sv)/norms[:,0]/svn
    med=np.median(cs)
    keep=cs>=max(0.0,med*0.8)
    ev=np.exp(-beta*np.array(us))
    w=np.where(keep, np.maximum(0,cs)*ev*np.array(ns,float), 0.0)
    if w.sum()<=0: w=np.array(ns,float)
    w=w/w.sum()
    Vn=V*(svn/norms)   # normalize each to root norm
    agg=(w[:,None]*Vn).sum(0)
    return _unflat(deltas[0],agg), w

AGGS['Guard-v4']=guard_v4
C,dt,da,K=6,16,16,20
atk=set(range(16,20))
rng=np.random.default_rng(200)
clients,test,_=build_clients(K,C,dt,da,rng,alpha=0.5,attack_clients=atk,n_per=200)
# inspect cosines under adaptive
import fedsim as fs
from fedsim import DualEDL
for attack in ['adaptive','sign-flip','label-flip']:
    line=[attack]
    for agg in ['Krum','EAFA-Guard','Guard-v4']:
        f1=[f1_macro(fed_train(clients,test,dt,da,C,agg,np.random.default_rng(s),rounds=35,lr=0.015,lam=0.05,anneal=15,beta=8.0,attack=attack,attack_clients=atk)['model'],test,C) for s in range(4)]
        line.append(f'{agg}=%.1f'%np.mean(f1))
    print('  '.join(line))
