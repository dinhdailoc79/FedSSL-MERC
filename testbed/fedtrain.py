"""Federated training driver over the DualEDL testbed."""
import numpy as np
from fedsim import (DualEDL, make_class_centers, make_client_data,
                    dirichlet_partition, agg_fedavg, agg_eafa, agg_eafa_guard,
                    agg_krum, agg_trimmed_mean, agg_median, agg_fltrust,
                    agg_foolsgold, _flat, _unflat)

AGGS = {
    "FedAvg": agg_fedavg, "EAFA": agg_eafa, "EAFA-Guard": agg_eafa_guard,
    "Krum": lambda *a, **k: agg_krum(*a, multi=False, **k),
    "Multi-Krum": lambda *a, **k: agg_krum(*a, multi=True, **k),
    "Trimmed-Mean": agg_trimmed_mean, "Coord-Median": agg_median,
    "FLTrust": agg_fltrust, "FoolsGold": agg_foolsgold,
}

def build_clients(K, C, dt, da, rng, alpha=0.5, n_per=220,
                  noise_clients=None, noise_rate=0.0, hard_clients=None,
                  hardness_hard=2.6, audio_missing=0.0, attack=None,
                  attack_clients=None, shift_clients=None, shift_mag=2.5):
    text_c = make_class_centers(C, dt, rng)
    audio_c = make_class_centers(C, da, rng)
    probs = dirichlet_partition(C, K, alpha, rng)
    noise_clients = noise_clients or set()
    hard_clients = hard_clients or set()
    attack_clients = attack_clients or set()
    shift_clients = shift_clients or set()
    clients = []
    for k in range(K):
        ln = noise_rate if k in noise_clients else 0.0
        hd = hardness_hard if k in hard_clients else 1.0
        cs = shift_mag if k in shift_clients else 0.0
        data = make_client_data(n_per, C, text_c, audio_c, rng,
                                label_noise=ln, hardness=hd,
                                audio_missing=audio_missing,
                                class_probs=probs[k], center_shift=cs)
        meta = dict(kind=("lowq" if (k in shift_clients or k in noise_clients)
                          else "attacker" if k in attack_clients else "clean"))
        clients.append((data, meta))
    # held-out global test
    yt = rng.choice(C, size=2000)
    Xt = text_c[yt] + rng.normal(0, 1.0, (2000, dt))
    Xa = audio_c[yt] + rng.normal(0, 1.0, (2000, da))
    test = dict(Xt=Xt, Xa=Xa, y=yt, y_true=yt, miss=np.zeros(2000, bool))
    return clients, test, (text_c, audio_c)

def apply_attack(model, data, attack, rng, C):
    """Return a client whose update is malicious."""
    if attack == "label-flip":
        d2 = dict(data); d2["y"] = (data["y"] + 1) % C
        return d2, False
    if attack == "sign-flip":
        return dict(data), "sign"
    if attack == "adaptive":
        # train on flipped labels while minimizing own uncertainty
        d2 = dict(data); d2["y"] = (data["y"] + 1) % C
        return d2, "adaptive"
    return dict(data), False

def fed_train(clients, test, dt, da, C, agg_name, rng, rounds=30, h=48,
              lr=0.02, epochs=1, lam=0.3, beta=4.0, frac=1.0, participate=1.0,
              ssl_mode=None, frac_labeled=1.0, attack=None, attack_clients=None,
              return_dynamics=False, gated=False, wd=1e-3, anneal=10):
    attack_clients = attack_clients or set()
    global_model = DualEDL(dt, da, h, C, rng, gated=gated)
    P = global_model.params()
    agg = AGGS[agg_name]
    K = len(clients)
    # clean server-held root set (FLTrust assumption): small clean sample
    root_data = dict(Xt=test["Xt"][:120], Xa=test["Xa"][:120],
                     y=test["y"][:120], y_true=test["y"][:120],
                     miss=test["miss"][:120])
    fool_hist = np.zeros((K, _flat(P).size))
    weight_log, unc_log = [], []
    for t in range(rounds):
        lam_t = lam * min(1.0, (t + 1) / max(1, anneal))
        sel = [k for k in range(K) if rng.random() < participate] or [0]
        deltas, ns, us = [], [], []
        server_delta = None
        # server-side root update for trust-based aggregators
        if ("Trust" in agg_name) or ("Guard" in agg_name):
            sm = DualEDL(dt, da, h, C, rng, gated=gated); sm.set_params(P)
            sm.local_train(root_data, lam_t, lr, epochs, rng=rng,
                           frac_labeled=frac_labeled,
                           ssl_mode=ssl_mode if frac_labeled < 1 else None)
            server_delta = {k: sm.params()[k] - P[k] for k in P}
        for k in sel:
            data, meta = clients[k]
            lm = DualEDL(dt, da, h, C, rng, gated=gated); lm.set_params(P)
            min_unc = False; sign = False
            if k in attack_clients and attack:
                data, mode = apply_attack(lm, data, attack, rng, C)
                if mode == "sign": sign = True
                if mode == "adaptive": min_unc = True
            # quality signal: incoming global model's evidential discrepancy
            # on the client's labeled data (privacy-preserving scalar)
            u_k = lm.quality_signal(data)
            lm.local_train(data, lam_t, lr, epochs, rng=rng,
                           ssl_mode=ssl_mode if frac_labeled < 1 else None,
                           frac_labeled=frac_labeled,
                           minimize_uncertainty=min_unc)
            d = {kk: lm.params()[kk] - P[kk] for kk in P}
            # clip local update norm (stabilizes EDL + bounds attacker leverage)
            dn = np.sqrt(sum(float((v**2).sum()) for v in d.values()))
            clip = 3.0
            if dn > clip:
                d = {kk: v * (clip / dn) for kk, v in d.items()}
            if sign:
                d = {kk: -3.0 * v for kk, v in d.items()}
                u_k = 0.02  # adaptive/sign attackers report low uncertainty
            if min_unc:
                u_k = min(u_k, 0.05)
            deltas.append(d); ns.append(len(data["y"])); us.append(u_k)
            fool_hist[k] += _flat(d)
        # aggregate
        kwargs = dict(beta=beta, server_delta=server_delta,
                      hist=fool_hist[sel])
        res = agg(deltas, ns, us, **kwargs)
        if isinstance(res, tuple):
            agg_delta, w = res
            if return_dynamics:
                weight_log.append(w.copy()); unc_log.append(np.array(us))
        else:
            agg_delta = res
        for k in P: P[k] = (1.0 - wd) * P[k] + agg_delta[k]
        global_model.set_params(P)
    # evaluate
    p, u = global_model.predict(test["Xt"], test["Xa"])
    pred = p.argmax(1)
    acc = float((pred == test["y"]).mean())
    out = dict(acc=acc, model=global_model)
    if return_dynamics:
        out["weight_log"] = weight_log; out["unc_log"] = unc_log
    return out

def f1_macro(model, test, C):
    p, _ = model.predict(test["Xt"], test["Xa"])
    pred = p.argmax(1); y = test["y"]
    f1s = []
    for c in range(C):
        tp = ((pred == c) & (y == c)).sum()
        fp = ((pred == c) & (y != c)).sum()
        fn = ((pred != c) & (y == c)).sum()
        prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
        f1s.append(2 * prec * rec / (prec + rec + 1e-9))
    return float(np.mean(f1s)) * 100
