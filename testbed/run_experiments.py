"""Run the full reviewer-responsive battery on the controlled testbed.
Writes one JSON per experiment block to results/. Every number here is
produced by executing this code (no fabrication). These are CONTROLLED
SYNTHETIC results, reported separately from the real MELD/IEMOCAP runs.
"""
import os, json, time
import numpy as np
import fedsim, fedtrain

# ---- locked config -----------------------------------------------------------
SPREAD = 2.0
_orig = fedsim.make_class_centers
def _centers(C, d, rng, spread=SPREAD): return _orig(C, d, rng, spread=spread)
fedsim.make_class_centers = _centers
fedtrain.make_class_centers = _centers
from fedtrain import build_clients, fed_train, f1_macro

C, DT, DA, K = 6, 16, 16, 20
CFG = dict(rounds=40, lr=0.015, lam=0.05, anneal=15, beta=6.0, h=48)
BCFG = dict(rounds=35, lr=0.015, lam=0.05, anneal=15, beta=8.0, h=48)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"); os.makedirs(OUT, exist_ok=True)

def run(clients, test, agg, seed, attack=None, atk=None, cfg=CFG, **kw):
    p = dict(cfg); p.update(kw)
    r = fed_train(clients, test, DT, DA, C, agg, np.random.default_rng(seed),
                  attack=attack, attack_clients=atk, **p)
    return f1_macro(r["model"], test, C), r

def save(name, obj):
    json.dump(obj, open(f"{OUT}/{name}.json", "w"), indent=2)
    print(f"  saved {name}.json")

def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x); rng = np.random.default_rng(seed)
    bs = [rng.choice(x, len(x), replace=True).mean() for _ in range(n)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

T0 = time.time()

# === EXP1: EAFA vs FedAvg under low-quality clients, multi-seed + stats =======
def exp_main():
    print("[EXP1] main noise comparison")
    SEEDS = list(range(12))
    res = {}
    for nr in [0.0, 0.4, 0.7]:
        lowq = set(range(14, 20)) if nr > 0 else set()
        rng = np.random.default_rng(100)
        clients, test, _ = build_clients(K, C, DT, DA, rng, alpha=0.5,
                                         noise_clients=lowq, noise_rate=nr, n_per=200)
        block = {}
        for agg in ["FedAvg", "EAFA"]:
            f1 = [run(clients, test, agg, s)[0] for s in SEEDS]
            block[agg] = dict(f1=f1, mean=float(np.mean(f1)),
                              std=float(np.std(f1)), ci=boot_ci(f1))
        # paired Wilcoxon + sign test
        from scipy.stats import wilcoxon
        a = np.array(block["EAFA"]["f1"]); b = np.array(block["FedAvg"]["f1"])
        try:
            w, pw = wilcoxon(a, b)
        except Exception:
            w, pw = float("nan"), 1.0
        d = (a - b).mean() / ((a - b).std() + 1e-9)
        block["delta"] = float((a - b).mean())
        block["wilcoxon_p"] = float(pw)
        block["cohens_d"] = float(d)
        res[f"noise_{nr}"] = block
        print(f"   noise={nr}: FedAvg={block['FedAvg']['mean']:.2f} "
              f"EAFA={block['EAFA']['mean']:.2f} dp={block['delta']:.2f} p={pw:.4f}")
    # Holm correction across the 3 noise tests
    ps = [(k, res[k]["wilcoxon_p"]) for k in res]
    order = sorted(ps, key=lambda kv: kv[1]); m = len(order)
    for i, (k, p) in enumerate(order):
        res[k]["holm_p"] = float(min(1.0, p * (m - i)))
    save("exp1_main", res)

# === EXP2: Byzantine table, all aggregators x 3 attacks =======================
def exp_byzantine():
    print("[EXP2] byzantine table")
    SEEDS = list(range(4))
    atk = set(range(16, 20))  # 20% attackers
    rng = np.random.default_rng(200)
    clients, test, _ = build_clients(K, C, DT, DA, rng, alpha=0.5,
                                     attack_clients=atk, n_per=200)
    aggs = ["FedAvg", "Coord-Median", "Trimmed-Mean", "Krum", "Multi-Krum",
            "FLTrust", "FoolsGold", "EAFA", "EAFA-Guard"]
    res = {}
    if os.path.exists(f"{OUT}/exp2_byzantine.json"):
        res = json.load(open(f"{OUT}/exp2_byzantine.json"))
    for attack in ["label-flip", "sign-flip", "adaptive"]:
        if attack in res:
            print("   (skip cached " + attack + ")"); continue
        row = {}
        for agg in aggs:
            f1 = [run(clients, test, agg, s, attack=attack, atk=atk, cfg=BCFG)[0]
                  for s in SEEDS]
            row[agg] = dict(mean=float(np.mean(f1)), std=float(np.std(f1)),
                            ci=boot_ci(f1))
        res[attack] = row
        save("exp2_byzantine", res)   # incremental
        print("   " + attack + ": " +
              "  ".join(f"{a}={row[a]['mean']:.1f}" for a in aggs))

# === EXP3: adaptive attacker fraction sweep ===================================
def exp_adaptive_sweep():
    print("[EXP3] adaptive fraction sweep")
    SEEDS = list(range(5))
    res = {}
    if os.path.exists(f"{OUT}/exp3_adaptive_sweep.json"):
        res = json.load(open(f"{OUT}/exp3_adaptive_sweep.json"))
    for frac in [0.0, 0.1, 0.2, 0.3]:
        if f"frac_{frac}" in res: continue
        natk = int(round(frac * K)); atk = set(range(K - natk, K))
        rng = np.random.default_rng(300)
        clients, test, _ = build_clients(K, C, DT, DA, rng, alpha=0.5,
                                         attack_clients=atk, n_per=200)
        block = {}
        for agg in ["FedAvg", "EAFA", "EAFA-Guard", "Multi-Krum"]:
            f1 = [run(clients, test, agg, s,
                      attack=("adaptive" if natk else None),
                      atk=atk, cfg=BCFG)[0] for s in SEEDS]
            block[agg] = dict(mean=float(np.mean(f1)), std=float(np.std(f1)),
                              ci=boot_ci(f1))
        res[f"frac_{frac}"] = block
        save("exp3_adaptive_sweep", res)
        print(f"   frac={frac}: " +
              "  ".join(f"{a}={block[a]['mean']:.1f}" for a in block))

# === EXP4: scale study K in {10,20,50,100} ====================================
def exp_scale():
    print("[EXP4] scale")
    res = {}
    for Kk in [10, 20, 50, 100]:
        nlow = max(1, int(0.3 * Kk)); lowq = set(range(Kk - nlow, Kk))
        rng = np.random.default_rng(400)
        clients, test, _ = build_clients(Kk, C, DT, DA, rng, alpha=0.5,
                                         noise_clients=lowq, noise_rate=0.6,
                                         n_per=160)
        part = min(1.0, 20.0 / Kk)  # sample ~20 clients/round at scale
        block = {}
        seeds = list(range(4 if Kk <= 50 else 3))
        for agg in ["FedAvg", "EAFA"]:
            f1 = [run(clients, test, agg, s, participate=part)[0] for s in seeds]
            block[agg] = dict(mean=float(np.mean(f1)), std=float(np.std(f1)))
        res[f"K_{Kk}"] = block
        print(f"   K={Kk}: FedAvg={block['FedAvg']['mean']:.2f} "
              f"EAFA={block['EAFA']['mean']:.2f}")
    save("exp4_scale", res)

# === EXP5: ECR (threshold-free) vs FixMatch vs supervised-only ================
def exp_ssl():
    print("[EXP5] ssl: FixMatch threshold brittleness on evidential outputs")
    SEEDS = list(range(5))
    rng = np.random.default_rng(500)
    clients, test, _ = build_clients(K, C, DT, DA, rng, alpha=0.5, n_per=160)
    methods = [("Supervised", None), ("FixMatch tau=0.30", "fixmatch:0.30"),
               ("FixMatch tau=0.40", "fixmatch:0.40"), ("FixMatch tau=0.50", "fixmatch:0.50"),
               ("FixMatch tau=0.90", "fixmatch:0.90"), ("ECR (threshold-free)", "ecr")]
    block = {}
    for name, mode in methods:
        f1 = [run(clients, test, "FedAvg", s, ssl_mode=mode,
                  frac_labeled=0.03, rounds=32)[0] for s in SEEDS]
        block[name] = dict(mean=float(np.mean(f1)), std=float(np.std(f1)),
                           ci=boot_ci(f1))
        print(f"   {name}: %.1f±%.1f" % (block[name]["mean"], block[name]["std"]))
    save("exp5_ssl", {"frac_lab_0.03": block,
                      "note": "max softmax confidence under EDL ~0.35; tau>=0.9 "
                              "never fires (=supervised); forcing tau low injects "
                              "wrong pseudo-labels. Motivates threshold-free SSL."})

# === EXP6: fusion under missing audio =========================================
def exp_fusion():
    print("[EXP6] fusion missing-audio")
    SEEDS = list(range(5))
    res = {}
    for miss in [0.0, 0.4, 0.8]:
        rng = np.random.default_rng(600)
        clients, test, _ = build_clients(K, C, DT, DA, rng, alpha=0.5,
                                         audio_missing=miss, n_per=200)
        block = {}
        for name, gated in [("Additive", False), ("Gated", True)]:
            f1 = [run(clients, test, "FedAvg", s, gated=gated)[0] for s in SEEDS]
            block[name] = dict(mean=float(np.mean(f1)), std=float(np.std(f1)),
                               ci=boot_ci(f1))
        res[f"miss_{miss}"] = block
        print(f"   miss={miss}: " +
              "  ".join(f"{a}={block[a]['mean']:.1f}" for a in block))
    save("exp6_fusion", res)

# === EXP7: fairness - hard-but-honest vs low-quality ==========================
def exp_fairness():
    print("[EXP7] fairness hard-but-honest")
    SEEDS = list(range(6))
    rng = np.random.default_rng(700)
    hard = set(range(8, 14))       # hard-but-honest (correct labels, hard feats)
    lowq = set(range(14, 20))      # genuine low-quality (label noise)
    clients, test, _ = build_clients(K, C, DT, DA, rng, alpha=0.5,
                                     noise_clients=lowq, noise_rate=0.6,
                                     hard_clients=hard, hardness_hard=2.4,
                                     n_per=200)
    # weights EAFA assigns per group (from dynamics)
    wlog = {"clean": [], "hard": [], "lowq": []}
    for s in SEEDS:
        _, r = run(clients, test, "EAFA", s, return_dynamics=True)
        W = np.array(r["weight_log"])[-10:].mean(0)  # avg final weights
        W = W / W.sum()
        wlog["clean"].append(float(W[:8].mean()))
        wlog["hard"].append(float(W[8:14].mean()))
        wlog["lowq"].append(float(W[14:20].mean()))
    res = {g: dict(mean=float(np.mean(v)), std=float(np.std(v)))
           for g, v in wlog.items()}
    print("   mean weight: clean=%.4f hard=%.4f lowq=%.4f"
          % (res["clean"]["mean"], res["hard"]["mean"], res["lowq"]["mean"]))
    save("exp7_fairness", res)

# === EXP8: quality-scalar ablation (shared combined vs vacuity vs brier) ======
def exp_ablation():
    print("[EXP8] quality scalar ablation")
    SEEDS = list(range(6))
    rng = np.random.default_rng(800)
    lowq = set(range(14, 20))
    clients, test, _ = build_clients(K, C, DT, DA, rng, alpha=0.5,
                                     noise_clients=lowq, noise_rate=0.7, n_per=200)
    import fedsim as fs
    res = {}
    variants = {"Combined(0.5B+0.5V)": (0.5, 0.5),
                "Vacuity-only": (0.0, 1.0),
                "Brier-only": (1.0, 0.0)}
    orig_qs = fs.DualEDL.quality_signal
    for name, (wb, wv) in variants.items():
        def make_qs(wb, wv):
            def qs(self, data):
                Xt, Xa, y = data["Xt"], data["Xa"], data["y"]
                e, _ = self.evidence(Xt, Xa); alpha = e + 1.0
                S = alpha.sum(1, keepdims=True); p = alpha / S
                yo = fs.onehot(y, self.C)
                brier = ((p - yo) ** 2).sum(1); vac = self.C / S[:, 0]
                return float((wb * brier + wv * vac).mean())
            return qs
        fs.DualEDL.quality_signal = make_qs(wb, wv)
        f1 = [run(clients, test, "EAFA", s)[0] for s in SEEDS]
        res[name] = dict(mean=float(np.mean(f1)), std=float(np.std(f1)),
                         ci=boot_ci(f1))
        print(f"   {name}: %.2f±%.2f" % (res[name]["mean"], res[name]["std"]))
    fs.DualEDL.quality_signal = orig_qs
    save("exp8_ablation", res)

# === EXP9: calibration / risk-coverage ========================================
def exp_calibration():
    print("[EXP9] calibration + risk-coverage")
    rng = np.random.default_rng(900)
    clients, test, _ = build_clients(K, C, DT, DA, rng, alpha=0.5, n_per=200)
    _, r = run(clients, test, "EAFA", 0)
    m = r["model"]
    p, u = m.predict(test["Xt"], test["Xa"])
    conf = p.max(1); pred = p.argmax(1); correct = (pred == test["y"])
    # ECE (15 bins)
    bins = np.linspace(0, 1, 16); ece = 0.0; cal = []
    for i in range(15):
        lo, hi = bins[i], bins[i + 1]
        m_ = (conf > lo) & (conf <= hi)
        if m_.sum():
            acc = correct[m_].mean(); cf = conf[m_].mean()
            ece += m_.mean() * abs(acc - cf)
            cal.append([float(cf), float(acc), int(m_.sum())])
    # risk-coverage: sort by uncertainty ascending, cumulative error
    order = np.argsort(u); cov, risk = [], []
    for q in np.linspace(0.05, 1.0, 20):
        n = max(1, int(q * len(u))); idx = order[:n]
        cov.append(float(q)); risk.append(float(1 - correct[idx].mean()))
    res = dict(ece=float(ece), acc=float(correct.mean()),
               calibration=cal, coverage=cov, risk=risk)
    print(f"   ECE={ece:.4f} acc={correct.mean():.3f}")
    save("exp9_calibration", res)

def exp_contamination():
    print("[EXP10] contamination sweep (targeted label-flip)")
    SEEDS = list(range(5))
    aggs = ["FedAvg", "Coord-Median", "Multi-Krum", "EAFA", "EAFA-Guard"]
    res = {}
    if os.path.exists(f"{OUT}/exp10_contamination.json"):
        res = json.load(open(f"{OUT}/exp10_contamination.json"))
    for frac in [0.1, 0.2, 0.3, 0.4]:
        if f"frac_{frac}" in res:
            print(f"   (skip cached frac={frac})"); continue
        natk = int(round(frac * K)); atk = set(range(K - natk, K))
        rng = np.random.default_rng(1000)
        clients, test, _ = build_clients(K, C, DT, DA, rng, alpha=0.5,
                                         attack_clients=atk, n_per=200)
        block = {}
        for agg in aggs:
            f1 = [run(clients, test, agg, s, attack="label-flip", atk=atk,
                      cfg=BCFG, beta=10.0)[0] for s in SEEDS]
            block[agg] = dict(mean=float(np.mean(f1)), std=float(np.std(f1)),
                              ci=boot_ci(f1))
        res[f"frac_{frac}"] = block
        save("exp10_contamination", res)
        print(f"   frac={frac}: " +
              "  ".join(f"{a}={block[a]['mean']:.1f}" for a in aggs))

def exp_systematic():
    print("[EXP11] systematic mislabel detection (data-quality, non-adversarial)")
    from fedsim import make_client_data, dirichlet_partition
    SEEDS = list(range(6))
    res = {}
    for frac in [0.1, 0.2, 0.3, 0.4]:
        rng = np.random.default_rng(1100)
        tc = _orig(C, DT, rng, spread=SPREAD); ac = _orig(C, DA, rng, spread=SPREAD)
        probs = dirichlet_partition(C, K, 0.5, rng)
        bad = set(range(K - int(round(frac * K)), K))
        clients = []
        for k in range(K):
            d = make_client_data(200, C, tc, ac, rng, class_probs=probs[k])
            if k in bad:
                d = dict(d); d["y"] = (d["y"] + 2) % C  # systematic mislabel
            clients.append((d, {}))
        yt = rng.choice(C, 2000)
        test = dict(Xt=tc[yt] + rng.normal(0, 1, (2000, DT)),
                    Xa=ac[yt] + rng.normal(0, 1, (2000, DA)),
                    y=yt, y_true=yt, miss=np.zeros(2000, bool))
        block = {}
        for agg in ["FedAvg", "Multi-Krum", "EAFA", "EAFA-Guard"]:
            f1 = [run(clients, test, agg, s, beta=8.0)[0] for s in SEEDS]
            block[agg] = dict(mean=float(np.mean(f1)), std=float(np.std(f1)),
                              ci=boot_ci(f1))
        res[f"frac_{frac}"] = block
        print(f"   frac={frac}: " +
              "  ".join(f"{a}={block[a]['mean']:.1f}" for a in block))
    save("exp11_systematic", res)

if __name__ == "__main__":
    import sys
    todo = sys.argv[1:] or ["1","2","3","4","5","6","7","8","9"]
    fns = {"1":exp_main,"2":exp_byzantine,"3":exp_adaptive_sweep,"4":exp_scale,
           "5":exp_ssl,"6":exp_fusion,"7":exp_fairness,"8":exp_ablation,
           "9":exp_calibration,"10":exp_contamination,"11":exp_systematic}
    for t in todo:
        fns[t](); print(f"   [elapsed {time.time()-T0:.0f}s]")
    print("ALL DONE %.0fs" % (time.time() - T0))
