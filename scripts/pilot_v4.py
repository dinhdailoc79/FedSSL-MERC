"""
FedE-SSL Pilot v4 — The "No Excuses" Rigorous Evaluation
=========================================================
Features:
1. FAIR COMPUTE: All methods get 5 local epochs and same LR.
2. STRONG BACKBONE: FLCNN (~820K params) with GroupNorm.
3. EXTENDED AGGREGATION: CWE^2 (head + projector).
4. CLASS-WISE GATING: AEC-CPA uses per-class uncertainty.
5. MULTIPLE SEEDS & SWEEP: α = [0.01, 0.1], Seeds = [42, 123, 456].
6. BASELINES: FedAvg (Sup), FedAvg (FixMatch), FedProx (FixMatch), EAFA (EDL).
"""

import os, sys, json, copy, time, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from collections import OrderedDict

sys.path.insert(0, '.')
from fessl.models import FedESSLModel
from fessl.losses import SupervisedEvidentialLoss, AECLoss, FixMatchLoss
from fessl.server import fedavg_aggregate, eafa_aggregate, cwe_avg_aggregate
from fessl.data import get_cifar10_transforms, SSLDataset, dirichlet_partition, ssl_split
from fessl.utils import classwise_uncertainty

CFG = {
    "num_clients": 10,
    "num_rounds": 40,
    "local_epochs": 5,      # FAIR: 5 for EVERYONE
    "batch_size": 64,
    "lr": 0.01,             # FAIR: 0.01 for EVERYONE
    "num_classes": 10,
    "feature_dim": 256,     # FLCNN feature dim
    "backbone_type": "flcnn",
    "alphas": [0.01, 0.1],  # Extreme and Standard non-IID
    "seeds": [42, 123, 456],
    "label_ratio": 0.1,
    "mu_fedprox": 0.01,     # FedProx hyperparameter
    "beta_eafa": 1.0,
    "beta_cwe": 1.0,
    "lambda_proto": 0.5,
    "warmup_rounds": 5,
    "rampup_rounds": 5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

_DATA_CACHE = {}

def get_data_cached():
    global _DATA_CACHE
    if "raw" not in _DATA_CACHE:
        print("  [Cache] Loading CIFAR-10 into RAM (one-time)...")
        w_t, s_t, t_t = get_cifar10_transforms()
        raw_ds = datasets.CIFAR10("./data", train=True, download=True, transform=None)
        raw_data = [(img, tg) for img, tg in raw_ds]
        test_ds = datasets.CIFAR10("./data", train=False, download=True, transform=t_t)
        _DATA_CACHE = {"raw": raw_data, "test": test_ds, "weak_t": w_t, "strong_t": s_t, "test_t": t_t}
        print("  [Cache] Done.")
    return _DATA_CACHE

def gpu_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def train_edl(model, data, client, cfg, prototypes, rnd):
    dev = cfg["device"]
    model.train()
    d = data
    lab_ds = Subset(datasets.CIFAR10("./data", train=True, transform=d["weak_t"]), client["lab"])
    lab_ld = DataLoader(lab_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
    ssl_ds = SSLDataset(Subset(d["raw"], client["unlab"]), d["weak_t"], d["strong_t"])
    ssl_ld = DataLoader(ssl_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)

    sup_fn = SupervisedEvidentialLoss(num_classes=cfg["num_classes"], annealing_epochs=cfg["local_epochs"]*3)
    aec_fn = AECLoss(num_classes=cfg["num_classes"], lambda_proto=cfg["lambda_proto"])
    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9, weight_decay=1e-4)

    aec_w = 0.0 if rnd < cfg["warmup_rounds"] else min(1.0, (rnd - cfg["warmup_rounds"]) / cfg["rampup_rounds"]) if rnd < cfg["warmup_rounds"] + cfg["rampup_rounds"] else 1.0

    tot_loss, nb, pc, pt = 0., 0, 0, 0
    all_u, all_e = [], []

    for ep in range(cfg["local_epochs"]):
        sup_fn.set_epoch(ep)
        aec_fn.set_epoch(ep)
        ssl_it = iter(ssl_ld)
        for imgs, tgts in lab_ld:
            imgs, tgts = imgs.to(dev), tgts.to(dev)
            out = model(imgs)
            l_ce = F.cross_entropy(out["evidence"], tgts)
            l_edl, _ = sup_fn(out["alpha"], tgts)
            l_sup = l_ce + l_edl

            l_ssl = torch.tensor(0., device=dev)
            try:
                wi, si, tl, _ = next(ssl_it)
                wi, si, tl = wi.to(dev), si.to(dev), tl.to(dev)
                with torch.no_grad():
                    ow = model(wi)
                os_ = model(si)
                pl = ow["pred"]
                pc += (pl == tl).sum().item()
                pt += len(tl)
                ie = (pl != tl).float()
                
                # We collect scalar mean uncertainty for tracking, but loss uses class-wise internally
                all_u.extend(ow["uncertainty"].cpu().tolist())
                all_e.extend(ie.cpu().tolist())
                
                if aec_w > 0:
                    la, _ = aec_fn(ow["alpha"], os_["alpha"], ow["uncertainty"], prototypes=prototypes, pseudo_labels=pl)
                    l_ssl = aec_w * la
            except StopIteration:
                pass

            loss = l_sup + l_ssl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot_loss += loss.item()
            nb += 1

    model.eval()
    with torch.no_grad():
        ev_ds = Subset(datasets.CIFAR10("./data", train=True, transform=d["test_t"]), client["all"])
        ev_ld = DataLoader(ev_ds, batch_size=256, shuffle=False)
        aa = []
        for im, _ in ev_ld:
            aa.append(model(im.to(dev))["alpha"].cpu())
        aa = torch.cat(aa, 0)
        ma = aa.mean(0)
        cwu = classwise_uncertainty(ma.unsqueeze(0), cfg["num_classes"]).squeeze(0)

    gpu_cleanup()
    return model.state_dict(), {
        "loss": tot_loss/max(nb,1), "purity": pc/max(pt,1),
        "su": cwu.mean().item(), "cwu": cwu,
        "u_raw": all_u, "e_raw": all_e,
        "sz": len(client["all"]), "aec_w": aec_w,
    }

def train_baseline(model, data, client, cfg, method, global_model=None):
    dev = cfg["device"]
    model.train()
    d = data
    lab_ds = Subset(datasets.CIFAR10("./data", train=True, transform=d["weak_t"]), client["lab"])
    lab_ld = DataLoader(lab_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
    
    if "fixmatch" in method:
        ssl_ds = SSLDataset(Subset(d["raw"], client["unlab"]), d["weak_t"], d["strong_t"])
        ssl_ld = DataLoader(ssl_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
        fm_fn = FixMatchLoss(threshold=0.95)

    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9, weight_decay=1e-4)
    tot, nb, pc, pt = 0., 0, 0, 0

    for ep in range(cfg["local_epochs"]):
        if "fixmatch" in method:
            si = iter(ssl_ld)
            
        for imgs, tgts in lab_ld:
            imgs, tgts = imgs.to(dev), tgts.to(dev)
            o = model(imgs)
            loss = F.cross_entropy(o["logits"], tgts)
            
            if "fixmatch" in method:
                try:
                    wi, sti, tl, _ = next(si)
                    wi, sti, tl = wi.to(dev), sti.to(dev), tl.to(dev)
                    with torch.no_grad():
                        ow = model(wi)
                    os_ = model(sti)
                    pl = ow["pred"]
                    pc += (pl == tl).sum().item()
                    pt += len(tl)
                    lu, _ = fm_fn(os_["logits"], ow["probs"])
                    loss += lu
                except StopIteration:
                    pass
            
            if "fedprox" in method and global_model is not None:
                prox_term = 0.0
                for local_param, global_param in zip(model.parameters(), global_model.parameters()):
                    prox_term += (local_param - global_param).norm(2)
                loss += (cfg["mu_fedprox"] / 2) * prox_term

            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1

    gpu_cleanup()
    return model.state_dict(), {"loss": tot/max(nb,1), "purity": pc/max(pt,1), "su": 0.5, "sz": len(client["all"])}

@torch.no_grad()
def evaluate(model, test_ld, dev):
    model.eval()
    c, t = 0, 0
    for im, tg in test_ld:
        im, tg = im.to(dev), tg.to(dev)
        c += (model(im)["pred"] == tg).sum().item()
        t += tg.size(0)
    return c / t

def run_one(method, alpha, seed, cfg, data):
    print(f"\n{'='*60}\n  M: {method} | a={alpha} | seed={seed}\n{'='*60}")
    
    # 1. Setup seed and data partition
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    targets = [t for _, t in data["raw"]]
    partitions = dirichlet_partition(targets, cfg["num_clients"], alpha=alpha, seed=seed)
    clients = []
    for k in range(cfg["num_clients"]):
        lab, unlab = ssl_split(partitions[k], cfg["label_ratio"], seed=seed+k)
        clients.append({"all": partitions[k], "lab": lab, "unlab": unlab})
    
    # 2. Setup model
    dev = cfg["device"]
    test_ld = DataLoader(data["test"], batch_size=256, shuffle=False)
    ht = "evidential" if "edl" in method else "softmax"
    gm = FedESSLModel(
        num_classes=cfg["num_classes"], 
        feature_dim=cfg["feature_dim"], 
        head_type=ht, 
        backbone_type=cfg["backbone_type"]
    ).to(dev)

    protos = torch.ones(cfg["num_classes"], cfg["num_classes"]).to(dev) * 2.0
    for c in range(cfg["num_classes"]): protos[c, c] = 10.0

    best_acc = 0.0

    for rnd in range(cfg["num_rounds"]):
        t0 = time.time()
        states, stats = [], []
        
        for k in range(cfg["num_clients"]):
            lm = copy.deepcopy(gm)
            if "edl" in method:
                s, st = train_edl(lm, data, clients[k], cfg, protos if "cwe" in method else None, rnd)
            else:
                s, st = train_baseline(lm, data, clients[k], cfg, method, gm if "fedprox" in method else None)
            states.append(s); stats.append(st)
            del lm; gpu_cleanup()

        sizes = [s["sz"] for s in stats]
        
        # Aggregation
        if method in ("fedavg_sup", "fedavg_fixmatch", "fedprox_fixmatch"):
            g_state = fedavg_aggregate(states, sizes)
        elif method == "eafa_edl":
            su = [s["su"] for s in stats]
            g_state, _ = eafa_aggregate(states, sizes, su, beta=cfg["beta_eafa"])
        elif method == "cwe2_edl":
            cwu = [s["cwu"] for s in stats]
            g_state, _ = cwe_avg_aggregate(
                states, sizes, cwu, beta=cfg["beta_cwe"],
                head_prefix="head.", projector_prefix="backbone.projector."
            )
        
        gm.load_state_dict(g_state)
        acc = evaluate(gm, test_ld, dev)
        best_acc = max(best_acc, acc)
        
        avg_pur = np.mean([s["purity"] for s in stats])
        phase = "FULL" if rnd >= cfg["warmup_rounds"] + cfg["rampup_rounds"] else "WARM/RAMP"
        
        print(f"  R{rnd+1:02d}/{cfg['num_rounds']} | Acc: {acc:.4f} | Best: {best_acc:.4f} | Pur: {avg_pur:.4f} | {int(time.time()-t0)}s")
        
    return best_acc

if __name__ == "__main__":
    data = get_data_cached()
    results = {}
    results_path = "./pilot_results/results_v4.json"
    
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            try:
                results = json.load(f)
                print(f"  [Resume] Loaded existing results from {results_path}")
            except json.JSONDecodeError:
                pass
                
    methods = ["fedavg_sup", "fedavg_fixmatch", "fedprox_fixmatch", "eafa_edl", "cwe2_edl"]
    
    for a in CFG["alphas"]:
        a_str = str(a)
        if a_str not in results:
            results[a_str] = {}
            
        for m in methods:
            # Check if this method was already fully completed for this alpha
            if m in results[a_str] and "runs" in results[a_str][m] and len(results[a_str][m]["runs"]) == len(CFG["seeds"]):
                print(f"\n=> [SKIP] {m} @ a={a} already completed. Mean Acc: {results[a_str][m]['mean']:.4f}")
                continue
                
            accs = []
            for s in CFG["seeds"]:
                best_acc = run_one(m, a, s, CFG, data)
                accs.append(best_acc)
            
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            results[a_str][m] = {"mean": mean_acc, "std": std_acc, "runs": accs}
            
            print(f"\n=> {m} @ a={a}: {mean_acc:.4f} ± {std_acc:.4f}")
            
            # Save incrementally
            os.makedirs("./pilot_results", exist_ok=True)
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
                
    print("\n================ FINAL REPORT ================")
    print(json.dumps(results, indent=2))
