"""
Controlled federated EDL testbed (pure numpy).
Every number reported in the revised paper that is NOT from the original
real MELD/IEMOCAP runs is produced by executing this module.

Design goals (mapped to reviewer points):
  - real multi-seed statistics with CIs and Holm correction  (R4, R11)
  - adaptive attacker that keeps evidential uncertainty low   (R5)
  - FLTrust + FoolsGold baselines actually implemented        (R5)
  - scale study K in {10,50,100,200}                          (R8)
  - hard-but-honest vs noisy client fairness                  (R6)
  - shared-vs-decoupled scalar ablation                       (R20)
  - threshold-free ECR vs FixMatch hard threshold             (R-SSL)
  - evidence-level fusion (honest naming, not Dempster's rule)
"""
import numpy as np
from scipy.special import digamma, polygamma, gammaln

# ----------------------------------------------------------------------------
# Synthetic multimodal "emotion-like" data
# ----------------------------------------------------------------------------
def make_class_centers(C, d, rng, spread=2.2):
    return rng.normal(0, spread, size=(C, d))

def sample_modality(y, centers, rng, noise_scale=1.0):
    n = len(y)
    d = centers.shape[1]
    X = centers[y] + rng.normal(0, noise_scale, size=(n, d))
    return X

def make_client_data(n, C, text_centers, audio_centers, rng,
                     label_noise=0.0, hardness=1.0, audio_missing=0.0,
                     class_probs=None, center_shift=0.0):
    if class_probs is None:
        class_probs = np.ones(C) / C
    y = rng.choice(C, size=n, p=class_probs)
    tc = text_centers; ac = audio_centers
    if center_shift > 0:
        # per-client covariate shift -> genuine epistemic (OOD) uncertainty
        sv_t = rng.normal(0, center_shift, size=text_centers.shape[1])
        sv_a = rng.normal(0, center_shift, size=audio_centers.shape[1])
        tc = text_centers + sv_t; ac = audio_centers + sv_a
    Xt = sample_modality(y, tc, rng, noise_scale=hardness)
    Xa = sample_modality(y, ac, rng, noise_scale=hardness)
    miss = rng.random(n) < audio_missing
    Xa[miss] = 0.0
    y_obs = y.copy()
    if label_noise > 0:
        flip = rng.random(n) < label_noise
        y_obs[flip] = rng.choice(C, size=flip.sum())
    return dict(Xt=Xt, Xa=Xa, y=y_obs, y_true=y, miss=miss)

def dirichlet_partition(C, K, alpha, rng):
    """Return per-client class probability vectors (non-IID)."""
    return rng.dirichlet([alpha] * C, size=K)

# ----------------------------------------------------------------------------
# EDL MLP (one hidden layer) per modality, evidence-level fusion
# ----------------------------------------------------------------------------
def softplus(z):
    return np.logaddexp(0.0, z)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

def relu(z):
    return np.maximum(z, 0.0)

def init_mlp(d, h, C, rng):
    s1 = np.sqrt(2.0 / d); s2 = np.sqrt(2.0 / h)
    return dict(W1=rng.normal(0, s1, (d, h)), b1=np.zeros(h),
                W2=rng.normal(0, s2, (h, C)), b2=np.zeros(C))

def mlp_evidence(p, X):
    """Return logits z and cache for backprop. evidence = softplus(z)."""
    a1 = X @ p["W1"] + p["b1"]
    h1 = relu(a1)
    z = h1 @ p["W2"] + p["b2"]
    return z, (X, a1, h1)

def edl_loss_grad(z, y_onehot, lam):
    """
    EDL Type-II MLE loss + annealed KL(Dir(alpha_tilde)||Dir(1)).
    Returns scalar loss and dL/dz (per sample).
    """
    EV_CLIP = 30.0
    e = np.clip(softplus(z), 1e-6, EV_CLIP)
    alpha = e + 1.0
    S = alpha.sum(1, keepdims=True)
    # MLE term
    mle = (y_onehot * (digamma(S) - digamma(alpha))).sum(1)
    # KL term on alpha_tilde = y + (1-y)*alpha
    at = y_onehot + (1 - y_onehot) * alpha
    Sat = at.sum(1, keepdims=True)
    C = z.shape[1]
    kl = (gammaln(Sat[:, 0]) - gammaln(at).sum(1)
          - gammaln(float(C))
          + ((at - 1.0) * (digamma(at) - digamma(Sat))).sum(1))
    loss = (mle + lam * kl).mean()

    # grad of MLE wrt alpha_j: psi'(S) - y_j psi'(alpha_j)
    dmle_da = polygamma(1, S) - y_onehot * polygamma(1, alpha)
    # grad of KL wrt a_j: (a_j-1)(psi'(a_j)-psi'(Sat)) - psi'(Sat)*sum(a-1)
    sum_am1 = (at - 1.0).sum(1, keepdims=True)
    dkl_da = (at - 1.0) * (polygamma(1, at) - polygamma(1, Sat)) \
             - polygamma(1, Sat) * sum_am1
    dkl_dalpha = dkl_da * (1 - y_onehot)  # da/dalpha = (1-y)
    dL_dalpha = (dmle_da + lam * dkl_dalpha) / z.shape[0]
    # dalpha/dz = sigmoid(z) (softplus'); clipped region grad ~ included
    dL_dz = dL_dalpha * sigmoid(z)
    return loss, dL_dz

def mlp_backward(p, cache, dL_dz):
    X, a1, h1 = cache
    grads = {}
    grads["W2"] = h1.T @ dL_dz
    grads["b2"] = dL_dz.sum(0)
    dh1 = dL_dz @ p["W2"].T
    da1 = dh1 * (a1 > 0)
    grads["W1"] = X.T @ da1
    grads["b1"] = da1.sum(0)
    return grads

def onehot(y, C):
    o = np.zeros((len(y), C)); o[np.arange(len(y)), y] = 1.0; return o

# ----------------------------------------------------------------------------
# A two-modality EDL model = text head + audio head, evidence summed
# ----------------------------------------------------------------------------
class DualEDL:
    def __init__(self, dt, da, h, C, rng, gated=False):
        self.t = init_mlp(dt, h, C, rng)
        self.a = init_mlp(da, h, C, rng)
        self.C = C
        self.gated = gated

    def evidence(self, Xt, Xa):
        zt, ct = mlp_evidence(self.t, Xt)
        za, ca = mlp_evidence(self.a, Xa)
        et = np.clip(softplus(zt), 0, 30.0)
        ea = np.clip(softplus(za), 0, 30.0)
        if self.gated:
            # gate audio by its own certainty (1 - vacuity)
            Sa = ea.sum(1, keepdims=True) + self.C
            g = 1.0 - self.C / Sa  # in (0,1)
            ea = ea * g
        e = et + ea
        return e, (zt, ct, za, ca, et, ea)

    def predict(self, Xt, Xa):
        e, _ = self.evidence(Xt, Xa)
        alpha = e + 1.0
        S = alpha.sum(1, keepdims=True)
        p = alpha / S
        u = self.C / S[:, 0]  # vacuity (epistemic) uncertainty
        return p, u

    def quality_signal(self, data):
        """Privacy-preserving scalar a client can report: mean evidential
        Bayes-risk discrepancy of the (incoming global) model on its labeled
        data. High when the consensus model's evidence contradicts the client's
        labels, i.e. label noise / OOD / adversarial. Combines vacuity and
        disagreement, so it tracks quality more reliably than raw vacuity."""
        Xt, Xa, y = data["Xt"], data["Xa"], data["y"]
        e, _ = self.evidence(Xt, Xa)
        alpha = e + 1.0
        S = alpha.sum(1, keepdims=True)
        p = alpha / S
        yo = onehot(y, self.C)
        # expected Brier under Dirichlet (mean) + vacuity term
        brier = ((p - yo) ** 2).sum(1)
        vac = self.C / S[:, 0]
        return float((0.5 * brier + 0.5 * vac).mean())

    def params(self):
        return {("t_" + k): v for k, v in self.t.items()} | \
               {("a_" + k): v for k, v in self.a.items()}

    def set_params(self, P):
        for k in self.t: self.t[k] = P["t_" + k].copy()
        for k in self.a: self.a[k] = P["a_" + k].copy()

    def local_train(self, data, lam, lr, epochs, batch=64, rng=None,
                    use_unlabeled=False, ssl_mode=None, frac_labeled=1.0,
                    minimize_uncertainty=False):
        """One client's local update. Returns mean labeled uncertainty.
        ssl_mode in {None,'fixmatch','ecr'}; frac_labeled controls SSL.
        minimize_uncertainty: adaptive attacker objective add-on."""
        Xt, Xa, y = data["Xt"], data["Xa"], data["y"]
        n = len(y)
        idx = np.arange(n)
        nlab = max(1, int(frac_labeled * n))
        lab_mask = np.zeros(n, bool); lab_mask[:nlab] = True
        for ep in range(epochs):
            rng.shuffle(idx)
            for s in range(0, n, batch):
                bi = idx[s:s + batch]
                self._step(Xt[bi], Xa[bi], y[bi], lab_mask[bi], lam, lr,
                           ssl_mode, minimize_uncertainty)
        # report mean vacuity on (labeled) data
        p, u = self.predict(Xt[lab_mask], Xa[lab_mask])
        return float(u.mean())

    def _step(self, Xt, Xa, y, lab, lam, lr, ssl_mode, min_unc):
        C = self.C
        e, (zt, ct, za, ca, et, ea) = self.evidence(Xt, Xa)
        yo = onehot(y, C)
        alpha = e + 1.0
        S = alpha.sum(1, keepdims=True)
        loss, dL_dalpha = self._edl_grad_fused(e, yo, lab, lam)
        if min_unc:
            dL_dalpha = dL_dalpha - 0.5 * (C / S**2) * np.ones_like(alpha)
        dL_de = dL_dalpha
        gnorm = np.linalg.norm(dL_de, axis=1, keepdims=True) + 1e-12
        dL_de = dL_de * np.minimum(1.0, 1.0 / gnorm)
        dL_dzt = dL_de * sigmoid(zt)
        dL_dza = dL_de * sigmoid(za)
        gt = mlp_backward(self.t, ct, dL_dzt)
        ga = mlp_backward(self.a, ca, dL_dza)
        for k in self.t: self.t[k] -= lr * gt[k]
        for k in self.a: self.a[k] -= lr * ga[k]
        if ssl_mode and ssl_mode.split(":")[0] in ("fixmatch", "ecr") and (~lab).any():
            self._ssl_step(Xt[~lab], Xa[~lab], lam, lr, ssl_mode)

    def _ssl_step(self, Xt, Xa, lam, lr, mode, sigma=0.25):
        """Pseudo-label from weak (clean) view; train a strong (noised) view.
        fixmatch: keep only conf>=0.95 (hard threshold). ecr: certainty-weighted
        (1-u)^2 soft weight, threshold-free."""
        C = self.C
        rng = np.random.default_rng()
        ew, _ = self.evidence(Xt, Xa)
        aw = ew + 1.0; Sw = aw.sum(1, keepdims=True)
        pw = aw / Sw; uw = C / Sw
        pl = pw.argmax(1); conf = pw.max(1, keepdims=True)
        target = onehot(pl, C)
        Xts = Xt + rng.normal(0, sigma, Xt.shape)
        Xas = Xa + rng.normal(0, sigma, Xa.shape)
        es, (zt, ct, za, ca, _, _) = self.evidence(Xts, Xas)
        a_s = es + 1.0; Ss = a_s.sum(1, keepdims=True)
        if mode.startswith("fixmatch"):
            thr = float(mode.split(":")[1]) if ":" in mode else 0.95
            w = (conf >= thr).astype(float)
        else:
            w = (1.0 - uw)  # evidential certainty, threshold-free
        dL_dalpha = (polygamma(1, Ss) - target * polygamma(1, a_s)) * w
        coef = 1.2 / max(1, len(Xt))
        dL_de = dL_dalpha * coef
        gn = np.linalg.norm(dL_de, axis=1, keepdims=True) + 1e-12
        dL_de = dL_de * np.minimum(1.0, 1.0 / gn)
        dL_dzt = dL_de * sigmoid(zt); dL_dza = dL_de * sigmoid(za)
        gt = mlp_backward(self.t, ct, dL_dzt)
        ga = mlp_backward(self.a, ca, dL_dza)
        for k in self.t: self.t[k] -= lr * gt[k]
        for k in self.a: self.a[k] -= lr * ga[k]


    def _edl_grad_fused(self, e, yo, lab, lam):
        C = self.C
        alpha = e + 1.0
        S = alpha.sum(1, keepdims=True)
        n = max(1, lab.sum())
        w = lab.astype(float)[:, None]
        dmle = (polygamma(1, S) - yo * polygamma(1, alpha)) * w
        at = yo + (1 - yo) * alpha
        Sat = at.sum(1, keepdims=True)
        sum_am1 = (at - 1.0).sum(1, keepdims=True)
        dkl = ((at - 1.0) * (polygamma(1, at) - polygamma(1, Sat))
               - polygamma(1, Sat) * sum_am1) * (1 - yo) * w
        loss = 0.0
        return loss, (dmle + lam * dkl) / n

    def _ssl_grad(self, e, unlab, mode):
        """Consistency on unlabeled rows. Pseudo-label from current evidence.
        fixmatch: hard threshold 0.95 on prob, hard target.
        ecr: certainty-weighted KL toward sharpened target (threshold-free)."""
        C = self.C
        alpha = e + 1.0
        S = alpha.sum(1, keepdims=True)
        p = alpha / S
        u = C / S  # vacuity
        conf = p.max(1, keepdims=True)
        pl = p.argmax(1)
        target = onehot(pl, C)
        w = unlab.astype(float)[:, None]
        if mode == "fixmatch":
            keep = (conf >= 0.95).astype(float)
            g = (polygamma(1, S) - target * polygamma(1, alpha))
            return 0.1 * g * w * keep / max(1, unlab.sum())
        else:  # ecr: weight by certainty (1-u), no threshold
            cw = (1.0 - u)
            g = (polygamma(1, S) - target * polygamma(1, alpha))
            return 0.1 * g * w * cw / max(1, unlab.sum())


# ----------------------------------------------------------------------------
# Aggregators. Each takes list of (delta_dict, n_k, u_k) -> aggregated delta
# ----------------------------------------------------------------------------
def _flat(d):
    return np.concatenate([v.ravel() for k, v in sorted(d.items())])

def _unflat(template, vec):
    out = {}; i = 0
    for k in sorted(template):
        sz = template[k].size
        out[k] = vec[i:i + sz].reshape(template[k].shape); i += sz
    return out

def agg_fedavg(deltas, ns, us, **kw):
    w = np.array(ns, float); w /= w.sum()
    return _weighted(deltas, w)

def agg_eafa(deltas, ns, us, beta=4.0, **kw):
    w = np.array(ns, float) * np.exp(-beta * np.array(us))
    w /= w.sum()
    return _weighted(deltas, w), w

def _adjacent_class_cosine_np(W):
    """Compute cosine similarity for each adjacent class pair (c, c+1) % C.

    Args:
        W: weight matrix of shape [C, H] or [H, C]. Detects orientation automatically.
    Returns:
        Array of shape [C] with cosine values.
    """
    if W.shape[0] <= W.shape[1]:
        W = W.T  # [H, C] -> [C, H]
    W = W.astype(np.float64)
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-12
    W_norm = W / norms
    C = W.shape[0]
    return np.array([W_norm[c] @ W_norm[(c + 1) % C] for c in range(C)])


def _get_text_head_w2(state):
    """Extract text classifier head W2 from a model state dict. Returns [H, C] or None."""
    if "t_W2" in state:
        return state["t_W2"]
    return None


def compute_lf_scores_np(deltas, global_state, num_attackers=0):
    """Compute per-client label-flip suspicion scores (NumPy version).

    Reconstructs full client params (global + delta) and compares the
    classifier head weight drift against the global baseline.
    Score = mean(cos(W_c, W_{c+1}) - baseline_cos_c) over all classes.
    Positive = adjacent class weights drifted toward each other (label-flip signal).
    """
    global_head = _get_text_head_w2(global_state)
    if global_head is None:
        return np.zeros(len(deltas))

    baseline = _adjacent_class_cosine_np(global_head)
    scores = []
    for d in deltas:
        # Reconstruct full client params: theta_client = theta_global + delta
        delta_head = _get_text_head_w2(d)
        global_w2 = global_head
        if delta_head is None:
            scores.append(0.0)
            continue
        client_w2 = global_w2 + delta_head  # full client classifier head
        client_cos = _adjacent_class_cosine_np(client_w2)
        scores.append(float(np.mean(client_cos - baseline)))
    return np.array(scores)


def agg_eafa_guard(deltas, ns, us, server_delta=None, beta=4.0,
                   use_lf_guard=False, num_attackers=0, **kw):
    """EAFA-Guard: server-root direction filter + magnitude cap + evidential
    quality weight. Robust to update poisoning (incl. adaptive) because the
    median-cosine filter and the norm cap bound an attacker's leverage even when
    it spoofs a low quality scalar; the evidential weight still rewards honest
    high-quality clients among the survivors.

    With use_lf_guard=True, also applies the Label-Flip Detector which catches
    label-flip attacks that survive the direction filter by analyzing classifier
    head weight drift (adjacent class cosine similarity)."""
    if server_delta is None:
        return agg_eafa(deltas, ns, us, beta=beta, **kw)
    sv = _flat(server_delta); svn = np.linalg.norm(sv) + 1e-12
    V = np.stack([_flat(d) for d in deltas]); K = len(deltas)
    norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    cs = (V @ sv) / norms[:, 0] / svn               # cosine to clean root
    med = np.median(cs)
    keep = cs >= med                                 # drop lower half by direction
    trust = np.maximum(0.0, cs) ** 2                 # sharpened trust
    ev = np.exp(-beta * np.array(us))                # evidential quality weight
    w = np.where(keep, trust * ev * np.array(ns, float), 0.0)

    # --- Label-Flip Detector ---
    # Note: kw may contain 'global_state' passed from fed_train
    lf_keep = np.ones(K, dtype=bool)
    lf_scores = np.zeros(K)
    lf_threshold = 0.0
    if use_lf_guard:
        global_state = kw.get('global_state', server_delta)  # prefer true global params
        lf_scores = compute_lf_scores_np(deltas, global_state, num_attackers=num_attackers)
        surviving_scores = lf_scores[keep]
        # Only apply LF filter if we have enough survivors AND VERY CLEAR label-flip signature
        # Label-flip creates large POSITIVE drift (> 0.1)
        # Sign-flip/adative typically create negative or near-zero drift
        if len(surviving_scores) >= 2:
            max_score = float(np.max(surviving_scores))
            # Very conservative: only filter if score > 0.1 (very clear label-flip signature)
            # This prevents false positives on sign-flip/adaptive attacks
            if max_score > 0.1:
                lf_threshold = 0.0  # Filter only positive outliers
                lf_keep = lf_scores <= lf_threshold
                keep = keep & lf_keep
    # --- End Label-Flip Detector ---

    # cap magnitudes to the median survivor norm to bound attacker leverage
    cap = np.median(norms[keep, 0]) if keep.sum() > 0 else norms[:, 0].max()
    scale = np.minimum(1.0, cap / norms[:, 0])
    if w.sum() <= 0:
        w = np.array(ns, float)
    w = w / w.sum()
    agg = ((w * scale)[:, None] * V).sum(0)
    return _unflat(deltas[0], agg), w

def agg_krum(deltas, ns, us, f=1, multi=False, **kw):
    V = np.stack([_flat(d) for d in deltas]); K = len(deltas)
    sc = []
    m = max(1, K - f - 2)
    for i in range(K):
        dist = np.sum((V - V[i])**2, 1); dist[i] = np.inf
        sc.append(np.sort(dist)[:m].sum())
    sc = np.array(sc)
    if multi:
        sel = np.argsort(sc)[:max(1, K - f)]
    else:
        sel = [int(np.argmin(sc))]
    w = np.zeros(K); w[sel] = 1.0 / len(sel)
    return _weighted(deltas, w)

def agg_trimmed_mean(deltas, ns, us, trim=1, **kw):
    V = np.stack([_flat(d) for d in deltas])
    Vs = np.sort(V, 0)
    if 2 * trim < len(deltas):
        Vt = Vs[trim:len(deltas) - trim].mean(0)
    else:
        Vt = Vs.mean(0)
    return _unflat(deltas[0], Vt)

def agg_median(deltas, ns, us, **kw):
    V = np.stack([_flat(d) for d in deltas])
    return _unflat(deltas[0], np.median(V, 0))

def agg_fltrust(deltas, ns, us, server_delta=None, **kw):
    sv = _flat(server_delta); svn = np.linalg.norm(sv) + 1e-12
    w = []
    for d in deltas:
        dv = _flat(d); cs = (dv @ sv) / (np.linalg.norm(dv) + 1e-12) / svn
        ts = max(0.0, cs)
        # normalize client update to server norm
        w.append(ts)
    w = np.array(w)
    # FLTrust normalizes each update to server norm; emulate by scaling
    scaled = []
    for d, ts in zip(deltas, w):
        dv = _flat(d); dn = np.linalg.norm(dv) + 1e-12
        scaled.append(_unflat(deltas[0], dv * (svn / dn)))
    if w.sum() <= 0: w = np.ones(len(deltas))
    w = w / w.sum()
    return _weighted(scaled, w)

def agg_foolsgold(deltas, ns, us, hist=None, **kw):
    """Down-weight Sybil-correlated updates via pairwise cosine of histories."""
    V = np.stack([_flat(d) for d in deltas]); K = len(deltas)
    if hist is None: hist = V
    H = hist / (np.linalg.norm(hist, axis=1, keepdims=True) + 1e-12)
    cs = H @ H.T; np.fill_diagonal(cs, 0.0)
    maxcs = cs.max(1)
    # pardoning + logit reweighting (Fung et al.)
    alpha = 1.0 - maxcs
    alpha = np.clip(alpha, 1e-3, 1.0)
    alpha = alpha / alpha.max()
    eps = 1e-5
    wlog = np.log(alpha / (1 - alpha + eps) + eps)
    wlog = np.clip(wlog, -5, 5)
    w = 1.0 / (1.0 + np.exp(-wlog))
    w = np.clip(w, 0, 1)
    if w.sum() <= 0: w = np.ones(K)
    w = w / w.sum()
    return _weighted(deltas, w)

def _weighted(deltas, w):
    out = {k: np.zeros_like(v) for k, v in deltas[0].items()}
    for d, wi in zip(deltas, w):
        for k in out: out[k] += wi * d[k]
    return out
