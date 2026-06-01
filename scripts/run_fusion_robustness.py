"""
DS Fusion Missing-Modality Robustness Test
==============================================
Compare 3 fusion strategies under simulated missing modalities:
  1. Concat: Feature concatenation → linear → EDL head
  2. Attention: Cross-modal attention → EDL head
  3. DS Fusion (Ours): Evidence-level Dempster-Shafer combination

Missing modality simulation:
  At test time, randomly zero-out audio features for X% of utterances.
  Missing rates: 0%, 20%, 40%, 60%, 80%, 100%

All methods trained on MELD (text+audio, full modality) then evaluated
with simulated missing audio at increasing rates.

Experiments: 3 methods × 6 missing rates × 3 seeds = 54 runs
Results saved to results_fusion_robustness.json

Usage:
    cd D:\\OJT\\FedSSL-MERC
    python scripts/run_fusion_robustness.py
"""

import sys, os, json, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_FILE = "results_fusion_robustness.json"
SEEDS = [42, 123, 2024]
MISSING_RATES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


# -------------------------------------------------------
# Dataset (reused from train_multimodal.py)
# -------------------------------------------------------
class MultimodalDialogueDataset(Dataset):
    def __init__(self, dialogues, text_cache, audio_cache, text_dim=768, audio_dim=768):
        self.dialogues = dialogues
        self.text_cache = text_cache
        self.audio_cache = audio_cache
        self.text_dim = text_dim
        self.audio_dim = audio_dim

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, idx):
        d = self.dialogues[idx]
        text_feats, audio_feats, labels, speakers = [], [], [], []
        for utt in d.utterances:
            key = f"{d.dialogue_id}_{utt.utterance_id}"
            text_feats.append(
                torch.from_numpy(self.text_cache[key]) if key in self.text_cache
                else torch.zeros(self.text_dim)
            )
            audio_feats.append(
                torch.from_numpy(self.audio_cache[key]) if key in self.audio_cache
                else torch.zeros(self.audio_dim)
            )
            labels.append(utt.emotion_idx)
            speakers.append(utt.speaker_id if hasattr(utt, 'speaker_id') else 0)
        return {
            "text_features": torch.stack(text_feats),
            "audio_features": torch.stack(audio_feats),
            "labels": torch.tensor(labels, dtype=torch.long),
            "speaker_ids": torch.tensor(speakers, dtype=torch.long),
        }


def collate_multimodal(batch):
    max_len = max(b["labels"].size(0) for b in batch)
    text_dim = batch[0]["text_features"].size(1)
    audio_dim = batch[0]["audio_features"].size(1)
    bs = len(batch)
    text_feat = torch.zeros(bs, max_len, text_dim)
    audio_feat = torch.zeros(bs, max_len, audio_dim)
    labels = torch.full((bs, max_len), -1, dtype=torch.long)
    speakers = torch.zeros(bs, max_len, dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["labels"].size(0)
        text_feat[i, :L] = b["text_features"]
        audio_feat[i, :L] = b["audio_features"]
        labels[i, :L] = b["labels"]
        speakers[i, :L] = b["speaker_ids"]
    return {
        "text_features": text_feat, "audio_features": audio_feat,
        "labels": labels, "speaker_ids": speakers,
    }


# -------------------------------------------------------
# Fusion Baselines
# -------------------------------------------------------
from models.erc.dialogue_rnn import DialogueRNN
from models.evidential.edl_head import EvidentialHead
from models.evidential.ds_fusion import DempsterShaferFusion


class ConcatFusionModel(nn.Module):
    """Baseline 1: Simple feature concatenation."""
    def __init__(self, text_dim=768, audio_dim=768, hidden_dim=256,
                 num_classes=7, num_speakers=10, dropout=0.3):
        super().__init__()
        self.encoder = DialogueRNN(
            input_dim=text_dim + audio_dim, hidden_dim=hidden_dim,
            num_classes=num_classes, num_speakers=num_speakers,
            dropout=dropout, use_attention=True,
        )
        self.projection = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.edl_head = EvidentialHead(hidden_dim, num_classes)
        self.num_classes = num_classes

    def forward(self, text_features, audio_features, speaker_ids, lengths=None):
        concat = torch.cat([text_features, audio_features], dim=-1)
        hidden = self.encoder.get_features(concat, speaker_ids)
        proj = self.projection(hidden)
        edl = self.edl_head(proj)
        return edl  # Dict with alpha, belief, uncertainty, evidence


class AttentionFusionModel(nn.Module):
    """Baseline 2: Cross-modal attention fusion."""
    def __init__(self, text_dim=768, audio_dim=768, hidden_dim=256,
                 num_classes=7, num_speakers=10, dropout=0.3):
        super().__init__()
        self.text_encoder = DialogueRNN(
            input_dim=text_dim, hidden_dim=hidden_dim,
            num_classes=num_classes, num_speakers=num_speakers,
            dropout=dropout, use_attention=True,
        )
        self.audio_encoder = DialogueRNN(
            input_dim=audio_dim, hidden_dim=hidden_dim,
            num_classes=num_classes, num_speakers=num_speakers,
            dropout=dropout, use_attention=True,
        )
        # Cross-modal attention: query=text, key/value=audio
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, dropout=dropout, batch_first=True,
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.projection = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.edl_head = EvidentialHead(hidden_dim, num_classes)
        self.num_classes = num_classes

    def forward(self, text_features, audio_features, speaker_ids, lengths=None):
        text_h = self.text_encoder.get_features(text_features, speaker_ids)
        audio_h = self.audio_encoder.get_features(audio_features, speaker_ids)
        
        # Cross-modal attention
        attn_out, _ = self.cross_attn(text_h, audio_h, audio_h)
        
        # Gated fusion
        gate = self.gate(torch.cat([text_h, attn_out], dim=-1))
        fused = gate * text_h + (1 - gate) * attn_out
        
        proj = self.projection(fused)
        edl = self.edl_head(proj)
        return edl


class DSFusionModel(nn.Module):
    """Our method: Dempster-Shafer evidence fusion."""
    def __init__(self, text_dim=768, audio_dim=768, hidden_dim=256,
                 num_classes=7, num_speakers=10, dropout=0.3):
        super().__init__()
        self.text_encoder = DialogueRNN(
            input_dim=text_dim, hidden_dim=hidden_dim,
            num_classes=num_classes, num_speakers=num_speakers,
            dropout=dropout, use_attention=True,
        )
        self.text_proj = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.text_edl = EvidentialHead(hidden_dim, num_classes)

        self.audio_encoder = DialogueRNN(
            input_dim=audio_dim, hidden_dim=hidden_dim,
            num_classes=num_classes, num_speakers=num_speakers,
            dropout=dropout, use_attention=True,
        )
        self.audio_proj = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.audio_edl = EvidentialHead(hidden_dim, num_classes)

        self.fusion = DempsterShaferFusion(num_classes=num_classes, mode="evidence_sum")
        self.num_classes = num_classes

    def forward(self, text_features, audio_features, speaker_ids, lengths=None):
        text_h = self.text_encoder.get_features(text_features, speaker_ids)
        text_proj = self.text_proj(text_h)
        text_edl = self.text_edl(text_proj)

        audio_h = self.audio_encoder.get_features(audio_features, speaker_ids)
        audio_proj = self.audio_proj(audio_h)
        audio_edl = self.audio_edl(audio_proj)

        fused = self.fusion([text_edl["evidence"], audio_edl["evidence"]])
        fused["text_alpha"] = text_edl["alpha"]
        fused["audio_alpha"] = audio_edl["alpha"]
        return fused


# -------------------------------------------------------
# Training & Evaluation
# -------------------------------------------------------
from models.evidential.losses import SupervisedEvidentialLoss


def load_data():
    """Load MELD text + audio features."""
    from data.datasets.meld import MELDDataset, MELD_EMOTIONS

    meld = MELDDataset(data_dir="data/raw/MELD")
    train_dias = meld.get_dialogues("train")
    dev_dias = meld.get_dialogues("dev")
    test_dias = meld.get_dialogues("test")
    weights = meld.get_emotion_weights("train")

    # Text features
    text_caches = {}
    text_path = "data/features/meld_text_roberta_finetuned.pt"
    if Path(text_path).exists():
        cached = torch.load(text_path, weights_only=False)
        for split in ["train", "dev", "test"]:
            if split in cached:
                feats = cached[split]["features"].numpy()
                dia_ids = cached[split]["dialogue_ids"]
                utt_ids = cached[split]["utterance_ids"]
                c = {}
                for i in range(len(feats)):
                    c[f"{dia_ids[i].item()}_{utt_ids[i].item()}"] = feats[i]
                text_caches[split] = c

    # Audio features
    audio_caches = {}
    audio_path = "data/features/meld_audio_wavlm.pt"
    if Path(audio_path).exists():
        cached = torch.load(audio_path, weights_only=False)
        for split in ["train", "dev", "test"]:
            if split in cached:
                feats = cached[split]["features"].numpy()
                dia_ids = cached[split]["dialogue_ids"]
                utt_ids = cached[split]["utterance_ids"]
                c = {}
                for i in range(len(feats)):
                    c[f"{dia_ids[i].item()}_{utt_ids[i].item()}"] = feats[i]
                audio_caches[split] = c

    return train_dias, dev_dias, test_dias, MELD_EMOTIONS, weights, text_caches, audio_caches


@torch.no_grad()
def evaluate_with_missing(model, loader, device, missing_rate=0.0, seed=42):
    """Evaluate with simulated missing audio at given rate."""
    model.eval()
    rng = np.random.RandomState(seed)
    all_preds, all_labels = [], []

    for batch in loader:
        text = batch["text_features"].to(device)
        audio = batch["audio_features"].to(device)
        speakers = batch["speaker_ids"].to(device)
        labels = batch["labels"].to(device)

        # Simulate missing audio: zero-out audio for random utterances
        if missing_rate > 0:
            mask_missing = torch.from_numpy(
                rng.random(audio.shape[:2]) < missing_rate
            ).bool().to(device)
            audio = audio.clone()
            audio[mask_missing] = 0.0

        out = model(text, audio, speakers)
        mask = labels != -1
        preds = out["belief"][mask].argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels[mask].cpu().numpy())

    wf1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return wf1


def train_one_model(method, seed, train_dias, dev_dias, text_caches, audio_caches,
                    emotions, weights):
    """Train one fusion model, return the best model state."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    num_classes = len(emotions)
    class_weights = torch.from_numpy(weights.astype(np.float32)).to(device)

    # Build loaders
    train_ds = MultimodalDialogueDataset(
        train_dias, text_caches.get("train", {}), audio_caches.get("train", {}))
    dev_ds = MultimodalDialogueDataset(
        dev_dias, text_caches.get("dev", {}), audio_caches.get("dev", {}))
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,
                              collate_fn=collate_multimodal)
    dev_loader = DataLoader(dev_ds, batch_size=32, shuffle=False,
                            collate_fn=collate_multimodal)

    # Create model
    if method == "concat":
        model = ConcatFusionModel(num_classes=num_classes).to(device)
    elif method == "attention":
        model = AttentionFusionModel(num_classes=num_classes).to(device)
    elif method == "ds_fusion":
        model = DSFusionModel(num_classes=num_classes).to(device)
    else:
        raise ValueError(f"Unknown method: {method}")

    loss_fn = SupervisedEvidentialLoss(
        num_classes=num_classes, annealing_epochs=30, class_weights=class_weights)
    
    # DS Fusion uses auxiliary per-modality losses
    aux_loss_fn = SupervisedEvidentialLoss(
        num_classes=num_classes, annealing_epochs=30, class_weights=class_weights
    ) if method == "ds_fusion" else None

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_wf1 = 0.0
    best_state = None
    patience = 0

    for epoch in range(1, 81):
        model.train()
        loss_fn.set_epoch(epoch)
        if aux_loss_fn:
            aux_loss_fn.set_epoch(epoch)

        for batch in train_loader:
            text = batch["text_features"].to(device)
            audio = batch["audio_features"].to(device)
            speakers = batch["speaker_ids"].to(device)
            labels = batch["labels"].to(device)

            out = model(text, audio, speakers)
            mask = labels != -1
            labels_flat = labels[mask]

            fused_loss, _ = loss_fn(out["alpha"][mask], labels_flat)

            if method == "ds_fusion" and aux_loss_fn:
                t_loss, _ = aux_loss_fn(out["text_alpha"][mask], labels_flat)
                a_loss, _ = aux_loss_fn(out["audio_alpha"][mask], labels_flat)
                loss = fused_loss + 0.3 * (t_loss + a_loss) / 2.0
            else:
                loss = fused_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        # Dev eval
        dev_wf1 = evaluate_with_missing(model, dev_loader, device, missing_rate=0.0)
        if dev_wf1 > best_wf1:
            best_wf1 = dev_wf1
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= 20:
                break

        if epoch % 20 == 0:
            logger.info(f"  {method} s{seed} E{epoch}: dev_wf1={dev_wf1:.4f} best={best_wf1:.4f}")

    return best_state, model


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)


def main():
    results = load_results()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("DS FUSION MISSING-MODALITY ROBUSTNESS TEST")
    print(f"  3 methods × 6 missing rates × 3 seeds = 54 eval points")
    print("=" * 60)

    train_dias, dev_dias, test_dias, emotions, weights, text_caches, audio_caches = load_data()

    methods = ["concat", "attention", "ds_fusion"]
    total = len(methods) * len(SEEDS)
    done = 0

    for method in methods:
        for seed in SEEDS:
            done += 1
            train_key = f"{method}_s{seed}_trained"

            # Check if all missing rates for this method+seed are done
            all_done = all(
                f"{method}_s{seed}_miss{int(mr*100)}" in results
                for mr in MISSING_RATES
            )
            if all_done:
                print(f"\n[{done}/{total}] SKIP {method} seed={seed} (all rates done)")
                continue

            print(f"\n[{done}/{total}] Training {method} seed={seed}...")
            t0 = time.time()

            best_state, model = train_one_model(
                method, seed, train_dias, dev_dias,
                text_caches, audio_caches, emotions, weights
            )
            train_time = time.time() - t0
            print(f"  Training: {train_time:.0f}s")

            # Evaluate at each missing rate
            if best_state:
                model.load_state_dict(best_state)

            test_ds = MultimodalDialogueDataset(
                test_dias, text_caches.get("test", {}), audio_caches.get("test", {}))
            test_loader = DataLoader(test_ds, batch_size=32, shuffle=False,
                                     collate_fn=collate_multimodal)

            for mr in MISSING_RATES:
                key = f"{method}_s{seed}_miss{int(mr*100)}"
                if key in results:
                    print(f"  SKIP {key}: WF1={results[key]['wf1']:.4f}")
                    continue

                wf1 = evaluate_with_missing(model, test_loader, device,
                                            missing_rate=mr, seed=seed)
                results[key] = {
                    "wf1": round(float(wf1), 4),
                    "method": method,
                    "seed": seed,
                    "missing_rate": mr,
                    "train_time": round(train_time, 1),
                }
                save_results(results)
                print(f"  miss={int(mr*100):3d}% | WF1={wf1:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print("FUSION ROBUSTNESS RESULTS (WF1 %)")
    print(f"{'='*60}")
    
    header = f"  {'Method':<12s} |"
    for mr in MISSING_RATES:
        header += f"  {int(mr*100):3d}%  |"
    print(header)
    print(f"  {'-'*65}")
    
    for method in methods:
        row = f"  {method:<12s} |"
        for mr in MISSING_RATES:
            vals = [results[f"{method}_s{s}_miss{int(mr*100)}"]["wf1"]
                    for s in SEEDS if f"{method}_s{s}_miss{int(mr*100)}" in results]
            if vals:
                m = np.mean(vals) * 100
                s = np.std(vals) * 100
                row += f" {m:4.1f}±{s:3.1f}|"
            else:
                row += f"   ---  |"
        print(row)

    # Degradation analysis
    print(f"\n  Performance Drop (0% → 100% missing):")
    for method in methods:
        vals_0 = [results[f"{method}_s{s}_miss0"]["wf1"]
                  for s in SEEDS if f"{method}_s{s}_miss0" in results]
        vals_100 = [results[f"{method}_s{s}_miss100"]["wf1"]
                    for s in SEEDS if f"{method}_s{s}_miss100" in results]
        if vals_0 and vals_100:
            drop = (np.mean(vals_0) - np.mean(vals_100)) * 100
            print(f"  {method:<12s}: -{drop:.1f}%")


if __name__ == "__main__":
    main()
