"""Quick evaluation of FedProx checkpoints."""
import sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.evidential.evidential_dialogue_rnn import EvidentialDialogueRNN
from scripts.train_multi_dataset import (
    load_meld, load_iemocap, GenericDialogueDataset, collate_dialogues, evaluate
)
from torch.utils.data import DataLoader

device = "cpu"

# MELD
print("=" * 60)
print("  Loading MELD FedProx checkpoint...")
train, dev, test, emotions, wts, cache, num_spk = load_meld(finetuned=True)
model = EvidentialDialogueRNN(
    input_dim=768, hidden_dim=256, num_classes=len(emotions),
    num_speakers=num_spk, dropout=0.3
)
ckpt = torch.load("checkpoints/best_fedavg_edl_meld.pt", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
test_ds = GenericDialogueDataset(test, cache.get("test", {}))
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_dialogues)
wf1, u, report, _ = evaluate(model, test_loader, device, emotions, "meld")
print(f"  MELD FedProx: WF1 = {wf1:.4f}, u = {u:.4f}")
rnd = ckpt.get("round", "N/A")
print(f"  Best round: {rnd}")
print(report)

# IEMOCAP-6
print("=" * 60)
print("  Loading IEMOCAP-6 FedProx checkpoint...")
train2, dev2, test2, emotions2, wts2, cache2, num_spk2 = load_iemocap(finetuned=True, num_classes=6)
model2 = EvidentialDialogueRNN(
    input_dim=768, hidden_dim=256, num_classes=len(emotions2),
    num_speakers=num_spk2, dropout=0.3
)
ckpt2 = torch.load("checkpoints/best_fedavg_edl_iemocap.pt", weights_only=False)
model2.load_state_dict(ckpt2["model_state_dict"])
test_ds2 = GenericDialogueDataset(test2, cache2.get("test", {}))
test_loader2 = DataLoader(test_ds2, batch_size=16, shuffle=False, collate_fn=collate_dialogues)
wf1_2, u2, report2, _ = evaluate(model2, test_loader2, device, emotions2, "iemocap")
print(f"  IEMOCAP-6 FedProx: WF1 = {wf1_2:.4f}, u = {u2:.4f}")
rnd2 = ckpt2.get("round", "N/A")
print(f"  Best round: {rnd2}")
print(report2)
print("=" * 60)
