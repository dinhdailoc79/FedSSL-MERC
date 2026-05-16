"""Quick eval: load checkpoints, evaluate on test set, print table."""
import sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

from scripts.train_multi_dataset import (
    load_dailydialog, evaluate, GenericDialogueDataset, collate_dialogues
)
from torch.utils.data import DataLoader
from models.evidential.evidential_dialogue_rnn import EvidentialDialogueRNN
from models.erc.dialogue_rnn import DialogueRNN

train, dev, test, emotions, weights, cache, ns = load_dailydialog(finetuned=True)
test_ds = GenericDialogueDataset(test, cache.get("test", {}))
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False,
                         collate_fn=collate_dialogues, num_workers=0)
device = "cuda" if torch.cuda.is_available() else "cpu"
nc = len(emotions)

configs = [
    ("CE FedAvg",       "checkpoints/best_fedavg_ce_dailydialog.pt",  "ce"),
    ("EDL FedAvg",      "checkpoints/best_fedavg_edl_dailydialog.pt", "edl"),
    ("EDL Centralized", "checkpoints/best_edl_dailydialog.pt",        "edl"),
    ("EDL EAFA",        "checkpoints/best_eafa_dailydialog.pt",       "edl"),
]

emo_str = ", ".join(emotions)
print(f"Dataset: DailyDialog | {nc} classes | Test: {len(test)} dialogues")
print(f"Emotions: {emo_str}")
print()
print(f"{'Config':20s} | {'WF1':>6s} | {'Micro':>6s} | {'Uncert':>6s}")
print("-" * 50)

for name, ckpt, loss_type in configs:
    if not os.path.exists(ckpt):
        print(f"{name:20s} | NOT FOUND: {ckpt}")
        continue

    if loss_type == "edl":
        model = EvidentialDialogueRNN(
            input_dim=768, hidden_dim=256, num_classes=nc,
            num_speakers=ns, dropout=0.3
        ).to(device)
    else:
        model = DialogueRNN(
            input_dim=768, hidden_dim=256, num_classes=nc,
            num_speakers=ns, dropout=0.3
        ).to(device)

    ckpt_data = torch.load(ckpt, map_location=device, weights_only=False)
    state = ckpt_data.get("model_state_dict", ckpt_data)
    model.load_state_dict(state)

    wf1, u, report, micro = evaluate(model, test_loader, device, emotions, "dailydialog")
    print(f"{name:20s} | {wf1:.4f} | {micro:.4f} | {u:.4f}")

print()
print("Note: CE Centralized checkpoint was overwritten by EDL Centralized.")
print("CE Centralized results must come from training logs.")
