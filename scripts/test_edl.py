"""
Smoke test for EDL components:
1. EvidentialHead
2. EvidentialDialogueRNN
3. SupervisedEvidentialLoss
4. EvidentialConsistencyRegularization
5. EAFAAggregator
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np

def test_edl_head():
    from models.evidential.edl_head import EvidentialHead
    
    head = EvidentialHead(input_dim=256, num_classes=7)
    hidden = torch.randn(4, 10, 256)  # batch=4, seq=10, hidden=256
    out = head(hidden)
    
    assert out["evidence"].shape == (4, 10, 7)
    assert out["alpha"].shape == (4, 10, 7)
    assert out["belief"].shape == (4, 10, 7)
    assert out["uncertainty"].shape == (4, 10)
    
    # Evidence must be non-negative
    assert (out["evidence"] >= 0).all(), "Evidence must be non-negative"
    # Alpha must be >= 1
    assert (out["alpha"] >= 1).all(), "Alpha must be >= 1"
    # Belief + uncertainty must sum to ~1
    total = out["belief"].sum(dim=-1) + out["uncertainty"]
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5), \
        f"Belief + uncertainty must sum to 1, got {total[0,0]:.4f}"
    # Uncertainty in [0, 1]
    assert (out["uncertainty"] >= 0).all() and (out["uncertainty"] <= 1).all()
    
    print("[PASS] EvidentialHead")

def test_evidential_dialogue_rnn():
    from models.evidential.evidential_dialogue_rnn import EvidentialDialogueRNN
    
    model = EvidentialDialogueRNN(
        input_dim=768, hidden_dim=128, num_classes=7,
        num_speakers=10, dropout=0.1,
    )
    
    utts = torch.randn(2, 5, 768)
    spks = torch.randint(0, 3, (2, 5))
    
    out = model(utts, spks)
    assert out["alpha"].shape == (2, 5, 7)
    assert out["uncertainty"].shape == (2, 5)
    
    # Test predict
    preds, uncert = model.predict(utts, spks)
    assert preds.shape == (2, 5)
    assert uncert.shape == (2, 5)
    
    # Test mean uncertainty
    labels = torch.randint(0, 7, (2, 5))
    labels[0, 3:] = -1  # padding
    u_mean = model.get_mean_uncertainty(utts, spks, labels)
    assert 0 <= u_mean <= 1, f"Mean uncertainty {u_mean} out of range"
    
    params = sum(p.numel() for p in model.parameters())
    print(f"[PASS] EvidentialDialogueRNN ({params:,} params)")

def test_supervised_loss():
    from models.evidential.losses import SupervisedEvidentialLoss
    
    loss_fn = SupervisedEvidentialLoss(num_classes=7, annealing_epochs=10)
    
    alpha = (torch.rand(16, 7) * 5 + 1).requires_grad_(True)  # alpha >= 1
    labels = torch.randint(0, 7, (16,))
    
    loss_fn.set_epoch(0)
    loss0, stats0 = loss_fn(alpha, labels)
    assert loss0.requires_grad
    assert not torch.isnan(loss0), "Loss is NaN"
    
    loss_fn.set_epoch(10)
    loss10, stats10 = loss_fn(alpha, labels)
    # At epoch 10, KL weight should be 1.0
    assert stats10["lambda_kl"] == 1.0
    
    print(f"[PASS] SupervisedEvidentialLoss (loss@ep0={loss0.item():.4f}, loss@ep10={loss10.item():.4f})")

def test_ecr_loss():
    from models.evidential.losses import EvidentialConsistencyRegularization
    
    ecr = EvidentialConsistencyRegularization(lambda_u=1.0)
    
    alpha_weak = torch.rand(16, 7) * 10 + 1
    alpha_strong = torch.rand(16, 7) * 10 + 1
    uncertainty = torch.rand(16) * 0.5  # Low uncertainty = high certainty
    
    loss, stats = ecr(alpha_weak, alpha_strong, uncertainty)
    assert not torch.isnan(loss), "ECR loss is NaN"
    assert stats["mean_certainty"] > 0.5  # Should be high since u < 0.5
    
    # Test with high uncertainty (should reduce loss)
    uncertainty_high = torch.ones(16) * 0.95
    loss_high, stats_high = ecr(alpha_weak, alpha_strong, uncertainty_high)
    # High uncertainty → loss should be smaller (auto-vanishing)
    
    print(f"[PASS] ECR Loss (certain={loss.item():.4f}, uncertain={loss_high.item():.4f})")

def test_eafa():
    from federated.aggregation.eafa import EAFAAggregator
    from collections import OrderedDict
    
    agg = EAFAAggregator(beta=1.0)
    
    # Simulate 3 clients
    states = []
    for _ in range(3):
        state = OrderedDict()
        state["layer1.weight"] = torch.randn(10, 10)
        state["layer1.bias"] = torch.randn(10)
        states.append(state)
    
    data_sizes = [100, 200, 50]
    uncertainties = [0.3, 0.1, 0.8]  # Client 1 is most certain, client 2 most uncertain
    
    global_state, stats = agg.aggregate(states, data_sizes, uncertainties, round_num=0)
    
    # Client 1 (large data, low uncertainty) should have highest weight
    assert stats["weights"][1] > stats["weights"][0] > stats["weights"][2], \
        f"Weights not properly ordered: {stats['weights']}"
    
    # Verify shape preserved
    assert global_state["layer1.weight"].shape == (10, 10)
    
    print(f"[PASS] EAFA (weights={[f'{w:.3f}' for w in stats['weights']]})")

def test_combined_loss():
    from models.evidential.losses import FedEvidenceLoss
    
    loss_fn = FedEvidenceLoss(
        num_classes=7, annealing_epochs=10,
        lambda_u=1.0, lambda_u_rampup_epochs=20,
    )
    
    # Supervised only
    alpha_l = torch.rand(16, 7) * 5 + 1
    labels = torch.randint(0, 7, (16,))
    
    loss_fn.set_epoch(5)
    loss, stats = loss_fn(alpha_l, labels)
    assert stats["loss_ecr"] == 0.0  # No unlabeled data
    
    # Supervised + ECR
    alpha_w = torch.rand(32, 7) * 5 + 1
    alpha_s = torch.rand(32, 7) * 5 + 1
    u_w = torch.rand(32) * 0.5
    
    loss2, stats2 = loss_fn(
        alpha_l, labels,
        alpha_weak=alpha_w, alpha_strong=alpha_s,
        uncertainty_weak=u_w,
    )
    assert stats2["loss_ecr"] > 0  # Should have ECR loss
    assert stats2["lambda_u"] > 0  # Ramp-up should be active
    
    print(f"[PASS] FedEvidenceLoss (sup_only={loss.item():.4f}, sup+ecr={loss2.item():.4f}, lambda_u={stats2['lambda_u']:.3f})")


if __name__ == "__main__":
    print("=" * 60)
    print("  Testing EDL Components")
    print("=" * 60)
    
    test_edl_head()
    test_evidential_dialogue_rnn()
    test_supervised_loss()
    test_ecr_loss()
    test_eafa()
    test_combined_loss()
    
    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED!")
    print("=" * 60)
