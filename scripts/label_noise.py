"""
Label Noise Injection for Federated Robustness Testing
=======================================================
Injects symmetric label noise into specific client partitions
to simulate real-world annotation quality heterogeneity.

In practice, different institutions may have:
- Expert annotators (clean labels)
- Novice annotators (10-20% noise)
- Crowd-sourced labels (30-40% noise)

EAFA should automatically detect and downweight noisy clients
via epistemic uncertainty.
"""

import copy
import logging
import numpy as np

logger = logging.getLogger(__name__)


def inject_label_noise(dialogues, noise_rate, num_classes, seed=42):
    """
    Inject symmetric label noise into dialogue utterances.
    
    For each utterance, with probability `noise_rate`, flip the label
    to a uniformly random OTHER class.
    
    Args:
        dialogues: List of Dialogue objects (will be deep copied)
        noise_rate: Probability of flipping each label (0.0 to 1.0)
        num_classes: Total number of emotion classes
        seed: Random seed for reproducibility
        
    Returns:
        noisy_dialogues: New list with noisy labels
        stats: Dict with noise injection statistics
    """
    if noise_rate <= 0:
        return dialogues, {"flipped": 0, "total": 0, "actual_rate": 0.0}
    
    rng = np.random.RandomState(seed)
    noisy_dialogues = copy.deepcopy(dialogues)
    
    total_utterances = 0
    flipped = 0
    
    for dialogue in noisy_dialogues:
        for utterance in dialogue.utterances:
            if utterance.emotion_idx < 0:  # Skip padding/unlabeled
                continue
            total_utterances += 1
            
            if rng.random() < noise_rate:
                # Flip to a random OTHER class
                original = utterance.emotion_idx
                candidates = [c for c in range(num_classes) if c != original]
                utterance.emotion_idx = rng.choice(candidates)
                flipped += 1
    
    actual_rate = flipped / total_utterances if total_utterances > 0 else 0.0
    stats = {
        "flipped": flipped,
        "total": total_utterances,
        "actual_rate": round(actual_rate, 4),
        "target_rate": noise_rate,
    }
    
    logger.info(
        f"  Label noise: flipped {flipped}/{total_utterances} "
        f"({actual_rate:.1%}) labels, target={noise_rate:.0%}"
    )
    
    return noisy_dialogues, stats
