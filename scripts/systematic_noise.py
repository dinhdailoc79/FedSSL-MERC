"""
Systematic Label Noise Injection for Federated Robustness Testing
===================================================================
Injects structured, systematic label noise based on common emotion confusion
pairs rather than symmetric random noise.

Confusion pairs:
  - MELD (7 classes):
    * anger (0) <-> disgust (1)
    * sadness (5) <-> fear (2)
    * surprise (6) <-> joy (3)
  - IEMOCAP (6 classes):
    * angry (2) <-> frustrated (4)
    * sad (1) <-> neutral (3)
    * excited (5) <-> happy (0)
"""

import copy
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Emotion indices mappings:
# MELD:
# 0: anger, 1: disgust, 2: fear, 3: joy, 4: neutral, 5: sadness, 6: surprise
MELD_CONFUSIONS = {
    0: 1,  # anger -> disgust
    1: 0,  # disgust -> anger
    5: 2,  # sadness -> fear
    2: 5,  # fear -> sadness
    6: 3,  # surprise -> joy
    3: 6,  # joy -> surprise
}

# IEMOCAP (6-class):
# 0: happy, 1: sad, 2: angry, 3: neutral, 4: frustrated, 5: excited
IEMOCAP_CONFUSIONS = {
    2: 4,  # angry -> frustrated
    4: 2,  # frustrated -> angry
    1: 3,  # sad -> neutral
    3: 1,  # neutral -> sad
    5: 0,  # excited -> happy
    0: 5,  # happy -> excited
}


def inject_systematic_noise(dialogues, dataset_name, noise_rate, seed=42):
    """
    Inject systematic label noise based on confusion mappings.
    
    For each utterance, if its label is in the confusion mapping,
    we flip it to the confused class with probability `noise_rate`.
    Otherwise, we do not flip.
    
    Args:
        dialogues: List of Dialogue/IEMOCAPDialogue objects (will be deep-copied)
        dataset_name: 'meld' or 'iemocap'
        noise_rate: Probability of flipping matching labels (0.0 to 1.0)
        seed: Random seed for reproducibility
        
    Returns:
        noisy_dialogues: Dialogue list with systematic noise injected
        stats: Dict of stats
    """
    if noise_rate <= 0:
        return dialogues, {"flipped": 0, "total": 0, "actual_rate": 0.0}
    
    dataset_name = dataset_name.lower()
    if "meld" in dataset_name:
        confusions = MELD_CONFUSIONS
    elif "iemocap" in dataset_name:
        confusions = IEMOCAP_CONFUSIONS
    else:
        raise ValueError(f"Unknown dataset for systematic noise: {dataset_name}")
        
    rng = np.random.RandomState(seed)
    noisy_dialogues = copy.deepcopy(dialogues)
    
    total_utterances = 0
    flipped = 0
    eligible = 0
    
    for dialogue in noisy_dialogues:
        for utterance in dialogue.utterances:
            if utterance.emotion_idx < 0:
                continue
            total_utterances += 1
            
            # Check if this class is eligible for systematic confusion
            orig = utterance.emotion_idx
            if orig in confusions:
                eligible += 1
                if rng.random() < noise_rate:
                    utterance.emotion_idx = confusions[orig]
                    flipped += 1
                    
    actual_rate = flipped / total_utterances if total_utterances > 0 else 0.0
    stats = {
        "flipped": flipped,
        "eligible": eligible,
        "total": total_utterances,
        "actual_rate": round(actual_rate, 4),
        "target_rate": noise_rate,
    }
    
    logger.info(
        f"  Systematic noise ({dataset_name}): flipped {flipped}/{total_utterances} "
        f"({actual_rate:.1%}) labels, target={noise_rate:.0%}, eligible={eligible}"
    )
    
    return noisy_dialogues, stats
