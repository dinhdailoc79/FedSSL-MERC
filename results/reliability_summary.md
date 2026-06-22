# Reliable Federated ERC - Comprehensive Evaluation Summary
This document summarizes the findings from RQ1-RQ4 on the expanded reliability modules.
## RQ1: Selective Prediction (AURC Comparison)
Lower AURC indicates better risk-coverage trade-off. We compare different uncertainty confidence signals:
| Dataset | Model | Max Prob (AURC) | Entropy (AURC) | Vacuity 1-u (AURC) |
|---------|-------|-----------------|----------------|--------------------|
| MELD | EDL | 0.2585±0.0132 | 0.2587±0.0131 | 0.2599±0.0122 |
| MELD | CE | 0.2404±0.0053 | 0.2403±0.0048 | N/A |
| IEMOCAP | EDL | 0.2702±0.0146 | 0.2879±0.0166 | 0.3002±0.0288 |
| IEMOCAP | CE | 0.2565±0.0085 | 0.2517±0.0070 | N/A |
| DAILYDIALOG | EDL | 0.0560±0.0102 | 0.0561±0.0102 | 0.0565±0.0100 |
| DAILYDIALOG | CE | 0.0425±0.0108 | 0.0429±0.0106 | N/A |

## RQ2: Conformal Prediction (Efficiency vs Coverage)
Target coverage: 90% (alpha = 0.1). We report actual coverage and average set size (lower set size at target coverage is better):
| Dataset | Model | Method | Actual Coverage | Avg Set Size |
|---------|-------|--------|-----------------|--------------|
| MELD | EDL | LAC | 0.9207±0.0033 | 4.77±0.04 |
| MELD | EDL | APS | 0.9460±0.0113 | 5.52±0.43 |
| MELD | EDL | APS_randomized | 0.9268±0.0087 | 5.02±0.44 |
| MELD | CE | LAC | 0.8953±0.0087 | 3.46±0.08 |
| MELD | CE | APS | 0.9484±0.0089 | 4.60±0.04 |
| MELD | CE | APS_randomized | 0.9221±0.0103 | 3.96±0.05 |
| IEMOCAP | EDL | LAC | 0.9402±0.0060 | 4.00±1.14 |
| IEMOCAP | EDL | APS | 0.9696±0.0074 | 3.95±0.26 |
| IEMOCAP | EDL | APS_randomized | 0.9519±0.0071 | 3.53±0.34 |
| IEMOCAP | CE | LAC | 0.9246±0.0066 | 2.60±0.10 |
| IEMOCAP | CE | APS | 0.9811±0.0036 | 3.87±0.11 |
| IEMOCAP | CE | APS_randomized | 0.9552±0.0068 | 3.09±0.10 |
| DAILYDIALOG | EDL | LAC | 0.8910±0.0043 | 1.01±0.01 |
| DAILYDIALOG | EDL | APS | 0.9575±0.0021 | 3.00±0.25 |
| DAILYDIALOG | EDL | APS_randomized | 0.9319±0.0050 | 1.54±0.11 |
| DAILYDIALOG | CE | LAC | 0.8873±0.0051 | 1.03±0.01 |
| DAILYDIALOG | CE | APS | 0.9887±0.0025 | 3.76±0.05 |
| DAILYDIALOG | CE | APS_randomized | 0.9302±0.0185 | 1.38±0.33 |

## RQ3: Federated Conformal Prediction (FCP vs Centralized)
Distributed quantile calibration (FCP) vs Centralized Conformal calibration:
| Dataset | Model | Method | FCP Coverage | FCP Set Size |
|---------|-------|--------|--------------|--------------|
| MELD | EDL | FCP_LAC | 0.9208±0.0031 | 4.78±0.04 |
| MELD | EDL | FCP_APS | 0.9466±0.0124 | 5.54±0.44 |
| MELD | CE | FCP_LAC | 0.8980±0.0102 | 3.51±0.04 |
| MELD | CE | FCP_APS | 0.9485±0.0087 | 4.61±0.04 |
| IEMOCAP | EDL | FCP_LAC | 0.9402±0.0060 | 4.00±1.14 |
| IEMOCAP | EDL | FCP_APS | 0.9696±0.0074 | 3.95±0.26 |
| IEMOCAP | CE | FCP_LAC | 0.9244±0.0068 | 2.60±0.10 |
| IEMOCAP | CE | FCP_APS | 0.9811±0.0036 | 3.87±0.11 |
| DAILYDIALOG | EDL | FCP_LAC | 0.8910±0.0043 | 1.01±0.01 |
| DAILYDIALOG | EDL | FCP_APS | 0.9575±0.0021 | 3.00±0.25 |
| DAILYDIALOG | CE | FCP_LAC | 0.8873±0.0051 | 1.03±0.01 |
| DAILYDIALOG | CE | FCP_APS | 0.9887±0.0025 | 3.76±0.05 |

## RQ4: Out-of-Distribution Detection (Speaker Hold-out on IEMOCAP)
Corrected split (ID = train sessions 1-3, OOD = session 5). AUROC for different uncertainty metrics:
### Model: EDL
- **entropy**: AUROC = 0.5742±0.0294
- **max_prob_inv**: AUROC = 0.5696±0.0198
- **vacuity_u**: AUROC = 0.5405±0.0181

### Model: CE
- **entropy**: AUROC = 0.5965±0.0037
- **max_prob_inv**: AUROC = 0.5831±0.0116

