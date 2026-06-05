from models.evidential.edl_head import EvidentialHead
from models.evidential.losses import (
    SupervisedEvidentialLoss,
    EvidentialConsistencyRegularization,
    FedEvidenceLoss,
    dirichlet_kl_divergence,
)
from models.evidential.calibration import (
    compute_calibration_metrics,
    plot_reliability_diagram,
)

