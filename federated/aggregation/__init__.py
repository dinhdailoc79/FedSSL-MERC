"""
FedSSL-MERC: Aggregation Strategies
"""

from .fedavg import fedavg_aggregate, fedavg_aggregate_state_dicts
from .fedprox import FedProxLoss
from .eafa import eafa_aggregate, EAFAAggregator
from .eafa_guard import eafa_guard_aggregate, EAFAGuardAggregator
from .cwe import cwe_aggregate, CWEAggregator

__all__ = [
    "fedavg_aggregate",
    "fedavg_aggregate_state_dicts",
    "FedProxLoss",
    "eafa_aggregate",
    "EAFAAggregator",
    "eafa_guard_aggregate",
    "EAFAGuardAggregator",
    "cwe_aggregate",
    "CWEAggregator",
]
