"""
FedSSL-MERC: Aggregation Strategies
"""

from .fedavg import fedavg_aggregate, fedavg_aggregate_state_dicts
from .fedprox import FedProxLoss
from .eafa import eafa_aggregate, EAFAAggregator
from .cwe import cwe_aggregate, CWEAggregator

__all__ = [
    "fedavg_aggregate",
    "fedavg_aggregate_state_dicts",
    "FedProxLoss",
    "eafa_aggregate",
    "EAFAAggregator",
    "cwe_aggregate",
    "CWEAggregator",
]
