"""
FedSSL-MERC: Data Processing Module
====================================
Dataset loaders, preprocessing, and federated data partitioning
for multimodal conversational emotion recognition.
"""

from data.datasets.meld import MELDDataset
from data.preprocessing import MultimodalPreprocessor

__all__ = ["MELDDataset", "MultimodalPreprocessor"]
