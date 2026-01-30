"""
Intermediate training module for CheMeleon.

This module provides tools for training CheMeleon encoder with contrastive loss
to improve its ability to separate activity cliffs in latent space.

Uses pairs instead of triplets for better data efficiency.
"""

from .encoder import CheMeleonEncoder
from .dataset import MoleculeACEPairDataset
from .model import CheMeleonContrastiveTrainer, weighted_contrastive_loss

__all__ = [
    "CheMeleonEncoder",
    "MoleculeACEPairDataset",
    "CheMeleonContrastiveTrainer",
    "weighted_contrastive_loss",
]
