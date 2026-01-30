"""
Lightning module for intermediate training with weighted contrastive loss.
"""

import torch
import torch.nn.functional as F
from lightning import LightningModule
from typing import Dict, Any

from .encoder import CheMeleonEncoder


def weighted_contrastive_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    pair_type: list,
    activity_diff: torch.Tensor,
    margin: float = 1.0,
    cliff_weight: float = 2.0,
) -> torch.Tensor:
    """
    Compute weighted contrastive loss (pairwise ranking loss).
    
    For consistent pairs (similar structure + similar activity):
    - Pull embeddings together (minimize distance)
    
    For cliff pairs (similar structure + large activity difference):
    - Push embeddings apart if closer than margin (maximize distance)
    
    Args:
        z1: First molecule embeddings [batch_size, embedding_dim]
        z2: Second molecule embeddings [batch_size, embedding_dim]
        pair_type: List of pair types ("consistent" or "cliff") [batch_size]
        activity_diff: Activity differences [batch_size]
        margin: Margin for cliff pairs (default: 1.0)
        cliff_weight: Weight for cliff pairs relative to consistent pairs (default: 2.0)
        
    Returns:
        Scalar loss tensor
    """
    # Compute pairwise distances
    distances = F.pairwise_distance(z1, z2)  # [batch_size]
    
    # Create masks for pair types
    is_consistent = torch.tensor(
        [pt == "consistent" for pt in pair_type],
        dtype=torch.bool,
        device=z1.device
    )
    is_cliff = torch.tensor(
        [pt == "cliff" for pt in pair_type],
        dtype=torch.bool,
        device=z1.device
    )
    
    # Consistent pairs: minimize distance (pull together)
    # Loss = distance^2 (squared distance for smooth gradients)
    consistent_loss = (distances[is_consistent] ** 2).mean() if is_consistent.any() else torch.tensor(0.0, device=z1.device)
    
    # Cliff pairs: push apart if closer than margin
    # Loss = max(0, margin - distance)^2 (hinge loss)
    cliff_distances = distances[is_cliff]
    cliff_loss = (F.relu(margin - cliff_distances) ** 2).mean() if is_cliff.any() else torch.tensor(0.0, device=z1.device)
    
    # Weight cliff pairs more heavily
    total_loss = consistent_loss + cliff_weight * cliff_loss
    
    return total_loss


class CheMeleonContrastiveTrainer(LightningModule):
    """
    Lightning module for training CheMeleon encoder with contrastive loss.
    
    This module trains the CheMeleon backbone to separate activity cliffs
    in the latent space using pairwise contrastive loss (more data-efficient than triplets).
    """

    def __init__(
        self,
        encoder: CheMeleonEncoder,
        lr: float = 1e-4,
        margin: float = 1.0,
        cliff_weight: float = 2.0,
    ):
        """
        Args:
            encoder: CheMeleonEncoder instance
            lr: Learning rate
            margin: Margin for cliff pairs (default: 1.0)
            cliff_weight: Weight for cliff pairs relative to consistent pairs (default: 2.0)
        """
        super().__init__()
        self.encoder = encoder
        self.lr = lr
        self.margin = margin
        self.cliff_weight = cliff_weight
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=["encoder"])

    def configure_optimizers(self):
        """Configure optimizer for training."""
        return torch.optim.Adam(self.encoder.parameters(), lr=self.lr)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step."""
        # Encode both molecules
        z1 = self.encoder(batch["mol1_smiles"])
        z2 = self.encoder(batch["mol2_smiles"])
        
        # Get pair information
        pair_type = batch["pair_type"]  # List of strings
        activity_diff = batch["activity_diff"].to(self.device)
        
        # Compute loss
        loss = weighted_contrastive_loss(
            z1, z2, pair_type, activity_diff,
            margin=self.margin,
            cliff_weight=self.cliff_weight,
        )
        
        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Also log distances for monitoring
        with torch.no_grad():
            distances = F.pairwise_distance(z1, z2)
            is_consistent = torch.tensor(
                [pt == "consistent" for pt in pair_type],
                dtype=torch.bool,
                device=self.device
            )
            is_cliff = torch.tensor(
                [pt == "cliff" for pt in pair_type],
                dtype=torch.bool,
                device=self.device
            )
            
            if is_consistent.any():
                self.log("train_d_consistent", distances[is_consistent].mean(), on_step=False, on_epoch=True)
            if is_cliff.any():
                self.log("train_d_cliff", distances[is_cliff].mean(), on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        z1 = self.encoder(batch["mol1_smiles"])
        z2 = self.encoder(batch["mol2_smiles"])
        
        pair_type = batch["pair_type"]
        activity_diff = batch["activity_diff"].to(self.device)
        
        loss = weighted_contrastive_loss(
            z1, z2, pair_type, activity_diff,
            margin=self.margin,
            cliff_weight=self.cliff_weight,
        )
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        with torch.no_grad():
            distances = F.pairwise_distance(z1, z2)
            is_consistent = torch.tensor(
                [pt == "consistent" for pt in pair_type],
                dtype=torch.bool,
                device=self.device
            )
            is_cliff = torch.tensor(
                [pt == "cliff" for pt in pair_type],
                dtype=torch.bool,
                device=self.device
            )
            
            if is_consistent.any():
                self.log("val_d_consistent", distances[is_consistent].mean(), on_step=False, on_epoch=True)
            if is_cliff.any():
                self.log("val_d_cliff", distances[is_cliff].mean(), on_step=False, on_epoch=True)
        
        return loss
