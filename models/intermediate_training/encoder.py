"""
CheMeleon encoder for intermediate training.

This module provides a trainable CheMeleon encoder that can be used
for triplet loss training. Unlike CheMeleonFingerprint, this allows
gradients to flow through the model.
"""

from pathlib import Path
from urllib.request import urlretrieve
from typing import List, Union

import torch
from chemprop import featurizers, nn
from chemprop.data import BatchMolGraph
from chemprop.models import MPNN
from chemprop.nn import RegressionFFN
from rdkit.Chem import Mol, MolFromSmiles


class CheMeleonEncoder(torch.nn.Module):
    """
    Trainable CheMeleon encoder for intermediate training.
    
    This encoder loads the CheMeleon message passing weights and allows
    gradients to flow through during training. Use this for triplet loss
    or other intermediate training objectives.
    """

    def __init__(self, device: Union[str, torch.device, None] = None):
        super().__init__()
        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        agg = nn.MeanAggregation()
        
        # Download CheMeleon weights if needed
        ckpt_dir = Path.home() / ".chemprop"
        ckpt_dir.mkdir(exist_ok=True)
        mp_path = ckpt_dir / "chemeleon_mp.pt"
        if not mp_path.exists():
            print(f"Downloading CheMeleon weights to {mp_path}...")
            urlretrieve(
                "https://zenodo.org/records/15460715/files/chemeleon_mp.pt",
                mp_path,
            )
        
        chemeleon_mp = torch.load(mp_path, weights_only=True, map_location="cpu")
        mp = nn.BondMessagePassing(**chemeleon_mp["hyper_parameters"])
        mp.load_state_dict(chemeleon_mp["state_dict"])
        
        # Create MPNN with dummy predictor (we only use fingerprint)
        predictor = RegressionFFN(input_dim=mp.output_dim)
        self.mpnn = MPNN(
            message_passing=mp,
            agg=agg,
            predictor=predictor,
        )
        
        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.to(self.device)
        
        # IMPORTANT: Set to train mode so gradients flow
        self.train()

    def forward(self, smiles_list: List[Union[str, Mol]]) -> torch.Tensor:
        """
        Encode SMILES strings to embeddings.
        
        Args:
            smiles_list: List of SMILES strings or RDKit Mol objects
            
        Returns:
            Tensor of shape [batch_size, embedding_dim] (typically 2048)
        """
        # Convert SMILES to mol graphs
        mol_graphs = [
            self.featurizer(MolFromSmiles(s) if isinstance(s, str) else s)
            for s in smiles_list
        ]
        
        # Create batch
        bmg = BatchMolGraph(mol_graphs)
        bmg.to(device=self.device)
        
        # IMPORTANT: No torch.no_grad() here - we want gradients!
        # Use fingerprint method which returns the latent representation
        # fingerprint takes (batch_mol_graph, V_d, X_d) - we pass None for descriptors
        Z = self.mpnn.fingerprint(bmg)
        
        return Z

    def get_message_passing(self):
        """Get the message passing module (useful for saving/loading)."""
        return self.mpnn.message_passing

    def get_aggregation(self):
        """Get the aggregation module."""
        return self.mpnn.agg
