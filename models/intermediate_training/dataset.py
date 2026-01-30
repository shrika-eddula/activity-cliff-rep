"""
Dataset classes for contrastive training with pairs.
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, Any


class MoleculeACEPairDataset(Dataset):
    """
    Dataset for MoleculeACE pairs (contrastive loss).
    
    Returns batches of (mol1_smiles, mol2_smiles, y1, y2, pair_type, activity_diff)
    for contrastive loss training.
    """

    def __init__(self, pairs_df, smiles_series, y_series):
        """
        Args:
            pairs_df: DataFrame with columns mol1_idx, mol2_idx, y1, y2, pair_type, activity_diff
            smiles_series: Series of SMILES strings indexed by the same indices as pairs_df
            y_series: Series of y values indexed by the same indices as pairs_df
        """
        self.pairs = pairs_df.reset_index(drop=True)
        self.smiles = smiles_series.reset_index(drop=True)
        self.y = y_series.reset_index(drop=True)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx) -> Dict[str, Any]:
        row = self.pairs.iloc[idx]
        i1 = int(row["mol1_idx"])
        i2 = int(row["mol2_idx"])

        return {
            "mol1_smiles": self.smiles[i1],
            "mol2_smiles": self.smiles[i2],
            "y1": torch.tensor(float(self.y[i1]), dtype=torch.float32),
            "y2": torch.tensor(float(self.y[i2]), dtype=torch.float32),
            "pair_type": row["pair_type"],  # "consistent" or "cliff"
            "activity_diff": torch.tensor(float(row["activity_diff"]), dtype=torch.float32),
        }
