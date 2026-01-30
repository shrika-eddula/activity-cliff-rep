"""
Example usage script for intermediate training.

This demonstrates the complete workflow:
1. Mine pairs from MoleculeACE (contrastive loss)
2. Train intermediate model
3. Use refined encoder for downstream fine-tuning
"""

import subprocess
import sys
from pathlib import Path

# Example: Mine pairs from a few targets
print("Step 1: Mining pairs...")
subprocess.run([
    sys.executable,
    "models/intermediate_training/mine_triplets.py",
    "--output", "data/pairs.pkl",
    "--targets", "CHEMBL1862_Ki", "CHEMBL2047_EC50", "CHEMBL204_Ki",
    "--similarity-thresh", "0.9",
    "--min-cliff-delta", "1.0",
    "--consistent-delta", "0.5",
], check=True)

# Example: Train intermediate model
print("\nStep 2: Training intermediate model...")
subprocess.run([
    sys.executable,
    "models/intermediate_training/train.py",
    "data/pairs.pkl",
    "data/pairs_mace_train.pkl",  # Use train split only
    "output/intermediate_training",
    "--batch-size", "64",
    "--max-epochs", "20",
    "--lr", "1e-4",
    "--margin", "1.0",
    "--cliff-weight", "2.0",
    "--patience", "5",
], check=True)

print("\nStep 3: Use refined encoder for fine-tuning")
print("See README.md for code example on using the refined encoder weights")
