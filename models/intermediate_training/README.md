# Intermediate Training for CheMeleon

This module implements intermediate pre-training for the CheMeleon foundation model using **contrastive loss** on MoleculeACE activity cliff datasets.

## Overview

CheMeleon was pre-trained on structural descriptors, which causes it to map structurally similar molecules (including activity cliffs) to nearby locations in latent space. This intermediate training step refines the CheMeleon encoder to separate activity cliffs before fine-tuning on downstream tasks.

**Why Contrastive Loss instead of Triplet Loss?**
- **Data Efficiency**: Triplet loss requires finding anchors with BOTH a positive and negative, which is data-sparse. Contrastive loss only needs pairs, allowing us to use ALL cliff pairs.
- **Better Coverage**: If you have 500 cliff pairs but only 50 can form triplets, triplet loss uses 10% of data. Contrastive loss uses 100%.
- **Sparse Datasets**: MoleculeACE is a sparse biological dataset - contrastive loss is more robust for this scenario.

## Architecture

- **Encoder**: `CheMeleonEncoder` - Loads CheMeleon weights and allows gradients to flow (no `no_grad()`)
- **Dataset**: `MoleculeACEPairDataset` - Provides pairs (consistent or cliff) from MoleculeACE
- **Model**: `CheMeleonContrastiveTrainer` - Lightning module with weighted contrastive loss
- **Training**: Uses PyTorch Lightning for fast, distributed training with automatic checkpointing

## Usage

### Step 1: Mine Pairs

First, mine pairs from MoleculeACE datasets:

```bash
python models/intermediate_training/mine_triplets.py \
    --output data/pairs.pkl \
    --similarity-thresh 0.9 \
    --min-cliff-delta 1.0 \
    --consistent-delta 0.5
```

This will:
- Load all MoleculeACE benchmark datasets
- **Filter to TRAIN split only** (respects MoleculeACE splits to avoid data leakage)
- Compute **MoleculeACE consensus similarities** using three metrics:
  1. **ECFP Tanimoto similarity** (Morgan fingerprints)
  2. **Scaffold ECFP Tanimoto similarity** (ECFP on molecular scaffolds)
  3. **SMILES Levenshtein similarity** (normalized Levenshtein distance)
- Molecules are considered "highly similar" if **ANY** of these metrics > 0.9
- Find pairs:
  - **Consistent pairs**: Highly similar structure (consensus > 0.9) + similar activity (delta < 0.5)
  - **Cliff pairs**: Highly similar structure (consensus > 0.9) + large activity difference (delta >= 1.0 = 10x)

**Important**: 
- Only pairs from the MoleculeACE train split are mined (no data leakage)
- Uses MoleculeACE consensus criteria for similarity (matches paper methodology)
- Direct comparability with other papers using MoleculeACE benchmark

Options:
- `--targets`: Specify specific targets (default: all MoleculeACE targets)
- `--similarity-thresh`: Consensus similarity threshold (default: 0.9, MoleculeACE standard)
- `--min-cliff-delta`: Minimum activity difference for cliff pair in log space (default: 1.0 = 10x difference)
- `--consistent-delta`: Maximum activity difference for consistent pair (default: 0.5)

### Step 2: Train Intermediate Model

Train the CheMeleon encoder with contrastive loss:

```bash
python models/intermediate_training/train.py \
    data/pairs.pkl \
    data/pairs_mace_train.pkl \
    output/intermediate_training \
    --batch-size 64 \
    --max-epochs 50 \
    --lr 1e-4 \
    --margin 1.0 \
    --cliff-weight 2.0 \
    --patience 10 \
    --save-top-k 3
```

This will:
- Load pairs and MoleculeACE TRAIN data (test split excluded)
- Create train/val splits from the training pairs
- Train CheMeleon encoder with weighted contrastive loss
- Save checkpoints and final encoder weights

**Note**: The train/val split here is for intermediate training only. The MoleculeACE test split remains untouched for final evaluation.

Options:
- `--batch-size`: Batch size (default: 64)
- `--max-epochs`: Maximum epochs (default: 50)
- `--lr`: Learning rate (default: 1e-4)
- `--margin`: Margin for cliff pairs (default: 1.0)
- `--cliff-weight`: Weight for cliff pairs relative to consistent pairs (default: 2.0)
- `--patience`: Early stopping patience (default: 10)
- `--save-top-k`: Number of best checkpoints to save (default: 3)

Output files:
- `output/intermediate_training/chemeleon_intermediate_encoder.pt` - Encoder weights for downstream use
- `output/intermediate_training/chemeleon_intermediate_full.pt` - Full model checkpoint
- `output/intermediate_training/checkpoints/` - Lightning checkpoints

**Mining output files**:
- `data/pairs.pkl` - Pairs DataFrame (from train split only)
- `data/pairs_mace_train.pkl` - MoleculeACE train data
- `data/pairs_mace_test.pkl` - MoleculeACE test data (preserved for final evaluation)

### Step 3: Evaluate on MoleculeACE Benchmark

After intermediate training, evaluate the refined encoder on the MoleculeACE benchmark:

```bash
python models/intermediate_training/evaluate.py \
    output/evaluation \
    output/intermediate_training/chemeleon_intermediate_encoder.pt
```

This will:
- Load the intermediate-trained encoder weights
- Fine-tune on each of 30 MoleculeACE targets (5 random seeds each)
- Evaluate on test set and compute cliff/noncliff RMSE
- Save results in `output/evaluation/train_results.md` (same format as CheMeleon evaluation)

**Note**: This evaluation follows the same protocol as CheMeleon evaluation:
- Uses MoleculeACE official train/test splits
- Creates 80/20 train/val split from training data
- Trains for up to 50 epochs with early stopping (patience=5)
- Reports overall, cliff, and noncliff RMSE on test set

#### Manual Fine-tuning (Alternative)

If you want to fine-tune on your own dataset instead of MoleculeACE:

```python
import torch
from chemprop import nn, models
from chemprop.nn import RegressionFFN

# Load intermediate encoder weights
intermediate_weights = torch.load("output/intermediate_training/chemeleon_intermediate_encoder.pt")

# Create message passing with intermediate weights
mp = nn.BondMessagePassing(**intermediate_weights["hyper_parameters"])
mp.load_state_dict(intermediate_weights["message_passing_state_dict"])

# Create aggregation
agg = nn.MeanAggregation()
agg.load_state_dict(intermediate_weights["aggregation_state_dict"])

# Create task-specific head
ffn = nn.RegressionFFN(input_dim=mp.output_dim, ...)

# Create MPNN model
mpnn = models.MPNN(mp, agg, ffn, ...)

# Proceed with standard ChemProp fine-tuning
```

## Contrastive Loss

The loss function encourages:
- **Consistent pairs**: Pull embeddings together (minimize distance)
- **Cliff pairs**: Push embeddings apart if closer than margin (maximize distance)
- **Weighting**: Cliff pairs are weighted more heavily (default: 2x) to focus on activity cliffs

Formula:
```
loss = consistent_loss + cliff_weight * cliff_loss

consistent_loss = mean(distance^2) for consistent pairs
cliff_loss = mean(max(0, margin - distance)^2) for cliff pairs
```

## Key Features

1. **Gradients Enabled**: Unlike `CheMeleonFingerprint`, this encoder allows gradients to flow through the message passing layers
2. **Respects MoleculeACE Splits**: Only mines pairs from train split, preserving test split for final evaluation
3. **No Data Leakage**: Test molecules never seen during intermediate training
4. **Data Efficient**: Uses pairs instead of triplets, allowing use of ALL cliff pairs
5. **Automatic Checkpointing**: Lightning saves best models automatically
6. **Weight Saving**: Saves encoder weights separately for easy downstream use
7. **Monitoring**: Logs distances and losses to TensorBoard

## Requirements

- `chemprop>=2.1`
- `lightning>=2.5`
- `torch`
- `pandas`
- `rdkit`
- `numpy`
- `python-Levenshtein` (optional but **highly recommended** for fast mining)

**Performance Note**: The `python-Levenshtein` library uses optimized C code and can make pair mining **10-100x faster** for large datasets. Install it with:
```bash
pip install python-Levenshtein
```

The script will work without it (using a slower manual implementation), but will print a warning.

See main repository `requirements.txt` for full dependencies.
