"""
Main training script for intermediate CheMeleon training.

This script trains the CheMeleon encoder using contrastive loss on MoleculeACE
pairs to improve its ability to separate activity cliffs in latent space.

Using pairs instead of triplets is more data-efficient and allows us to use
ALL cliff pairs, not just those that can form complete triplets.
"""

import sys
from pathlib import Path

import pandas as pd
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.intermediate_training.dataset import MoleculeACEPairDataset
from models.intermediate_training.encoder import CheMeleonEncoder
from models.intermediate_training.model import CheMeleonContrastiveTrainer


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train CheMeleon encoder with contrastive loss for activity cliff separation"
    )
    parser.add_argument(
        "pairs_path",
        type=str,
        help="Path to pairs DataFrame (pickle or CSV)",
    )
    parser.add_argument(
        "mace_data_path",
        type=str,
        help="Path to MoleculeACE DataFrame (pickle or CSV). Should be TRAIN split only.",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory for checkpoints and final model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=50,
        help="Maximum number of epochs (default: 50)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=1.0,
        help="Contrastive loss margin for cliff pairs (default: 1.0)",
    )
    parser.add_argument(
        "--cliff-weight",
        type=float,
        default=2.0,
        help="Weight for cliff pairs relative to consistent pairs (default: 2.0)",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.9,
        help="Train/val split ratio (default: 0.9)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers (default: 4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: auto-detect)",
    )
    parser.add_argument(
        "--save-top-k",
        type=int,
        default=1,
        help="Number of best checkpoints to save (default: 3)",
    )
    
    args = parser.parse_args()
    
    # Load data
    pairs_path = Path(args.pairs_path)
    mace_path = Path(args.mace_data_path)
    
    print(f"Loading pairs from {pairs_path}...")
    if pairs_path.suffix in [".pkl", ".pickle"]:
        pairs_df = pd.read_pickle(pairs_path)
    else:
        pairs_df = pd.read_csv(pairs_path)
    
    print(f"Loading MoleculeACE data from {mace_path}...")
    if mace_path.suffix in [".pkl", ".pickle"]:
        mace_df = pd.read_pickle(mace_path)
    else:
        mace_df = pd.read_csv(mace_path)
    
    # Verify we're using train split only (to avoid data leakage)
    if "split" in mace_df.columns:
        train_only = mace_df[mace_df["split"] == "train"]
        if len(train_only) < len(mace_df):
            print(f"Warning: Found test data in MoleculeACE file!")
            print(f"Using only train split: {len(train_only)}/{len(mace_df)} molecules")
            mace_df = train_only.reset_index(drop=True)
        else:
            print(f"Confirmed: Using train split only ({len(mace_df)} molecules)")
    else:
        print("Warning: No 'split' column found. Assuming all data is train split.")
    
    print(f"Loaded {len(pairs_df)} pairs")
    print(f"Pair types: {pairs_df['pair_type'].value_counts().to_dict()}")
    print(f"Loaded {len(mace_df)} molecules (train split only)")
    
    # Create dataset
    dataset = MoleculeACEPairDataset(
        pairs_df,
        mace_df["smiles"],
        mace_df["y"],
    )
    
    # Split train/val
    n_train = int(args.train_split * len(dataset))
    n_val = len(dataset) - n_train
    train_dataset, val_dataset = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    
    # Create encoder and trainer module
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    encoder = CheMeleonEncoder(device=device)
    model = CheMeleonContrastiveTrainer(
        encoder=encoder,
        lr=args.lr,
        margin=args.margin,
        cliff_weight=args.cliff_weight,
    )
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup callbacks
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=args.patience,
            verbose=True,
        ),
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            dirpath=checkpoint_dir,
            filename="best-{epoch:02d}-{val_loss:.4f}",
            save_top_k=args.save_top_k,
            save_last=True,
            verbose=True,
        ),
    ]
    
    # Setup logger
    logger = TensorBoardLogger(
        output_dir,
        name="tensorboard_logs",
        default_hp_metric=False,
    )
    
    # Create trainer
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
    )
    
    # Train
    print("\nStarting training...")
    trainer.fit(model, train_loader, val_loader)
    
    # Load best model and save final weights
    best_ckpt = trainer.checkpoint_callback.best_model_path
    print(f"\nBest checkpoint: {best_ckpt}")
    
    if best_ckpt:
        print("Loading best model...")
        best_model = CheMeleonContrastiveTrainer.load_from_checkpoint(
            best_ckpt,
            encoder=encoder,
        )
        
        # Save the encoder weights (message passing + aggregation)
        # This is what we'll use for downstream fine-tuning
        final_encoder_path = output_dir / "chemeleon_intermediate_encoder.pt"
        torch.save(
            {
                "message_passing_state_dict": best_model.encoder.get_message_passing().state_dict(),
                "aggregation_state_dict": best_model.encoder.get_aggregation().state_dict(),
                "hyper_parameters": best_model.encoder.get_message_passing().__dict__,
            },
            final_encoder_path,
        )
        print(f"Saved encoder weights to {final_encoder_path}")
        
        # Also save full model checkpoint for reference
        full_model_path = output_dir / "chemeleon_intermediate_full.pt"
        torch.save(best_model.state_dict(), full_model_path)
        print(f"Saved full model to {full_model_path}")
    else:
        print("Warning: No best checkpoint found!")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
