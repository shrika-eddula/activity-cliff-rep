"""
Evaluate intermediate-trained CheMeleon encoder on MoleculeACE benchmark.

This script loads the intermediate-trained encoder weights and fine-tunes
on each MoleculeACE target, then evaluates on the test set with cliff/noncliff metrics.
Results are saved in the same format as CheMeleon evaluation.
"""

import datetime
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from astartes import train_test_split
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.models import MPNN
from chemprop.nn import (BondMessagePassing, RegressionFFN, UnscaleTransform)
from chemprop.nn.agg import MeanAggregation
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.metrics import root_mean_squared_error

# MoleculeACE benchmarks
MOLECULEACE_BENCHMARKS = (
    "CHEMBL1862_Ki",
    "CHEMBL1871_Ki",
    "CHEMBL2034_Ki",
    "CHEMBL2047_EC50",
    "CHEMBL204_Ki",
    "CHEMBL2147_Ki",
    "CHEMBL214_Ki",
    "CHEMBL218_EC50",
    "CHEMBL219_Ki",
    "CHEMBL228_Ki",
    "CHEMBL231_Ki",
    "CHEMBL233_Ki",
    "CHEMBL234_Ki",
    "CHEMBL235_EC50",
    "CHEMBL236_Ki",
    "CHEMBL237_EC50",
    "CHEMBL237_Ki",
    "CHEMBL238_Ki",
    "CHEMBL239_EC50",
    "CHEMBL244_Ki",
    "CHEMBL262_Ki",
    "CHEMBL264_Ki",
    "CHEMBL2835_Ki",
    "CHEMBL287_Ki",
    "CHEMBL2971_Ki",
    "CHEMBL3979_EC50",
    "CHEMBL4005_Ki",
    "CHEMBL4203_Ki",
    "CHEMBL4616_EC50",
    "CHEMBL4792_Ki",
)


if __name__ == "__main__":
    try:
        output_dir = Path(sys.argv[1])
        encoder_pt = Path(sys.argv[2]) if len(sys.argv) >= 3 else None
    except:
        print("usage: python evaluate.py OUTPUT_DIR ENCODER_PT")
        print("  OUTPUT_DIR: Directory to save results")
        print("  ENCODER_PT: Path to chemeleon_intermediate_encoder.pt")
        exit(1)
    
    if encoder_pt is None:
        print("ERROR: Must provide path to intermediate encoder weights")
        print("  Example: python evaluate.py output/evaluation output/intermediate_training/chemeleon_intermediate_encoder.pt")
        exit(1)
    
    if not encoder_pt.exists():
        print(f"ERROR: Encoder weights not found at {encoder_pt}")
        exit(1)
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    # Load intermediate encoder weights
    print(f"Loading intermediate encoder weights from {encoder_pt}...")
    # weights_only=False is safe here since we trust our own saved weights
    intermediate_weights = torch.load(encoder_pt, map_location="cpu", weights_only=False)
    mp_hyperparams = intermediate_weights["hyper_parameters"]
    
    # Filter out non-constructor arguments (runtime state like 'training', '_modules', etc.)
    # These are common PyTorch module attributes that aren't constructor params
    exclude_keys = {
        'training', '_modules', '_parameters', '_buffers', '_non_persistent_buffers_set',
        '_backward_hooks', '_backward_pre_hooks', '_forward_hooks', '_forward_pre_hooks',
        '_state_dict_hooks', '_load_state_dict_pre_hooks', '_state_dict_pre_hooks',
        '_load_state_dict_post_hooks', '_full_backward_hooks', '_full_backward_pre_hooks',
        '_forward_hooks_with_kwargs', '_backward_hooks_with_kwargs', '_forward_pre_hooks_with_kwargs',
        '_backward_pre_hooks_with_kwargs', '_state_dict_hooks_with_kwargs', '_load_state_dict_pre_hooks_with_kwargs',
        '_state_dict_pre_hooks_with_kwargs', '_load_state_dict_post_hooks_with_kwargs',
    }
    
    # Extract the actual hidden dimension from the saved state_dict
    # Chemprop v2 uses 'd_h' as the parameter name, and we need to infer it from weights
    saved_state_dict = intermediate_weights["message_passing_state_dict"]
    if "W_i.weight" in saved_state_dict:
        # W_i is [d_h, input_dim], so shape[0] gives us d_h
        actual_d_h = saved_state_dict["W_i.weight"].shape[0]
    else:
        raise ValueError("Cannot infer d_h from saved weights: W_i.weight not found")
    
    print(f"Detected hidden dimension (d_h) from weights: {actual_d_h}")
    
    # Filter hyperparameters to only include constructor arguments
    # CheMeleon's hyper_parameters contains: depth, activation, dropout, undirected, d_h
    # We need to exclude module attributes like W_d, W_i, W_h, W_o, etc. (these are in state_dict)
    # Also exclude PyTorch internals
    known_constructor_params = {'depth', 'activation', 'dropout', 'undirected', 'd_h', 'hidden_dim'}
    
    valid_params = {}
    for k, v in mp_hyperparams.items():
        if k in exclude_keys or k.startswith('_'):
            continue
        # Only include known constructor parameters or simple types that look like constructor args
        # Exclude anything that looks like a module parameter (W_d, W_i, W_h, W_o, etc.)
        if k in known_constructor_params:
            valid_params[k] = v
        elif isinstance(v, (int, float, bool, str, type, type(None))) and not k.startswith('W_'):
            # Include other simple types that aren't module parameters
            valid_params[k] = v
        elif hasattr(v, '__name__') and not k.startswith('W_'):  # Functions/classes
            valid_params[k] = v
    
    # Override with the actual d_h from weights (this is the critical fix!)
    valid_params['d_h'] = actual_d_h
    
    # Get hidden_size by creating a temporary instance (like CheMeleon does)
    try:
        temp_mp = BondMessagePassing(**valid_params)
        hidden_size = temp_mp.output_dim
        del temp_mp
        print(f"Loaded encoder: d_h={actual_d_h}, output_dim={hidden_size}")
    except TypeError as e:
        print(f"Error creating BondMessagePassing with params: {list(valid_params.keys())}")
        print(f"Error: {e}")
        raise
    
    # Create output file
    output_file = open(output_dir / "train_results.md", "w")
    output_file.write(
        f"""# Intermediate-Trained CheMeleon Results
timestamp: {datetime.datetime.now()}
encoder_path: {encoder_pt}
"""
    )
    performance_dict = {}
    
    # Run evaluation for each random seed
    for random_seed in (42, 117, 709, 1701, 9001):
        output_file.write(f"## Random Seed {random_seed}\n")
        seed_dir = output_dir / f"seed_{random_seed}"
        
        for benchmark_name in MOLECULEACE_BENCHMARKS:
            print(f"\n{'='*60}")
            print(f"Benchmark: {benchmark_name}, Seed: {random_seed}")
            print(f"{'='*60}")
            
            # Load MoleculeACE data
            smiles_col = "smiles"
            target_cols = ["y"]
            df = pd.read_csv(
                f"https://raw.githubusercontent.com/molML/MoleculeACE/7e6de0bd2968c56589c580f2a397f01c531ede26/MoleculeACE/Data/benchmark_data/{benchmark_name}.csv"
            )
            train_df, test_df = (
                df[df["split"] == "train"],
                df[df["split"] == "test"],
            )
            # MoleculeACE is always regression
            
            targets = train_df[target_cols]
            targets = targets.fillna(targets.mean(axis=0)).to_numpy()
            
            # Create train/val split from training data
            train_idxs, val_idxs = train_test_split(
                np.arange(len(targets)),
                train_size=0.80,
                test_size=0.20,
                random_state=random_seed,
            )
            
            train_data = [
                MoleculeDatapoint.from_smi(smi, y)
                for smi, y in zip(
                    train_df[smiles_col].iloc[train_idxs], targets[train_idxs]
                )
            ]
            val_data = [
                MoleculeDatapoint.from_smi(smi, y)
                for smi, y in zip(
                    train_df[smiles_col].iloc[val_idxs], targets[val_idxs]
                )
            ]
            test_data = list(map(MoleculeDatapoint.from_smi, test_df[smiles_col]))
            
            # Create datasets
            featurizer = SimpleMoleculeMolGraphFeaturizer()
            train_dataset = MoleculeDataset(train_data, featurizer)
            val_dataset = MoleculeDataset(val_data, featurizer)
            test_dataset = MoleculeDataset(test_data, featurizer)
            
            # Normalize targets (MoleculeACE is regression)
            scaler = train_dataset.normalize_targets()
            val_dataset.normalize_targets(scaler)
            
            # Create dataloaders
            train_dataloader = build_dataloader(train_dataset, num_workers=1)
            val_dataloader = build_dataloader(val_dataset, num_workers=1, shuffle=False)
            test_dataloader = build_dataloader(
                test_dataset, num_workers=1, shuffle=False
            )
            
            # Create output transform (always regression for MoleculeACE)
            output_transform = UnscaleTransform.from_standard_scaler(scaler)
            
            # Create model with intermediate-trained encoder
            # Note: We create new MP and agg instances for each benchmark to avoid state leakage
            mp_instance = BondMessagePassing(**valid_params)
            mp_instance.load_state_dict(intermediate_weights["message_passing_state_dict"])
            agg_instance = MeanAggregation()
            agg_instance.load_state_dict(intermediate_weights["aggregation_state_dict"])
            
            fnn = RegressionFFN(
                output_transform=output_transform,
                input_dim=hidden_size,
                hidden_dim=2048,  # Match CheMeleon's FFN hidden dimension
            )
            model = MPNN(mp_instance, agg_instance, fnn)
            
            # Setup training
            _subdir = "".join(c if c.isalnum() else "_" for c in benchmark_name)
            tensorboard_logger = TensorBoardLogger(
                seed_dir / _subdir,
                name="tensorboard_logs",
                default_hp_metric=False,
            )
            callbacks = [
                EarlyStopping(
                    monitor="val_loss",
                    mode="min",
                    verbose=False,
                    patience=5,
                ),
                ModelCheckpoint(
                    monitor="val_loss",
                    save_top_k=2,
                    mode="min",
                    dirpath=seed_dir / _subdir / "checkpoints",
                ),
            ]
            trainer = Trainer(
                max_epochs=50,
                logger=tensorboard_logger,
                log_every_n_steps=1,
                enable_checkpointing=True,
                check_val_every_n_epoch=1,
                callbacks=callbacks,
            )
            
            # Train
            trainer.fit(model, train_dataloader, val_dataloader)
            ckpt_path = trainer.checkpoint_callback.best_model_path
            print(f"Reloading best model from checkpoint: {ckpt_path}")
            
            # Clean up and reload
            del model, train_dataloader, train_dataset, val_dataloader, val_dataset
            torch.cuda.empty_cache()
            model = MPNN.load_from_checkpoint(ckpt_path)
            trainer = Trainer(logger=tensorboard_logger)
            
            # Predict on test set
            predictions = (
                torch.vstack(trainer.predict(model, test_dataloader))
                .numpy(force=True)
                .flatten()
            )
            
            # Compute metrics
            results = pd.DataFrame.from_records(
                [
                    dict(
                        metric="overall test rmse",
                        value=root_mean_squared_error(
                            predictions, test_df["y"]
                        ),
                    ),
                    dict(
                        metric="noncliff test rmse",
                        value=root_mean_squared_error(
                            predictions[test_df["cliff_mol"] == 0],
                            test_df[test_df["cliff_mol"] == 0]["y"],
                        ),
                    ),
                    dict(
                        metric="cliff test rmse",
                        value=root_mean_squared_error(
                            predictions[test_df["cliff_mol"] == 1],
                            test_df[test_df["cliff_mol"] == 1]["y"],
                        ),
                    ),
                ],
                index="metric",
            )
            performance = {
                "cliff": results.at["cliff test rmse", "value"],
                "noncliff": results.at["noncliff test rmse", "value"],
            }
            
            # Write results
            output_file.write(
                f"""
### `{benchmark_name}`

{results.to_markdown()}

"""
            )
            performance_dict[benchmark_name] = performance
            
            # Clean up checkpoints to save disk space
            shutil.rmtree(seed_dir / _subdir / "checkpoints")
    
    # Write summary
    output_file.write(
        f"""
### Summary

```
results_dict = {json.dumps(performance_dict, indent=4)}
```
"""
    )
    output_file.close()
    print(f"\n{'='*60}")
    print(f"Evaluation complete! Results saved to {output_dir / 'train_results.md'}")
    print(f"{'='*60}")
