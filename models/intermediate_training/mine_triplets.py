"""
Mine pairs from MoleculeACE datasets for intermediate training using contrastive loss.

This script implements MoleculeACE consensus similarity criteria:
- Uses THREE similarity metrics: ECFP Tanimoto, Scaffold ECFP Tanimoto, SMILES Levenshtein
- Molecules are "highly similar" if ANY metric > 0.9 (MoleculeACE standard)
- Mines pairs (more data-efficient than triplets):
  - Consistent pairs: Highly similar structure (consensus > 0.9) + similar activity (delta < 0.5)
  - Cliff pairs: Highly similar structure (consensus > 0.9) + large activity difference (delta >= 1.0)

Using pairs instead of triplets allows us to use ALL cliff pairs, not just those
that can form complete triplets, significantly increasing dataset size.
"""

import argparse
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

# MoleculeACE commit used in this repo
MOLECULEACE_COMMIT = "7e6de0bd2968c56589c580f2a397f01c531ede26"
BASE_URL = (
    "https://raw.githubusercontent.com/molML/MoleculeACE/"
    f"{MOLECULEACE_COMMIT}/MoleculeACE/Data/benchmark_data"
)

# All MoleculeACE benchmarks
MOLECULEACE_TARGETS = (
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


def load_mace_target(name: str) -> pd.DataFrame:
    """Load a MoleculeACE target dataset."""
    url = f"{BASE_URL}/{name}.csv"
    df = pd.read_csv(url)
    df["target_name"] = name
    return df


def compute_fingerprints(smiles_list: List[str], radius: int = 2, n_bits: int = 2048):
    """
    Compute Morgan fingerprints (ECFP) for SMILES strings.
    
    Returns:
        mols: List of RDKit Mol objects (or None for invalid SMILES)
        fps: List of ECFP fingerprints (or None for invalid SMILES)
    """
    mols = []
    fps = []
    for smi in tqdm(smiles_list, desc="Computing ECFP fingerprints"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            mols.append(None)
            fps.append(None)
        else:
            mols.append(mol)
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits))
    return mols, fps


def compute_scaffold_fingerprints(mols: List, radius: int = 2, n_bits: int = 2048):
    """
    Compute ECFP fingerprints on molecular scaffolds.
    
    Args:
        mols: List of RDKit Mol objects (can contain None)
        radius: Morgan fingerprint radius
        n_bits: Number of bits in fingerprint
        
    Returns:
        List of scaffold fingerprints (or None for invalid molecules)
    """
    scaffold_fps = []
    for mol in tqdm(mols, desc="Computing scaffold fingerprints"):
        if mol is None:
            scaffold_fps.append(None)
        else:
            try:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_fp = AllChem.GetMorganFingerprintAsBitVect(
                    scaffold, radius=radius, nBits=n_bits
                )
                scaffold_fps.append(scaffold_fp)
            except:
                scaffold_fps.append(None)
    return scaffold_fps


def compute_smiles_similarity(smi1: str, smi2: str) -> float:
    """
    Compute SMILES similarity using normalized Levenshtein distance.
    
    MoleculeACE uses Levenshtein distance on SMILES strings.
    We convert to similarity: 1 - (normalized_levenshtein_distance)
    
    Uses optimized python-Levenshtein library if available, otherwise falls back
    to manual implementation (much slower for large datasets).
    
    Args:
        smi1, smi2: SMILES strings
        
    Returns:
        Similarity score between 0 and 1 (1 = identical)
    """
    # Try to use optimized C implementation
    try:
        import Levenshtein
        use_optimized = True
    except ImportError:
        use_optimized = False
    
    if smi1 == smi2:
        return 1.0
    
    max_len = max(len(smi1), len(smi2))
    if max_len == 0:
        return 1.0
    
    if use_optimized:
        # Use optimized C implementation (much faster)
        distance = Levenshtein.distance(smi1, smi2)
    else:
        # Fallback to manual implementation (slower, O(n*m))
        def levenshtein_distance(s1: str, s2: str) -> int:
            """Compute Levenshtein distance between two strings."""
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            if len(s2) == 0:
                return len(s1)
            
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            return previous_row[-1]
        
        distance = levenshtein_distance(smi1, smi2)
    
    normalized_distance = distance / max_len
    similarity = 1.0 - normalized_distance
    return similarity


def compute_consensus_similarity(
    fp1, fp2, scaffold_fp1, scaffold_fp2, smi1: str, smi2: str, thresh: float = 0.9
) -> Tuple[bool, float]:
    """
    Compute consensus similarity using MoleculeACE criteria.
    
    MoleculeACE considers molecules "highly similar" if ANY of these > 0.9:
    1. ECFP Tanimoto similarity
    2. Scaffold ECFP Tanimoto similarity  
    3. SMILES Levenshtein-based similarity
    
    Args:
        fp1, fp2: ECFP fingerprints
        scaffold_fp1, scaffold_fp2: Scaffold ECFP fingerprints
        smi1, smi2: SMILES strings
        thresh: Similarity threshold (default 0.9)
        
    Returns:
        (is_similar, max_similarity): Tuple of (bool, float)
    """
    similarities = []
    
    # 1. ECFP Tanimoto
    if fp1 is not None and fp2 is not None:
        ecfp_sim = DataStructs.TanimotoSimilarity(fp1, fp2)
        similarities.append(("ECFP", ecfp_sim))
    
    # 2. Scaffold ECFP Tanimoto
    if scaffold_fp1 is not None and scaffold_fp2 is not None:
        scaffold_sim = DataStructs.TanimotoSimilarity(scaffold_fp1, scaffold_fp2)
        similarities.append(("Scaffold", scaffold_sim))
    
    # 3. SMILES Levenshtein similarity
    smiles_sim = compute_smiles_similarity(smi1, smi2)
    similarities.append(("SMILES", smiles_sim))
    
    if not similarities:
        return False, 0.0
    
    max_sim = max(sim for _, sim in similarities)
    is_similar = max_sim > thresh
    
    return is_similar, max_sim


def mine_pairs(
    df: pd.DataFrame,
    fps: List,
    scaffold_fps: List,
    smiles_list: List[str],
    similarity_thresh: float = 0.9,
    min_cliff_delta: float = 1.0,
    consistent_delta: float = 0.5,
) -> pd.DataFrame:
    """
    Mine pairs from a MoleculeACE dataset using MoleculeACE consensus criteria.
    
    This is more data-efficient than triplets because it only requires pairs:
    - Consistent pairs: Similar structure (consensus > 0.9) + similar activity (delta < 0.5)
    - Cliff pairs: Similar structure (consensus > 0.9) + large activity difference (delta >= 1.0)
    
    MoleculeACE considers molecules "highly similar" if ANY of these > 0.9:
    1. ECFP Tanimoto similarity
    2. Scaffold ECFP Tanimoto similarity
    3. SMILES Levenshtein-based similarity
    
    Args:
        df: DataFrame with columns 'smiles', 'y', 'cliff_mol'
        fps: List of ECFP fingerprints (can contain None for invalid SMILES)
        scaffold_fps: List of scaffold ECFP fingerprints (can contain None)
        smiles_list: List of SMILES strings
        similarity_thresh: Minimum similarity threshold (default: 0.9, MoleculeACE standard)
        min_cliff_delta: Minimum activity difference for cliff pair (default: 1.0 = 10x in log space)
        consistent_delta: Maximum activity difference for consistent pair (default: 0.5)
        
    Returns:
        DataFrame with columns: mol1_idx, mol2_idx, y1, y2, max_similarity, activity_diff,
        pair_type ('consistent' or 'cliff'), target_name
    """
    pairs = []
    n = len(df)
    
    # Filter valid fingerprints (need at least ECFP)
    valid_mask = [fp is not None for fp in fps]
    valid_indices = np.where(valid_mask)[0]
    
    print(f"Mining pairs from {len(valid_indices)} valid molecules...")
    print(f"Using MoleculeACE consensus similarity (ECFP, Scaffold, SMILES) with threshold {similarity_thresh}")
    print(f"Cliff pairs: activity_diff >= {min_cliff_delta}, Consistent pairs: activity_diff < {consistent_delta}")
    
    # Find all similar pairs
    for i in tqdm(valid_indices, desc="Mining pairs"):
        fp_i = fps[i]
        scaffold_fp_i = scaffold_fps[i] if i < len(scaffold_fps) else None
        smi_i = smiles_list[i]
        y_i = float(df.iloc[i]["y"])
        
        # Only check pairs where j > i to avoid duplicates
        for j in valid_indices:
            if j <= i:
                continue
            
            fp_j = fps[j]
            scaffold_fp_j = scaffold_fps[j] if j < len(scaffold_fps) else None
            smi_j = smiles_list[j]
            y_j = float(df.iloc[j]["y"])
            
            # Use consensus similarity (any metric > threshold qualifies)
            is_similar, max_sim = compute_consensus_similarity(
                fp_i, fp_j,
                scaffold_fp_i, scaffold_fp_j,
                smi_i, smi_j,
                thresh=similarity_thresh
            )
            
            if not is_similar:
                continue
            
            # Calculate activity difference
            activity_diff = abs(y_i - y_j)
            
            # Classify pair type
            if activity_diff >= min_cliff_delta:
                pair_type = "cliff"
            elif activity_diff < consistent_delta:
                pair_type = "consistent"
            else:
                # Skip pairs in the middle zone (not clearly consistent or cliff)
                continue
            
            pairs.append({
                "mol1_idx": i,
                "mol2_idx": j,
                "y1": y_i,
                "y2": y_j,
                "max_similarity": float(max_sim),
                "activity_diff": float(activity_diff),
                "pair_type": pair_type,
                "target_name": df.iloc[i]["target_name"],
            })
    
    return pd.DataFrame(pairs)


def mine_triplets(
    df: pd.DataFrame,
    fps: List,
    scaffold_fps: List,
    smiles_list: List[str],
    similarity_thresh: float = 0.9,
    min_cliff_delta: float = 1.0,
    prioritize_cliffs: bool = True,
) -> pd.DataFrame:
    """
    Mine triplets from a MoleculeACE dataset using MoleculeACE consensus criteria.
    
    MoleculeACE considers molecules "highly similar" if ANY of these > 0.9:
    1. ECFP Tanimoto similarity
    2. Scaffold ECFP Tanimoto similarity
    3. SMILES Levenshtein-based similarity
    
    Args:
        df: DataFrame with columns 'smiles', 'y', 'cliff_mol'
        fps: List of ECFP fingerprints (can contain None for invalid SMILES)
        scaffold_fps: List of scaffold ECFP fingerprints (can contain None)
        smiles_list: List of SMILES strings
        similarity_thresh: Minimum similarity threshold (default: 0.9, MoleculeACE standard)
        min_cliff_delta: Minimum activity difference for negative pair (default: 1.0 = 10x in log space)
        prioritize_cliffs: If True, prioritize cliff compounds as anchors but allow all
        
    Returns:
        DataFrame with columns: anchor_idx, pos_idx, neg_idx, y_anchor, y_pos, y_neg,
        max_similarity_anchor_pos, max_similarity_anchor_neg, activity_diff_pos, activity_diff_neg,
        is_cliff_anchor, target_name
    """
    triplets = []
    n = len(df)
    
    # Filter valid fingerprints (need at least ECFP)
    valid_mask = [fp is not None for fp in fps]
    valid_indices = np.where(valid_mask)[0]
    
    print(f"Mining triplets from {len(valid_indices)} valid molecules...")
    print(f"Using MoleculeACE consensus similarity (ECFP, Scaffold, SMILES) with threshold {similarity_thresh}")
    
    # Separate cliff and non-cliff anchors for prioritization
    cliff_anchors = []
    non_cliff_anchors = []
    
    for i in valid_indices:
        is_cliff = df.iloc[i].get("cliff_mol", 0) == 1
        if is_cliff:
            cliff_anchors.append(i)
        else:
            non_cliff_anchors.append(i)
    
    # Prioritize cliff compounds: process them first
    if prioritize_cliffs:
        anchor_candidates = cliff_anchors + non_cliff_anchors
        print(f"Prioritizing {len(cliff_anchors)} cliff compounds as anchors, then {len(non_cliff_anchors)} others")
    else:
        anchor_candidates = valid_indices.tolist()
        print(f"Using all {len(anchor_candidates)} molecules as potential anchors")
    
    for i in tqdm(anchor_candidates, desc="Mining triplets"):
        fp_anchor = fps[i]
        scaffold_fp_anchor = scaffold_fps[i] if i < len(scaffold_fps) else None
        smi_anchor = smiles_list[i]
        y_anchor = float(df.iloc[i]["y"])
        is_cliff_anchor = df.iloc[i].get("cliff_mol", 0) == 1
        
        # Find all similar molecules using consensus similarity
        similar_indices = []
        similar_sims = []
        
        for j in valid_indices:
            if j == i:
                continue
            
            fp_j = fps[j]
            scaffold_fp_j = scaffold_fps[j] if j < len(scaffold_fps) else None
            smi_j = smiles_list[j]
            
            # Use consensus similarity (any metric > threshold qualifies)
            is_similar, max_sim = compute_consensus_similarity(
                fp_anchor, fp_j,
                scaffold_fp_anchor, scaffold_fp_j,
                smi_anchor, smi_j,
                thresh=similarity_thresh
            )
            
            if is_similar:
                similar_indices.append(j)
                similar_sims.append(max_sim)
        
        if len(similar_indices) < 2:
            continue  # Need at least 2 similar molecules
        
        # Sort by similarity (highest first)
        similar_indices = np.array(similar_indices)
        similar_sims = np.array(similar_sims)
        sorted_idx = np.argsort(similar_sims)[::-1]
        similar_indices = similar_indices[sorted_idx]
        similar_sims = similar_sims[sorted_idx]
        
        # Find positive (similar structure, similar activity)
        # and negative (similar structure, large activity difference)
        pos_idx = None
        neg_idx = None
        pos_sim = None
        neg_sim = None
        
        for j, sim in zip(similar_indices, similar_sims):
            y_j = float(df.iloc[j]["y"])
            delta = abs(y_j - y_anchor)
            
            # Positive: similar structure, similar activity (small delta)
            if pos_idx is None and delta < (min_cliff_delta / 2.0):
                pos_idx = j
                pos_sim = sim
            
            # Negative: similar structure, large activity difference (cliff)
            if neg_idx is None and delta >= min_cliff_delta:
                neg_idx = j
                neg_sim = sim
            
            if pos_idx is not None and neg_idx is not None:
                break
        
        if pos_idx is not None and neg_idx is not None:
            triplets.append({
                "anchor_idx": i,
                "pos_idx": int(pos_idx),
                "neg_idx": int(neg_idx),
                "y_anchor": y_anchor,
                "y_pos": float(df.iloc[pos_idx]["y"]),
                "y_neg": float(df.iloc[neg_idx]["y"]),
                "max_similarity_anchor_pos": float(pos_sim),
                "max_similarity_anchor_neg": float(neg_sim),
                "activity_diff_pos": abs(y_anchor - float(df.iloc[pos_idx]["y"])),
                "activity_diff_neg": abs(y_anchor - float(df.iloc[neg_idx]["y"])),
                "is_cliff_anchor": bool(is_cliff_anchor),
                "target_name": df.iloc[i]["target_name"],
            })
    
    return pd.DataFrame(triplets)


def main():
    parser = argparse.ArgumentParser(description="Mine pairs from MoleculeACE datasets for contrastive loss")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for pairs DataFrame (pickle or CSV)",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help="Specific targets to mine (default: all MoleculeACE targets)",
    )
    parser.add_argument(
        "--similarity-thresh",
        type=float,
        default=0.9,
        help="Minimum similarity threshold for MoleculeACE consensus (default: 0.9)",
    )
    parser.add_argument(
        "--min-cliff-delta",
        type=float,
        default=1.0,
        help="Minimum activity difference for cliff pair in log space (default: 1.0 = 10x difference)",
    )
    parser.add_argument(
        "--consistent-delta",
        type=float,
        default=0.5,
        help="Maximum activity difference for consistent pair (default: 0.5)",
    )
    
    args = parser.parse_args()
    
    # Load datasets
    targets = args.targets if args.targets else MOLECULEACE_TARGETS
    print(f"Loading {len(targets)} MoleculeACE targets...")
    
    dfs = []
    for t in tqdm(targets, desc="Loading datasets"):
        try:
            df = load_mace_target(t)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Failed to load {t}: {e}")
            continue
    
    if len(dfs) == 0:
        raise ValueError("No datasets loaded successfully!")
    
    mace = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(mace)} total molecules")
    
    # IMPORTANT: Respect MoleculeACE train/test splits to avoid data leakage
    # Only mine triplets from training data
    if "split" in mace.columns:
        mace_train = mace[mace["split"] == "train"].copy()
        mace_test = mace[mace["split"] == "test"].copy()
        print(f"MoleculeACE splits: {len(mace_train)} train, {len(mace_test)} test")
        print(f"Using only TRAIN split for triplet mining to avoid data leakage")
    else:
        print("Warning: No 'split' column found in MoleculeACE data!")
        print("This may cause data leakage. Proceeding with all data.")
        mace_train = mace.copy()
        mace_test = pd.DataFrame()
    
    # Filter to valid SMILES and numeric y (only for train)
    mace_train = mace_train.dropna(subset=["smiles", "y"]).reset_index(drop=True)
    mace_train["y"] = pd.to_numeric(mace_train["y"], errors="coerce")
    mace_train = mace_train.dropna(subset=["y"]).reset_index(drop=True)
    print(f"After filtering train data: {len(mace_train)} molecules")
    
    # Check for optimized Levenshtein library (much faster for large datasets)
    try:
        import Levenshtein
        print("✓ Using optimized python-Levenshtein library for SMILES similarity (fast)")
    except ImportError:
        print("⚠ WARNING: python-Levenshtein not installed. Using slow manual implementation.")
        print("   For 10-100x faster mining, install: pip install python-Levenshtein")
        print("   This is especially important for large datasets!")
    
    # Compute fingerprints (only for train data)
    smiles_list = mace_train["smiles"].tolist()
    mols, fps = compute_fingerprints(smiles_list)
    
    # Compute scaffold fingerprints for MoleculeACE consensus
    scaffold_fps = compute_scaffold_fingerprints(mols)
    
    # Mine pairs (only from train data) using MoleculeACE consensus criteria
    # Using pairs instead of triplets for better data efficiency
    pairs_df = mine_pairs(
        mace_train,
        fps,
        scaffold_fps,
        smiles_list,
        similarity_thresh=args.similarity_thresh,
        min_cliff_delta=args.min_cliff_delta,
        consistent_delta=args.consistent_delta,
    )
    
    print(f"\nMined {len(pairs_df)} pairs")
    print(f"Pair types:")
    print(pairs_df["pair_type"].value_counts())
    print(f"\nPairs per target:")
    print(pairs_df["target_name"].value_counts())
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix == ".pkl" or output_path.suffix == ".pickle":
        pairs_df.to_pickle(output_path)
    else:
        pairs_df.to_csv(output_path, index=False)
    
    print(f"\nSaved pairs to {output_path}")
    
    # Save both train and test data separately for reference
    # This preserves the MoleculeACE splits for downstream use
    train_path = output_path.parent / f"{output_path.stem}_mace_train.pkl"
    mace_train.to_pickle(train_path)
    print(f"Saved MoleculeACE TRAIN data to {train_path}")
    
    if len(mace_test) > 0:
        # Also save test data if available (for final evaluation)
        test_path = output_path.parent / f"{output_path.stem}_mace_test.pkl"
        mace_test.to_pickle(test_path)
        print(f"Saved MoleculeACE TEST data to {test_path}")
    
    # Also save combined for backward compatibility (but warn about split)
    ref_path = output_path.parent / f"{output_path.stem}_mace_data.pkl"
    mace_train.to_pickle(ref_path)  # Save train as "mace_data" for backward compat
    print(f"Saved MoleculeACE data (TRAIN only) to {ref_path} for backward compatibility")


if __name__ == "__main__":
    main()
