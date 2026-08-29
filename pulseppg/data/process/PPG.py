import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch_ecg._preprocessors import Normalize
from utils import resample_batch_signal, preprocess_one_ppg_signal
import joblib
from sklearn.model_selection import train_test_split

WAVELENGTH_DIRS = ["ppg_660nm", "ppg_730nm", "ppg_850nm", "ppg_940nm"]


# ==========================
# Data Augmentation Functions
# ==========================

def augment_ppg_signal(signal: np.ndarray, aug_type: str = "jitter") -> np.ndarray:
    """
    Apply augmentation to a PPG signal (B, 1, L) or (1, L).
    
    Args:
        signal: PPG signal array, shape (1, L) or (1, 1, L)
        aug_type: Type of augmentation ('jitter', 'scaling', 'rotation', 'random')
    
    Returns:
        Augmented signal with same shape as input
    """
    # Handle both (1, L) and (1, 1, L) shapes
    original_shape = signal.shape
    if signal.ndim == 3:
        signal = signal[0]  # Remove batch dim if present
    
    signal = signal.ravel()
    
    if aug_type == "jitter":
        # Add small Gaussian noise
        noise = np.random.normal(0, 0.02 * np.std(signal), len(signal))
        augmented = signal + noise
        
    elif aug_type == "scaling":
        # Scale signal amplitude (preserve DC component)
        scale_factor = np.random.uniform(0.9, 1.1)
        mean_val = np.mean(signal)
        augmented = mean_val + scale_factor * (signal - mean_val)
        
    elif aug_type == "rotation":
        # Random rotation: shift and roll the signal
        shift = np.random.randint(1, max(2, len(signal) // 10))
        augmented = np.roll(signal, shift)
        
    elif aug_type == "mixup":
        # Mix with slightly different version of same signal
        noise = np.random.normal(0, 0.01 * np.std(signal), len(signal))
        augmented = 0.95 * signal + 0.05 * (signal + noise)
        
    else:
        # Random choice of augmentation
        aug_types = ["jitter", "scaling", "mixup"]
        chosen = np.random.choice(aug_types)
        augmented = augment_ppg_signal(signal.reshape(1, -1), aug_type=chosen)
        augmented = augmented.ravel()
    
    # Reshape back to original shape
    if len(original_shape) == 3:
        augmented = augmented.reshape(original_shape)
    else:
        augmented = augmented.reshape(original_shape)
    
    return augmented


def augment_high_glucose_samples(
    ppgpath: str,
    glucose_threshold: float = 7.0,
    num_augmentations: int = 3,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int | None = None,
) -> dict:
    """
    Find samples with glucose > threshold, augment them, and randomly distribute across splits.
    
    Augmented samples are NOT added back to their original split, but randomly assigned
    to train/val/test according to the specified ratios.
    
    Args:
        ppgpath: Path to dataset directory with NPY files
        glucose_threshold: Glucose level threshold (mmol/L or mg/dL depending on data)
        num_augmentations: Number of augmented copies per high-glucose sample
        train_ratio: Fraction of augmented samples for training (default: 0.7)
        val_ratio: Fraction of augmented samples for validation (default: 0.15)
        test_ratio: Fraction of augmented samples for testing (default: 0.15)
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with augmentation statistics
    """
    if seed is not None:
        np.random.seed(seed)
    
    wavelengths = ("660", "730", "850", "940")
    
    # Normalize ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    train_ratio /= total_ratio
    val_ratio /= total_ratio
    test_ratio /= total_ratio
    
    stats = {
        "high_glucose_samples_found": 0,
        "augmented_samples_created": 0,
        "augmented_per_split": {"train": 0, "val": 0, "test": 0},
        "splits_modified": [],
    }
    
    print(f"\n{'='*70}")
    print(f"AUGMENTING HIGH GLUCOSE SAMPLES WITH RANDOM SPLIT DISTRIBUTION")
    print(f"Target ratio: train={train_ratio:.1%}, val={val_ratio:.1%}, test={test_ratio:.1%}")
    print(f"Glucose threshold: > {glucose_threshold}")
    print(f"{'='*70}\n")
    
    # Step 1: Identify all high glucose samples across all splits
    all_high_glucose_samples = []  # List of (split_name, sample_idx, signal, label)
    
    for split_name in ["train", "val", "test"]:
        glucose_file = os.path.join(ppgpath, f"{split_name}_y_glucose_660nm.npy")
        if not os.path.exists(glucose_file):
            print(f"⚠ Glucose file not found: {glucose_file}")
            continue
        
        glucose_labels = np.load(glucose_file)
        high_glucose_mask = glucose_labels > glucose_threshold
        high_glucose_indices = np.where(high_glucose_mask)[0]
        
        print(f"{split_name.upper()}: Found {len(high_glucose_indices)} high glucose samples")
        
        for idx in high_glucose_indices:
            all_high_glucose_samples.append({
                'split': split_name,
                'idx': idx,
                'glucose': glucose_labels[idx]
            })
        
        stats["high_glucose_samples_found"] += len(high_glucose_indices)
    
    if stats["high_glucose_samples_found"] == 0:
        print(f"\n⚠ No high glucose samples found with glucose > {glucose_threshold}")
        return stats
    
    print(f"\nTotal high glucose samples found: {stats['high_glucose_samples_found']}")
    
    # Step 2: Generate augmented samples for each high glucose sample
    print(f"\nGenerating augmented samples...\n")
    
    augmented_samples = {split_name: {"data": [], "labels": [], "subjects": []} 
                         for split_name in ["train", "val", "test"]}
    
    subject_id_counter = 20000  # Use high offset for augmented subject IDs
    
    for sample_info in all_high_glucose_samples:
        original_split = sample_info['split']
        original_idx = sample_info['idx']
        glucose_label = sample_info['glucose']
        
        for aug_num in range(num_augmentations):
            # Randomly assign this augmented sample to a split
            rand_val = np.random.uniform(0, 1)
            if rand_val < train_ratio:
                target_split = "train"
            elif rand_val < train_ratio + val_ratio:
                target_split = "val"
            else:
                target_split = "test"
            
            # For each wavelength, augment and store
            for wl in wavelengths:
                data_file = os.path.join(ppgpath, f"{original_split}_X_ppg_50Hz_{wl}nm.npy")
                data = np.load(data_file)
                original_signal = data[original_idx]  # (1, L)
                
                # Augment
                aug_type = np.random.choice(["jitter", "scaling", "mixup"])
                augmented_signal = augment_ppg_signal(original_signal, aug_type=aug_type)
                
                augmented_samples[target_split]["data"].append((wl, augmented_signal))
            
            augmented_samples[target_split]["labels"].append(glucose_label)
            augmented_samples[target_split]["subjects"].append(subject_id_counter)
            
            subject_id_counter += 1
            stats["augmented_samples_created"] += 1
    
    # Step 3: Add augmented samples to each split's files
    print(f"\nAdding augmented samples to splits...\n")
    
    for split_name in ["train", "val", "test"]:
        if len(augmented_samples[split_name]["labels"]) == 0:
            print(f"{split_name.upper()}: No augmented samples to add")
            continue
        
        n_aug = len(augmented_samples[split_name]["labels"])
        stats["augmented_per_split"][split_name] = n_aug
        stats["splits_modified"].append(split_name)
        
        print(f"{split_name.upper()}: Adding {n_aug} augmented samples")
        
        # Group augmented data by wavelength
        aug_by_wl = {wl: [] for wl in wavelengths}
        for wl_signal_pair in augmented_samples[split_name]["data"]:
            wl, signal = wl_signal_pair
            aug_by_wl[wl].append(signal)
        
        # Add to each wavelength's files
        for wl in wavelengths:
            data_file = os.path.join(ppgpath, f"{split_name}_X_ppg_50Hz_{wl}nm.npy")
            label_file = os.path.join(ppgpath, f"{split_name}_y_glucose_{wl}nm.npy")
            subject_file = os.path.join(ppgpath, f"{split_name}_subject_ids_{wl}nm.npy")
            
            # Load existing
            data = np.load(data_file)
            labels = np.load(label_file)
            subject_ids = np.load(subject_file)
            
            # Prepare augmented data for this wavelength (should have n_aug samples)
            # Note: Each augmented sample has data for each wavelength
            n_expected = n_aug
            if len(aug_by_wl[wl]) != n_expected:
                print(f"  ⚠ Warning: Expected {n_expected} augmented samples for {wl}, got {len(aug_by_wl[wl])}")
                continue
            
            augmented_data = np.array(aug_by_wl[wl])  # (n_aug, 1, L)
            augmented_labels = np.array(augmented_samples[split_name]["labels"], dtype=np.float32)
            augmented_subjects = np.array(augmented_samples[split_name]["subjects"], dtype=np.int32)
            
            # Concatenate
            combined_data = np.concatenate([data, augmented_data], axis=0)
            combined_labels = np.concatenate([labels, augmented_labels])
            combined_subjects = np.concatenate([subject_ids, augmented_subjects])
            
            # Save
            np.save(data_file, combined_data)
            np.save(label_file, combined_labels)
            np.save(subject_file, combined_subjects)
        
        # Verify
        new_glucose = np.load(os.path.join(ppgpath, f"{split_name}_y_glucose_660nm.npy"))
        n_total = len(new_glucose)
        n_high = np.sum(new_glucose > glucose_threshold)
        print(f"  Result: {n_total} total samples, {n_high} high glucose ({100*n_high/n_total:.1f}%)")
    
    print(f"\n{'='*70}")
    print(f"AUGMENTATION COMPLETE")
    print(f"{'='*70}")
    print(f"High glucose samples found: {stats['high_glucose_samples_found']}")
    print(f"Augmented samples created: {stats['augmented_samples_created']}")
    print(f"Distribution:")
    print(f"  train: {stats['augmented_per_split']['train']}")
    print(f"  val:   {stats['augmented_per_split']['val']}")
    print(f"  test:  {stats['augmented_per_split']['test']}")
    print(f"\nAugmented samples randomly distributed across all splits!")
    print(f"{'='*70}\n")
    
    return stats


def _load_snr_filter(ppgpath: str, snr_threshold: float = 40.0) -> set | None:
    """
    Load SNR results if available and filter subjects by SNR_Mean threshold.
    
    Args:
        ppgpath: Path to dataset root
        snr_threshold: Minimum SNR_Mean value to keep subject (default: 40.0 dB)
    
    Returns:
        Set of subject IDs with SNR >= threshold, or None if no SNR file found
    """
    snr_file_candidates = [
        os.path.join(ppgpath, "snr_results.csv"),
        os.path.join(ppgpath, "Data File", "snr_results.csv"),
    ]
    
    snr_file = None
    for candidate in snr_file_candidates:
        if os.path.exists(candidate):
            snr_file = candidate
            break
    
    if snr_file is None:
        print(f"⚠ SNR results file not found. Processing all subjects.")
        return None
    
    try:
        snr_df = pd.read_csv(snr_file)
        if 'File' not in snr_df.columns or 'SNR_Mean' not in snr_df.columns:
            print(f"⚠ SNR file is missing 'File' or 'SNR_Mean' columns. Processing all subjects.")
            return None
        
        # Filter by SNR threshold
        high_quality = snr_df[snr_df['SNR_Mean'] >= snr_threshold]['File'].astype(int).tolist()
        high_quality_set = set(high_quality)
        
        print(f"✓ Loaded SNR results: {len(high_quality)} subjects with SNR_Mean >= {snr_threshold} dB")
        print(f"  Subject IDs: {sorted(high_quality)}")
        
        return high_quality_set
    
    except Exception as e:
        print(f"⚠ Error loading SNR results: {e}. Processing all subjects.")
        return None

# ==========================
# Helper functions
# ==========================

def _load_subjects_metadata(ppgpath: str) -> tuple[pd.DataFrame, str]:
    """
    Load subject metadata from Excel or CSV file.
    
    Supported formats:
      - 'Subjects Information.xlsx' (Excel format)
      - 'Subjects Information.csv' (CSV format)
      - Any file matching pattern 'Subjects*' in Data File/ or root directory
    
    Expected columns (flexible naming):
        ID (or subject_ID), SBP(mmHg) or SBP, DBP(mmHg) or DBP, Age, Gender, etc.
    """
    # Try loading from different locations in order of preference
    search_paths = [
        os.path.join(ppgpath, "Data File", "Subjects Information.xlsx"),
        os.path.join(ppgpath, "Data File", "Subjects Information.csv"),
        os.path.join(ppgpath, "Subjects Information.xlsx"),
        os.path.join(ppgpath, "Subjects Information.csv"),
    ]
    
    df = None
    loaded_from = None
    
    # Try explicit filenames first
    for path in search_paths:
        if os.path.exists(path):
            try:
                if path.endswith('.xlsx'):
                    df = pd.read_excel(path, header=0)
                    loaded_from = path
                    break
                elif path.endswith('.csv'):
                    df = pd.read_csv(path, header=0)
                    loaded_from = path
                    break
            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")
                continue
    
    # Fallback: search for any 'Subjects*' files in Data File/ directory
    if df is None:
        data_file_dir = os.path.join(ppgpath, "Data File")
        if os.path.isdir(data_file_dir):
            for fname in os.listdir(data_file_dir):
                if fname.lower().startswith("subjects") and (fname.endswith(".xlsx") or fname.endswith(".csv")):
                    path = os.path.join(data_file_dir, fname)
                    try:
                        if fname.endswith('.xlsx'):
                            df = pd.read_excel(path, header=0)
                        else:
                            df = pd.read_csv(path, header=0)
                        loaded_from = path
                        break
                    except Exception as e:
                        print(f"Warning: Failed to load {path}: {e}")
                        continue
    
    if df is None:
        raise FileNotFoundError(
            f"Subject metadata not found. Searched for 'Subjects Information.xlsx/csv' in:\n"
            f"  - {os.path.join(ppgpath, 'Data File')}\n"
            f"  - {ppgpath}"
        )
    
    # Normalize column names (strip whitespace, standardize case)
    df.columns = [c.strip() for c in df.columns]
    print(f"✓ Loaded metadata from: {loaded_from}")
    print(f"  Columns: {', '.join(df.columns)}")

    # Determine ID column name
    id_col = None
    for col_candidate in ["ID", "subject_ID", "Subject_ID", "id"]:
        if col_candidate in df.columns:
            id_col = col_candidate
            break
    
    if id_col is None:
        # Default to first column if identified
        id_col = df.columns[0]
        print(f"  Warning: No explicit ID column found. Using first column: '{id_col}'")
    
    # Convert ID to numeric (Int64 for nullable integer support)
    df[id_col] = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
    
    print(f"  ID column: '{id_col}'")
    print(f"  Records loaded: {len(df)}")

    return df, id_col


def _build_random_splits(
    df: pd.DataFrame,
    id_col: str,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int | None,
    stratify_col: str | None = None,
):
    """
    Return (train_ids, val_ids, test_ids) as Python int lists.
    
    Args:
        df: DataFrame with subject metadata
        id_col: Column name for subject IDs
        train_frac, val_frac, test_frac: Split fractions
        seed: Random seed for reproducibility
        stratify_col: Column to stratify by (e.g., "glucose" for balanced distribution)
                     If None, use random split. If specified, ensures label distribution
                     is preserved across train/val/test splits.
    """
    unique_ids = df[id_col].dropna().unique()
    unique_ids = np.array(unique_ids)

    # normalize fractions if they don't sum to 1
    total_frac = train_frac + val_frac + test_frac
    if total_frac <= 0:
        raise ValueError("train_frac + val_frac + test_frac must be > 0")
    train_frac /= total_frac
    val_frac /= total_frac
    test_frac /= total_frac

    # Stratified split based on label distribution
    if stratify_col is not None and stratify_col in df.columns:
        print(f"  Using stratified split based on '{stratify_col}' distribution")
        
        # Get stratification column for each unique subject
        strat_values = []
        strat_ids = []
        for uid in unique_ids:
            val = df[df[id_col] == uid][stratify_col].iloc[0]
            strat_values.append(val)
            strat_ids.append(uid)
        
        strat_values = np.array(strat_values)
        strat_ids = np.array(strat_ids)
        
        # Discretize continuous values into bins for stratification
        # Use quantile-based binning to create balanced strata
        n_strata = min(5, len(unique_ids) // 3)  # 3-5 strata
        strat_bins = pd.qcut(strat_values, q=n_strata, labels=False, duplicates='drop')
        
        print(f"  Stratification bins: {n_strata} strata from {stratify_col}")
        print(f"  Distribution: {np.bincount(strat_bins)}")
        
        # First split: train vs (val+test)
        temp_train_ids, temp_valtest_ids, temp_train_strat, temp_valtest_strat = train_test_split(
            strat_ids, strat_bins, test_size=(1 - train_frac), stratify=strat_bins, random_state=seed
        )
        
        # Second split: val vs test (from remaining data)
        test_size_ratio = test_frac / (val_frac + test_frac) if (val_frac + test_frac) > 0 else 0.5
        val_ids, test_ids, _, _ = train_test_split(
            temp_valtest_ids, temp_valtest_strat, test_size=test_size_ratio, stratify=temp_valtest_strat, random_state=seed
        )
        
        train_ids = temp_train_ids.tolist()
        val_ids = val_ids.tolist()
        test_ids = test_ids.tolist()
        
        # Print stratification verification
        print(f"  Train {stratify_col} range: [{df[df[id_col].isin(train_ids)][stratify_col].min():.2f}, {df[df[id_col].isin(train_ids)][stratify_col].max():.2f}]")
        print(f"  Val {stratify_col} range: [{df[df[id_col].isin(val_ids)][stratify_col].min():.2f}, {df[df[id_col].isin(val_ids)][stratify_col].max():.2f}]")
        print(f"  Test {stratify_col} range: [{df[df[id_col].isin(test_ids)][stratify_col].min():.2f}, {df[df[id_col].isin(test_ids)][stratify_col].max():.2f}]")
    
    else:
        # Standard random split (no stratification)
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(unique_ids)

        n = len(shuffled)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        if n_train < 1:
            n_train = max(1, n - n_val)
        n_test = n - n_train - n_val
        if n_test < 0:
            n_val = max(0, n - n_train)
            n_test = n - n_train - n_val

        train_ids = shuffled[:n_train].tolist()
        val_ids = shuffled[n_train : n_train + n_val].tolist()
        test_ids = shuffled[n_train + n_val :].tolist()

    def to_py_int_list(lst):
        return [int(x) for x in lst]

    return to_py_int_list(train_ids), to_py_int_list(val_ids), to_py_int_list(test_ids)


def _normalize_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize metadata column names to standardized names:
        - age, sysbp, diasbp, sex
        - glucose (blood glucose)
        - hemoglobin (hemoglobin)
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    rename_map: dict[str, str] = {}

    # Age variants
    for col in ["Age", "Age (year)", "age"]:
        if col in df.columns:
            rename_map[col] = "age"
            break

    # SBP variants
    for col in ["SBP(mmHg)", "SBP (mmHg)", "sbp"]:
        if col in df.columns:
            rename_map[col] = "sysbp"
            break

    # DBP variants
    for col in ["DBP(mmHg)", "DBP (mmHg)", "dbp"]:
        if col in df.columns:
            rename_map[col] = "diasbp"
            break

    # Gender variants
    for col in ["Gender", "Gender:", "Gender: ", "Gender ", "sex"]:
        if col in df.columns:
            rename_map[col] = "sex"
            break

    # Glucose variants
    for col in ["Blood glucose (mmol/L)", "Blood glucose", "Glucose", "glucose"]:
        if col in df.columns:
            rename_map[col] = "glucose"
            break

    # Hemoglobin variants
    for col in ["Hemoglobin (g/L)", "Hemoglobin", "hemoglobin", "Hb"]:
        if col in df.columns:
            rename_map[col] = "hemoglobin"
            break

    df = df.rename(columns=rename_map).fillna(0)

    # Require SBP and DBP (core labels)
    for col in ["sysbp", "diasbp"]:
        if col not in df.columns:
            raise KeyError(
                f"Expected column '{col}' in subjects file after renaming, "
                f"but it is missing. Got columns: {df.columns}"
            )

    return df


# ==========================
# CSV Processor (1 channel)
# ==========================

class MWPPGCSVDataProcessor:
    """
    Reads per-subject CSV files with format:

        Channel0
        val1
        val2
        ...

    One CSV per subject, filename = "<ID>.csv" inside <ppgpath>/ppg.
    """

    def __init__(
        self,
        zippath: str,
        ppgpath: str,
        fs_target: int,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        seed: int | None = None,
        snr_threshold: float | None = None,
        label_targets: list[str] | None = None,
    ):
        self.zippath = zippath
        self.ppgpath = ppgpath

        self.fs_target = fs_target
        self.fs = 200  # original sampling rate
        self.norm = Normalize(method="z-score")
        self.df: pd.DataFrame | None = None
        self.id_col: str | None = None

        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.seed = seed
        self.snr_threshold = snr_threshold
        self.high_quality_subjects: set | None = None
        
        # Label targets: which columns to save as labels
        # Default: ['sysbp', 'diasbp'] for blood pressure
        # Can also use: ['glucose'], ['hemoglobin'], ['sysbp', 'diasbp', 'glucose'], etc.
        self.label_targets = label_targets

        self.train_ids: list[int] = []
        self.val_ids: list[int] = []
        self.test_ids: list[int] = []

    def process_data(self):
        # ---- 1. metadata & splits ----
        self.df, self.id_col = _load_subjects_metadata(self.ppgpath)

        # ---- 2. Load SNR filter if threshold specified ----
        if self.snr_threshold is not None:
            self.high_quality_subjects = _load_snr_filter(self.ppgpath, self.snr_threshold)
            if self.high_quality_subjects is not None:
                # Filter metadata to high-quality subjects only
                self.df = self.df[self.df[self.id_col].isin(self.high_quality_subjects)].reset_index(drop=True)
                print(f"After SNR filtering: {len(self.df)} subjects remaining")

        # ---- 3. Normalize label columns ----
        self.df = _normalize_label_columns(self.df)

        # ---- 4. Validate that all requested labels exist ----
        missing_labels = [t for t in self.label_targets if t not in self.df.columns]
        if missing_labels:
            raise ValueError(
                f"Requested labels {missing_labels} not found in metadata columns: {list(self.df.columns)}. "
                f"Available columns: {', '.join(self.df.columns)}"
            )
        print(f"✓ All label targets found: {self.label_targets}")

        # data file dir (where split files live)
        data_file_dir = os.path.join(self.ppgpath, "Data File")
        os.makedirs(data_file_dir, exist_ok=True)

        # If split numpy files exist, load them to preserve the same split across runs
        train_path = os.path.join(data_file_dir, 'train_subject_ids.npy')
        val_path = os.path.join(data_file_dir, 'val_subject_ids.npy')
        test_path = os.path.join(data_file_dir, 'test_subject_ids.npy')

        if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
            try:
                self.train_ids = np.load(train_path).tolist()
                self.val_ids = np.load(val_path).tolist()
                self.test_ids = np.load(test_path).tolist()
                print(f"Loaded existing splits from {data_file_dir}: train={len(self.train_ids)}, val={len(self.val_ids)}, test={len(self.test_ids)}")
            except Exception:
                # fallback to building new splits with stratification
                stratify_col = self.label_targets[0] if self.label_targets else None
                self.train_ids, self.val_ids, self.test_ids = _build_random_splits(
                    self.df, self.id_col, self.train_frac, self.val_frac, self.test_frac, self.seed,
                    stratify_col=stratify_col
                )
                np.save(train_path, np.array(self.train_ids, dtype=np.int32))
                np.save(val_path, np.array(self.val_ids, dtype=np.int32))
                np.save(test_path, np.array(self.test_ids, dtype=np.int32))
                print(f"Saved new splits to {data_file_dir}: train={len(self.train_ids)}, val={len(self.val_ids)}, test={len(self.test_ids)}")
        else:
            # create and save splits with stratification based on first label
            stratify_col = self.label_targets[0] if self.label_targets else None
            self.train_ids, self.val_ids, self.test_ids = _build_random_splits(
                self.df, self.id_col, self.train_frac, self.val_frac, self.test_frac, self.seed,
                stratify_col=stratify_col
            )
            np.save(train_path, np.array(self.train_ids, dtype=np.int32))
            np.save(val_path, np.array(self.val_ids, dtype=np.int32))
            np.save(test_path, np.array(self.test_ids, dtype=np.int32))
            print(f"Saved new splits to {data_file_dir}: train={len(self.train_ids)}, val={len(self.val_ids)}, test={len(self.test_ids)}")

        # ---- 2. process CSV PPG files ----
        # Process all wavelength directories
        ppg_base_dir = os.path.join(self.ppgpath, "Data File", "ppg")
        os.makedirs(ppg_base_dir, exist_ok=True)

        for wl_dir in WAVELENGTH_DIRS:
            main_dir = os.path.join(self.ppgpath, wl_dir)
            if not os.path.isdir(main_dir):
                print(f"Skipping {wl_dir}: directory not found")
                continue

            filenames = [f for f in os.listdir(main_dir) if f.lower().endswith(".csv")]

            for fname in tqdm(filenames, desc=f"Processing {wl_dir}"):
                file_path = os.path.join(main_dir, fname)
                id_str = os.path.splitext(fname)[0]

                try:
                    patient_id = int(id_str)
                except ValueError:
                    print(f"Warning: filename '{fname}' is not numeric; skipping.")
                    continue

                child_dir = f"{patient_id:04d}"
                subject_dir = os.path.join(ppg_base_dir, child_dir, wl_dir)
                os.makedirs(subject_dir, exist_ok=True)

                try:
                    df_file = pd.read_csv(file_path)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}; skipping.")
                    continue

                df_file.columns = [c.strip() for c in df_file.columns]

                if "Channel0" in df_file.columns:
                    sig = df_file["Channel0"].to_numpy()
                else:
                    # fallback: first numeric column
                    num_cols = [
                        c for c in df_file.columns
                        if np.issubdtype(df_file[c].dtype, np.number)
                    ]
                    if not num_cols:
                        print(f"Warning: no numeric column in {fname}; skipping.")
                        continue
                    sig = df_file[num_cols[0]].to_numpy()

                sig = np.asarray(sig).ravel()
                if sig.size > 0:
                    sig = sig[:-1]  

                #plot before and after processing for debugging
                #import matplotlib.pyplot as plt
                #plt.figure(figsize=(12, 6))
                #plt.subplot(2, 1, 1)
                #plt.plot(sig)
                #plt.title("Original Signal")
                #plt.subplot(2, 1, 2)
                #processed_signal = self._process_one_signal(sig)
                #plt.plot(processed_signal)
                #plt.title("Processed Signal")
                #plt.tight_layout()
                #plt.show()

                padded_signal = self._process_one_signal(sig)

                # Extract MULTIPLE 10-second segments per subject
                # This gives us more training data while keeping labels aligned
                segment_length = 10 * self.fs_target  # 500 samples @ 50Hz
                
                if len(padded_signal) >= segment_length:
                    num_segments = len(padded_signal) // segment_length
                    for i in range(num_segments):
                        segment = padded_signal[i * segment_length:(i + 1) * segment_length]
                        joblib.dump(segment, os.path.join(subject_dir, f"{i}.p"))
                else:
                    # Pad if too short (shouldn't happen with 60s data)
                    segment = np.pad(
                        padded_signal,
                        pad_width=(0, max(0, segment_length - len(padded_signal))),
                        mode="constant",
                        constant_values=0.0,
                    )
                    joblib.dump(segment, os.path.join(subject_dir, "0.p"))

        # ---- 3. labels & final numpy datasets ----
        self.split_and_save_labels()
        
        # Process each wavelength
        for wl_dir in WAVELENGTH_DIRS:
            self.process_additional_data(ppg_base_dir, wl_dir)

    def _process_one_signal(self, signal_1d: np.ndarray) -> np.ndarray:
        normalized_signal, _ = self.norm.apply(signal_1d, fs=self.fs)
        try:
            proc_signal, _, _, _ = preprocess_one_ppg_signal(
                waveform=normalized_signal, frequency=self.fs
            )
        except Exception:
            print("Preprocessing failed; using normalized signal directly.")
            proc_signal = normalized_signal

        resampled_signal = resample_batch_signal(
            proc_signal, fs_original=self.fs, fs_target=self.fs_target, axis=0
        )

        target_len = 10 * self.fs_target
        padding_needed = max(0, target_len - len(resampled_signal))
        pad_left = padding_needed // 2
        pad_right = padding_needed - pad_left
        padded_signal = np.pad(
            resampled_signal,
            pad_width=(pad_left, pad_right),
            mode="constant",
            constant_values=0.0,
        )
        return padded_signal

    def split_and_save_labels(self):
        if self.df is None:
            raise RuntimeError("Subjects DataFrame (self.df) is not initialized.")

        self.df = _normalize_label_columns(self.df)
        id_col = self.id_col

        df_train = self.df[self.df[id_col].isin(self.train_ids)]
        df_val = self.df[self.df[id_col].isin(self.val_ids)]
        df_test = self.df[self.df[id_col].isin(self.test_ids)]

        data_file_dir = os.path.join(self.ppgpath, "Data File")
        os.makedirs(data_file_dir, exist_ok=True)

        df_train.to_csv(os.path.join(data_file_dir, "train.csv"), index=False)
        df_val.to_csv(os.path.join(data_file_dir, "val.csv"), index=False)
        df_test.to_csv(os.path.join(data_file_dir, "test.csv"), index=False)

    def process_additional_data(self, ppg_base_dir: str, wl_dir: str):
        """
        Load per-subject segments and export:
          - <split>_X_ppg_<fs>Hz_<wl>.npy  => shape (N, 1, L)
          - <split>_y_<target>_<wl>.npy    => shape (N,) for each label in label_targets
        
        Supports multiple label targets: sysbp, diasbp, glucose, hemoglobin, age, etc.
        CRITICAL: Each wavelength gets its own labels, not shared across wavelengths!
        """
        if self.df is None:
            raise RuntimeError("Subjects DataFrame (self.df) is not initialized.")

        self.df = _normalize_label_columns(self.df)
        
        print(f"\nProcessing with labels: {self.label_targets}")

        for name, ids in zip(
            ["train", "val", "test"], [self.train_ids, self.val_ids, self.test_ids]
        ):
            data = []
            labels = {target: [] for target in self.label_targets}
            subject_ids = []
            missing_subjects = []
            missing_segments = []

            print(f"Collecting {wl_dir} data for split: {name}")

            for id_i in ids:
                subj_dir = os.path.join(ppg_base_dir, f"{id_i:04d}", wl_dir)
                if not os.path.isdir(subj_dir):
                    missing_subjects.append(id_i)
                    continue

                row = self.df[self.df[self.id_col] == id_i]
                if row.empty:
                    continue

                seg_files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.p')])
                if not seg_files:
                    missing_segments.append(id_i)
                    continue

                # Extract label values for this subject
                label_vals = {}
                for target in self.label_targets:
                    if target in row.columns:
                        label_vals[target] = row[target].values[0]
                    else:
                        print(f"  Warning: Column '{target}' not found for subject {id_i}. Using 0.")
                        label_vals[target] = 0

                for seg_file in seg_files:
                    seg_path = os.path.join(subj_dir, seg_file)
                    try:
                        loaded = joblib.load(seg_path)
                    except Exception as e:
                        print(f"  Warning: Failed to load {seg_path}: {e}")
                        continue
                    
                    sig = np.asarray(loaded).ravel()
                    sig = sig.reshape(1, -1)         # (1, L)
                    sig = sig[np.newaxis, :, :]      # (1, 1, L)
                    data.append(sig)
                    
                    # Add labels for THIS segment
                    for target, val in label_vals.items():
                        labels[target].append(val)
                    subject_ids.append(int(id_i))

            if len(data) == 0:
                print(f"  No data for {name}/{wl_dir}")
                continue

            data = np.concatenate(data, axis=0)
            subject_ids = np.array(subject_ids, dtype=np.int32)

            # Convert labels to numpy arrays
            for target in self.label_targets:
                labels[target] = np.array(labels[target], dtype=np.float32)

            # VERIFY alignment
            for target in self.label_targets:
                assert len(labels[target]) == len(data), \
                    f"{wl_dir} {name} {target}: {len(labels[target])} labels vs {len(data)} data samples!"

            # Report differences
            print(f"  Missing subjects (no directory): {len(missing_subjects)} - {missing_subjects[:5]}")
            print(f"  Missing segments (no .p files): {len(missing_segments)} - {missing_segments[:5]}")
            
            wl_suffix = wl_dir.replace("ppg_", "").replace("nm", "")
            
            # Save data
            np.save(
                os.path.join(self.ppgpath, f"{name}_X_ppg_{self.fs_target}Hz_{wl_suffix}nm.npy"),
                data,
            )
            
            # Save each label target
            for target in self.label_targets:
                np.save(
                    os.path.join(self.ppgpath, f"{name}_y_{target}_{wl_suffix}nm.npy"),
                    labels[target],
                )
            
            # Save subject IDs
            np.save(
                os.path.join(self.ppgpath, f"{name}_subject_ids_{wl_suffix}nm.npy"),
                subject_ids,
            )
            
            label_shapes = {t: labels[t].shape for t in self.label_targets}
            print(f"  Saved {name}: data={data.shape}, labels={label_shapes}")
            print(f"  Unique subjects: {len(np.unique(subject_ids))}")
            print(f"  ✓ Labels and data are ALIGNED\n")


# ==========================
# main
# ==========================

def main(newhz: int, zippath: str, ppgpath: str, snr_threshold: float | None = None, label_targets: list[str] | None = None):
    """
    Process PPG data with optional SNR filtering and flexible label targets.
    
    Args:
        newhz: Target sampling frequency (Hz)
        zippath: Path to zip file (if applicable)
        ppgpath: Path to dataset directory
        snr_threshold: Minimum SNR_Mean value to include subject (None = use all subjects)
                      Recommended value: 40.0 dB for high-quality signals
        label_targets: List of label columns to save (e.g., ["sysbp", "diasbp", "glucose"])
                      Default: ["sysbp", "diasbp"]
                      Options: "sysbp", "diasbp", "glucose", "hemoglobin", "age", "sex"
    """
    processor = MWPPGCSVDataProcessor(
        zippath=zippath,
        ppgpath=ppgpath,
        fs_target=newhz,
        snr_threshold=snr_threshold,
        label_targets=label_targets,
    )
    processor.process_data()
    print(
        f"MWPPG CSV PPG data files are ready in {os.path.abspath(processor.ppgpath)}"
    )


if __name__ == "__main__":
    # Example 1: Process all subjects with default labels (SBP/DBP)
    # main(
    #     newhz=50,
    #     zippath="",
    #     ppgpath="pulseppg/data/datasets/Hb_PPG_Dataset"
    # )
    
    # Example 2: Process only high-quality subjects (SNR_Mean >= 40 dB)
    # main(
    #     newhz=50,
    #     zippath="",
    #     ppgpath="pulseppg/data/datasets/Hb_PPG_Dataset",
    #     snr_threshold=40.0,
    # )
    
    # Example 3: Use glucose as label target
    # main(
    #     newhz=50,
    #     zippath="",
    #     ppgpath="pulseppg/data/datasets/Hb_PPG_Dataset",
    #     label_targets=["glucose"],
    # )
    
    # Example 4: Multi-label with SNR filtering (BP + Glucose + Hemoglobin)
    main(
        newhz=50,
        zippath="",
        ppgpath="pulseppg/data/datasets/Hb_PPG_Dataset_Glucose_Hb",
        label_targets=["glucose", "hemoglobin"]
    )
    
    # Example 5: Augment high glucose samples and randomly distribute across splits
    # IMPORTANT: Run this AFTER running main() above!
    #
    # This will:
    # - Find all samples with glucose > 7.0 across train, val, test
    # - Create 3 augmented copies of each high-glucose sample (with different augmentation types)
    # - Assign new unique subject IDs to augmented samples (20000+)
    # - RANDOMLY distribute augmented samples to train/val/test according to ratios
    # - Result: Augmented samples are NOT in same split as original, but spread across all splits
    augment_high_glucose_samples(
        ppgpath="pulseppg/data/datasets/Hb_PPG_Dataset_Glucose_Hb",
        glucose_threshold=7.0,      # Augment samples with glucose > 7 mmol/L
        num_augmentations=3,         # Create 3 augmented copies per high-glucose sample
        train_ratio=0.7,             # 70% of augmented samples go to train
        val_ratio=0.15,              # 15% of augmented samples go to val
        test_ratio=0.15,             # 15% of augmented samples go to test
        seed=42
    )
