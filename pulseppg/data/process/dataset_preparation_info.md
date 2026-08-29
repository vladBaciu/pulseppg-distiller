# PPG dataset preparation

This folder contains the scripts used to prepare the Hb_PPG_Dataset for glucose regression.

The main entry point is:

- `prepare_ppg_dataset.py`

It performs the dataset preparation pipeline and produces the final processed arrays used for training.

## What the script does

The workflow is:

1. Resolve the dataset input
   - Looks for the dataset directory or a zip archive.
   - Keeps the zip file intact.
   - Uses the zip stem name as the processed output folder name.

2. Extract the archive to a temporary staging folder
   - The zip is not deleted.
   - Files are extracted into a temporary folder only during preparation.

3. Split raw CSV data by wavelength
   - Reads `data_csv/*.csv`
   - Creates:
     - `ppg_660nm/`
     - `ppg_730nm/`
     - `ppg_850nm/`
     - `ppg_940nm/`

4. Process PPG signals
   - Reads each subject CSV
   - Normalizes and preprocesses the waveform
   - Resamples to the target frequency
   - Splits the signal into 10-second segments
   - Saves the processed segments under a `Data File/ppg/...` structure

5. Build train/val/test subject splits
   - Uses subject metadata to map patient IDs
   - Applies stratification based on the glucose label
   - Saves split IDs as `.npy` arrays

6. Export final numpy datasets
   - Saves per-split arrays for each wavelength
   - Keeps the label target as glucose by default

7. Optional augmentation
   - If `--augment` is enabled, high-glucose samples are augmented and added back into train/val/test splits

---

## Final dataset layout after a successful run

After running the script, the output looks like this:

```text
<dataset_parent>/
├── Hb_PPG_Dataset.zip
├── Hb_PPG_Dataset/
│   ├── data_csv/
│   │   ├── ...
│   │   └── ...
│   │
│   ├── ppg_660nm/
│   │   ├── 0001.csv
│   │   ├── 0002.csv
│   │   └── ...
│   ├── ppg_730nm/
│   ├── ppg_850nm/
│   ├── ppg_940nm/
│   │
│   ├── Data File/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   ├── test.csv
│   │   ├── train_subject_ids.npy
│   │   ├── val_subject_ids.npy
│   │   ├── test_subject_ids.npy
│   │   └── ppg/
│   │       └── <subject_id>/
│   │           └── ppg_660nm/
│   │               ├── 0.p
│   │               ├── 1.p
│   │               └── ...
│   │
│   ├── train_X_ppg_50Hz_660nm.npy
│   ├── train_y_glucose_660nm.npy
│   ├── train_subject_ids_660nm.npy
│   ├── val_X_ppg_50Hz_660nm.npy
│   ├── val_y_glucose_660nm.npy
│   ├── val_subject_ids_660nm.npy
│   ├── test_X_ppg_50Hz_660nm.npy
│   ├── test_y_glucose_660nm.npy
│   ├── test_subject_ids_660nm.npy
│   │
│   ├── train_X_ppg_50Hz_730nm.npy
│   ├── train_y_glucose_730nm.npy
│   ├── train_subject_ids_730nm.npy
│   ├── val_X_ppg_50Hz_730nm.npy
│   ├── val_y_glucose_730nm.npy
│   ├── val_subject_ids_730nm.npy
│   ├── test_X_ppg_50Hz_730nm.npy
│   ├── test_y_glucose_730nm.npy
│   ├── test_subject_ids_730nm.npy
│   │
│   ├── train_X_ppg_50Hz_850nm.npy
│   ├── train_y_glucose_850nm.npy
│   ├── train_subject_ids_850nm.npy
│   ├── val_X_ppg_50Hz_850nm.npy
│   ├── val_y_glucose_850nm.npy
│   ├── val_subject_ids_850nm.npy
│   ├── test_X_ppg_50Hz_850nm.npy
│   ├── test_y_glucose_850nm.npy
│   ├── test_subject_ids_850nm.npy
│   │
│   ├── train_X_ppg_50Hz_940nm.npy
│   ├── train_y_glucose_940nm.npy
│   ├── train_subject_ids_940nm.npy
│   ├── val_X_ppg_50Hz_940nm.npy
│   ├── val_y_glucose_940nm.npy
│   ├── val_subject_ids_940nm.npy
│   ├── test_X_ppg_50Hz_940nm.npy
│   ├── test_y_glucose_940nm.npy
│   └── test_subject_ids_940nm.npy
```

---

## Meaning of the final arrays

### `*_X_ppg_50Hz_<wavelength>.npy`
These arrays contain the processed PPG segments.

Typical shape:

```python
(N, 1, 500)
```

Meaning:

- `N`: number of samples in the split
- `1`: one signal channel
- `500`: 10 seconds at 50 Hz

### `*_y_glucose_<wavelength>.npy`
These arrays contain the corresponding glucose labels for each sample.

Typical shape:

```python
(N,)
```

### `*_subject_ids_<wavelength>.npy`
These arrays contain the subject IDs associated with each sample.

This is useful for sample tracking and subject-level analysis.

---

## Optional augmentation

If `--augment` is used, the script will:

- find high-glucose samples
- create augmented versions of those signals
- randomly assign them to train/val/test
- append them to the existing `.npy` files

This is done only for glucose regression and is optional.

---

## How to run

From the project folder:

```bash
python pulseppg/data/process/prepare_ppg_dataset.py --augment
```

This prepares the dataset and keeps the original zip archive together with the processed folder.

---

## Notes

- The original `.zip` file is preserved.
- The processed dataset is created in a sibling folder with the same stem name as the zip.
- Temporary extraction folders are used only during preparation and are removed afterward.
- The final output is intended for training and evaluation of glucose estimation models.
