"""Split CSV files by wavelength channels into the final postprocessed dataset folder."""

import sys
from pathlib import Path

import pandas as pd


DATASETS_ROOT = Path(__file__).resolve().parents[1] / "data" / "datasets"
POSTPROCESSED_DATASETS_ROOT = DATASETS_ROOT / "postprocessed"


def split_csv_by_wavelength(data_csv_dir, parent_output_dir=None):
    """Split a data_csv folder into ppg_660nm/730nm/850nm/940nm subfolders.

    By default, outputs go under data/datasets/postprocessed/<dataset-name>/.
    """
    data_csv_path = Path(data_csv_dir).expanduser().resolve()

    if not data_csv_path.exists():
        print(f"Error: Directory '{data_csv_path}' does not exist.")
        return

    if parent_output_dir is None:
        parent_output_dir = POSTPROCESSED_DATASETS_ROOT / data_csv_path.parent.name
    else:
        parent_output_dir = Path(parent_output_dir).expanduser().resolve()

    wavelengths = ["660nm", "730nm", "850nm", "940nm"]
    output_dirs = {}

    for wl in wavelengths:
        output_dir = parent_output_dir / f"ppg_{wl}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dirs[wl] = output_dir
        print(f"Created/verified directory: {output_dir}")

    csv_files = sorted(data_csv_path.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in '{data_csv_path}'")
        return

    print(f"\nFound {len(csv_files)} CSV files to process.\n")

    for csv_file in csv_files:
        print(f"Processing: {csv_file.name}")
        try:
            df = pd.read_csv(csv_file)
            expected_cols = ["660nm", "730nm", "850nm", "940nm"]
            if not all(col in df.columns for col in expected_cols):
                print(f"  Warning: '{csv_file.name}' does not have all expected columns.")
                print(f"  Found columns: {list(df.columns)}")
                continue

            for wl in wavelengths:
                output_file = output_dirs[wl] / csv_file.name
                df[[wl]].copy().astype(int).to_csv(output_file, index=False)
                print(f"  ✓ Saved to ppg_{wl}/{csv_file.name}")
        except Exception as exc:
            print(f"  Error processing '{csv_file.name}': {exc}")

    print("\nDone!")
    return str(parent_output_dir)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_csv_folder = sys.argv[1]
    else:
        default_dataset = DATASETS_ROOT / "Hb_PPG_Dataset" / "data_csv"
        data_csv_folder = str(default_dataset)

    if len(sys.argv) > 2:
        parent_output_dir = sys.argv[2]
    else:
        parent_output_dir = None

    print(f"Processing CSV folder: {data_csv_folder}\n")
    split_csv_by_wavelength(data_csv_folder, parent_output_dir=parent_output_dir)
