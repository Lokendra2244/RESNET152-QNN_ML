import os
import pandas as pd
import shutil
from tqdm import tqdm

# ==========================================
# 1. Configuration (Point to your generated CSVs)
# ==========================================
# Change these to 'clean_data_set.csv' and 'clean_test_set.csv' if you ran in TEST_MODE
TRAIN_CSV = "clean_data_set_full.csv"
TEST_CSV = "clean_test_set_full.csv"

OUTPUT_DIR = "./cbis_ddsm"


def setup_directories():
    print(f"Creating output structure in {OUTPUT_DIR}...")
    for split in ["train", "test"]:
        for label in ["BENIGN", "MALIGNANT"]:
            os.makedirs(os.path.join(OUTPUT_DIR, split, label), exist_ok=True)


def copy_images_from_csv(csv_path, split):
    if not os.path.exists(csv_path):
        print(f"⚠️ Warning: {csv_path} not found. Skipping.")
        return 0, 0

    print(f"\nProcessing {csv_path} for {split} split...")
    df = pd.read_csv(csv_path)

    copied = 0
    missing = 0

    # Iterate through the generated CSV
    for index, row in tqdm(df.iterrows(), total=len(df)):
        img_path = str(row["cropped image file path"])
        pathology_code = row["pathology"]

        # Based on your previous script's mapping: 1 = BENIGN, 0 = MALIGNANT
        label = "BENIGN" if pathology_code == 1 else "MALIGNANT"

        # If the file exists exactly where the CSV says it is, copy it
        if os.path.exists(img_path):
            # Extract the 1.3.6... UID from the path to use as a unique filename
            parts = img_path.split("/")
            uid = "unknown_uid"
            for part in parts:
                if part.startswith("1.3.6"):
                    uid = part
                    break

            ext = os.path.splitext(img_path)[1]
            new_filename = f"{uid}{ext}"
            dest_path = os.path.join(OUTPUT_DIR, split, label, new_filename)

            # Copy the file
            if not os.path.exists(dest_path):
                shutil.copy2(img_path, dest_path)
            copied += 1
        else:
            missing += 1

    return copied, missing


if __name__ == "__main__":
    setup_directories()

    # Process Train Data
    train_copied, train_missing = copy_images_from_csv(TRAIN_CSV, "train")

    # Process Test Data
    test_copied, test_missing = copy_images_from_csv(TEST_CSV, "test")

    # Final Summary
    print("\n" + "=" * 50)
    print("✅ FAST DATASET GENERATION COMPLETE")
    print("=" * 50)
    print(f"Total Train Images Copied: {train_copied}")
    print(f"Total Test Images Copied:  {test_copied}")
    print(f"Total Missing/Errors:      {train_missing + test_missing}")
    print(f"Dataset ready at:          {OUTPUT_DIR}")
    print("=" * 50)
