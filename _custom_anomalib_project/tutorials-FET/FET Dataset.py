import os
import shutil
from pathlib import Path
from anomalib.data import Folder

# --- Configuration ---
ROOT_DIR = "./temp_folder_data"
NORMAL_DIR = "good_items"
ABNORMAL_DIR = "defective_items"
MASK_DIR = "masks"
IMAGE_SIZE = 64
BATCH_SIZE = 4

# --- Implementation ---

try:
    # 1. Instantiate the Folder DataModule
    datamodule = Folder(
        name="custom_project_data",
        root=ROOT_DIR,
        normal_dir=NORMAL_DIR,
        abnormal_dir=ABNORMAL_DIR,
        mask_dir=MASK_DIR,
        # Configuration for splitting and batching
        train_batch_size=BATCH_SIZE,
        eval_batch_size=BATCH_SIZE,
        num_workers=0,  # Use 0 for reliable testing/debugging
        # Data Splitting Control:
        # We use all normal images for training, and all abnormal/normal_test for testing.
        # This setup ensures a clean train/test separation based on directories.
        normal_split_ratio=0.0, # Do not split normal images for testing
        # test_split_mode=TestSplitMode.FROM_DIR, # Use images found in abnormal_dir (and optional normal_test_dir)
        # val_split_mode=ValSplitMode.SPLIT_FROM_TRAIN, # Split validation from the training set
        val_split_ratio=0.2, # 20% of normal images will be used for validation
    )

    # 2. Setup the DataModule
    # This call finds the files, creates the samples DataFrame, and sets up the Datasets.
    datamodule.setup()
    print("Folder DataModule successfully instantiated and set up.")

except Exception as e:
    print(f"Error during DataModule setup: {e}")
    datamodule = None

