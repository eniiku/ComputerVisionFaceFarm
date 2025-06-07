# data_splitting_script.py
import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm 

# --- Configuration ---
# Path to your raw dataset where all images are initially stored,
# organized into subdirectories per class.
RAW_DATA_DIR = '../data/raw_corpus_sheep_facial_expression_dataset/'
# Path where the split dataset (train, validation, test folders) will be created.
# This should be the 'sheep_pain_dataset' directory as expected by train_model.py
OUTPUT_DATA_DIR = '../datasets/sheep_pain_dataset/'

# Split ratios
TRAIN_RATIO = 0.8  # 80% for training
VAL_RATIO = 0.17    # 17% for validation (of the original dataset)
TEST_RATIO = 0.03   # 3% for testing (of the original dataset)
assert TRAIN_RATIO + VAL_RATIO + TEST_RATIO == 1.0, "Ratios must sum to 1.0"

RANDOM_SEED = 42

# --- Setup Logging ---
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def split_and_copy_data(raw_data_dir, output_data_dir, train_ratio, val_ratio, test_ratio, random_seed):
    """
    Splits the raw image dataset into training, validation, and test sets
    and copies them to the specified output directory structure.

    Args:
        raw_data_dir (str): Path to the directory containing raw image data,
                            with class subdirectories.
        output_data_dir (str): Path where the split dataset will be created.
        train_ratio (float): Proportion of data for the training set.
        val_ratio (float): Proportion of data for the validation set.
        test_ratio (float): Proportion of data for the test set.
        random_seed (int): Seed for reproducibility of the split.
    """
    logger.info(f"Starting data splitting from '{raw_data_dir}' to '{output_data_dir}'")
    logger.info(f"Train Ratio: {train_ratio}, Validation Ratio: {val_ratio}, Test Ratio: {test_ratio}")

    # Create main output directories if they don't exist
    os.makedirs(os.path.join(output_data_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_data_dir, 'validation'), exist_ok=True)
    os.makedirs(os.path.join(output_data_dir, 'test'), exist_ok=True)

    class_names = [d for d in os.listdir(raw_data_dir) if os.path.isdir(os.path.join(raw_data_dir, d))]
    if not class_names:
        logger.error(f"No class subdirectories found in '{raw_data_dir}'. Please organize your data.")
        return

    logger.info(f"Found classes: {class_names}")

    for class_name in class_names:
        class_path = os.path.join(raw_data_dir, class_name)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        if not images:
            logger.warning(f"No image files found in class directory: {class_path}")
            continue

        logger.info(f"Processing class '{class_name}': Found {len(images)} images.")

        # Split into training and temp (validation + test)
        train_images, temp_images = train_test_split(
            images,
            test_size=(val_ratio + test_ratio),
            random_state=random_seed,
            shuffle=True
        )

        # Split temp into validation and test
        if (val_ratio + test_ratio) > 0: 
            # Calculate test_size for the second split relative to temp_images size
            test_size_relative = test_ratio / (val_ratio + test_ratio)
            val_images, test_images = train_test_split(
                temp_images,
                test_size=test_size_relative,
                random_state=random_seed,
                shuffle=True
            )
        else:
            val_images = []
            test_images = []


        splits = {
            'train': train_images,
            'validation': val_images,
            'test': test_images
        }

        # Create class subdirectories in output folders and copy images
        for split_name, image_list in splits.items():
            dest_dir = os.path.join(output_data_dir, split_name, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            logger.info(f"Copying {len(image_list)} images to {split_name}/{class_name}/")

            for img_file in tqdm(image_list, desc=f"Copying {class_name} to {split_name}"):
                src_path = os.path.join(class_path, img_file)
                dest_path = os.path.join(dest_dir, img_file)
                try:
                    shutil.copy(src_path, dest_path)
                except Exception as e:
                    logger.error(f"Failed to copy {src_path} to {dest_path}: {e}")

    logger.info("Data splitting and copying complete!")

if __name__ == "__main__":
    split_and_copy_data(RAW_DATA_DIR, OUTPUT_DATA_DIR, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED)


