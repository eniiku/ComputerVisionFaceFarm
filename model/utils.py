import os
import numpy as np
import logging
from PIL import Image
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def oversample_minority_class(
    train_data_root_dir, target_ratio, augmentation_params, save_format
):
    """
    Oversamples the minority class in the training dataset using image augmentation.
    This function will modify the files on disk in the specified directory.
    """
    logging.info(
        f"Starting oversampling process for training data in: {train_data_root_dir}"
    )

    class_dirs = [
        d
        for d in os.listdir(train_data_root_dir)
        if os.path.isdir(os.path.join(train_data_root_dir, d))
    ]

    if len(class_dirs) < 2:
        logging.warning(
            f"Less than two class directories found in {train_data_root_dir}. Oversampling not applicable."
        )
        return

    class_counts = {}
    for class_name in class_dirs:
        class_path = os.path.join(train_data_root_dir, class_name)
        images = [
            f
            for f in os.listdir(class_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))
        ]
        class_counts[class_name] = len(images)

    if not class_counts:
        logging.error(
            f"No images found in any class directory under {train_data_root_dir}. Exiting."
        )
        return

    # Sort classes by count to easily identify majority/minority
    sorted_classes = sorted(class_counts.items(), key=lambda item: item[1])

    minority_class_name, minority_count = sorted_classes[0]
    majority_class_name, majority_count = sorted_classes[-1]

    if majority_count == minority_count:
        logging.info("Classes are already balanced. No oversampling needed.")
        return

    # Ensure 'Pain' is correctly identified as minority if it exists
    # This might be needed if "No Pain" had fewer images than "Pain"
    # For binary 'Pain'/'No Pain', simple min/max is usually fine.
    if "Pain" in class_counts and class_counts["Pain"] < majority_count:
        minority_class_name = "Pain"
        minority_count = class_counts["Pain"]
        # Find majority class that is not 'Pain'
        for cls, count in class_counts.items():
            if cls != "Pain" and count == majority_count:
                majority_class_name = cls
                break
    elif "No Pain" in class_counts and class_counts["No Pain"] < majority_count:
        # This case implies 'No Pain' is the minority, which is unlikely for pain detection
        # but included for robustness.
        minority_class_name = "No Pain"
        minority_count = class_counts["No Pain"]
        for cls, count in class_counts.items():
            if cls != "No Pain" and count == majority_count:
                majority_class_name = cls
                break
    else:
        # Fallback if specific class names are not found or counts are same
        logging.info("Auto-identified majority and minority classes based on counts.")

    logging.info(f"Majority Class: '{majority_class_name}' ({majority_count} images)")
    logging.info(f"Minority Class: '{minority_class_name}' ({minority_count} images)")

    target_minority_count = int(majority_count * target_ratio)
    images_to_generate = target_minority_count - minority_count

    if images_to_generate <= 0:
        logging.info(
            f"Minority class ({minority_count}) is already sufficiently balanced or larger than target. No images to generate."
        )
        return

    logging.info(f"Target minority class count: {target_minority_count}")
    logging.info(
        f"Number of images to generate for '{minority_class_name}': {images_to_generate}"
    )

    minority_class_path = os.path.join(train_data_root_dir, minority_class_name)
    minority_images = [
        f
        for f in os.listdir(minority_class_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))
    ]

    if not minority_images:
        logging.error(
            f"No original images found in minority class directory {minority_class_path}. Cannot oversample."
        )
        return

    # Setup ImageDataGenerator for oversampling
    oversample_datagen = ImageDataGenerator(rescale=1.0 / 255, **augmentation_params)

    temp_gen_dir = os.path.join(train_data_root_dir, "temp_oversample_gen")
    os.makedirs(temp_gen_dir, exist_ok=True)

    generated_count = 0
    logging.info(
        f"Generating {images_to_generate} images for '{minority_class_name}'..."
    )

    import tqdm  # For progress bar

    minority_image_paths = [
        os.path.join(minority_class_path, img_name) for img_name in minority_images
    ]

    image_idx = 0
    with tqdm.tqdm(
        total=images_to_generate, desc=f"Generating for {minority_class_name}"
    ) as pbar:
        while generated_count < images_to_generate:
            if not minority_image_paths:
                logging.warning(
                    f"Ran out of source images for {minority_class_name} before reaching target."
                )
                break

            img_path = minority_image_paths[image_idx % len(minority_image_paths)]

            try:
                img = Image.open(img_path)
                img = img.convert("RGB")
                x = np.asarray(img)
                x = np.expand_dims(x, axis=0)

                batch_flow = oversample_datagen.flow(
                    x,
                    batch_size=1,
                    save_to_dir=temp_gen_dir,
                    save_prefix=f"aug_{os.path.splitext(os.path.basename(img_path))[0]}",
                    save_format=save_format,
                )
                _ = next(batch_flow)  # Get one augmented image
                generated_count += 1
                pbar.update(1)

            except Exception as e:
                logging.error(
                    f"Error processing image {os.path.basename(img_path)} for augmentation: {e}"
                )

            image_idx += 1

    logging.info(f"Finished generating {generated_count} augmented images.")

    # Move generated images to the actual minority class directory
    logging.info(f"Moving generated images to '{minority_class_path}'...")
    generated_files = [
        f for f in os.listdir(temp_gen_dir) if f.lower().endswith((f".{save_format}"))
    ]
    for filename in tqdm.tqdm(generated_files, desc="Moving generated files"):
        src_path = os.path.join(temp_gen_dir, filename)
        dest_path = os.path.join(minority_class_path, filename)
        try:
            if os.path.exists(dest_path):
                base, ext = os.path.splitext(dest_path)
                count = 1
                new_dest_path = f"{base}_{count}{ext}"
                while os.path.exists(new_dest_path):
                    count += 1
                    new_dest_path = f"{base}_{count}{ext}"
                shutil.move(src_path, new_dest_path)
            else:
                shutil.move(src_path, dest_path)
        except Exception as e:
            logging.error(f"Failed to move {src_path} to {dest_path}: {e}")

    # Clean up the temporary directory
    try:
        if os.path.exists(temp_gen_dir):
            shutil.rmtree(temp_gen_dir)
            logging.info("Cleaned up temporary generation directory.")
    except Exception as e:
        logging.warning(f"Could not remove temporary directory {temp_gen_dir}: {e}")

    final_minority_count = len(
        [
            f
            for f in os.listdir(minority_class_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))
        ]
    )
    logging.info(
        f"Oversampling complete. Final count for '{minority_class_name}': {final_minority_count} images."
    )
    logging.info("Remember to run your model training script after oversampling.")
