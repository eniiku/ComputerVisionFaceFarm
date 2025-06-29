import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os
import numpy as np
import logging
from sklearn.utils import class_weight

from .utils import oversample_minority_class
from .evaluators import (
    plot_training_history,
    plot_confusion_matrix,
    print_classification_report,
    plot_roc_curve,
    plot_precision_recall_curve,
)

# --- Configuration ---
# IMPORTANT:
# 1. Ensure your dataset is uploaded to Colab (or accessible locally) and organized as follows:
#    sheep_pain_dataset/
#    ├── train/
#    │   ├── No Pain/
#    │   └── Pain/
#    ├── validation/
#    │   ├── No Pain/
#    │   └── Pain/
#    └── test/
#        ├── No Pain/
#        └── Pain/
DATA_DIR = "datasets/sheep_pain_dataset"
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
NUM_CLASSES = 2  # 'No Pain', 'Pain' - This code is for binary classification
EPOCHS = 30

# Define paths for your dataset splits
train_dir = os.path.join(DATA_DIR, "train")
validation_dir = os.path.join(DATA_DIR, "validation")
test_dir = os.path.join(DATA_DIR, "test")

# --- Oversampling Configuration ---
TRAIN_DATA_ROOT_DIR_OVERSAMPLE = train_dir
TARGET_CLASS_BALANCE_RATIO = 0.95
AUGMENTATION_PARAMS = {
    "rotation_range": 30,
    "width_shift_range": 0.25,
    "height_shift_range": 0.25,
    "shear_range": 0.2,
    "zoom_range": 0.25,
    "horizontal_flip": True,
    "brightness_range": [0.7, 1.3],
    "fill_mode": "nearest",
}
SAVE_FORMAT = "png"


def main():
    logging.info(
        "Starting Sheep Pain Detection Model Training (Binary Classification Focus)."
    )

    # --- Step 1: Oversample Minority Class (if desired) ---
    # You might want to comment this out after the first run
    # to avoid regenerating images every time you train.
    oversample_minority_class(
        TRAIN_DATA_ROOT_DIR_OVERSAMPLE,
        TARGET_CLASS_BALANCE_RATIO,
        AUGMENTATION_PARAMS,
        SAVE_FORMAT,
    )

    # --- Data Augmentation and Loading ---
    logging.info("Setting up ImageDataGenerators and loading data...")

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )

    validation_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    try:
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode="binary",
        )

        validation_generator = validation_test_datagen.flow_from_directory(
            validation_dir,
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode="binary",
        )

        test_generator = validation_test_datagen.flow_from_directory(
            test_dir,
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode="binary",
            shuffle=False,
        )

        class_names = list(train_generator.class_indices.keys())
        logging.info(f"Detected classes: {class_names}")

        logging.info(
            f"Found {train_generator.samples} training images belonging to {train_generator.num_classes} classes."
        )
        logging.info(
            f"Found {validation_generator.samples} validation images belonging to {validation_generator.num_classes} classes."
        )
        logging.info(
            f"Found {test_generator.samples} test images belonging to {test_generator.num_classes} classes."
        )

    except Exception as e:
        logging.error(
            f"Error loading data from directories. Please check DATA_DIR path and folder structure: {e}"
        )
        raise

    # --- Handle Class Imbalance with Class Weighting ---
    logging.info("Computing class weights to handle dataset imbalance...")
    try:
        class_weights_array = class_weight.compute_class_weight(
            class_weight="balanced",
            classes=np.unique(train_generator.classes),
            y=train_generator.classes,
        )
        class_weights_dict = dict(enumerate(class_weights_array))
        logging.info(f"Computed Class Weights: {class_weights_dict}")
    except Exception as e:
        logging.error(f"Failed to compute class weights: {e}")
        class_weights_dict = None

    # --- Model Architecture (Transfer Learning with MobileNetV2) ---
    logging.info(
        "Building the model architecture using MobileNetV2 for transfer learning..."
    )

    base_model = MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights="imagenet"
    )

    base_model.trainable = False
    logging.info("Base model (MobileNetV2) layers frozen for initial training.")

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.summary()

    # --- Model Compilation ---
    logging.info("Compiling the model for initial training phase...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # --- Callbacks for Initial Training ---
    logging.info(
        "Setting up callbacks for initial training (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau)..."
    )
    initial_callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            "best_sheep_pain_model_stage1.h5",
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1
        ),
    ]

    # --- Initial Model Training (Feature Extraction) ---
    logging.info(
        f"Starting initial model training (feature extraction) for {EPOCHS} epochs..."
    )
    history_initial = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        class_weight=class_weights_dict,
        callbacks=initial_callbacks,
    )
    logging.info("Initial model training finished.")

    # --- Fine-tuning Phase ---
    logging.info("Starting fine-tuning phase...")
    model.load_weights("best_sheep_pain_model_stage1.h5")
    logging.info("Loaded best weights from initial training for fine-tuning.")

    base_model.trainable = True
    logging.info("Base model layers unfrozen for fine-tuning.")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    logging.info("Model re-compiled for fine-tuning with a very low learning rate.")

    fine_tune_epochs = 20
    fine_tune_callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=7, restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            "best_sheep_pain_model_final.h5",
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-8, verbose=1
        ),
    ]

    logging.info(f"Starting fine-tuning for {fine_tune_epochs} additional epochs...")
    history_fine_tune = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS + fine_tune_epochs,
        initial_epoch=history_initial.epoch[-1],
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        class_weight=class_weights_dict,
        callbacks=fine_tune_callbacks,
    )
    logging.info("Fine-tuning finished.")

    # Combine histories for plotting comprehensive curves
    combined_history = {
        "accuracy": history_initial.history["accuracy"]
        + history_fine_tune.history["accuracy"],
        "val_accuracy": history_initial.history["val_accuracy"]
        + history_fine_tune.history["val_accuracy"],
        "loss": history_initial.history["loss"] + history_fine_tune.history["loss"],
        "val_loss": history_initial.history["val_loss"]
        + history_fine_tune.history["val_loss"],
    }

    # --- Evaluation and Reporting ---
    logging.info("Generating evaluation reports and plots...")

    plot_training_history(combined_history)

    model.load_weights("best_sheep_pain_model_final.h5")
    logging.info("Loaded final best weights for test set evaluation.")

    logging.info("Evaluating model on the test set...")
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    logging.info(f"Test Loss: {test_loss:.4f}")
    logging.info(f"Test Accuracy: {test_accuracy:.4f}")

    logging.info("Generating predictions on test data for detailed report...")
    test_generator.reset()
    num_test_steps = test_generator.samples // test_generator.batch_size + (
        test_generator.samples % test_generator.batch_size != 0
    )

    # For binary classification, model.predict returns probabilities (1D array)
    Y_pred_probs = model.predict(test_generator, steps=num_test_steps).flatten()
    y_pred_classes = (Y_pred_probs > 0.5).astype(int)

    y_true = test_generator.classes[test_generator.index_array][: len(y_pred_classes)]

    logging.info("Plotting Confusion Matrix...")
    plot_confusion_matrix(y_true, y_pred_classes, class_names)

    logging.info("Printing Classification Report...")
    print_classification_report(y_true, y_pred_classes, class_names)

    logging.info("Plotting ROC Curve for 'Pain' class...")
    plot_roc_curve(y_true, Y_pred_probs, ["No Pain", "Pain"])

    logging.info("Plotting Precision-Recall Curve for 'Pain' class...")
    plot_precision_recall_curve(y_true, Y_pred_probs, ["No Pain", "Pain"])

    # --- Save the Final Trained Model in SavedModel format ---
    MODEL_SAVE_PATH = "sheep_pain_detection_model_binary"
    logging.info(
        f"Attempting to save final model to: {MODEL_SAVE_PATH} (SavedModel format)"
    )
    try:
        model.export(MODEL_SAVE_PATH)
        logging.info(f"Final trained model successfully saved to: {MODEL_SAVE_PATH}")
    except Exception as e:
        logging.error(
            f"ERROR: Failed to save final model to {MODEL_SAVE_PATH}. Reason: {e}"
        )
        logging.error(
            "Please check file permissions, disk space, and TensorFlow compatibility."
        )
        logging.error(
            "You can still use 'best_sheep_pain_model_final.h5' for deployment if needed."
        )

    logging.info("Sheep Pain Detection Model Training script completed successfully.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()
