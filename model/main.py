import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import numpy as np
import logging

from evaluators import (
    plot_training_history,
    plot_confusion_matrix,
    print_classification_report,
    plot_roc_curve,
    plot_precision_recall_curve
)

DATA_DIR = 'datasets/sheep_pain_dataset'
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
NUM_CLASSES = 2
EPOCHS = 20

# Define paths for your dataset
train_dir = os.path.join(DATA_DIR, 'train')
validation_dir = os.path.join(DATA_DIR, 'validation')
test_dir = os.path.join(DATA_DIR, 'test')

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Sheep Pain Detection Model Training.")

    # --- Data Augmentation and Loading ---
    # ImageDataGenerator is used to load images from directories and apply real-time
    # data augmentation to the training set. This helps prevent overfitting and
    # improves the model's ability to generalize to unseen data.
    # For validation and test sets, only rescaling is applied.

    logger.info("Setting up ImageDataGenerators and loading data...")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # For validation and test sets, only rescale. No augmentation should be applied.
    validation_test_datagen = ImageDataGenerator(rescale=1./255)

    try:
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='binary'
        )

        validation_generator = validation_test_datagen.flow_from_directory(
            validation_dir,
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='binary'
        )

        # Load test data. Set shuffle=False for reproducible evaluation.
        test_generator = validation_test_datagen.flow_from_directory(
            test_dir,
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='binary',
            shuffle=False # IMPORTANT: Do not shuffle test data for proper evaluation metrics
        )

        class_names = list(train_generator.class_indices.keys())
        logger.info(f"Detected classes: {class_names}")

        logger.info(f"Found {train_generator.samples} training images belonging to {train_generator.num_classes} classes.")
        logger.info(f"Found {validation_generator.samples} validation images belonging to {validation_generator.num_classes} classes.")
        logger.info(f"Found {test_generator.samples} test images belonging to {test_generator.num_classes} classes.")

    except Exception as e:
        logger.error(f"Error loading data from directories. Please check DATA_DIR path and folder structure: {e}")
        raise

    # --- Model Architecture (Transfer Learning with MobileNetV2) ---
    # Transfer learning is used to leverage a pre-trained model (MobileNetV2)
    # that has learned powerful features from a very large dataset (ImageNet).
    # We "freeze" its layers and add our own classification layers on top.

    logger.info("Building the model architecture using MobileNetV2 for transfer learning...")

    # Load the MobileNetV2 model pre-trained on ImageNet, excluding its original top (classification) layer.
    base_model = MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), # Input shape matching our image dimensions
        include_top=False, # We don't want the original ImageNet classification head
        weights='imagenet' # Use weights trained on ImageNet
    )

    # Freeze the base model layers. This means their weights will not be updated during training.
    # This is common in the initial phase of transfer learning to preserve learned features.
    base_model.trainable = False
    logger.info("Base model (MobileNetV2) layers frozen.")

    # Add custom classification layers on top of the base model's output.
    x = base_model.output
    x = GlobalAveragePooling2D()(x) # Reduces spatial dimensions to a single vector per feature map
    x = Dense(128, activation='relu')(x) # A dense layer with ReLU activation
    x = Dropout(0.5)(x) # Dropout layer for regularization to prevent overfitting
    predictions = Dense(1, activation='sigmoid')(x) # Output layer for binary classification (sigmoid for probability)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.summary()

    # --- Model Compilation ---
    # Compile the model with an optimizer, loss function, and metrics.
    # Adam optimizer is a good general-purpose choice.
    # Binary Crossentropy is suitable for binary classification with a sigmoid output.
    logger.info("Compiling the model...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), # Small learning rate for initial training
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # --- Callbacks for Training ---
    # Callbacks are functions that can be applied at certain stages of the training process.
    # EarlyStopping: Stops training if validation loss does not improve for a certain number of epochs.
    # ModelCheckpoint: Saves the best model weights based on validation accuracy.
    logger.info("Setting up training callbacks (EarlyStopping, ModelCheckpoint)...")
    callbacks = [
        # Monitor validation loss and restore the best weights when training stops
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        # Save the model that achieves the highest validation accuracy
        ModelCheckpoint('best_sheep_pain_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    ]

    # --- Model Training ---
    logger.info(f"Starting model training for {EPOCHS} epochs...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE, # Number of batches per epoch
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        callbacks=callbacks
    )
    logger.info("Model training finished.")

    # --- Evaluation and Reporting ---
    logger.info("Generating evaluation reports and plots...")

    # Plot training history (accuracy and loss curves)
    plot_training_history(history)

    # Evaluate model on the unseen test set
    logger.info("Evaluating model on the test set...")
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")

    # Generate predictions on the test set for detailed metrics
    logger.info("Generating predictions on test data for detailed report...")
    test_generator.reset() # Important: Reset generator to ensure predictions match labels order
    num_test_steps = test_generator.samples // test_generator.batch_size + \
                     (test_generator.samples % test_generator.batch_size != 0)

    # Get raw probabilities for ROC/PR curves
    Y_pred_probs = model.predict(test_generator, steps=num_test_steps)
    # Convert probabilities to binary class labels using 0.5 threshold
    y_pred_classes = (Y_pred_probs > 0.5).astype(int).flatten()

    # Get true labels directly from the generator after reset.
    # Slicing is important because the last batch might be incomplete.
    y_true = test_generator.classes[test_generator.index_array][:len(y_pred_classes)]

    # Plot Confusion Matrix
    logger.info("Plotting Confusion Matrix...")
    plot_confusion_matrix(y_true, y_pred_classes, class_names)

    # Print Classification Report
    logger.info("Printing Classification Report...")
    print_classification_report(y_true, y_pred_classes, class_names)

    # Plot ROC Curve
    logger.info("Plotting ROC Curve...")
    plot_roc_curve(y_true, Y_pred_probs.flatten(), class_names)

    # Plot Precision-Recall Curve
    logger.info("Plotting Precision-Recall Curve...")
    plot_precision_recall_curve(y_true, Y_pred_probs.flatten(), class_names)

    # --- Save the Final Trained Model ---
    # Save the model in TensorFlow's SavedModel format, which is recommended for production deployment.
    # This format saves the model's architecture, weights, and training configuration.
    MODEL_SAVE_PATH = 'sheep_pain_detection_model'
    model.export(MODEL_SAVE_PATH)
    logger.info(f"\nFinal trained model saved to: {MODEL_SAVE_PATH}")

    logger.info("Sheep Pain Detection Model Training script completed successfully.")

if __name__ == '__main__':
    main()
