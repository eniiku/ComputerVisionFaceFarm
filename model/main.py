import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os
import numpy as np
import logging
from sklearn.utils import class_weight # For class imbalance handling
from evaluators import (
    plot_training_history,
    plot_confusion_matrix,
    print_classification_report,
    plot_roc_curve,
    plot_precision_recall_curve
)

# --- Configuration ---
DATA_DIR = 'datasets/sheep_pain_dataset'
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
NUM_CLASSES = 2
EPOCHS = 30 # Increased epochs; EarlyStopping will manage actual training length

# Define paths for your dataset
train_dir = os.path.join(DATA_DIR, 'train')
validation_dir = os.path.join(DATA_DIR, 'validation')
test_dir = os.path.join(DATA_DIR, 'test')

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Sheep Pain Detection Model Training (Accuracy Focus).")

    # --- Data Augmentation and Loading ---
    logger.info("Setting up ImageDataGenerators and loading data...")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

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

    # --- Handle Class Imbalance with Class Weighting ---
    # This calculates weights to give more importance to the minority class (e.g., 'Pain').
    # The 'balanced' mode automatically computes weights inversely proportional to class frequencies.
    logger.info("Computing class weights to handle dataset imbalance...")
    try:
        class_weights_array = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_generator.classes),
            y=train_generator.classes
        )
        class_weights_dict = dict(enumerate(class_weights_array))
        logger.info(f"Computed Class Weights: {class_weights_dict}")
    except Exception as e:
        logger.error(f"Failed to compute class weights: {e}")
        class_weights_dict = None # Proceed without class weights if calculation fails

    # --- Model Architecture (Transfer Learning with MobileNetV2) ---
    logger.info("Building the model architecture using MobileNetV2 for transfer learning...")

    base_model = MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet'
    )

    base_model.trainable = False # Freeze layers for initial training
    logger.info("Base model (MobileNetV2) layers frozen for initial training.")

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.summary()

    # --- Model Compilation ---
    logger.info("Compiling the model for initial training phase...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), # Small learning rate
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # --- Callbacks for Initial Training ---
    logger.info("Setting up callbacks for initial training (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau)...")
    initial_callbacks = [
        # Stop training if validation loss doesn't improve for 5 epochs
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        # Save the best model based on validation accuracy during this phase
        ModelCheckpoint('best_sheep_pain_model_stage1.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        # Reduce learning rate if validation loss plateaus for 3 epochs
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
    ]

    # --- Initial Model Training (Feature Extraction) ---
    logger.info(f"Starting initial model training (feature extraction) for {EPOCHS} epochs...")
    history_initial = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        class_weight=class_weights_dict,
        callbacks=initial_callbacks
    )
    logger.info("Initial model training finished.")

    # --- Fine-tuning Phase ---
    # This is critical for adapting the pre-trained features to your specific dataset.
    # Unfreeze some layers of the base model and train with a very low learning rate.
    logger.info("Starting fine-tuning phase...")
    # Load the best model from the previous stage to ensure we continue from the best point
    model.load_weights('best_sheep_pain_model_stage1.h5')
    logger.info("Loaded best weights from initial training for fine-tuning.")

    # Unfreeze the base model
    base_model.trainable = True
    logger.info("Base model layers unfrozen for fine-tuning.")

    # Re-compile the model with a much lower learning rate for fine-tuning
    # This allows the base model's weights to be slightly adjusted without forgetting
    # the general features it learned.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), # Very small learning rate for fine-tuning
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    logger.info("Model re-compiled for fine-tuning with a very low learning rate.")

    # Callbacks for Fine-tuning
    fine_tune_epochs = 20 # Additional epochs for fine-tuning
    fine_tune_callbacks = [
        # Stop training if validation loss doesn't improve for 7 epochs (slightly more patience)
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
        # Save the overall best model (including fine-tuned weights)
        ModelCheckpoint('best_sheep_pain_model_final.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        # Reduce learning rate again if validation loss plateaus for 4 epochs
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-8, verbose=1)
    ]

    logger.info(f"Starting fine-tuning for {fine_tune_epochs} additional epochs...")
    history_fine_tune = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS + fine_tune_epochs, # Total epochs include previous and new
        initial_epoch=history_initial.epoch[-1], # Start from where initial training left off
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        class_weight=class_weights_dict, # Continue applying class weights
        callbacks=fine_tune_callbacks
    )
    logger.info("Fine-tuning finished.")

    # Combine histories for plotting comprehensive curves
    combined_history = {
        'accuracy': history_initial.history['accuracy'] + history_fine_tune.history['accuracy'],
        'val_accuracy': history_initial.history['val_accuracy'] + history_fine_tune.history['val_accuracy'],
        'loss': history_initial.history['loss'] + history_fine_tune.history['loss'],
        'val_loss': history_initial.history['val_loss'] + history_fine_tune.history['val_loss']
    }

    # --- Evaluation and Reporting ---
    logger.info("Generating evaluation reports and plots...")

    # Plot comprehensive training history (accuracy and loss curves)
    plot_training_history(combined_history)

    # Evaluate model on the unseen test set (load the best overall weights first)
    model.load_weights('best_sheep_pain_model_final.h5') # Ensure the best fine-tuned model is used
    logger.info("Loaded final best weights for test set evaluation.")

    logger.info("Evaluating model on the test set...")
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")

    # Generate predictions on the test set for detailed metrics
    logger.info("Generating predictions on test data for detailed report...")
    test_generator.reset() # Important: Reset generator to ensure predictions match labels order
    num_test_steps = test_generator.samples // test_generator.batch_size + \
                     (test_generator.samples % test_generator.batch_size != 0)

    Y_pred_probs = model.predict(test_generator, steps=num_test_steps)
    y_pred_classes = (Y_pred_probs > 0.5).astype(int).flatten()

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

    # --- Save the Final Trained Model in SavedModel format ---
    MODEL_SAVE_PATH = 'sheep_pain_detection_model'
    logger.info(f"Attempting to save final model to: {MODEL_SAVE_PATH} (SavedModel format)")
    try:
        model.export(MODEL_SAVE_PATH) # Use model.export() for SavedModel format
        logger.info(f"Final trained model successfully saved to: {MODEL_SAVE_PATH}")
    except Exception as e:
        logger.error(f"ERROR: Failed to save final model to {MODEL_SAVE_PATH}. Reason: {e}")
        logger.error("Please check file permissions, disk space, and TensorFlow compatibility.")
        logger.error("You can still use 'best_sheep_pain_model_final.h5' for deployment if needed.")


    logger.info("Sheep Pain Detection Model Training script completed successfully.")

if __name__ == '__main__':
    main()

