import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import logging

# --- Configuration ---
DATA_DIR = 'sheep_pain_dataset'
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
NUM_CLASSES = 2
EPOCHS = 20

train_dir = os.path.join(DATA_DIR, 'train')
validation_dir = os.path.join(DATA_DIR, 'validation')
test_dir = os.path.join(DATA_DIR, 'test') # Assuming a test directory exists

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Data Augmentation and Loading ---
# ImageDataGenerator is used to load images from directories and apply real-time
# data augmentation to the training set. This helps prevent overfitting and
# improves the model's ability to generalize to unseen data.
# For validation and test sets, only rescaling is applied.

logger.info("Setting up ImageDataGenerators and loading data...")

train_datagen = ImageDataGenerator(
    rescale=1./255,                 # Normalize pixel values to [0, 1]
    rotation_range=20,              # Random rotations up to 20 degrees
    width_shift_range=0.2,          # Random horizontal shifts (fraction of total width)
    height_shift_range=0.2,         # Random vertical shifts (fraction of total height)
    shear_range=0.2,                # Shear transformations
    zoom_range=0.2,                 # Random zooms
    horizontal_flip=True,           # Random horizontal flips
    fill_mode='nearest'             # Strategy for filling in new pixels created by transformations
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

    test_generator = validation_test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False # shuffle=False for reproducible evaluation.
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
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_sheep_pain_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
]

# --- Model Training ---
logger.info(f"Starting model training for {EPOCHS} epochs...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=callbacks
)
logger.info("Model training finished.")

# --- Plotting Training History ---
# Visualize training and validation accuracy/loss to understand model performance
# and identify potential overfitting.
logger.info("Plotting training history...")
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
logger.info("Training history plots displayed.")

# --- Model Evaluation on Test Set ---
# Evaluate the final model on the unseen test set to get an unbiased estimate
# of its performance.
logger.info("Evaluating model on the test set...")
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
logger.info(f"Test Loss: {test_loss:.4f}")
logger.info(f"Test Accuracy: {test_accuracy:.4f}")

# --- Detailed Classification Report and Confusion Matrix ---
# These metrics provide a deeper insight into the model's performance,
# especially important for imbalanced datasets.
logger.info("Generating detailed classification report and confusion matrix on test data...")

# Predict probabilities on the test set
test_generator.reset() # Reset generator to ensure predictions match labels order
Y_pred_probs = model.predict(test_generator, steps=test_generator.samples // BATCH_SIZE + 1)
y_pred_classes = (Y_pred_probs > 0.5).astype(int).flatten() # Convert probabilities to binary class labels

# Get true labels from the test generator
y_true = test_generator.classes[test_generator.index_array] # Correct way to get ordered labels

# Ensure lengths match in case of incomplete last batch
y_true = y_true[:len(y_pred_classes)]

print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_true, y_pred_classes))
logger.info("Classification report and confusion matrix displayed.")

# --- Save the Trained Model for Deployment ---
# Save the model in TensorFlow's SavedModel format, which is recommended for production.
# This format saves the model's architecture, weights, and training configuration.
MODEL_SAVE_PATH = 'sheep_pain_detection_model'
model.save(MODEL_SAVE_PATH)
logger.info(f"\nFinal trained model saved to: {MODEL_SAVE_PATH}")

# If you specifically need the .h5 format (e.g., for compatibility):
# MODEL_H5_PATH = 'sheep_pain_detection_model.h5'
# model.save(MODEL_H5_PATH)
# logger.info(f"Final trained model also saved as .h5 to: {MODEL_H5_PATH}")

logger.info("Model training and saving script completed.")
