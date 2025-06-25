# Machine Learning Models: Sheep Pain Assessment

This section details the development, training, and evaluation of the deep learning models used for automated sheep pain assessment. The core task is a binary classification problem: identifying whether a sheep is in 'Pain' or 'No Pain' based on facial expressions captured in images.

## Table of Contents

1.  [Project Description](#project-description)
2.  [Problem Statement](#problem-statement)
3.  [Dataset](#dataset)
4.  [Model Architecture](#model-architecture)
5.  [Training Strategy](#training-strategy)
    - [Data Preprocessing](#data-preprocessing)
    - [Data Balancing](#data-balancing)
    - [Training Phases](#training-phases)
6.  [Evaluation Metrics](#evaluation-metrics)
7.  [Running the Training Script (Google Colab)](#running-the-training-script-google-colab)
8.  [Output](#output)

## 1. Project Description

This module encompasses the machine learning pipeline, from raw data to a deployable model. It includes scripts for dataset preparation (splitting, augmentation, oversampling), model training, and performance evaluation, focusing on robust pain assessment in an imbalanced data context.

## 2. Problem Statement

Automated sheep pain assessment is critical for animal welfare. However, real-world datasets for pain are often severely imbalanced (many more 'No Pain' instances than 'Pain'). This imbalance poses a significant challenge for deep learning models, as they tend to become biased towards the majority class, leading to poor assessment of the crucial minority 'Pain' class. This project addresses this challenge through targeted data balancing techniques.

## 3. Dataset

The dataset consists of images of sheep facial expressions, categorized into 'No Pain' and 'Pain'.

- **Raw Data Structure:**
  ```
  my_raw_sheep_data/
  ├── No Pain/
  │   ├── image_001.jpg
  │   └── ...
  └── Pain/
      ├── image_X01.jpg
      └── ...
  ```
- **Initial Imbalance:** The raw dataset typically has a significant class imbalance (e.g., 7:1 ratio of 'No Pain' to 'Pain').

- **Preprocessed and Split Data Structure:** The training scripts will automatically split the raw data into:
  ```
  sheep_pain_dataset_split/
  ├── train/
  │   ├── No Pain/
  │   └── Pain/ (will contain original + augmented images for balanced training)
  ├── validation/
  │   ├── No Pain/
  │   └── Pain/
  └── test/
      ├── No Pain/
      └── Pain/
  ```
  - **Split Ratios:** 80% for training, 17% for validation, 3% for testing. Validation and test sets retain their original class imbalance to ensure realistic evaluation.

## 4. Model Architecture

A **transfer learning** approach is employed using **MobileNetV2** as the pre-trained backbone. MobileNetV2 is chosen for its efficiency, making it suitable for potential mobile or edge deployments.

- **Base Model:** MobileNetV2 (pre-trained on ImageNet, without its top classification layer).
- **Custom Classification Head:**
  - `GlobalAveragePooling2D`: Reduces spatial dimensions, suitable for transfer learning.
  - `Dense` layer (128 units, ReLU activation): For learning high-level features.
  - `Dropout` layer (0.5 rate): Regularization to prevent overfitting.
  - `Dense` layer (1 unit, Sigmoid activation): Final output for binary classification (probability of 'Pain').

## 5. Training Strategy

The training process is designed to address the inherent class imbalance and optimize model performance.

### Data Preprocessing

- **Resizing:** All images are uniformly resized to `224x224` pixels.
- **Normalization:** Pixel values are scaled to the `[0, 1]` range.

### Data Balancing

Two key strategies are applied to the **training dataset only**:

1.  **Class Weighting:** During model training, higher importance is assigned to the minority 'Pain' class's contribution to the loss function using `sklearn.utils.class_weight.compute_class_weight('balanced')`.
2.  **Oversampling via Data Augmentation:** The minority 'Pain' class images in the training set are augmented (rotated, shifted, sheared, zoomed, flipped, brightness adjusted) until their count reaches approximately 95% of the majority 'No Pain' class count.

### Training Phases

The model undergoes two distinct training phases:

1.  **Feature Extraction Phase:**
    - MobileNetV2 base layers are **frozen**.
    - Only the newly added custom classification head is trained.
    - **Optimizer:** Adam with a learning rate of `0.0001`.
    - **Loss Function:** `binary_crossentropy`.
2.  **Fine-tuning Phase:**
    - Best weights from the first phase are loaded.
    - MobileNetV2 base layers are **unfrozen**.
    - The entire model is further trained with a significantly reduced learning rate (`0.00001`) to fine-tune the pre-trained weights for the specific sheep dataset.

### Callbacks

- **`EarlyStopping`:** Stops training if validation loss doesn't improve for a specified number of epochs (patience).
- **`ModelCheckpoint`:** Saves the model's weights when validation accuracy is at its highest.
- **`ReduceLROnPlateau`:** Reduces the learning rate if validation loss plateaus.

## 6. Evaluation Metrics

Model performance is rigorously evaluated on the held-out, naturally imbalanced test set using the following metrics, crucial for imbalanced classification:

- **Accuracy:** Overall correct predictions.
- **Precision (Pain):** Proportion of true 'Pain' predictions among all predicted 'Pain' instances.
- **Recall (Pain):** Proportion of true 'Pain' predictions among all actual 'Pain' instances (critical for avoiding missed pain).
- **F1-score (Pain):** Harmonic mean of Precision and Recall for the 'Pain' class, balancing both.
- **Confusion Matrix:** Detailed breakdown of TP, TN, FP, FN.
- **ROC Curve & AUC (Area Under Curve):** Overall discriminatory power across all thresholds, less sensitive to imbalance.
- **Precision-Recall Curve & Average Precision (AP):** Focuses on positive class performance, especially useful for imbalanced data.
- **Predicted Probabilities Scatter Plot:** Visualizes the distribution of model outputs vs. true labels.
- **Prediction Confidence Bubble Plot:** Scatter plot where marker size indicates prediction certainty.

## 7. Running the Training Script (Google Colab)

The training script (`sheep_pain_assessment_training.ipynb` or the combined Python file) is designed to run in a Google Colab environment.

1.  **Open Google Colab:** Go to [Colab](https://colab.research.google.com/) and create a new notebook.
2.  **Set Runtime Type:** Go to `Runtime` -> `Change runtime type` -> select `GPU` as the hardware accelerator.
3.  **Upload Raw Dataset:** Upload your `my_raw_sheep_data` folder (containing `No Pain` and `Pain` subfolders) directly to your Colab session's file system.
4.  **Install Libraries:** Run the following cell in your Colab notebook:
    ```python
    !pip install tensorflow==2.16.1 matplotlib seaborn scikit-learn tqdm Pillow
    ```
5.  **Paste Code:** Copy and paste the sections from the training Python file into separate cells in your Colab notebook, executing them sequentially.
6.  **Adjust `RAW_DATA_INPUT_DIR`:** In the `config.py` (or the Configuration section of the combined script), ensure `RAW_DATA_INPUT_DIR` points to the correct path of your uploaded raw dataset (e.g., `/content/my_raw_sheep_data`).
7.  **Run All Cells:** Execute all cells in order.

The script will perform:

- Data splitting (creates `sheep_pain_dataset_split` folder).
- Oversampling of the minority class in the training split.
- Model training (two phases).
- Model evaluation and generation of all specified plots and reports.

## 8. Output

Upon successful training, the script will:

- Print test loss and accuracy.
- Print a detailed classification report (precision, recall, f1-score).
- Generate and display plots for:
  - Training/Validation Accuracy & Loss
  - Confusion Matrix
  - ROC Curve
  - Precision-Recall Curve
  - Predicted Probabilities Scatter Plot
  - Prediction Confidence Bubble Plot
- Save the best trained model (in TensorFlow SavedModel format) to your local Colab session directory, e.g., `sheep_pain_assessment_model_balanced`. This folder needs to be downloaded and used by the backend service.
