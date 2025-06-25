import tensorflow as tf
import numpy as np
import logging

from config import MODEL_PATH, IMG_HEIGHT, IMG_WIDTH

logger = logging.getLogger(__name__)

# Global variable to store the loaded model layer
inference_model_layer = None


def load_inference_model():
    """
    Loads the TensorFlow SavedModel using TFSMLayer and performs a dummy prediction
    to ensure it's functional. Stores the loaded model in a global variable.
    """
    global inference_model_layer
    if inference_model_layer is not None:
        logger.info("Model already loaded. Skipping re-load.")
        return inference_model_layer

    logger.info(f"Attempting to load AI model layer from: {MODEL_PATH}")
    try:
        inference_model_layer = tf.keras.layers.TFSMLayer(
            MODEL_PATH, call_endpoint="serving_default"
        )

        # Run a dummy prediction to ensure the model layer is loaded correctly and is functional.
        # This also pre-warms the model, reducing latency on the first real request.
        dummy_input = tf.constant(
            np.zeros((1, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
        )
        _ = inference_model_layer(dummy_input)
        logger.info(
            f"Model layer loaded successfully from {MODEL_PATH} and verified functional."
        )
        return inference_model_layer
    except Exception as e:
        logger.error(
            f"FATAL ERROR: Could not load AI model layer from {MODEL_PATH}. Reason: {e}"
        )
        logger.error(
            "Please ensure the MODEL_PATH is correct and the SavedModel is valid."
        )
        raise RuntimeError(f"Failed to load AI model layer. Check MODEL_PATH: {e}")


def get_loaded_model():
    """
    Returns the globally loaded inference model layer.
    """
    if inference_model_layer is None:
        # Attempt to load if not already loaded (e.g., if called directly without explicit load_inference_model)
        # This can be helpful but `load_inference_model` should ideally be called once at app startup.
        return load_inference_model()
    return inference_model_layer
