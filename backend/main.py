from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np
import tensorflow as tf # Import TensorFlow
import logging
import os

# --- Configuration ---
MODEL_PATH = "ml/sheep_pain_detection_model"
CLASS_NAMES = ["corpus_sheep_face_no_pain", "corpus_sheep_face_pain"]
IMG_HEIGHT = 224 # Must match the input size your model was trained with
IMG_WIDTH = 224  # Must match the input size your model was trained with

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Load Model ---
inference_model_layer = None # Initialize to None
try:
    inference_model_layer = tf.keras.layers.TFSMLayer(MODEL_PATH, call_endpoint='serving_default')

    # Run a dummy prediction to ensure the model layer is loaded correctly and is functional.
    dummy_input = tf.constant(np.zeros((1, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32))
    _ = inference_model_layer(dummy_input)
    logger.info(f"Model layer loaded successfully from {MODEL_PATH} using TFSMLayer.")
except Exception as e:
    logger.error(f"FATAL ERROR: Could not load AI model layer from {MODEL_PATH}. Reason: {e}")
    raise RuntimeError(f"Failed to load AI model layer. Please check MODEL_PATH and model files: {e}")

# --- FastAPI Application Instance ---
app = FastAPI(
    title="Sheep Pain Detection API",
    description="API for classifying sheep images as in pain or not in pain using a deep learning model.",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  #NOTE: Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Health Check Endpoint ---
@app.get("/health", summary="Health Check", response_model=dict)
async def health_check():
    """
    Checks the health of the API and ensures the model layer is loaded and responsive.
    """
    if inference_model_layer is None:
        logger.warning("Health check: Model layer is not loaded (should not happen if startup was successful).")
        return {"status": "error", "model_loaded": False, "detail": "Model layer not loaded"}
    try:
        # Perform a quick dummy inference to verify model layer responsiveness
        dummy_input = tf.constant(np.zeros((1, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32))
        _ = inference_model_layer(dummy_input)
        logger.info("Health check: Model layer responsive.")
        return {"status": "ok", "model_loaded": True}
    except Exception as e:
        logger.error(f"Health check failed due to model layer inference error: {e}")
        return {"status": "error", "model_loaded": False, "detail": f"Model layer inference failed: {e}"}

# --- Prediction Endpoint ---
@app.post("/predict", summary="Predict Sheep Pain", response_model=dict)
async def predict_pain(file: UploadFile = File(...)):
    """
    Accepts an image file of a sheep, preprocesses it, and predicts
    whether the sheep is in pain based on the trained deep learning model.

    - **file**: Upload a sheep image in JPG or PNG format.
    """
    # 1. Validate input file type
    if not file.content_type.startswith("image/"):
        logger.warning(f"Invalid file type uploaded: {file.content_type}")
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image (JPG, PNG).")

    try:
        # 2. Read image content
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB") # Ensure 3 channels for consistency

        # 3. Preprocess image for model input
        # Resize the image to the dimensions expected by the model.
        image = image.resize((IMG_WIDTH, IMG_HEIGHT))
        # Convert PIL image to NumPy array and normalize pixel values to [0, 1].
        image_array = np.array(image) / 255.0
        # Add a batch dimension and convert to TensorFlow tensor
        input_tensor = tf.constant(np.expand_dims(image_array, axis=0)) # Shape becomes (1, IMG_HEIGHT, IMG_WIDTH, 3)

        # 4. Perform inference using the loaded model layer
        logger.info(f"Performing inference for file: {file.filename}")
        # Call the TFSMLayer directly with the input tensor
        predictions = inference_model_layer(input_tensor)

        # 5. Interpret predictions
        # The output of TFSMLayer might be a dictionary if the SavedModel has named outputs.
        # For a simple classification model exported with model.export(), it often returns
        # a single tensor, but it's safer to check the structure.
        # Assuming the output is a single tensor representing probability.
        # If it returns a dictionary like {'output_0': tensor}, you might need predictions['output_0'].numpy()[0][0]
        pain_probability = float(predictions.numpy()[0][0]) # Extract value from TensorFlow tensor

        # Classify based on a threshold (e.g., 0.5 probability)
        if pain_probability >= 0.5:
            prediction_label = CLASS_NAMES[1]
        else:
            prediction_label = CLASS_NAMES[0]

        # Calculate a confidence score (the higher probability of the two classes)
        confidence = max(pain_probability, 1 - pain_probability)

        logger.info(f"Prediction for {file.filename}: '{prediction_label}' (Pain Probability: {pain_probability:.4f}, Confidence: {confidence:.4f})")

        # 6. Return structured JSON response
        return JSONResponse(content={
            "filename": file.filename,
            "prediction": prediction_label,
            "pain_probability": pain_probability,
            "confidence": confidence
        })

    except Exception as e:
        logger.exception(f"Error processing prediction for {file.filename}: {e}")
        # Return a 500 Internal Server Error for unhandled exceptions during prediction
        raise HTTPException(status_code=500, detail=f"An internal server error occurred during prediction: {e}")


