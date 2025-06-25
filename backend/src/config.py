import os
import logging
from dotenv import load_dotenv

load_dotenv()

# --- Application Configuration ---
APP_TITLE = "Sheep Pain Detection API"
APP_DESCRIPTION = "API for classifying sheep images as in pain or not in pain, storing records based on device ID."
APP_VERSION = "1.0.0"

# --- Machine Learning Model Configuration ---
MODEL_PATH = "ml/sheep_pain_detection_model"
CLASS_NAMES = ["corpus_sheep_face_no_pain", "corpus_sheep_face_pain"]
IMG_HEIGHT = 224  # Must match the input size your model was trained with
IMG_WIDTH = 224  # Must match the input size your model was trained with

# --- MongoDB Configuration ---
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.environ.get("MONGO_DB_NAME", "sheep_pain_db")
# Ensure MongoDB URI is loaded, raise error if critical for startup
if MONGO_URI is None:
    raise ValueError(
        "MONGO_URI environment variable is not set. Please create a .env file or set the variable."
    )
MONGO_COLLECTION_NAME = os.environ.get("MONGO_COLLECTION_NAME", "sheep_records")

# --- CORS Configuration ---
# NOTE: Configure this properly in production!
ALLOWED_ORIGINS = [
    "*",
    "https://face-farm.vercel.app",
    "http://localhost:3000",
]
ALLOWED_METHODS = ["*"]
ALLOWED_HEADERS = [
    "*",
    "X-Device-ID",
]  # X-Device-ID must be allowed for FastAPI Header() to work
ALLOW_CREDENTIALS = True

# --- Logging Configuration ---
LOGGING_LEVEL = logging.INFO
LOGGING_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT)
