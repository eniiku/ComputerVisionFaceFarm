# Backend Service: Sheep Pain Assessment API

This repository contains the FastAPI backend service for the Sheep Pain Assessment system. This service is responsible for handling incoming image uploads, performing inference using a pre-trained deep learning model, and storing prediction records in a MongoDB database.

## Table of Contents

1.  [Project Description](#project-description)
2.  [Features](#features)
3.  [Technologies Used](#technologies-used)
4.  [Setup and Installation](#setup-and-installation)
    - [Prerequisites](#prerequisites)
    - [Cloning the Repository](#cloning-the-repository)
    - [MongoDB Atlas Setup](#mongodb-atlas-setup)
    - [Environment Variables (`.env`)](#environment-variables-env)
    - [Install Python Dependencies](#install-python-dependencies)
    - [Place Machine Learning Model](#place-machine-learning-model)
5.  [Running the Backend](#running-the-backend)
6.  [API Endpoints](#api-endpoints)
7.  [Security Considerations](#security-considerations)
8.  [Troubleshooting](#troubleshooting)

## 1. Project Description

The backend service acts as the bridge between the Nextjs-based frontend application and the deep learning model. It exposes RESTful API endpoints for image classification and for managing historical prediction records, providing a robust and scalable solution for automated sheep pain detection.

## 2. Features

- **Image Prediction:** Accepts sheep images, processes them, and returns a pain/no-pain classification with confidence.
- **Record Storage:** Stores prediction results (filename, prediction, probability, confidence, timestamp) in a MongoDB database, linked to a unique device ID.
- **Record Retrieval:** Allows retrieval of all historical prediction records for a given device ID.
- **Health Checks:** Provides an endpoint to verify the operational status of the API, including ML model and database connectivity.
- **Containerization Ready:** Designed for easy deployment in containerized environments.

## 3. Technologies Used

- **Python 3.9+**
- **FastAPI:** High-performance web framework for building APIs.
- **Uvicorn:** ASGI server for running FastAPI applications.
- **TensorFlow/Keras:** For loading and running the deep learning inference model (`TFSMLayer`).
- **PyMongo:** MongoDB driver for Python.
- **Pillow (PIL):** For image processing (reading, resizing).
- **python-dotenv:** For managing environment variables.
- **MongoDB Atlas:** Cloud-hosted NoSQL database (free tier available).

## 4. Setup and Installation

### Prerequisites

- Python 3.9 or higher.
- `pip` (Python package installer).
- A virtual environment (highly recommended).
- Access to a Google Cloud Platform project or Firebase project if you used that path for model saving, or ensure your `tensorflow` installation can access pre-trained weights.

### Cloning the Repository

First, clone your project repository (assuming this backend code is within a `backend` directory in your main project):

```bash
git clone <your-project-repo-url>
cd <your-project-folder>/backend
```

MongoDB Atlas Setup

You need a MongoDB Atlas account and a free tier cluster.

    Create a MongoDB Atlas Account and Free Tier Cluster:

        Go to MongoDB Atlas.

        Follow instructions to create a free (M0 Sandbox) cluster.

    Create a Database User:

        In Atlas, go to "Database Access" under "Security."

        Add a new user with a strong username and password. Grant "Read and write to any database" privileges (for simplicity in development).

        Keep these credentials secure!

    Whitelist Your IP Address:

        In Atlas, go to "Network Access" under "Security."

        Add your current IP address. For development, you can temporarily "Allow Access From Anywhere" (0.0.0.0/0), but never use this in production.

    Get Your Connection String:

        Go to "Databases" -> "Connect" on your cluster -> "Connect your application."

        Copy the connection string. It will look like mongodb+srv://<username>:<password>@clustername.xxxxx.mongodb.net/?retryWrites=true&w=majority.

        Replace <username> and <password> with the credentials you created. This is your MONGO_URI.

Environment Variables (.env)

Create a file named .env in the root of your backend directory. This file will store your sensitive MongoDB connection string and database name.
Code snippet

# .env

MONGO_URI="mongodb+srv://your_username:your_password@clustername.xxxxx.mongodb.net/?retryWrites=true&w=majority"
MONGO_DB_NAME="sheep_pain_db"

IMPORTANT:

    Replace the placeholder values with your actual MongoDB Atlas connection string and chosen database name.

    Add .env to your .gitignore file to prevent committing sensitive information to version control.

Install Python Dependencies

It's highly recommended to use a Python virtual environment.
Bash

# Create a virtual environment

python3 -m venv .venv

# Activate the virtual environment

source .venv/bin/activate # On Linux/macOS

# .venv\Scripts\activate # On Windows CMD

# .venv\Scripts\Activate.ps1 # On Windows PowerShell

# Install dependencies

pip install fastapi uvicorn "python-multipart" Pillow numpy tensorflow "pymongo[srv]" python-dotenv

Place Machine Learning Model

Ensure your trained TensorFlow SavedModel is located at the path specified in config.py. By default, this is expected to be:

backend/
├── ml/
│ └── sheep_pain_detection_model_balanced/ # Your SavedModel directory
│ ├── keras_metadata.pb
│ ├── saved_model.pb
│ └── variables/
└── (other backend files: main.py, config.py, etc.)

Adjust the MODEL_PATH variable in config.py if your model is located elsewhere.

5. Running the Backend

Make sure your virtual environment is active (source .venv/bin/activate).
Bash

uvicorn main:app --host 0.0.0.0 --port 8000 --reload

    main:app: Refers to the app FastAPI instance in main.py.

    --host 0.0.0.0: Makes the server accessible externally (e.g., from your frontend or Postman).

    --port 8000: Runs the server on port 8000.

    --reload: (Development only) Automatically reloads the server on code changes.

You should see output indicating that Uvicorn has started the application.

6.  API Endpoints

    /health (GET)

        Description: Checks the health status of the API, including the ML model loading and MongoDB connection.

        Response: {"status": "ok", "model_loaded": true, "mongodb_ok": true} or an error status.

    /predict (POST)

        Description: Accepts an image file, predicts sheep pain, and stores the record.

        Headers: X-Device-ID: [your_unique_device_id] (a UUID generated by the client, e.g., a1b2c3d4-e5f6-7890-abcd-ef0123456789)

        Body: multipart/form-data with a file field containing the image (JPG or PNG).

        Response:
        JSON

    {
    "filename": "my_sheep_photo.jpg",
    "prediction": "pain",
    "pain_probability": 0.85,
    "confidence": 0.85,
    "record_id": "65b7d8c9e0f1g2h3i4j5k6l7"
    }

/records (GET)

    Description: Retrieves all stored sheep pain prediction records for a specific device.

    Headers: X-Device-ID: [your_unique_device_id]

    Response:
    JSON

        {
            "records": [
                {
                    "id": "65b7d8c9e0f1g2h3i4j5k6l7",
                    "timestamp": "2024-01-30T10:30:00.123456",
                    "filename": "my_sheep_photo.jpg",
                    "prediction": "pain",
                    "pain_probability": 0.85,
                    "confidence": 0.85,
                    "device_id": "a1b2c3d4-e5f6-7890-abcd-ef0123456789"
                },
                // ... more records
            ]
        }

7.  Security Considerations

    CORS (allow_origins=["*"]): In production, change ALLOWED_ORIGINS in config.py to explicitly list the domains from which your frontend will access the API (e.g., ["https://face-farm.vercel.app"]).

    .env File: Never commit your .env file to version control.

    MongoDB Network Access: In production, restrict MongoDB Atlas "Network Access" to only allow IPs from your deployed backend service.

8.  Troubleshooting

    Backend doesn't start (stuck at "Started reloader process"):

        Check MONGO_URI in .env: Ensure it's correctly formatted and includes your correct username, password, and cluster details.

        MongoDB Atlas Network Access: Verify your IP is whitelisted in MongoDB Atlas.

        MODEL_PATH in config.py: Ensure the path to your ML model is correct and the directory exists and contains a valid SavedModel.

        Review Logs: The server's console output will contain ERROR or CRITICAL messages during startup if there are issues with model loading or MongoDB connection. Look for these specific messages.

    X-Device-ID Header Missing: Ensure your frontend or API testing tool is sending the X-Device-ID header for /predict and /records requests.
