from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
import logging
import datetime
from config import MONGO_URI, MONGO_DB_NAME, MONGO_COLLECTION_NAME

logger = logging.getLogger(__name__)

mongo_client: MongoClient = None
mongo_db = None


def initialize_mongodb():
    """
    Initializes the MongoDB client and connects to the database.
    This should be called once at application startup.
    """
    global mongo_client, mongo_db
    if mongo_client is not None and mongo_client.admin.command("ping"):
        logger.info("MongoDB client already connected. Skipping re-initialization.")
        return mongo_db

    logger.info("Attempting to initialize MongoDB client and connect to database...")
    try:
        # Connect to MongoDB Atlas cluster
        mongo_client = MongoClient(MONGO_URI)

        mongo_client.admin.command("ping")
        mongo_db = mongo_client[MONGO_DB_NAME]
        logger.info(f"Successfully connected to MongoDB database: '{MONGO_DB_NAME}'")
        return mongo_db
    except ConnectionFailure as e:
        logger.critical(
            f"FATAL ERROR: MongoDB connection failed. Please check MONGO_URI and network access. Reason: {e}"
        )
        raise RuntimeError(f"Failed to connect to MongoDB: {e}")
    except Exception as e:
        logger.critical(f"FATAL ERROR: Could not initialize MongoDB. Reason: {e}")
        raise RuntimeError(f"Failed to initialize MongoDB: {e}")


def get_mongo_db():
    """
    Returns the globally initialized MongoDB database instance.
    Ensures connection is live, or attempts to re-initialize.
    """
    global mongo_client, mongo_db
    if mongo_db is None:
        return initialize_mongodb()

    try:
        if mongo_client and mongo_client.admin.command("ping"):
            return mongo_db
        else:
            logger.warning("MongoDB connection lost. Attempting to re-initialize.")
            return initialize_mongodb()
    except Exception as e:
        logger.error(f"MongoDB ping failed: {e}. Attempting re-initialization.")
        return initialize_mongodb()


async def insert_record(device_id: str, record_data: dict) -> str:
    """
    Inserts a new sheep pain record into MongoDB for a given device ID.
    Returns the inserted document's ID.
    """
    db_instance = get_mongo_db()
    if db_instance is None:
        raise ConnectionError("MongoDB database not initialized.")

    collection = db_instance[MONGO_COLLECTION_NAME]

    # Add device_id and timestamp to the record data
    record_data["device_id"] = device_id
    record_data["timestamp"] = datetime.datetime.utcnow()  # UTC timestamp

    try:
        result = collection.insert_one(record_data)
        return str(result.inserted_id)
    except OperationFailure as e:
        logger.error(f"MongoDB insert operation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Error inserting record into MongoDB: {e}")
        raise


async def get_records_by_device_id(device_id: str) -> list[dict]:
    """
    Retrieves all records for a given device ID from MongoDB.
    """
    db_instance = get_mongo_db()
    if db_instance is None:
        raise ConnectionError("MongoDB database not initialized.")

    collection = db_instance[MONGO_COLLECTION_NAME]

    try:
        # Find all documents where 'device_id' matches
        records_cursor = collection.find({"device_id": device_id}).sort(
            "timestamp", -1
        )  # Sort by most recent first
        records = []
        for doc in records_cursor:
            doc_copy = dict(doc)  # Create a copy to modify
            doc_copy["id"] = str(doc_copy["_id"])  # Convert ObjectId to string
            del doc_copy["_id"]  # Remove ObjectId field for JSON serialization
            if "timestamp" in doc_copy and isinstance(
                doc_copy["timestamp"], datetime.datetime
            ):
                doc_copy["timestamp"] = doc_copy[
                    "timestamp"
                ].isoformat()  # Convert datetime to ISO string
            records.append(doc_copy)
        return records
    except OperationFailure as e:
        logger.error(f"MongoDB find operation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Error retrieving records from MongoDB: {e}")
        raise
