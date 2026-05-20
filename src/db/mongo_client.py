import os
import logging
from pymongo import MongoClient
from dotenv import load_dotenv

logger = logging.getLogger("rag-service.db")
logging.basicConfig(level=logging.INFO)

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")

class MongoDB:
    client: MongoClient = None
    db = None

db_connection = MongoDB()

def connect_to_mongo():
    logger.info("Connecting to MongoDB via pymongo...")
    try:
        db_connection.client = MongoClient(MONGODB_URI)
        try:
            db_connection.db = db_connection.client.get_default_database()
        except Exception:
            db_connection.db = db_connection.client["rag_db"]
            
        db_connection.client.admin.command('ping')
        logger.info(f"Successfully connected to MongoDB! Database: {db_connection.db.name}")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise e

def close_mongo_connection():
    if db_connection.client:
        logger.info("Closing connection to MongoDB...")
        db_connection.client.close()
        logger.info("MongoDB connection closed.")

def get_database():
    if db_connection.db is None:
        raise RuntimeError("Database connection has not been initialized. Call connect_to_mongo() first.")
    return db_connection.db
