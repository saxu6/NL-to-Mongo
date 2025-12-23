from pymongo import MongoClient
from backend.config import settings

client = None
database = None

def connect_to_mongo():
    global client, database
    try:
        client = MongoClient(settings.MONGODB_URI)
        database = client[settings.DATABASE_NAME]
        client.admin.command('ping')
        print("Successfully connected to MongoDB!")
        return database
    except Exception as e:
        print(f"MongoDB connection error: {e}")
        return None

def close_mongo_connection():
    global client
    if client:
        client.close()

def get_database():
    return database

