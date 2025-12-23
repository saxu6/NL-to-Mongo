import os
import sys
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv() 


def connect_mongodb(uri: str = os.getenv("MONGODB_URI"), db_name: str = os.getenv("ATLAS_DATABASE_NAME")):
    try:
        client = MongoClient(uri)
        db = client[db_name]
        client.admin.command('ping')
        print("Successfully connected to MongoDB Atlas!")
        return db
        
    except Exception as e:
        print(f"Connection Error: {e}")
        return None

def get_collection(db, collection_name: str):
    if db is not None:
        return db[collection_name]
    return None

if __name__ == "__main__":
    
    database = connect_mongodb()
    if database is None: 
        print("Failed to initialize database.")
        sys.exit(1)
        
    
    # collection = get_collection(database, "events")
    # doc = collection.find_one()
    # if doc:
    #     print("Successfully retrieved a document:")
    #     print(doc)
    # else:
    #     print("Connection successful, but the collection 'events' is empty.")

    # count = collection.count_documents({})
    # print(f"Total documents in 'events': {count}")
