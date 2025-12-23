from pymongo_schema.extract import extract_pymongo_client_schema
from connection import connect_mongodb
import json

db = connect_mongodb()
client = db.client

schema = extract_pymongo_client_schema(client)

with open("full_schema.json", "w") as f:
    json.dump(schema, f, indent=4, default=str)
