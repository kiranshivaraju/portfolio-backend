import datetime
import os
import warnings

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

load_dotenv()
mongo_uri = os.getenv("MONGO_URI")
db_name = os.getenv("DB_NAME")
collection_name = os.getenv("COLLECTION_NAME")

# Suppress the specific warning related to DocumentDB
warnings.filterwarnings("ignore", message="You appear to be connected to a DocumentDB cluster")

class MongoDB:
    def __init__(self):
        self.uri = mongo_uri
        self.client = MongoClient(self.uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def _create_document(self, document):
        try:
            result = self.collection.insert_one(document)
            return result.inserted_id
        except OperationFailure as e:
            print(f"Failed to insert document: {e}")
        except Exception as e:
            print(f"An error occurred during document insertion: {e}")

    def read_document(self, query):
        try:
            document = self.collection.find_one(query)
            return document
        except OperationFailure as e:
            print(f"Failed to find document: {e}")
        except Exception as e:
            print(f"An error occurred during document retrieval: {e}")

    def read_documents(self, query, limit=None, sort_field=None, sort_order=None):
        try:
            cursor = self.collection.find(query)
            if sort_field:
                cursor = cursor.sort(sort_field, sort_order)
            if limit:
                cursor = cursor.limit(limit)
            documents = list(cursor)  # Convert cursor to list
            return documents
        except OperationFailure as e:
            print(f"Failed to find documents: {e}")
        except Exception as e:
            print(f"An error occurred during document retrieval: {e}")

    def update_document(self, query, update):
        try:
            result = self.collection.update_one(query, {"$set": update})
            return result.modified_count
        except OperationFailure as e:
            print(f"Failed to update document: {e}")
        except Exception as e:
            print(f"An error occurred during document update: {e}")

    def delete_document(self, query):
        try:
            result = self.collection.delete_one(query)
            return result.deleted_count
        except OperationFailure as e:
            print(f"Failed to delete document: {e}")
        except Exception as e:
            print(f"An error occurred during document deletion: {e}")

    def insert_document(self, session_id, doc_type, content_text):
        document = {
            "session_id": session_id,
            "type": doc_type,  # AI, HUMAN
            "content": content_text,
            "created_at": datetime.datetime.now().isoformat(),
            "updated_at": datetime.datetime.now().isoformat()
        }
        return self._create_document(document)

# Example of how to use the MongoDB class
def main():
    mongo_db = MongoDB()
    # Example usage:
    mongo_db.insert_document('session1', 'AI', 'Hello World')

# Run the main function
if __name__ == "__main__":
    main()
