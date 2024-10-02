from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
import os
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
import json
from langchain.schema import Document
from tqdm import tqdm, trange
from tqdm import tqdm
import time
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1536, openai_api_key=OPENAI_API_KEY)
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
chunk_size = 50
data = []

with open("src/rag/data/data_to_index.jsonl", "r") as file:
    for line in file:
        # Parse each line (JSON object) and append to the list
        data.append(json.loads(line))

def create_vectordb_index(index_name):
    client.create_collection(
        collection_name=index_name,
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
    )

def add_data_to_index(index_name, data):
    vector_store = QdrantVectorStore(client=client, collection_name=index_name, embedding=embeddings)
    documents = []
    for item in data:
        try: 
            instruction = item.get('question', '')
            output = item.get('answer', '')
            document = Document(page_content=f"Question: {instruction} Answer: {output}")
            documents.append(document)
        except Exception as e:
            print(e)
            continue
    for i in tqdm(range(0, len(documents), chunk_size)):
        try:
            chunk = documents[i:i+chunk_size]
            vector_store.add_documents(chunk)
        except Exception as e:
            print(e)
            continue
    return {"message": "Data indexed successfully!"}

# a = add_data_to_index("kiran_portfolio", data)
# print(a)