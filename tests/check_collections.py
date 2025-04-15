from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Qdrant client
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

# Check document count in both collections
for lang in ["en", "ar"]:
    collection_name = f"rag_docs_{lang}"
    count = client.count(collection_name)
    print(f"Collection '{collection_name}' has {count} documents") 