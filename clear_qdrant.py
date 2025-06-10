from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

# List all collections before clearing
collections_before_clearing = client.get_collections().collections
print("Collections in Qdrant before clearing:")
if collections_before_clearing:
    for collection in collections_before_clearing:
        print(f"- {collection.name}")
else:
    print("No collections found before clearing.")

# Delete the 'documents' collection if it exists
collection_name = "documents"
try:
    print(f"\nAttempting to delete collection '{collection_name}'...")
    client.delete_collection(collection_name=collection_name)
    print(f"Collection '{collection_name}' deleted successfully.")
except Exception as e:
    print(f"Could not delete collection '{collection_name}': {e}. It might not exist.")

# Verify collections after clearing
collections_after_clearing = client.get_collections().collections
print("\nCollections in Qdrant after clearing:")
if collections_after_clearing:
    for collection in collections_after_clearing:
        print(f"- {collection.name}")
else:
    print("No collections found after clearing.") 