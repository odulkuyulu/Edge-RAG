"""
Vector Database Cleaner Module

This module provides functionality to clean up the Qdrant vector database by:
1. Listing all existing collections
2. Deleting specified collections
3. Verifying the cleanup operation

This is useful for:
- Resetting the database to a clean state
- Removing test data
- Troubleshooting database issues
"""

from vector_db import VectorDBClient

def main():
    """
    Main function to clean up the vector database.
    
    This function:
    1. Lists all collections before deletion
    2. Attempts to delete the 'documents' collection
    3. Verifies the deletion by listing remaining collections
    """
    # Initialize vector store client
    vector_store = VectorDBClient()

    # List all collections before clearing
    collections_before_clearing = vector_store.client.get_collections().collections
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
        vector_store.delete_collection()
        print(f"Collection '{collection_name}' deleted successfully.")
    except Exception as e:
        print(f"Could not delete collection '{collection_name}': {e}. It might not exist.")

    # Verify collections after clearing
    collections_after_clearing = vector_store.client.get_collections().collections
    print("\nCollections in Qdrant after clearing:")
    if collections_after_clearing:
        for collection in collections_after_clearing:
            print(f"- {collection.name}")
    else:
        print("No collections found after clearing.")

if __name__ == "__main__":
    main() 