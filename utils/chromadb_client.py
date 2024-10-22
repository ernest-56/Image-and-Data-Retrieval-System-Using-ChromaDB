import chromadb

def setup_chromadb():
    client = chromadb.Client()
    # Create or load a collection for image/text embeddings
    collection = client.create_collection("image_text_collection")
    return collection

def add_embeddings_to_collection(collection, embeddings, metadatas, ids):
    collection.add(
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
  