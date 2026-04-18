import chromadb
from sentence_transformers import SentenceTransformer

VECTORSTORE = "data/vectorstore"
COLLECTION  = "legal_docs"
MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"

# Load existing vectorstore — no re-embedding
client     = chromadb.PersistentClient(path=VECTORSTORE)
collection = client.get_collection(COLLECTION)
model      = SentenceTransformer(MODEL_NAME)

print(f"✅ Loaded {collection.count()} vectors\n")

while True:
    query = input("🔍 Enter query (or 'quit'): ").strip()
    if query.lower() == "quit":
        break

    results = collection.query(
        query_texts = [query],
        n_results   = 3
    )

    for i in range(len(results["ids"][0])):
        print(f"\n  Result {i+1}")
        print(f"  Doc     : {results['metadatas'][0][i]['doc_name']}")
        print(f"  Title   : {results['metadatas'][0][i]['title']}")
        print(f"  Chunk ID: {results['ids'][0][i]}")
        print(f"  Distance: {results['distances'][0][i]:.4f}")
        print(f"  Preview : {results['documents'][0][i][:150]}...")