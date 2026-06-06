import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb

#Load match data
print("[INFO] Loading match data...")
df = pd.read_csv("match_data.csv")

# Drop rows with missing key fields so sentences are complete
df = df.dropna(subset=['team1', 'team2', 'venue', 'result'])
df = df.fillna("Unknown")

print(f"[INFO] Loaded {len(df)} matches")


#Convert each match row into a natural language sentence
print("[INFO] Converting match records to text sentences...")

sentences = []
ids = []
for i, (_, row) in enumerate(df.iterrows()):
    scorer_part = (
        f"Top scorer: {row['top_scorer']} with {row['top_score']} runs."
        if str(row.get("top_scorer", "Unknown")) not in ("Unknown", "")
        else "Top scorer data unavailable."
    )
    text = (
        f"{row['team1']} vs {row['team2']} at {row['venue']} "
        f"on {row['match_date']}. {row['result']}. {scorer_part}"
    )
    sentences.append(text)
    ids.append(str(i))

print(f"[INFO] Created {len(sentences)} sentences")
print(f"[INFO] Example: {sentences[0]}")

#Generate vector embeddings
print("[INFO] Loading sentence transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("[INFO] Generating embeddings...")
embeddings = model.encode(sentences, show_progress_bar=True).tolist()
print(f"[INFO] Generated {len(embeddings)} embeddings")


# Store embeddings in ChromaDB (in-memory vector database)
print("[INFO] Building ChromaDB vector index...")
client = chromadb.Client()  # in-memory client
collection = client.create_collection("cricket_matches")

collection.add(
    documents=sentences,
    embeddings=embeddings,
    ids=ids,
)
print(f"[INFO] Indexed {len(sentences)} match records into ChromaDB")


#Semantic search function
def search(query, top_k=3):
    """
    Search for matches semantically similar to the query.

    Args:
        query  : natural language search string
        top_k  : number of top results to return (default 3)

    Returns:
        list of matching sentences
    """
    # Convert query to embedding using the same model
    query_vec = model.encode([query]).tolist()

    # ChromaDB finds top_k most similar documents
    results = collection.query(query_embeddings=query_vec, n_results=top_k)
    return results["documents"][0]


# Interactive query loop
print("\n" + "="*60)
print("[INFO] Semantic Search Ready. Type 'quit' to exit.")
print("="*60)

while True:
    query = input("\nEnter your query: ").strip()
    if query.lower() in ("quit", "exit", "q"):
        break
    if not query:
        continue

    results = search(query)
    print(f"\nTop {len(results)} results for: '{query}'")
    print("-" * 40)
    for i, doc in enumerate(results, 1):
        print(f"  {i}. {doc}")

print("\n" + "="*60)
print("[INFO] Semantic search complete")
print("="*60)
