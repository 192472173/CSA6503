
import pandas as pd
import faiss
import chromadb
from sentence_transformers import SentenceTransformer
import time

# -----------------------------------
# 1. Load dataset
# -----------------------------------
df = pd.read_csv("dataset/courses.csv")

print("Dataset loaded:", len(df), "courses")

texts = (
    df["title"].astype(str) + " " +
    df["description"].astype(str) + " " +
    df["category"].astype(str) + " " +
    df["level"].astype(str)
).tolist()

# -----------------------------------
# 2. Load embedding model
# -----------------------------------
print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("Embedding shape:", embeddings.shape)

# -----------------------------------
# 3. FAISS
# -----------------------------------
dimension = embeddings.shape[1]

faiss_index = faiss.IndexFlatIP(dimension)

faiss_index.add(
    embeddings.astype("float32")
)

# -----------------------------------
# 4. ChromaDB
# -----------------------------------
client = chromadb.Client()

try:
    client.delete_collection("courses")
except Exception:
    pass

collection = client.create_collection(
    name="courses"
)

collection.add(
    ids=df["course_id"].astype(str).tolist(),
    documents=texts,
    embeddings=embeddings.tolist(),
    metadatas=df[
        ["title", "category", "level"]
    ].to_dict("records")
)

# -----------------------------------
# 5. FAISS search
# -----------------------------------
def search_faiss(query, top_k=5):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    start = time.perf_counter()

    scores, indices = faiss_index.search(
        query_embedding,
        top_k
    )

    end = time.perf_counter()

    results = []

    for score, index in zip(scores[0], indices[0]):

        results.append({
            "id": int(df.iloc[index]["course_id"]),
            "title": df.iloc[index]["title"],
            "category": df.iloc[index]["category"],
            "level": df.iloc[index]["level"],
            "score": float(score)
        })

    return results, (end - start) * 1000


# -----------------------------------
# 6. ChromaDB search
# -----------------------------------
def search_chroma(query, top_k=5):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0].tolist()

    start = time.perf_counter()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    end = time.perf_counter()

    output = []

    for i in range(len(results["ids"][0])):

        metadata = results["metadatas"][0][i]

        output.append({
            "id": int(results["ids"][0][i]),
            "title": metadata["title"],
            "category": metadata["category"],
            "level": metadata["level"]
        })

    return output, (end - start) * 1000


# -----------------------------------
# 7. Define benchmark queries
# -----------------------------------
queries = [
    "learn PyTorch",
    "learn deep learning",
    "machine learning for beginners",
    "natural language processing",
    "cyber security"
]

# -----------------------------------
# 8. Run benchmark
# -----------------------------------
benchmark_results = []

for query in queries:

    print("\n" + "=" * 70)
    print("QUERY:", query)
    print("=" * 70)

    # FAISS
    faiss_results, faiss_time = search_faiss(query)

    print("\nFAISS")
    print("Search Time:", round(faiss_time, 3), "ms")

    for r in faiss_results:
        print(
            r["id"],
            "|",
            r["title"],
            "|",
            r["category"],
            "|",
            round(r["score"], 4)
        )

    # ChromaDB
    chroma_results, chroma_time = search_chroma(query)

    print("\nCHROMADB")
    print("Search Time:", round(chroma_time, 3), "ms")

    for r in chroma_results:
        print(
            r["id"],
            "|",
            r["title"],
            "|",
            r["category"]
        )

    # -----------------------------------
    # Compare Top-5 IDs
    # -----------------------------------
    faiss_ids = [r["id"] for r in faiss_results]
    chroma_ids = [r["id"] for r in chroma_results]

    common = set(faiss_ids).intersection(
        set(chroma_ids)
    )

    overlap = len(common)

    print("\nTop-5 overlap:", overlap, "/ 5")

    benchmark_results.append({
        "query": query,
        "FAISS_Time_ms": round(faiss_time, 3),
        "ChromaDB_Time_ms": round(chroma_time, 3),
        "Top5_Overlap": overlap
    })


# -----------------------------------
# 9. Display benchmark summary
# -----------------------------------
results_df = pd.DataFrame(
    benchmark_results
)

print("\n\n")
print("=" * 70)
print("BENCHMARK SUMMARY")
print("=" * 70)

print(results_df.to_string(index=False))

# -----------------------------------
# 10. Calculate averages
# -----------------------------------
avg_faiss = results_df["FAISS_Time_ms"].mean()
avg_chroma = results_df["ChromaDB_Time_ms"].mean()
avg_overlap = results_df["Top5_Overlap"].mean()

print("\nAverage FAISS Search Time:",
      round(avg_faiss, 3), "ms")

print("Average ChromaDB Search Time:",
      round(avg_chroma, 3), "ms")

print("Average Top-5 Overlap:",
      round(avg_overlap, 2), "/ 5")

# -----------------------------------
# 11. Save results
# -----------------------------------
results_df.to_csv(
    "benchmark_results.csv",
    index=False
)

print("\nBenchmark results saved to:")
print("benchmark_results.csv")