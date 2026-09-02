import streamlit as st
import pandas as pd
import faiss
import chromadb
from sentence_transformers import SentenceTransformer

# ----------------------------------------
# Page Configuration
# ----------------------------------------
st.set_page_config(
    page_title="AI Course Semantic Search",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 AI Course Semantic Search")
st.markdown("Search AI courses using **FAISS** or **ChromaDB** semantic search.")

# ----------------------------------------
# Load Dataset
# ----------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("dataset/courses.csv")

df = load_data()

# ----------------------------------------
# Load Sentence Transformer Model
# ----------------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ----------------------------------------
# Prepare Text Data
# ----------------------------------------
texts = (
    df["title"].astype(str)
    + " "
    + df["description"].astype(str)
    + " "
    + df["category"].astype(str)
    + " "
    + df["level"].astype(str)
).tolist()

# ----------------------------------------
# Create Embeddings
# ----------------------------------------
@st.cache_resource
def create_embeddings():
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings

embeddings = create_embeddings()

# ----------------------------------------
# Create FAISS Index
# ----------------------------------------
@st.cache_resource
def create_faiss():
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype("float32"))
    return index

faiss_index = create_faiss()

# ----------------------------------------
# Create ChromaDB Collection
# ----------------------------------------
@st.cache_resource
def create_chroma():

    client = chromadb.Client()

    try:
        client.delete_collection("courses")
    except:
        pass

    collection = client.create_collection("courses")

    collection.add(
        ids=df["course_id"].astype(str).tolist(),
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=df[
            ["title", "category", "level"]
        ].to_dict("records")
    )

    return collection

collection = create_chroma()

# ----------------------------------------
# Sidebar
# ----------------------------------------
st.sidebar.header("Search Settings")

engine = st.sidebar.radio(
    "Search Engine",
    ["FAISS", "ChromaDB"]
)

top_k = st.sidebar.slider(
    "Top Results",
    min_value=1,
    max_value=10,
    value=5
)

# ----------------------------------------
# Search Box
# ----------------------------------------
query = st.text_input(
    "Enter your query",
    placeholder="Example: learn PyTorch"
)

# ----------------------------------------
# Search
# ----------------------------------------
if st.button("Search"):

    if query.strip() == "":
        st.warning("Please enter a search query.")
        st.stop()

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    if engine == "FAISS":

        scores, indices = faiss_index.search(
            query_embedding.astype("float32"),
            top_k
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):

            row = df.iloc[idx]

            results.append({
                "Course ID": row["course_id"],
                "Title": row["title"],
                "Category": row["category"],
                "Level": row["level"],
                "Similarity Score": round(float(score), 4)
            })

    else:

        chroma_results = collection.query(
            query_embeddings=[
                query_embedding[0].tolist()
            ],
            n_results=top_k
        )

        results = []

        for i in range(len(chroma_results["ids"][0])):

            meta = chroma_results["metadatas"][0][i]

            results.append({
                "Course ID": chroma_results["ids"][0][i],
                "Title": meta["title"],
                "Category": meta["category"],
                "Level": meta["level"]
            })

    st.success(f"Showing Top {top_k} results using {engine}")

    st.dataframe(
        pd.DataFrame(results),
        use_container_width=True,
        hide_index=True
    )

# ----------------------------------------
# Dataset Information
# ----------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Info")
st.sidebar.write(f"**Total Courses:** {len(df)}")
st.sidebar.write("**Embedding Model:** all-MiniLM-L6-v2")
st.sidebar.write(f"**Vector Dimension:** {embeddings.shape[1]}")

# ----------------------------------------
# Footer
# ----------------------------------------
st.markdown("---")
st.caption(
    "AI Course Recommendation System using Sentence Transformers, FAISS, and ChromaDB"
)