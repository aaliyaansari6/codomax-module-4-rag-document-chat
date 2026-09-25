import streamlit as st
from google import genai
import chromadb
from pypdf import PdfReader

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Chat With Your Documents",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Chat With Your Documents")
st.caption("Codomax Internship – Module 4: RAG, Embeddings & Vector Databases")

# -----------------------------
# Gemini Client
# -----------------------------
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# -----------------------------
# ChromaDB
# -----------------------------
chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_or_create_collection(
    name="documents"
)

# -----------------------------
# Helper Functions
# -----------------------------
def extract_text(uploaded_file):
    """Extract text from PDF or TXT files."""

    if uploaded_file.name.lower().endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        return text

    elif uploaded_file.name.lower().endswith(".txt"):
        return uploaded_file.read().decode("utf-8")

    return ""


def chunk_text(text, chunk_size=1000, overlap=200):
    """Split document text into overlapping chunks."""

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def create_embeddings(texts):
    """Generate embeddings using Gemini."""

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts
    )

    return [embedding.values for embedding in response.embeddings]


def add_documents(files):
    """Process documents and store their embeddings."""

    all_chunks = []
    all_ids = []

    for file_index, uploaded_file in enumerate(files):

        text = extract_text(uploaded_file)

        if not text.strip():
            continue

        chunks = chunk_text(text)

        for chunk_index, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(
                f"{uploaded_file.name}_{file_index}_{chunk_index}"
            )

    if not all_chunks:
        return 0

    embeddings = create_embeddings(all_chunks)

    collection.add(
        documents=all_chunks,
        embeddings=embeddings,
        ids=all_ids
    )

    return len(all_chunks)


def retrieve_context(question, top_k=3):
    """Retrieve the most relevant document chunks."""

    query_embedding = create_embeddings([question])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    if not results["documents"]:
        return []

    return results["documents"][0]


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("📄 Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if st.sidebar.button("➕ Process Documents"):

    if not uploaded_files:
        st.sidebar.warning("Please upload a document first.")

    else:
        with st.spinner("Processing documents..."):

            try:
                count = add_documents(uploaded_files)

                st.sidebar.success(
                    f"Processed {count} document chunks successfully!"
                )

            except Exception as e:
                st.sidebar.error(f"Error: {e}")


if st.sidebar.button("🗑️ Clear Database"):

    try:
        chroma_client.delete_collection("documents")

        collection = chroma_client.get_or_create_collection(
            name="documents"
        )

        st.sidebar.success("Document database cleared.")

    except Exception as e:
        st.sidebar.error(f"Error: {e}")


# -----------------------------
# Main Chat
# -----------------------------
st.subheader("💬 Ask Questions About Your Documents")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


question = st.chat_input(
    "Ask something about your uploaded documents..."
)


if question:

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    try:

        with st.spinner("Searching your documents..."):

            context_chunks = retrieve_context(question)

        if not context_chunks:

            answer = (
                "I couldn't find any relevant information. "
                "Please upload and process a document first."
            )

        else:

            context = "\n\n---\n\n".join(context_chunks)

            prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the provided document context.

If the answer cannot be found in the context, clearly say:
"I could not find this information in the uploaded documents."

Do not invent information.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}
"""

            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt
            )

            answer = response.text

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

    except Exception as e:

        st.error(f"Error while processing your question: {e}")
