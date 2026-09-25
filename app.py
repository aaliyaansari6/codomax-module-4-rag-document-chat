import streamlit as st
from google import genai
import chromadb
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(
    page_title="Chat With Your Documents",
    page_icon="📚"
)

st.title("📚 Chat With Your Documents")
st.caption("Codomax Internship – Module 4: RAG, Embeddings & Vector Databases")

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None


def extract_text(file):
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join(
            page.extract_text() or "" for page in reader.pages
        )

    return file.read().decode("utf-8")


def chunk_text(text, size=1000, overlap=200):
    chunks = []
    start = 0

    while start < len(text):
        chunk = text[start:start + size].strip()

        if chunk:
            chunks.append(chunk)

        start += size - overlap

    return chunks


def process_documents(files):
    all_chunks = []

    for file in files:
        text = extract_text(file)

        if text.strip():
            all_chunks.extend(chunk_text(text))

    if not all_chunks:
        return 0

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(all_chunks).toarray()

    collection.delete(
        ids=collection.get()["ids"]
    ) if collection.count() > 0 else None

    collection.add(
        documents=all_chunks,
        embeddings=vectors.tolist(),
        ids=[f"chunk_{i}" for i in range(len(all_chunks))]
    )

    st.session_state.chunks = all_chunks
    st.session_state.vectorizer = vectorizer

    return len(all_chunks)


def retrieve(question, top_k=3):
    vectorizer = st.session_state.vectorizer

    if vectorizer is None:
        return []

    question_vector = vectorizer.transform([question]).toarray()

    results = collection.query(
        query_embeddings=question_vector.tolist(),
        n_results=min(top_k, collection.count())
    )

    return results["documents"][0]


st.sidebar.header("📄 Upload Documents")

files = st.sidebar.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if st.sidebar.button("➕ Process Documents"):

    if not files:
        st.sidebar.warning("Upload a document first.")

    else:
        with st.spinner("Processing document..."):

            try:
                count = process_documents(files)

                st.sidebar.success(
                    f"Processed {count} document chunks!"
                )

            except Exception as e:
                st.sidebar.error(f"Error: {e}")


if st.sidebar.button("🗑️ Clear Database"):

    try:
        ids = collection.get()["ids"]

        if ids:
            collection.delete(ids=ids)

        st.session_state.chunks = []
        st.session_state.vectorizer = None

        st.sidebar.success("Database cleared.")

    except Exception as e:
        st.sidebar.error(f"Error: {e}")


st.subheader("💬 Ask Questions About Your Documents")

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


question = st.chat_input(
    "Ask something about your uploaded documents..."
)


if question:

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    try:

        if collection.count() == 0:
            answer = "Please upload and process a document first."

        else:

            context_chunks = retrieve(question)

            context = "\n\n---\n\n".join(context_chunks)

            prompt = f"""
You are a document question-answering assistant.

Use ONLY the provided document context to answer the question.

If the answer is not present in the context, say:
"I could not find this information in the uploaded documents."

Do not invent information.

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}
"""

            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt
            )

            answer = response.text

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

    except Exception as e:
        st.error(f"Error: {e}")
