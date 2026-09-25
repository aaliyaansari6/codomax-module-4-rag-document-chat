# 📚 Chat With Your Documents

A Retrieval-Augmented Generation (RAG) based document question-answering application developed for the Codomax Internship – Module 4.

## Project Overview

This application allows users to upload PDF or TXT documents and ask questions about their content.

The system follows the RAG pipeline:

Document → Chunking → Embeddings → Vector Database → Retrieval → LLM → Answer

## Features

- 📄 Upload PDF and TXT documents
- ✂️ Document text chunking
- 🧠 Gemini embeddings
- 🗄️ ChromaDB vector database
- 🔎 Semantic similarity search
- 🤖 Gemini-powered answers
- 💬 Interactive chat interface
- 🗑️ Clear document database
- 🔐 Secure API key management

## Technologies Used

- Python
- Streamlit
- Google Gemini API
- Gemini Embeddings
- ChromaDB
- PyPDF
- GitHub

## RAG Pipeline

1. User uploads a document.
2. Text is extracted from the document.
3. The text is divided into smaller chunks.
4. Embeddings are generated for each chunk.
5. Embeddings are stored in ChromaDB.
6. A user asks a question.
7. The question is converted into an embedding.
8. Relevant document chunks are retrieved.
9. The retrieved context is sent to Gemini.
10. Gemini generates an answer based on the retrieved context.

## API Key Security

The Gemini API key is stored using Streamlit Secrets and is not included in the GitHub repository.

Example:

```toml
GEMINI_API_KEY = "your_api_key_here"
