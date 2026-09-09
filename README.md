# HR Policy Assistant

An AI-powered HR Policy Assistant built using Retrieval-Augmented Generation (RAG).

The application allows users to upload an HR Policy PDF and ask questions about its content. The system extracts text from the PDF, divides the text into smaller chunks, converts the chunks into embeddings, stores them in FAISS, retrieves the most relevant information for the user's question, and uses Groq to generate an answer based only on the retrieved HR policy content.

## Features

- Upload HR Policy PDF
- Extract text from PDF
- Divide document into chunks
- Generate text embeddings
- Store embeddings in FAISS
- Retrieve relevant policy information
- Ask questions about the HR policy
- Generate answers using Groq
- Display relevant pages and similarity scores
- Simple Streamlit interface

## Technologies

- Python
- Streamlit
- RAG
- FAISS
- Sentence Transformers
- PyMuPDF
- Groq
- NumPy

## How It Works

PDF Upload
↓
Text Extraction
↓
Text Chunking
↓
Embeddings
↓
FAISS Vector Database
↓
User Question
↓
Question Embedding
↓
Similarity Search
↓
Relevant HR Policy Chunks
↓
Groq
↓
Final Answer

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt