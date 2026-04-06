# AI Data Analyst RAG

An AI-powered document question-answering system that allows users to upload PDF files and interact with them using a Retrieval-Augmented Generation (RAG) pipeline.

## Features

- Upload PDF documents
- Extract and chunk text from PDFs
- Generate semantic embeddings using Sentence Transformers
- Store and retrieve relevant chunks using FAISS
- Ask natural language questions about the uploaded document
- Generate structured answers using a lightweight Hugging Face LLM
- Streamlit-based chat-style interface

## Tech Stack

- Python
- Streamlit
- FAISS
- Sentence Transformers
- Hugging Face Transformers
- PyPDF2

## Project Workflow

1. Upload a PDF document
2. Extract text from the PDF
3. Split the text into smaller chunks
4. Convert chunks into embeddings
5. Store embeddings in a FAISS index
6. Convert the user query into an embedding
7. Retrieve the most relevant chunks
8. Pass retrieved context to the LLM
9. Generate a structured response

## Files

- `app.py` → Streamlit frontend and app workflow
- `rag.py` → PDF extraction, chunking, embeddings, retrieval
- `llm.py` → Hugging Face model loading and response generation

## Installation

```bash
git clone https://github.com/your-username/ai-data-analyst-rag.git
cd ai-data-analyst-rag
python -m venv myenv
myenv\Scripts\activate
pip install -r requirements.txt
