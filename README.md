# GenAI PDF Study Assistant

GenAI PDF Study Assistant is an AI-powered Streamlit web application that helps users study from PDF documents. Users can upload a PDF, generate structured study notes, and ask questions based on the uploaded document using a Retrieval-Augmented Generation (RAG) pipeline.

## Features

- Upload PDF documents
- Extract text from uploaded PDFs
- Generate structured notes from PDF content
- Ask questions based on the uploaded PDF
- Retrieve relevant chunks using vector search
- Generate answers using an LLM through the Groq API
- Simple and interactive Streamlit interface

## Tech Stack

- Python
- Streamlit
- PyPDF
- Sentence Transformers
- FAISS
- Groq API
- NumPy
- python-dotenv

## Project Structure

```text
SMART-STUDY-ASSISTANT/
│
├── .streamlit/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .env