# Document-Retrieval-System-using-RAG-
An AI-powered RAG chatbot that lets you upload one or more PDFs and ask natural language questions. It splits documents into chunks, stores them in a vector database, retrieves relevant sections, and uses GPT-OSS-120B to generate clear, context-based answers. Built with Streamlit, LangChain, and Hugging Face.


🤖 AI Chat with Your Documents (RAG Assistant)

This project is an AI-powered chatbot that lets you upload one or more PDF documents and ask natural language questions about their content.
It uses Retrieval-Augmented Generation (RAG) to combine the power of vector search with a large language model (LLM). Instead of making up answers, the chatbot retrieves the most relevant sections from your PDFs and generates clear, structured, and context-aware responses.

🔹 Key Features

📄 Upload single or multiple PDFs at once.

🧩 Splits documents into chunks and stores them in a Chroma vector database.

🔍 Retrieves the most relevant sections using semantic search.

🤖 Uses Mistral-7B-Instruct v0.3 (via Hugging Face Transformers) to generate detailed answers.

💬 Provides a chat-like interface built with Streamlit.

✅ Answers are grounded in your uploaded documents — if something isn’t in the file, the assistant clearly says “I don’t know based on the provided document.”

🔧 Tech Stack

Python 🐍

Streamlit → interactive UI

LangChain → RAG pipeline and document handling

ChromaDB → vector database

Hugging Face Transformers → embeddings + LLM

PyTorch / Accelerate → model loading with CPU/GPU offloading

🚀 Use Cases

Quickly extracting answers from long research papers.

Chatting with contracts, policies, or legal docs.

Summarizing reports and technical manuals.

Making knowledge locked in PDFs instantly accessible.