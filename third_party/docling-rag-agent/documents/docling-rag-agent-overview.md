# Docling RAG Agent Overview

The Docling RAG Agent is an interactive CLI assistant that performs retrieval-augmented generation (RAG) over an organization’s knowledge base. It ingests documents in multiple formats, creates semantic chunks, and stores vector embeddings so that questions can be answered with grounded, cited context.

## Core Capabilities

- Conversational command-line interface with streaming responses.
- Semantic search across PDFs, Office documents, Markdown, and Whisper-transcribed audio files.
- Automatic source citation that returns the full document path for every supporting chunk.
- Conversation history that allows for multi-turn follow-up questions.

## Processing Pipeline

1. **Ingestion** – Docling converts supported documents to Markdown, while Whisper handles MP3 transcription.
2. **Chunking** – Content is split into 500–1,000 token semantic chunks using Docling’s HybridChunker.
3. **Embedding Generation** – Each chunk is embedded with `jinaai/jina-embeddings-v2-base-en`.
4. **Vector Storage** – Embeddings and metadata are stored in ChromaDB for similarity search.
5. **Retrieval & Generation** – At query time, the agent retrieves the most relevant chunks and feeds them to the local Ollama LLM (`gemma3:1b`) to produce a cited response.

These components allow the agent to answer questions using only verified knowledge from the document collection while showing exactly where the information came from.

