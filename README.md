# Serbian Educational RAG System

A Retrieval-Augmented Generation (RAG) system for answering questions from Serbian 4th grade textbooks. The system processes multiple PDF sources and uses Serbian-specialized embeddings with Llama 3.2 to provide accurate answers.

## Features

- **Multi-PDF Processing**: Handles multiple textbooks simultaneously
- **Sentence-Level Chunking**: Creates precise 2-3 sentence chunks for accurate retrieval
- **Serbian-Specialized Embeddings**: Uses `djovak/embedic-large` model fine-tuned for Serbian language
- **Hybrid Search**: Combines semantic similarity with keyword matching for better results
- **Llama 3.2 Integration**: Uses Meta's Llama 3.2 1B Instruct for answer generation

