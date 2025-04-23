# QA Wiki

A knowledge base system powered by embeddings and retrieval-augmented generation to answer questions from your documents in jira.

## Overview

QA Wiki creates a searchable knowledge base from your documents from jira. It uses ChromaDB for vector search and modern language models to provide accurate answers to your questions.

## Features

- **Document Embedding**: Convert documents into vector representations for semantic search
- **Semantic Search**: Find relevant content based on meaning, not just keywords
- **RAG-based Question Answering**: Combine retrieval with generation for accurate answers
- **Google Cloud Integration**: Seamless deployment on Google Cloud infrastructure

## Project Structure

- `chromadb_function/`: ChromaDB operations (embedding, backing up, fetching, updating)
- `model_function/`: Model operations (embedding, prompting, retrieval, utilities)
- `gce_instance_setup/`: Scripts for Google Compute Engine instance setup
