# Notesidian

<br>
<p align="center">
  <img src="logo.png" alt="Notesidian Logo" width="400">
</p>
<br>

## Overview

Notesidian is a note management and analysis application that uses machine learning to cluster, search, and visualize relationships between notes. It is designed for researchers, students, and professionals who want to organize and analyze large volumes of information efficiently. Key features include semantic clustering, similarity search, and a knowledge graph for visualizing connections.

The application also integrates GPT-based Retrieval-Augmented Generation (RAG) for Q&A functionality using your notes.

## Features

- **📝 Note Management**

  - Create, edit, and delete notes through an intuitive interface.
  - Upload and process text or PDF files as notes.
  - Automatically generate summaries for your notes with distilbart-cnn-12-6 (optimized for gpu with FP16).

- **📊 Semantic Clustering**

  - Automatically group notes into clusters using MiniLM-L6-v2 embeddings and Affinity Propagation.
  - Visualize clusters in 2D with PCA and Plotly.

- **🔍 Note Similarity Search**

  - Search for similar notes using semantic embeddings.
  - View summaries and content of similar notes.

- **🌐 Knowledge Graph**

  - Visualize notes as an interactive graph using PyVis.
  - Manually add or remove relationships between notes to represent relationships.

- **💬 Q&A with GPT**
  - Use RAG to retrieve relevant notes and answer questions contextually.
  - View summaries of retrieved notes for each query.

## Video

A walkthrough video demonstrating how to get started with Notesidian and its key features will be added soon!

## Setup Instructions

### Clone the Repository

```bash
git clone https://github.com/benjaminfranklin03/CS-Project.git
cd CS-Project
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Set the OpenAI API Key

The application requires an OpenAI API key to use the RAG feature with GPT models. You can set it up using any of the following methods:

#### A. Set it temporarily in your shell

- **For Bash (Linux/MacOS):**

```bash
export OPENAI_API_KEY=your-api-key
```

- **For Windows Command Prompt:**

```bash
set OPENAI_API_KEY=your-api-key
```

- **For PowerShell:**

```bash
$env:OPENAI_API_KEY="your-api-key"
```

### Running the App

```bash
streamlit run app.py
```

## Contribution Matrix
