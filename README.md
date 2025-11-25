# NIC Code Search Web Application

This is a Flask web application that allows you to search for NIC (National Industrial Classification) codes using different machine learning models. It leverages natural language processing and semantic search to find the most relevant industry codes for a given query.

## Overview

The application provides an interface to search for NIC codes using natural language queries. It supports multiple models, each with different approaches to finding the best match, ranging from hierarchical graph traversal to advanced sentence embeddings and AI-enhanced query processing.

## Key Features

- **Multiple Search Models**: Choose from 4 distinct models tailored for different search needs.
- **Natural Language Support**: Enter queries in plain English or mixed languages (Hinglish).
- **Real-time Performance**: Fast search results using FAISS indexing.
- **Confidence Scores**: View top 5 matching NIC codes with associated confidence levels.
- **Query Enhancement**: Advanced model uses AI to refine user queries for better accuracy.

## Technology Stack

- **Backend**: Flask (Python)
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Sentence Transformers (all-MiniLM-L12-v2), PyTorch
- **Search Indexing**: FAISS (Facebook AI Similarity Search)
- **NLP**: NLTK (Natural Language Toolkit)
- **AI Integration**: OpenAI API (for query enhancement)

## Getting Started

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. Clone the repository or navigate to the project directory.
2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. Start the Flask server:

   ```bash
   python app.py
   ```

2. Open your web browser and navigate to:

   ```
   http://localhost:5002
   ```

## Usage

1. **Select a Model**: Choose one of the 4 available models from the dropdown menu.
2. **Enter Query**: Type your business description or industry query in the search box.
3. **Search**: Click the "Search" button.
4. **View Results**: The application will display the top 5 matching NIC codes along with their descriptions and confidence scores.

## Models

The application supports 4 different models:

1. **Model 1: Hierarchical Search Model**
   - Uses a graph-based approach to traverse the NIC hierarchy (Section -> Division -> Group -> Class -> Subclass).
   - Applies score decay at each level to prioritize matches that are consistent across the hierarchy.

2. **Model 2: Multilingual Sentence Transformer**
   - Uses `all-MiniLM-L12-v2` to generate embeddings for NIC descriptions.
   - Performs semantic search using FAISS.
   - Trained on the basic NIC dataset.

3. **Model 3: Sentence Transformer with Enhanced Data**
   - Similar to Model 2 but trained on an enhanced dataset (`nic_data_descriptions_full.csv`).
   - Includes richer descriptions for better semantic matching.

4. **Model 4: Sentence Transformer with AI Query Enhancement**
   - Builds upon Model 3.
   - Uses an LLM (via OpenAI API) to pre-process and "enhance" the user's query into standard industry terminology before searching.
   - Ideal for vague or non-standard queries.

## Evaluation Dataset

The project includes an evaluation dataset for testing model performance:
- `nic_paraphrased_openai.txt`: Contains semantic queries generated/paraphrased by AI.
- `nic_subclass_codes.txt`: Contains exact queries corresponding to subclass codes.
- `Vague_Queries.txt`: Contains vague or ambiguous queries to test robustness.
