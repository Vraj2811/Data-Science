# Hierarchical Model

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
import networkx as nx
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re
import string

# Download NLTK resources if not already present
def download_nltk_resources():
    # Set NLTK data path to the model folder
    nltk_data_path = os.path.join(os.getcwd(), "model", "nltk_data")
    os.makedirs(nltk_data_path, exist_ok=True)
    nltk.data.path.append(nltk_data_path)

    # Resources to download
    resources = ["punkt", "stopwords", "wordnet", "omw-1.4"]

    # Download each resource
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
            print(f"Resource '{resource}' already downloaded")
        except LookupError:
            print(f"Downloading NLTK resource: {resource}")
            nltk.download(resource, download_dir=nltk_data_path, quiet=False)

    # Explicitly download punkt_tab if needed
    try:
        nltk.data.find('tokenizers/punkt_tab')
        print("Resource 'punkt_tab' already downloaded")
    except LookupError:
        print("Downloading NLTK resource: punkt_tab")
        nltk.download('punkt', download_dir=nltk_data_path, quiet=False)

def preprocess_text(text):
    """
    Cleans and processes the text by:
    - Converting to lowercase
    - Removing special characters and punctuation
    - Tokenizing
    - Removing stopwords
    - Lemmatizing words
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Simple tokenization by splitting on whitespace
    # This avoids using word_tokenize which might require punkt_tab
    tokens = text.split()

    # Remove stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize words
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    return " ".join(lemmatized_tokens)  # Return cleaned text

def train_and_save_model():
    """Train and save the hierarchical model"""
    # Download NLTK resources
    download_nltk_resources()

    # Load data
    df = pd.read_csv("nic_data.csv")

    # Create a directed graph
    G = nx.DiGraph()

    # Function to add a node with attributes if not already present
    def add_node_if_not_exists(graph, node_id, level, name):
        if node_id not in graph:
            graph.add_node(node_id, level=level, name=name)

    # Add nodes and edges with attributes
    for _, row in df.iterrows():
        section_id = f"Section {row['Section']}"
        section_name = row["Section Name"]

        division_id = f"Division {row['Division Code']}"
        division_name = row["Division Name"]

        group_id = f"Group {row['Group Code']}"
        group_name = row["Group Name"]

        class_id = f"Class {row['Class Code']}"
        class_name = row["Class Name"]

        subclass_id = f"Subclass {row['Sub-class Code']}"
        subclass_name = row["Sub-class Name"]

        # Add nodes with attributes
        add_node_if_not_exists(G, section_id, level=1, name=section_name)
        add_node_if_not_exists(G, division_id, level=2, name=division_name)
        add_node_if_not_exists(G, group_id, level=3, name=group_name)
        add_node_if_not_exists(G, class_id, level=4, name=class_name)
        add_node_if_not_exists(G, subclass_id, level=5, name=subclass_name)

        # Add edges
        if not G.has_edge(section_id, division_id):
            G.add_edge(section_id, division_id)
        if not G.has_edge(division_id, group_id):
            G.add_edge(division_id, group_id)
        if not G.has_edge(group_id, class_id):
            G.add_edge(group_id, class_id)
        if not G.has_edge(class_id, subclass_id):
            G.add_edge(class_id, subclass_id)

    # Preprocess node text and generate embeddings
    sbert_model = SentenceTransformer("all-MiniLM-L12-v2")

    for node in G.nodes:
        raw_text = G.nodes[node]["name"]
        cleaned_text = preprocess_text(raw_text)
        G.nodes[node]["processed_name"] = cleaned_text
        embedding = sbert_model.encode(cleaned_text, convert_to_numpy=True)
        G.nodes[node]["embedding"] = embedding

    # Create a mapping from subclass IDs to NIC codes
    subclass_to_nic = {}
    for _, row in df.iterrows():
        subclass_id = f"Subclass {row['Sub-class Code']}"
        subclass_to_nic[subclass_id] = row["Sub-class Code"]

    # Save the model components
    os.makedirs("model", exist_ok=True)

    model_components = {
        "graph": G,
        "sbert_model_name": "all-MiniLM-L12-v2",
        "subclass_to_nic": subclass_to_nic,
        "df": df
    }

    with open("model/model1.pkl", "wb") as f:
        pickle.dump(model_components, f)

    print("Model components saved successfully.")

    return G, sbert_model, subclass_to_nic, df

def load_or_train_model():
    """Load the model if it exists, otherwise train and save it"""
    if os.path.exists("model/model1.pkl"):
        print("Loading saved model components...")
        with open("model/model1.pkl", "rb") as f:
            model_components = pickle.load(f)

        G = model_components["graph"]
        sbert_model = SentenceTransformer(model_components["sbert_model_name"])
        subclass_to_nic = model_components["subclass_to_nic"]
        df = model_components["df"]

        print("Model components loaded successfully.")
    else:
        print("No saved model found. Training new model...")
        G, sbert_model, subclass_to_nic, df = train_and_save_model()

    return G, sbert_model, subclass_to_nic, df

# Load or train the model
G, sbert_model, subclass_to_nic, df = load_or_train_model()

def preprocess_query(query):
    """Preprocess the query text"""
    # Download NLTK resources if needed
    download_nltk_resources()

    # Convert to lowercase
    query = query.lower()

    # Remove special characters
    query = re.sub(r"[^a-zA-Z0-9\s]", "", query).strip()

    # Simple tokenization by splitting on whitespace
    # This avoids using word_tokenize which might require punkt_tab
    tokens = query.split()

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    return " ".join(lemmatized_tokens)

def get_top5_nic_codes(query_text, G=G, sbert_model=sbert_model, subclass_to_nic=subclass_to_nic, df=df):
    """Get top 5 NIC codes for a given query using hierarchical search"""
    # Preprocess the query
    processed_query = preprocess_query(query_text)

    # Encode the query
    query_embedding = sbert_model.encode(processed_query, convert_to_numpy=True)

    def search_at_level(parent_nodes, level_name, prev_scores, decay_factor=0.8):
        """Search within the given level of the hierarchy, applying score decay."""
        level_nodes = [node for node in parent_nodes if G.nodes[node]["level"] == level_name]

        if not level_nodes:
            return [], {}

        level_embeddings = np.array([G.nodes[node]["embedding"] for node in level_nodes])
        level_ids = np.array(level_nodes)

        # Create FAISS index for this level
        dimension = level_embeddings.shape[1]
        level_index = faiss.IndexFlatL2(dimension)
        level_index.add(level_embeddings)

        # Search in this level
        k = min(5, len(level_nodes))
        distances, indices = level_index.search(np.array([query_embedding]), k)

        new_scores = {}
        selected_nodes = []
        for i, idx in enumerate(indices[0]):
            node_id = level_ids[idx]
            similarity_score = 1 / (1 + distances[0][i])  # Convert L2 distance to similarity

            # Apply decay factor
            parent_node = next((p for p in G.predecessors(node_id)), None)
            prev_score = prev_scores.get(parent_node, 1)  # Default to 1 if root level
            new_scores[node_id] = prev_score * similarity_score * decay_factor

            selected_nodes.append(node_id)

        return selected_nodes, new_scores

    # Start hierarchical search
    results = []

    # Search in Sections (Root level)
    section_nodes = [node for node in G.nodes if G.nodes[node]["level"] == 1]
    top_sections, section_scores = search_at_level(section_nodes, 1, {})

    for section in top_sections:
        # Search in Divisions within this Section
        division_nodes = list(G.successors(section))
        top_divisions, division_scores = search_at_level(division_nodes, 2, section_scores)

        for division in top_divisions:
            # Search in Groups within this Division
            group_nodes = list(G.successors(division))
            top_groups, group_scores = search_at_level(group_nodes, 3, division_scores)

            for group in top_groups:
                # Search in Classes within this Group
                class_nodes = list(G.successors(group))
                top_classes, class_scores = search_at_level(class_nodes, 4, group_scores)

                for class_node in top_classes:
                    # Search in Subclasses within this Class
                    subclass_nodes = list(G.successors(class_node))
                    top_subclasses, subclass_scores = search_at_level(subclass_nodes, 5, class_scores)

                    for node in top_subclasses:
                        if node in subclass_to_nic:
                            nic_code = subclass_to_nic[node]
                            results.append((nic_code, subclass_scores[node]))

    # Sort results by final score in descending order
    results.sort(key=lambda x: x[1], reverse=True)
    results = results[:5]

    # Convert to DataFrame format similar to model2.py
    if not results:
        return pd.DataFrame(columns=["Sub-class Code", "Sub-class Name", "confidence"])

    # Get the NIC codes from results
    nic_codes = [code for code, _ in results]

    # Get the corresponding rows from the original dataframe
    result_df = df[df["Sub-class Code"].isin(nic_codes)].copy()

    # Add confidence scores and ensure they're between 0 and 1
    confidence_map = {code: min(max(score, 0), 1) for code, score in results}
    result_df["confidence"] = result_df["Sub-class Code"].map(confidence_map)

    # Sort by confidence
    result_df = result_df.sort_values("confidence", ascending=False)

    return result_df[["Sub-class Code", "Sub-class Name", "confidence"]]