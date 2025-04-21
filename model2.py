# Multilinugual Embedding Model

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os

def train_and_save_model():
    # Load and preprocess data
    df = pd.read_csv("nic_data.csv")

    # Enhance the combined description with better structure and weighting
    df["combined_description"] = (
    "Section " + ": " + df["Section"] +  "; " +
    "Division " + df["Division Code"] + ": " + df["Division Name"] + "; " +
    "Group " + df["Group Code"] + ": " + df["Group Name"] + "; " +
    "Class " + df["Class Code"] + ": " + df["Class Name"] + "; " +
    "Sub-class " + df["Sub-class Code"] + ": " + df["Sub-class Name"]
    )

    # Load multilingual model
    model = SentenceTransformer("all-MiniLM-L12-v2")

    # Generate embeddings for all NIC descriptions
    nic_embeddings = model.encode(df["combined_description"].tolist(), show_progress_bar=True)

    # Build FAISS index
    dimension = nic_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(nic_embeddings)

    # Save the model components with the name 'model/model2'
    faiss.write_index(index, "model/model2.faiss")
    df.to_pickle("model/model2_dataframe.pkl")

    # We don't need to save the SentenceTransformer model as it can be loaded by name
    # But we'll save the model name for reference
    with open("model/model2_info.pkl", "wb") as f:
        pickle.dump({"model_name": "all-MiniLM-L12-v2"}, f)

    print("Model, index, and dataframe saved successfully.")

    return model, index, df

# Check if saved model exists, otherwise train and save
def load_or_train_model():
    if os.path.exists("model/model2.faiss") and os.path.exists("model/model2_dataframe.pkl"):
        print("Loading saved model components...")
        index = faiss.read_index("model/model2.faiss")
        df = pd.read_pickle("model/model2_dataframe.pkl")
        with open("model/model2_info.pkl", "rb") as f:
            model_info = pickle.load(f)
        model = SentenceTransformer(model_info["model_name"])
        print("Model components loaded successfully.")
    else:
        print("No saved model found. Training new model...")
        model, index, df = train_and_save_model()

    return model, index, df

# Load or train the model
model, index, df = load_or_train_model()

def get_top5_nic_codes(query_text, model=model, index=index, df=df):
    # Encode query (works for any language)
    query_embedding = model.encode([query_text])[0]

    # FAISS search
    distances, indices = index.search(np.array([query_embedding]), 5)

    # Format results
    results = df.iloc[indices[0]].copy()
    # Calculate confidence scores and ensure they're between 0 and 1
    confidence_scores = (2 - distances[0])/2
    confidence_scores = np.clip(confidence_scores, 0, 1)
    results["confidence"] = confidence_scores

    return results[["Sub-class Code", "Sub-class Name", "confidence"]]
