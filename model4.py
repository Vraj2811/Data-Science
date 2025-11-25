# Model with API for Query Enhancement and Enhanced databse

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)

def train_and_save_model():
    # Load and preprocess data
    df = pd.read_csv("nic_data_descriptions_full.csv")

    df["combined_description"] = (
    "Section " + ": " + df["Section"] +  "; " +
    "Division " + df["Division Code"] + ": " + df["Division Name"] + "; " +
    "Group " + df["Group Code"] + ": " + df["Group Name"] + "; " +
    "Class " + df["Class Code"] + ": " + df["Class Name"] + "; " +
    "Sub-class " + df["Sub-class Code"] + ": " + df["Sub-class Name"] + "; " +
    "Description " + df["Description"]
    )

    # Load multilingual model
    model = SentenceTransformer("all-MiniLM-L12-v2")

    # Generate embeddings for all NIC descriptions
    nic_embeddings = model.encode(df["combined_description"].tolist(), show_progress_bar=True)

    # Build FAISS index
    dimension = nic_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(nic_embeddings)

    # Save the model components with the name 'model3'
    os.makedirs("model", exist_ok=True)  # Create model directory if it doesn't exist
    faiss.write_index(index, "model/model3.faiss")
    df.to_pickle("model/model3_dataframe.pkl")

    # We don't need to save the SentenceTransformer model as it can be loaded by name
    # But we'll save the model name for reference
    with open("model/model3_info.pkl", "wb") as f:
        pickle.dump({"model_name": "all-MiniLM-L12-v2"}, f)

    print("Model, index, and dataframe saved successfully.")

    return model, index, df

# Check if saved model exists, otherwise train and save
def load_or_train_model():
    if os.path.exists("model/model3.faiss") and os.path.exists("model/model3_dataframe.pkl"):
        print("Loading saved model components...")
        index = faiss.read_index("model/model3.faiss")
        df = pd.read_pickle("model/model3_dataframe.pkl")
        with open("model/model3_info.pkl", "rb") as f:
            model_info = pickle.load(f)
        model = SentenceTransformer(model_info["model_name"])
        print("Model components loaded successfully.")
    else:
        print("No saved model found. Training new model...")
        model, index, df = train_and_save_model()

    return model, index, df

# Load or train the model
model, index, df = load_or_train_model()

def enhance_query_with_openai(query_text):
    """Use OpenAI API to enhance the query by extracting key industry/business terms"""
    try:

        completion = client.chat.completions.create(
        # model="mistral/ministral-8b",
        model="openai/gpt-4.1-nano",
        messages=[
            {
            "role": "system",
            "content": "You are a specialized assistant that identifies the precise industry category from queries for NIC (National Industrial Classification) code matching. Convert queries in any language (including Hinglish/Hindi-English mix) to industry classifications using standard industry terminology. Your response should be at least 5 words and include relevant industry terms like 'production', 'cultivation', 'manufacturing', 'industry', 'processing', etc. as appropriate to the query. Use terminology that would appear in official classification systems like NIC codes. Do not include phrases like 'I want to' or 'I am interested in'.You are an expert assistant for mapping user queries to industry categories as per the NIC (National Industrial Classification). Convert queries in any language (including Hinglish or mixed language) into a clear, concise, one-line industry classification using formal industry terms like 'manufacturing', 'production, 'cultivation, 'processing', etc. Avoid explanations. Do not mention the word 'processing' or restate the user's input. Return the most relevant category in 5-15 words using standard classification terminology. Give atleast 5 words"
            },
            {
            "role": "user",
            "content": f"Identify the precise industry category from this query: '{query_text}'"
            }
        ]
        )

        enhanced_query = completion.choices[0].message.content.strip()
        # Remove quotes if present
        enhanced_query = enhanced_query.strip('"\'')
        print(f"Original query: '{query_text}'")
        print(f"Enhanced query: '{enhanced_query}'")
        return enhanced_query

    except Exception as e:
        print(f"Error using API: {e}")
        print("Using original query instead.")
        return query_text

def get_top5_nic_codes(query_text, model=model, index=index, df=df, enhance_query=True):
    # Enhance the query using OpenAI if requested
    query_text = enhance_query_with_openai(query_text)

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

