import re
import json
import torch
import numpy as np

from pathlib import Path
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from transformers import AutoModel, AutoTokenizer

from utils import encode_document, clean_text

app = FastAPI(title="Document Search Service")

# Charger le modèle et le tokenizer

model_name = "sentence-transformers/all-mpnet-base-v2"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Structure de la requête

class Query(BaseModel):
    query: str

# Fonction de calcul de la similarité cosinus

def cosine_similarity(vec1, vec2):

    v1 = np.array(vec1)
    v2 = np.array(vec2)

    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

@app.post("/search_documents")
async def search_documents(data: Query):

    try:
        # Encoder la requête utilisateur
        query_embedding = encode_document(data.query)
        file_path = Path("data/document_embeddings.json")

        with file_path.open("r", encoding="utf-8") as f:
            documents = json.load(f)
        
        # Calculer la similarité cosinus
        similarities = []

        for doc in documents:

            score = cosine_similarity(query_embedding, doc["embedding"])
            similarities.append({"id": doc["id"], "text": doc["text"], "similarity_score": round(score, 2)})
        
        # Trier les documents par similarité décroissante
        similarities = sorted(similarities, key=lambda x: x["similarity_score"], reverse=True)
        
        # Retourner les résultats les plus pertinents (par exemple, les 3 premiers)
        return {"relevant_documents": similarities[:3]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
