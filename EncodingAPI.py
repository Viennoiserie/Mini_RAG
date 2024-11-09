import re
import json
import uuid
import torch
import string

from pathlib import Path
from fastapi import FastAPI, HTTPException
from transformers import AutoModel, AutoTokenizer

from utils import encode_document, clean_text

MAX_LENGTH = 1500

app = FastAPI(title="Document Encoding Service")

# Charger le modèle et le tokenizer

model_name = "sentence-transformers/all-mpnet-base-v2"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Fonction de troncage
 
def chunk_document(text, max_char_length):

    chunks = []

    while len(text) > max_char_length:
        
        cut_point = text.rfind(' ', 0, max_char_length)

        if cut_point == -1:
            cut_point = max_char_length  

        chunks.append(text[:cut_point])
        text = text[cut_point:].strip()  

    if text:
        chunks.append(text)  

    return chunks

@app.post("/encode_documents")
async def encode_documents():

    try:
        file_path = Path("data/mixed_samples.txt")
        output_path = Path("data/document_embeddings.json")

        encoded_data = []
        
        # Charger les documents du fichier texte
        with file_path.open("r", encoding="utf-8") as file:
            documents = [line.strip() for line in file if line.strip()]
        
        for doc in documents:

            # Découper le document en chunks de 1500 caractères
            chunks = chunk_document(doc, max_char_length=MAX_LENGTH)

            for i, chunk in enumerate(chunks):

                # Création d'un ID
                chunk_id = f"{uuid.uuid4()}_chunk_{i}"

                # Encoder le chunk
                embedding = encode_document(chunk)

                # Ajouter les résultats dans la liste
                encoded_data.append({"id": chunk_id, "text": chunk, "embedding": embedding})
            
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(encoded_data, f, indent=4)
        
        return {"status": "success", "message": "Documents encoded and stored successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
