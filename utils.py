import re
import json
import uuid
import torch
import string

from pathlib import Path
from transformers import AutoModel, AutoTokenizer

model_name = "sentence-transformers/all-mpnet-base-v2"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Fonction de nettoyage

def clean_text(text):

    # 1. Conversion en minuscules
    text = text.lower()

    # 2. Suppression des caractères spéciaux et ponctuation
    text = re.sub(r'[^\w\s]', '', text)  

    # 3. Suppression des espaces en trop
    text = re.sub(r'\s+', ' ', text).strip() 

    return text

# Fonction d'encodage

def encode_document(text):

    # cleaned_text = clean_text(text)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)

    return embeddings.squeeze().tolist()
