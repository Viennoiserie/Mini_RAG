import json
import requests

SEARCH_URL = "http://127.0.0.1:8002/search_documents"
ENCODE_URL = "http://127.0.0.1:8001/encode_documents"

def encode_documents():

    url = ENCODE_URL
    response = requests.post(url)

    if response.status_code == 200:
        print("\nDocuments encodés avec succès et stockés.")

    else:
        print("Erreur:", response.json())

def search_documents():

    url = SEARCH_URL
    query = input("\nEntrez la requête de recherche: ")

    payload = {"query": query}
    response = requests.post(url, json=payload)

    if response.status_code == 200:

        results = response.json()["relevant_documents"]
        print("\nDocuments les plus pertinents:\n")

        for result in results:

            print(f"ID: {result['id']}")
            print(f"Texte: {result['text']}")
            print(f"\nScore de similarité: {result['similarity_score']}")

    else:
        print("Erreur:", response.json())

def main():

    choice = True

    while(choice):

        print("\nChoisissez une option:\n")
        print("1: Encoder les documents dans mixed_samples.txt")
        print("2: Rechercher des documents par une requête")
        print("3: Quitter")
        
        choice = input("\nEntrez votre choix (1 ou 2 ou 3): ")

        if choice == '1':
            encode_documents()

        elif choice == '2':
            search_documents()
            
        elif choice == '3':
            choice = False

        else:
            print("\nChoix invalide\n")

if __name__ == "__main__":
    main()
