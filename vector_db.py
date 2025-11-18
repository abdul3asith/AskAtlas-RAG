import os

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
chroma_client = chromadb.Client()

collection = chroma_client.create_collection(name="first_file")

files = [
    "Python is a high-level programming language.",
    "Machine learning is a subset of artificial intelligence.",
    "Neural networks are inspired by the human brain.",
    "Deep learning uses multiple layers of neural networks.",
    "Natural language processing helps computers understand text."
] 

# store with ids
for i, file in enumerate(files):
    collection.add(
        documents=[file],
        ids=[f"file_{i}"]
    )

print(f"Stored {len(files)} documents")



query = "How neural networks work?"
results = collection.query(
    query_texts=[query],
    n_results=2  
)

print(f"Query is - {query}")
print("Top Matches:")
for file in results['documents'][0]:
    print(file)


# Output

'''
Stored 5 documents
Query is - How neural networks work?
Top Matches:
Neural networks are inspired by the human brain.
Deep learning uses multiple layers of neural networks.
'''