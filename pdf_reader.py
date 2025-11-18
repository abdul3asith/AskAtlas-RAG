import os

# from pathlib import Path
import chromadb
import PyPDF2
from dotenv import load_dotenv
from openai import OpenAI


def read_pdf(file_path):
    file_path = "data/sample_pdf_2.pdf"
    text = ""

    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            text += f"\n--- Page {page_num + 1} ---\n{page_text}"
    
    return text

def chunk_text(text, chunk_size=1000):
    """Simple chunking"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks

def store_pdf_in_db(pdf_path):
    """Read PDF and store in ChromaDB"""
    
    # Read PDF
    print(f" Reading {pdf_path}...")
    text = read_pdf(pdf_path)
    
    # Chunk it
    chunks = chunk_text(text, chunk_size=200)
    print(f" Created {len(chunks)} chunks")
    
    # Store in ChromaDB
    client = chromadb.Client()
    collection = client.create_collection(name="pdf_docs")
    
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            metadatas=[{"source": pdf_path, "chunk": i}],
            ids=[f"chunk_{i}"]
        )
    
    print("Stored in database")
    return collection


collection = store_pdf_in_db("sample_pdf_2.pdf")

# search
query = "What are Large Language Models?"
results = collection.query(
    query_texts=[query],
    n_results=2
)
print(f"Query - {query}")
print(results['documents'])

def rag_query(question, collection, llm_client):
    """
    RAG pipeline:
    1. Search for relevant chunks
    2. Send to LLM with context
    3. Get answer
    """

    # Step 1: Retrieve relevant documents
    results = collection.query(
        query_texts=[question],
        n_results=3
    )

    context = "\n\n".join(results['documents'][0])

    # Step 2: Generate answer with context
    prompt = f"""
    Answer the question based ONLY on the following context.
    If the answer is not in the context, respond with:
    "I don't have that information."

    Context:
    {context}

    Question: {question}

    Answer:
    """

    # Using OpenAI
    response = llm_client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[
            {"role": "system", "content": "You are a helpful RAG question-answering assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.0
    )

    answer = response.choices[0].message.content
    sources = results['metadatas'][0]

    return answer, sources


# Initialize clients
chroma_client = chromadb.Client()

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Example usage:
collection = chroma_client.get_collection("pdf_docs")

answer, sources = rag_query(
    "What is AI agent?",
    collection,
    client
)

print("Answer:", answer)
print("Sources:", sources)
