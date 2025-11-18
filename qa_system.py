import os

import chromadb
from dotenv import load_dotenv
from openai import OpenAI


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
    "What is large language model?",
    collection,
    client
)

print("Answer:", answer)
print("Sources:", sources)
