import os
from io import BytesIO

import chromadb
import PyPDF2
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


if 'collection' not in st.session_state:
    chroma_client = chromadb.Client()
    st.session_state.collection = chroma_client.create_collection(
        name=f"docs_{st.session_state.get('session_id', 'default')}"
    )

st.title("AskAtlas - Your Personal RAG")


with st.sidebar:
    st.header("Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Upload PDFs or text files",
        type=['pdf', 'txt'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for file in uploaded_files:

            if file.type == "application/pdf":
                pdf_reader = PyPDF2.PdfReader(BytesIO(file.read()))
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
            else:
                text = file.read().decode()

            words = text.split()
            chunks = [" ".join(words[i:i + 500]) for i in range(0, len(words), 500)]

            for i, chunk in enumerate(chunks):
                st.session_state.collection.add(
                    documents=[chunk],
                    metadatas=[{"source": file.name}],
                    ids=[f"{file.name}_chunk_{i}"]
                )

        st.success(f"Uploaded {len(uploaded_files)} files")

st.header("Ask Questions")

question = st.text_input("Your question:")

if question:

    results = st.session_state.collection.query(
        query_texts=[question],
        n_results=3
    )

    context = "\n\n".join(results["documents"][0])

    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"Answer the question using ONLY the context.\n\nContext:\n{context}\n\nQuestion: {question}"
            }
        ],
        max_tokens=500,
        temperature=0
    )

    answer = response.choices[0].message.content

    st.write("### Answer:")
    st.write(answer)

    st.write("### Sources:")
    for metadata in results["metadatas"][0]:
        st.write(f"- {metadata['source']}")
