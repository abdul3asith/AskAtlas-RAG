import os

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

text = "The quick brown fox jumps over the lazy dog and a lazy cat"

# call openAI API to create an embedding
response = client.embeddings.create(
    model='text-embedding-3-small',
    input=text
)

# extracting actual embedding vector from the API response
# data[0] means using first element from array but we have only one item in text
embedding = response.data[0].embedding

print(f"Text: {text}")
print(f"Embedding length: {len(embedding)}")  
print(f"First 5 values: {embedding[:5]}")

''' 
---------- Up until now we translated text into embeddings(list of numbers) using 
openAI's model 
'''

# Let us define similarity between texts

def get_embedding(text):
    response = client.embeddings.create(
    model='text-embedding-3-small',
    input=text
    )    
    return response.data[0].embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

text1 = "I love dogs and cats"
text2 = "I adore puppies and kitten"
text3 = "Modi won elections"

emb1 = get_embedding(text1)
emb2 = get_embedding(text2)
emb3 = get_embedding(text3)

print(f"{cosine_similarity(emb1, emb2):.3f}")
print(f"{cosine_similarity(emb2, emb3):.3f}")
# print(cosine_similarity(emb2, emb3))



#output
'''
Text: The quick brown fox jumps over the lazy dog and a lazy cat
Embedding length: 1536
First 5 values: [-0.006066478323191404, -0.020251331850886345, 0.0001831055269576609, -0.038743894547224045, -0.016109302639961243]
0.738
0.067
'''