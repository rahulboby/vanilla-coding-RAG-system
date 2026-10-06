import pickle

import sys
sys.path.append("/home/rahulboby/projects/chatbot")


chunks = pickle.load(open("data/ingestion/chunks.pkl", 'rb'))
embeddings = pickle.load(open("data/ingestion/embeddings.pkl", 'rb'))

print(f"Number of chunks: {len(chunks)}")
print(f"Number of embeddings: {len(embeddings)}")

for chunk, embedding in zip(chunks, embeddings):
    print(f"CHUNK {chunk}")
    print(chunks[chunk])
    # print("Embedding", end=': ')
    # print(embeddings[embedding])

