from pathlib import Path
import os
import pickle
from rag.ingest import embed
import math

# custom linkedlist class to store embeddings, embedding distance and index

class node:
    def __init__(self):
        self.dist = None
        self.index = None
        self.next = None

# load both thepickle file

ingestion_path = Path("data/ingestion")

embeddings = pickle.load(open(ingestion_path / "embeddings.pkl", "rb"))
chunks = pickle.load(open(ingestion_path / "chunks.pkl", "rb"))

def distance(e1, e2):
    # returns the distance between the two embeddings e1 and e2
    sum_dist = 0
    for i in range(len(e1)):
        sum_dist += (e1[i]-e2[i])**2

    return math.sqrt(sum_dist)
    

def retrieve(query: str, top_k: int = 3) -> list:
    # Use the same embed function from ingestion.py for continuity
    q_embedding = embed(query) 

    # let's assume top_k is 3 - then we have to create top 3 values 

    # To store the top_k values, we will use a custom linkedlist setup.
    # I know this is an overkill - we could have just used the built-in heapq module in python that was introduced in python 2.3
    # But, I want this project to have as much less imports as possible - completely hardcode everything to test my python skills.

    head = None
    ptr = None

    # Create a linkedlist of nodes containing index and dist of each - to take the top k by traversing.
    for i in range(len(embeddings)):
        dist = distance(q_embedding, embeddings[i])
        if head == None:
            head = node()
            head.dist = dist
            head.index = i
        else:
            ptr = head
            while True:
                if ptr.next == None:
                    ptr.next = node()
                    ptr.next.index = i
                    ptr.next.dist = dist
                    break
                ptr = ptr.next

                # Mimic a do-while loo[]
                if ptr.next.dist < dist:
                    break

    # Create a context
    context = []
    ptr = head
    for i in range(len(chunks)):
        if ptr == None:
            break
        context.append(chunks[ptr.index])
        ptr = ptr.next

    return context
            

