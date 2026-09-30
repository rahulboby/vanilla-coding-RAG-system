"""
This is a module that contains the class ChatBot(). 
This class has all the features of a chatbot like clear session history, and all that.
For this version 1, I will be using only the terminal for chatting, and not streamlit.
"""

from openai import OpenAI
from rag.ingest import embed
import pickle
import math
from pathlib import Path

api_key = "placeholder"
api_url = "https://localhost:11434/v1"

# Mistak: Initially, this was assigned in the ChatBot class as a reference, and not a copy
# This made additions to self.history_messages also change DEFAULT_HISTORY_MESSAGES because both point to the same reference
DEFAULT_HISTORY_MESSAGES = [{
    'role' : 'system', 
    'content' : """
        DO not answer the question asked by the user. 
        In neat bullet points, list out the topics in the context given for that question. 
        Also include a very brief info of what was explained in each topic. 
    """
}] 

# custom linkedlist class to store embeddings, embedding distance and index
class node:
    def __init__(self):
        self.dist = None
        self.index = None
        self.next = None

class ChatBot():

    def __init__(
            self, api_key: str = "placeholder", 
            base_url: str = "http://localhost:11434/v1", 
            model_name: str = "gemma4:e4b", 
            verbose: bool = False,
            ingestion_path_str: str = "data/ingestion"
    )-> None:
        
        self.model_name = model_name
        self.history_messages = DEFAULT_HISTORY_MESSAGES
        # Let's keep the history and the messages as the same thing in the namespace (avoid appending every time)
        self.verbose = verbose
        self.client = OpenAI(
            base_url = base_url,
            api_key = api_key
        )
        self.ingestion_path = Path(ingestion_path_str)

        print("Loading embeddings...")
        self.embeddings = pickle.load(open(self.ingestion_path / "embeddings.pkl", "rb"))
        print("Loading chunks...")
        self.chunks = pickle.load(open(self.ingestion_path / "chunks.pkl", "rb"))
        print("DONE")


    def clear_history(self):
        self.history_messages = DEFAULT_HISTORY_MESSAGES

    def get_history(self):
        return self.history_messages
    
    def get_model_name(self):
        return self.model_name

    def build_prompt(self, question: str = '101'):
        if question == '101':
            return 'No prompt - say "NO PROMPT GIVEN"'
        context = self.retrieve(query=question, top_k = 5)
        return f"question: {question} \n context: {context}"

    def distance(self, e1, e2):
        # returns the distance between the two embeddings e1 and e2
        sum_dist = 0
        for i in range(len(e1)):
            sum_dist += (e1[i]-e2[i])**2

        return math.sqrt(sum_dist)

    def retrieve(self, query: str, top_k: int = 3) -> list:
        # Use the same embed function from ingestion.py for continuity
        q_embedding = embed(query) 

        # let's assume top_k is 3 - then we have to create top 3 values 

        """
        To store the top_k values, we will use a custom linkedlist setup.
        I know this is an overkill - we could have just used the built-in heapq module 
        in python that was introduced in python 2.3
        But, I want this project to have as much less imports as possible - 
        completely vanilla code everything to test my python skills.
        """
        head = None
        ptr = None

        # Create a linkedlist of nodes containing index and dist of each - to take the top k by traversing.
        # MISTAKE: I made it in such a way head was always the chunk 0 -because no check if the ptr was at head was done, everytime the second chunk would be added after the head, making the head stay there no matter what. 
        # Fixed? 
        for i in range(len(self.embeddings)):
            dist = self.distance(q_embedding, self.embeddings[i])
            if head == None:
                head = node()
                head.dist = dist
                head.index = i
                continue
            if dist<head.dist:
                new_node = node()
                new_node.dist = dist
                new_node.index = i
                new_node.next = head
                head = new_node
            else:
                ptr = head
                while True:
                    if ptr.next == None or ptr.next.dist > dist:
                        new_node = node()
                        new_node.index = i
                        new_node.dist = dist
                        new_node.next = ptr.next
                        ptr.next = new_node
                        break
                    ptr = ptr.next

        if self.verbose:
            # Print the linkedlist formed for retrieval
            print("Linked list formed for this retrieval: ")
            ptr = head
            while ptr!=None:
                print(ptr.index, end='->')
                ptr = ptr.next
            print('X')

        # Create a context
        context = []
        ptr = head
        for i in range(top_k):
            if ptr == None:
                print("Lesser number of chunks than top_k: retrieving all chunks")
                break
            context.append(self.chunks[ptr.index])
            if self.verbose:
                print(f"Retrieved Chunk {ptr.index}")
            ptr = ptr.next
        else:
            print(f"Retrieved {top_k} chunks")
        
        
        return context
    
    def generate_response(self, question: str = "101"):
        if question == "101":
            return "No message given!"

        self.history_messages.append({'role' : 'user', 'content' : self.build_prompt(question)})
        
        response_object = self.client.chat.completions.create(
            model = self.model_name,
            messages = self.history_messages,
            temperature = 0.7
        )
        
        response = response_object.choices[0].message.content

        if self.verbose:
            print(f"Number of input tokens: {response_object.usage.prompt_tokens}")
            print(f"Completion tokens: {response_object.usage.completion_tokens}")

        # Don't add the entire context into history - so replace the content with just the question
        self.history_messages[-1]['content'] = question 
        self.history_messages.append({'role':'assistant', 'content':response})

        print(response)

    def generate_response_stream(self, question: str = "101"):
        if question == "101":
            return "No message given"

        built_prompt = self.build_prompt(question=question)

        # if self.verbose:
        #     print(('-'*80 +'\n')*3)
        #     print(f"Sending built prompt: \n {built_prompt}")
        #     print(('-'*80 +'\n')*3)
        self.history_messages.append({'role' : 'user', 'content' : built_prompt})

        stream = self.client.chat.completions.create(
            model = self.model_name,
            messages = self.history_messages,
            temperature = 0.7,
            stream = True,
            max_tokens=50000,
            extra_body={
                "think": "none" # To turn thinking off - no reasoning
            }
        )

        response = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                token = chunk.choices[0].delta.content
                response += token
                print(token, end="", flush=True)

        # Don't add the entire context into history - so replace the content with just the question
        self.history_messages[-1]['content'] = question 
        self.history_messages.append({'role':'assistant', 'content':response})

    

