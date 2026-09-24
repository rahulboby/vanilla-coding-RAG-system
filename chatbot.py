"""
This is a module that contains the class ChatBot(). 
This class has all the features of a chatbot like clear session history, and all that.
For this version 1, I will be using only the terminal for chatting, and not streamlit.
"""

import pandas as pd
from openai import OpenAI

from rag.retrieval import retrieve # RAG retrieval function

api_key = "placeholder"
api_url = "https://localhost:11434/v1"
model_name = "gemma4:e4b"

class ChatBot():

    def __init__(self, api_key: str = "placeholder", base_url: str = "http://localhost:11434/v1", model_name: str = "gemma4:e4b")-> None:
        self.model_name = model_name
        self.history_messages = [
            {'role' : 'system', 'content' : 'Answer the questions asked by the user with crisp and clear points, without overexplaning or overengineering. Straightly come to the point, instead of building up stories.'}
        ] 
        # Let's keep the history and the messages as the same thing in the namespace (avoid appending every time)

        self.client = OpenAI(
            base_url = base_url,
            api_key = api_key
        )


    def clear_history(self):
        self.history_messages = [
            {'role' : 'system', 'content' : 'Answer the question by starting with "This is Rahul\'s chatbot...\n"'}
        ] 

    def get_history(self):
        return self.history_messages
    
    def get_model_name(self):
        return self.model_name

    def build_prompt(self, question: str = '101'):
        if question == '101':
            return 'No prompt - say "NO PROMPT GIVEN"'
        context = retrieve(query=question, top_k = 3)
        return f"question: {question} \n Answer this question strictly based on the context given. If the answer to the question is not mentioned in the context given below, please say exactly this: 'THE CONTEXT PROVIDED DOES NOT CONTAIN THE ANSWER TO YOUR QUESTION' \n context: {context}"
    
    def generate_response(self, question: str = "101"):
        if question == "101":
            return "No message given!"

        self.history_messages.append({'role' : 'user', 'content' : self.build_prompt(question)})
        
        response_object = self.client.chat.completions.create(
            model = model_name,
            messages = self.history_messages,
            temperature = 0.7
        )

        response = response_object.choices[0].message.content

        # Don't add the entire context into history - so replace the content with just the question
        self.history_messages[-1]['content'] = question 
        self.history_messages.append({'role':'assistant', 'content':response})

        print(response)

    def generate_response_stream(self, question: str = "101"):
        if question == "101":
            return "No message given"
       
        self.history_messages.append({'role' : 'user', 'content' : self.build_prompt(question)})

        stream = self.client.chat.completions.create(
            model = model_name,
            messages = self.history_messages,
            temperature = 0.7,
            stream = True
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

    

