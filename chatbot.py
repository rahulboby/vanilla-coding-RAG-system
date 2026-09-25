"""
This is a module that contains the class ChatBot(). 
This class has all the features of a chatbot like clear session history, and all that.
For this version 1, I will be using only the terminal for chatting, and not streamlit.
"""

from openai import OpenAI

from rag.retrieval import retrieve # RAG retrieval function

api_key = "placeholder"
api_url = "https://localhost:11434/v1"
model_name = "gpt-oss:20b"

DEFAULT_HISTORY_MESSAGES = history_messages = [
            {'role' : 'system', 'content' : 'Just list out the concepts that were given in the context with the question. DO NOT answer the question, just tell the user if the answer to the question is available in the context, and list all the topics that were given in the context.'}
        ] 

class ChatBot():

    def __init__(self, api_key: str = "placeholder", base_url: str = "http://localhost:11434/v1", model_name: str = "gemma4:e4b", verbose: bool = False)-> None:
        self.model_name = model_name
        self.history_messages = DEFAULT_HISTORY_MESSAGES
        # Let's keep the history and the messages as the same thing in the namespace (avoid appending every time)
        self.verbose = verbose
        self.client = OpenAI(
            base_url = base_url,
            api_key = api_key
        )


    def clear_history(self):
        self.history_messages = DEFAULT_HISTORY_MESSAGES

    def get_history(self):
        return self.history_messages
    
    def get_model_name(self):
        return self.model_name

    def build_prompt(self, question: str = '101'):
        if question == '101':
            return 'No prompt - say "NO PROMPT GIVEN"'
        context = retrieve(query=question, top_k = 1)
        return f"question: {question} \n context: {context}"
    
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
            model = model_name,
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

    

