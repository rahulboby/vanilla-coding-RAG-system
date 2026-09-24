"""
This is the chat interface that uses ChatBot class
Terminal based chatbot (no streamlit)
"""

from chatbot import ChatBot
import sys

mode = str(input("Mode: (groq(g)/ollama(o)): "))

if mode in ['g', 'groq', 'online']:
	print("Using groq with gpt-oss:120b")
	bot = ChatBot(model_name = "gpt-oss:120b")

elif mode in ['o', 'ollama', 'offline']:
	print("Using local ollama endpoint with gemma4:e4b.")
	bot = ChatBot(model_name = "gemma4:e4b")

else:
	print("INVALID OPTION. Available options: 'g' for groq and  'o' for ollama.")
	sys.exit(0)	

prompt = ""

while True:
    print("\nUser: ")
    prompt = input()
    
    if prompt.lower().strip() == '/bye':
        break
    
    if prompt.lower().strip() == '/clear':
        bot.clear_history()
        print("Cleared chat history.")
        continue

    print("\nAssistant: ")

    bot.generate_response_stream(prompt)

    print('\n' + '-'*50)

print("BYE BYE!")
