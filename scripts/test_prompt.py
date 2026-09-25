"""
Just to test one single prompt
"""
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(root_dir)
sys.path.append("/home/rahulboby/projects/chatbot")
from core.chatbot import ChatBot

mode = str(input("Mode: (groq(g)/ollama(o)): "))
print("Automatically going with verbose mode")
verbose = True 

if mode in ['g', 'groq', 'online']:
	print("Using groq with gpt-oss:120b")
	bot = ChatBot(model_name = "gpt-oss:120b")

elif mode in ['o', 'ollama', 'offline']:
	print("Using local ollama endpoint with gemma4:e4b.")
	bot = ChatBot(model_name = "gemma4:e4b", verbose=verbose)

else:
	print("INVALID OPTION. Available options: 'g' for groq and  'o' for ollama.")
	sys.exit(0)	

prompt = sys.argv[1]
print(f"Prompt: {prompt}")

print("\nAssistant: ")

bot.generate_response(question=prompt)

print('\n' + '-'*50)

print("BYE BYE!")