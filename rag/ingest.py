"""
To be run as a main module using terminal. 
Takes in a folder of .txt files, and chunks them and uses sentencetransformers to create embeddings.
Stores both embeddings and chunks as dictionary values.

"""
from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

MAX_CHUNK_SIZE = 1200 # 1200 char limit for each chunk
EMBEDDER_MODEL = "all-MiniLM-L6-v2" # Name of the embedder model to use

embedder = SentenceTransformer(EMBEDDER_MODEL)

# Takes in a string and gives out a vector embedding
def embed(chunk: str):
	
	return embedder.encode(chunk)

# The main function
def get_ingestion(raw_folder: str, verbose: bool = True):
	# Takes in a raw_path from which all the text files get chunked
	raw_folder = Path(raw_folder)
	txt_files = list(raw_folder.glob("*.txt"))

	print(f"{len(txt_files)} txt files taken for ingestion.")

	if verbose:
		print("Files taken for ingestion: ")
		print(txt_files)
    
	chunks = {}
	chunk_list = [] # Contains all chunks as string list, which will be passed in as whole to the sentence transformers. Batch processing is far more efficient.
	embeddings = {}
	embedding_list = [] # Consists of all embeddings as list - for batch processing output using chunk_list
	chunk = ""

	# Source - https://stackoverflow.com/a/10378012
	for filename in txt_files:
		with open(filename, 'r') as file:
			line = None
			while line != '':
				# global chunk # Refer to chunk in global namespace
				# global line # Refer to line in global namespace
				line = file.readline()
				print(f"Line "+'-'*80)
				print(line)
				# Check if this line would cause the max limit to exceed in chunks
				if len(line) + len(chunks) > MAX_CHUNK_SIZE:
					# Add each chunk to the chunk_list
					chunk_list.append(chunk)
					# Set chunk as the line
					chunk = line
				else:
					chunk = chunk + '\n' + line # Append the line to the chunk, in as a new line
	
	# If chunk is not empty, add that as a final new chunk into the list.
	if chunk:
		print("Extra chunk added to the chunk_list")
		chunk_list.append(chunk)
	
	embedding_list = embed(chunk_list)

	# Converting the chunks and imbeddings to dictionary
	for i in range(len(chunk_list)):
		chunks[i] = chunk_list[i]
		embeddings[i] = embedding_list[i]
	
	return embeddings, chunks

def save_ingestion(raw_folder: str, output_folder: str):
	raw_folder = Path(raw_folder)
	output_folder = Path(output_folder)

	import pickle
	embeddings, chunks = get_ingestion(raw_folder = raw_folder, verbose = True)

	embeddings_output_path = output_folder / "embeddings.pkl"
	chunks_output_path = output_folder / "chunks.pkl"
	
	try:
		with open(embeddings_output_path, 'wb') as file:
			pickle.dump(embeddings, file)
		print(f"Saved embeddings to {embeddings_output_path}")

		with open(chunks_output_path, 'wb') as file:
			pickle.dump(chunks, file)
		print(f"Saved chunks to {chunks_output_path}")
	
	except Exception as e:
		print(f"Exception: {e}")
	

if __name__ == "__main__":
	# Run the ingestion only if called directly
	raw_folder = str(input("Enter the raw folder: "))
	output_folder = str(input("Enter the output path: "))
	save_ingestion(raw_folder, output_folder)


