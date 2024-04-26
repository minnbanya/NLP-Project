import os
from datasets import load_from_disk
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
import torch
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Specify the model name and device
model_name = 'hkunlp/instructor-base'

# Initialize the embedding model
embedding_model = HuggingFaceInstructEmbeddings(
    model_name=model_name,
    model_kwargs={"device": device}
)


# Base directory where datasets are stored
base_dir = '../data/meta/'
# sizes = ['100', '500', '1000']
sizes = ['100']
categories = [
    'Toys_and_Games', 'Health_and_Personal_Care', 'Cell_Phones_and_Accessories', 
    'Home_and_Kitchen', 'Musical_Instruments', 'Baby_Products', 'Sports_and_Outdoors', 
    'Patio_Lawn_and_Garden', 'Video_Games', 'Pet_Supplies', 'Tools_and_Home_Improvement', 
    'Beauty_and_Personal_Care', 'Electronics', 'Automotive', 'Office_Products', 
    'Amazon_Fashion'
]

for size in sizes:
    for category in categories:
        dataset_path = os.path.join(base_dir, size, category)
        if os.path.exists(dataset_path):
            # Load the dataset from disk
            dataset = load_from_disk(dataset_path)
            # Build text from all fields
            texts = [json.dumps(entry) for entry in dataset]

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 700,
                chunk_overlap = 100
            )

            split_text = []    
            for text in texts:
                txt = text_splitter.split_text(text)
                for t in txt:
                    split_text.append(t)

            # Create and populate the FAISS index
            vectordb = FAISS.from_texts(split_text, embedding_model)

            # Directory to save the vector store
            output_dir = os.path.join('vector_stores', size, category)
            os.makedirs(output_dir, exist_ok=True)

            # Save the vector store
            vectordb.save_local(
                folder_path=output_dir,
                index_name=f'{category}'
            )

            print(f"Vector store saved for {category} of size {size} at {output_dir}")

