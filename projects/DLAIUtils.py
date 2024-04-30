import os
import sys

from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv, find_dotenv


class Utils:
    def __init__(self):
        pass

    def create_dlai_index_name(self, index_name):
        openai_key = ''
        if self.is_colab():  # google colab
            from google.colab import userdata
            openai_key = userdata.get("OPENAI_API_KEY")
        else:  # jupyter notebook
            openai_key = os.getenv("OPENAI_API_KEY")
        return f'{index_name}-{openai_key[-36:].lower().replace("_", "-")}'

    def is_colab(self):
        return 'google.colab' in sys.modules

    def get_openai_api_key(self):
        _ = load_dotenv(find_dotenv())
        return os.getenv("OPENAI_API_KEY")

    def get_pinecone_api_key(self):
        _ = load_dotenv(find_dotenv())
        return os.getenv("PINECONE_API_KEY")

    def get_pinecone_index(self, INDEX_NAME, pinecone, dimension=1536, metric='cosine'):
        
        if INDEX_NAME in [index.name for index in pinecone.list_indexes()]:
            pinecone.delete_index(INDEX_NAME)
        
        pinecone.create_index(
            name=INDEX_NAME, 
            dimension=dimension, 
            metric=metric,
            spec=ServerlessSpec(cloud='aws', region='us-east-1'))
        
        return pinecone.Index(INDEX_NAME)
