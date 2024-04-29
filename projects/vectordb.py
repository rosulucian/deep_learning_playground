# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# based on https://learn.deeplearning.ai/courses/building-applications-vector-databases

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import warnings
warnings.filterwarnings('ignore')

# %%
# !conda list sentence

# %%
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm
from DLAIUtils import Utils
import DLAIUtils

import os
import time
import torch
import json

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device != 'cuda':
    print('Sorry no cuda.')

# %% [markdown]
# ### Load dataset

# %%
dataset = load_dataset('quora', split='train[240000:290000]')

# %%
dataset.shape

# %%
dataset[5:10]

# %%
questions = [y for x in dataset['questions'] for y in x['text']]

# %%
questions[:5]

# %%
len(questions)

# %% [markdown]
# ### Load model

# %%
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# %%
query = 'which city is the most populated in the world?'
xq = model.encode(query)
xq.shape

# %%
model.get_sentence_embedding_dimension()

# %% [markdown]
# ### Setup Pinecone

# %%
utils = Utils()

# %%
PINECONE_API_KEY = utils.get_pinecone_api_key()
pinecone = Pinecone(api_key=PINECONE_API_KEY)

# %%
INDEX_NAME = utils.create_dlai_index_name('dl-ai')

if INDEX_NAME in [index.name for index in pinecone.list_indexes()]:
    pinecone.delete_index(INDEX_NAME)
print(INDEX_NAME)

# %%
pinecone.create_index(name=INDEX_NAME, 
    dimension=model.get_sentence_embedding_dimension(), 
    metric='cosine',
    spec=ServerlessSpec(cloud='aws', region="us-east-1"))

index = pinecone.Index(INDEX_NAME)
print(index)

# %% [markdown]
# ### Embeddings

# %%
batch_size=200
vector_limit=10000

questions = questions[:vector_limit]

# %%
for i in tqdm(range(0, len(questions), batch_size)):
    # find end of batch
    i_end = min(i+batch_size, len(questions))
    # create IDs batch
    ids = [str(x) for x in range(i, i_end)]
    # create metadata batch
    metadatas = [{'text': text} for text in questions[i:i_end]]
    # create embeddings
    xc = model.encode(questions[i:i_end])
    # create records list for upsert
    records = zip(ids, xc, metadatas)
    # upsert to Pinecone
    index.upsert(vectors=records)

# %%
index.describe_index_stats()


# %% [markdown]
# ### Query

# %%
# small helper function so we can repeat queries later
def run_query(query):
  embedding = model.encode(query).tolist()
  results = index.query(top_k=10, vector=embedding, include_metadata=True, include_values=False)
  for result in results['matches']:
    print(f"{round(result['score'], 2)}: {result['metadata']['text']}")


# %%
query = 'what is zionism?'
run_query(query)

# %%

# %%

# %%

# %%
