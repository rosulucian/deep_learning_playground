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

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import warnings
warnings.filterwarnings('ignore')

# %%
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm, trange
from DLAIUtils import Utils

import pandas as pd
import time
import os

# %% [markdown]
# ### Setup APIs

# %%
utils = Utils()
PINECONE_API_KEY = utils.get_pinecone_api_key()
OPENAI_API_KEY = utils.get_openai_api_key()
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# %%
# INDEX_NAME = utils.create_dlai_index_name('dl-ai')
# pinecone = Pinecone(api_key=PINECONE_API_KEY)

# if INDEX_NAME in [index.name for index in pinecone.list_indexes()]:
#   pinecone.delete_index(INDEX_NAME)

# pinecone.create_index(name=INDEX_NAME, dimension=1536, metric='cosine',
#   spec=ServerlessSpec(cloud='aws', region='us-east-1'))

# index = pinecone.Index(INDEX_NAME)

# %%
# pinecone.list_indexes()

# %%
pinecone = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = utils.create_dlai_index_name('dl-ai')

index = utils.get_pinecone_index(INDEX_NAME, pinecone)

# %% [markdown]
# ### Load data

# %%
data_csv = "E:\data\dl_ai_data\\all-the-news-3.csv"

# %%
with open(data_csv, 'r') as f:
    header = f.readline()
    print(header)

# %%
df = pd.read_csv(data_csv, nrows=99)
df.head()

# %%
df.iloc[0]


# %% [markdown]
# ### Embedd titles

# %%
def get_embeddings(text, model="text-embedding-ada-002"):
   return openai_client.embeddings.create(input = text, model=model)


# %%
CHUNK_SIZE=400
TOTAL_ROWS=10000

progress_bar = tqdm(total=TOTAL_ROWS)

chunks = pd.read_csv(data_csv, chunksize=CHUNK_SIZE, 
                     nrows=TOTAL_ROWS)
chunk_num = 0

for chunk in chunks:
    titles = chunk['title'].tolist()
    embeddings = get_embeddings(titles)
    
    prepped = [
        {
            'id':str(chunk_num*CHUNK_SIZE+i),
            'values':embeddings.data[i].embedding,
            'metadata': {
                'title':titles[i]
            },
            } for i in range(0,len(titles))
    ]
    
    chunk_num = chunk_num + 1
    
    if len(prepped) >= 200:
      index.upsert(prepped)
      prepped = []
    
    progress_bar.update(len(chunk))

# %%
index.describe_index_stats()


# %% [markdown]
# ### Recommender

# %%
def get_recommendations(pinecone_index, search_term, top_k=10):
  embed = get_embeddings([search_term]).data[0].embedding
  res = pinecone_index.query(vector=embed, top_k=top_k, include_metadata=True)
  return res


# %%
reco = get_recommendations(index, 'lucian')
for r in reco.matches:
    print(f'{r.score} : {r.metadata["title"]}')

# %% [markdown]
# ### Embedd articles

# %%
# reset index and embedd articles
index = utils.get_pinecone_index(INDEX_NAME, pinecone)


# %%
def embed_articles(embeddings, title, prepped, embed_num):
    for embedding in embeddings.data:
        prepped.append(
            {
                'id':str(embed_num), 
                'values':embedding.embedding, 
                'metadata':{'title':title}
            }
        )
        embed_num += 1
        
        if len(prepped) >= 100:
            index.upsert(prepped)
            prepped.clear()

    return embed_num


# %%
news_data_rows_num = 100

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400, 
    chunk_overlap=20
) # how to chunk each article

df = pd.read_csv(data_csv, nrows=news_data_rows_num)
articles_list = df['article'].tolist()
titles_list = df['title'].tolist()

# %%
prepped = []
embed_num = 0 #keep track of embedding number for 'id'

for i in range(0, len(articles_list)):
    print(".",end="")
    art = articles_list[i]
    title = titles_list[i]
    
    if art is not None and isinstance(art, str):
      texts = text_splitter.split_text(art)
      embeddings = get_embeddings(texts)
      embed_num = embed_articles(embeddings, title, prepped, embed_num)

# %%
reco = get_recommendations(articles_index, 'obama', top_k=100)
seen = {}
for r in reco.matches:
    title = r.metadata['title']
    if title not in seen:
        print(f'{r.score} : {title}')
        seen[title] = '.'

# %%
