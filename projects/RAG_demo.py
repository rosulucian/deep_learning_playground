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
from datasets import load_dataset
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm
from DLAIUtils import Utils

import ast
import os
import pandas as pd

# %% [markdown]
# ### Setup pinecone

# %%
# get api key
utils = Utils()
PINECONE_API_KEY = utils.get_pinecone_api_key()

pinecone = Pinecone(api_key=PINECONE_API_KEY)

# %%
utils = Utils()
INDEX_NAME = utils.create_dlai_index_name('dl-ai')
if INDEX_NAME in [index.name for index in pinecone.list_indexes()]:
  pinecone.delete_index(INDEX_NAME)

# %%
pinecone.create_index(name=INDEX_NAME, dimension=1536, metric='cosine',
  spec=ServerlessSpec(cloud='aws', region='us-east-1'))

index = pinecone.Index(INDEX_NAME)

# %%
INDEX_NAME

# %% [markdown]
# ### Store embeddings

# %%
data_csv = "E:\data\wiki.csv"

# %%
max_articles_num = 2000
df = pd.read_csv(data_csv, nrows=max_articles_num)
df.head()

# %%
ast.literal_eval(df.iloc[0].metadata)

# %%
len(ast.literal_eval(df.iloc[1].metadata)['text'])

# %%
len(ast.literal_eval(df.iloc[0]['values']))

# %%
df.columns, df.shape

# %%
prepped = []

for i, row in tqdm(df.iterrows(), total=df.shape[0]):
    meta = ast.literal_eval(row['metadata'])
    values = ast.literal_eval(row['values'])
    
    prepped.append({'id':row['id'], 
                    'values':values, 
                    'metadata':meta})
    if len(prepped) >= 250:
        index.upsert(prepped)
        prepped = []


# %%
index.describe_index_stats()

# %% [markdown]
# ### Connect to OpenAI

# %%
OPENAI_API_KEY = utils.get_openai_api_key()
openai_client = OpenAI(api_key=OPENAI_API_KEY)


# %%
def get_embeddings(articles, model="text-embedding-ada-002"):
   return openai_client.embeddings.create(input = articles, model=model)


# %% [markdown]
# ### Embedd query

# %%
query = "what is the berlin wall?"

embed = get_embeddings([query])
res = index.query(vector=embed.data[0].embedding, top_k=3, include_metadata=True)
text = [r['metadata']['text'] for r in res['matches']]
print('\n'.join(text))

# %%
len(res.matches)

# %%
res.matches[0]

# %% [markdown]
# ### Build prompt

# %%
query = "write an article titled: what is the berlin wall?"
embed = get_embeddings([query])
res = index.query(vector=embed.data[0].embedding, top_k=3, include_metadata=True)

contexts = [
    x['metadata']['text'] for x in res['matches']
]

prompt_start = (
    "Answer the question based on the context below.\n\n"+
    "Context:\n"
)

prompt_end = (
    f"\n\nQuestion: {query}\nAnswer:"
)

prompt = (
    prompt_start + "\n\n---\n\n".join(contexts) + 
    prompt_end
)

print(prompt)

# %% [markdown]
# ### RAG

# %%
res = openai_client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt=prompt,
    temperature=0,
    max_tokens=636,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None
)
print('-' * 80)
print(res.choices[0].text)

# %%

# %%

# %%
