# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import io
import time

import pandas as pd

from urllib.parse import urlparse
from datetime import datetime, date, UTC
from dateutil.parser import parse
from pymongo import MongoClient, UpdateOne

# %%
url = "https://hotnews.ro/sitemap.xml?yyyy=2001&mm=09&dd=21"

# %%
parsed = urlparse(url)

parsed

# %%
parsed.query

# %%
# Extract the date part (remove labels)
formatted_date_str = parsed.query.replace("yyyy=", "").replace("&mm=", "-").replace("&dd=", "-")

# Parse the date using datetime.strptime
date_object = datetime.strptime(formatted_date_str, "%Y-%m-%d")

date_object

# %%

# %%
date_str = '2024-11-05T05:22:14+00:00'
parse(date_str)

# %%
datetime.now().isoformat()

# %% [markdown]
# ### Mongo queries

# %%
# Connect to the MongoDB server
client = MongoClient("mongodb://localhost:27017/")  # Replace with your MongoDB connection string
db = client['crawler']  # Replace with your database name
batches_coll = db['batches']  # Replace with your collection name
cache_coll = db['httpcache']
urls_coll = db['urls_db'] 
content_coll = db['content'] 

# %%
# db.getCollectionInfos({'name': content_coll})[0].options.validator

# %%
# Query for all documents where 'batch_id' is True
query = {"scraped": False}
documents = list(batches_coll.find(query))

# Iterate and print the documents
for doc in documents:
    print(doc)

ids = [doc['batch_id'] for doc in documents]
ids

# %%

query = {'batch_id': {'$in': ids}}
projection = {'url': 1, '_id': 0}
# query = {"scraped": False}
documents = list(urls_coll.find(query, projection))
documents =[doc['url'] for doc in documents]
len(documents)

# %%
documents[:4]

# %% [markdown]
# ### BRAT

# %%
query = {"url": 'https://www.brat.ro/sati/export-rezultate/export/xls/type/site/c/all/period_type/day/category/all/editor/all/regie/all/period_filter/2024-12-3/order_by/name/order/asc/'}
projection = {'url': 1, 'body': 1, '_id': 0}
documents = list(cache_coll.find(query, projection))

len(documents)

# %%
data = documents[0]['body']

df = pd.read_excel(io.BytesIO(data))
df.columns = map(str.lower, df.columns)
df['tip trafic'] = df['tip trafic'].fillna('total')
df['cat'] = df['categorie'].apply(lambda x: ''.join(i[0:3] for i in x.lower().split()))

df.shape

# %%
df.head()

# %%
df['categorie'].nunique(), df['cat'].nunique()

# %%
df_total = df[df['tip trafic'] == 'total']
df_total = df_total.sort_values(by=['afisari'], ascending=False)

df_total.shape

# %%
df_total['categorie'].value_counts()

# %%
df_total[:20]

# %%
columns = ['site', 'sitecode', 'cat', 'categorie', 'afisari', 'vizite', 'clienti unici']

df_total[columns].head()

# %%

# %%
df_total.site.unique()

# %%
df_total.categorie.unique()

# %%
df_total[df_total['categorie'] == 'Video & TV online']

# %%
exclude_list = ['Comunitati online', 'Imobiliare', 'Portaluri & motoare de cautare', 'Altele', 'Filme & cinema']

# %%

# %%

# %% [markdown]
# ### Create collection to store websites

# %%
df_total.columns

# %%
exclude = ['nr.', 'tip trafic', 'afisari', 'vizite', 'clienti unici', ]

exclude = ['nr.', 'tip trafic', 'vizite', 'clienti unici', ]
websites = df_total[df_total.columns[~df_total.columns.isin(exclude)]]

print(websites.shape)
websites.head(2)

# %%
unique_sites = db['brat'].distinct("site")

len(unique_sites)

# %%
unique_sites

# %%
ops = []

for idx, row in websites.iterrows():
    row = row.to_dict()
    
    ops.append(UpdateOne({'site': row.pop('site', None)}, {'$set': row}, upsert=True))

print(len(ops))
    
result = db['websites'].bulk_write(ops)

# %%
# result

# %%
ops[0]

# %%

# %%

# %%

# %%
df_total[:20].site.tolist()

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
