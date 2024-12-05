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
import os
import json
import openai
import asyncio
import tiktoken

from bs4 import BeautifulSoup
from pymongo import MongoClient
from scrapy.selector import Selector
from crawl4ai import AsyncWebCrawler
from crawl4ai.extraction_strategy import LLMExtractionStrategy

# %%
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# %% [markdown]
# ### Connect to db, etc

# %%
# Connect to the MongoDB server
client = MongoClient("mongodb://localhost:27017/")  # Replace with your MongoDB connection string
db = client['crawler']  # Replace with your database name
cache_coll = db['httpcache']
batches_coll = db['batches']  # Replace with your collection name
urls_coll = db['urls_db'] 
content_coll = db['content'] 


# %%
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens



# %%

# %%

# %% [markdown]
# ### Strategy 1

# %%
def get_completion(prompt, model="gpt-3.5-turbo", temperature=0):
    messages = [{"role": "user", "content": prompt}]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        response_format={ "type": "json_object" },
        temperature=temperature, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message.content


# %%
query = {"fingerprint": '4f834bcc47a545cb55894d101761bf5ee80684e0'}
projection = {'url': 1, 'body': 1, '_id': 0}
documents = list(content_coll.find(query, projection))

doc = documents[0]

len(documents)

# %%
doc

# %%
body = doc['body']
soup = BeautifulSoup(body, "html.parser")

type(body), len(body), len(soup.prettify())

# %%
print(len(soup.prettify()))

for tag in soup(['script', 'style', 'meta', 'head', 'nav', 'footer', 'aside', 'svg']):
    tag.decompose()

print(len(soup.prettify()))

# %%
soup.prettify()

# %%
text = soup.get_text(separator=' ', strip=True)

len(text), num_tokens_from_string(text, 'gpt-4o')

# %%
text

# %% [markdown]
# #### Strategy no 1:
# * clean html file of polutting elemnents, 'script', 'style', 'meta', 'head', 'nav', 'footer', 'aside'
# * extract all text. at this point we should have the article text, title, date, etc we need but also some garbage from other element
# * prompt llm to sieve out the garbage and keep only the relevant info
#
# ##### Features
# * low token count ~3-4 tokens -> low cost, fast
# * gpt-4o-mini appears to handle the job well
# * could

# %%
prompt = f"""the following text is a news article scraped off a news webpage and is polluted with bits of text that are not related to the subject matter of the article.
determine the relevant data and leave out any unrelated data.
return a json object with the article date, title and body
the text is bellow:
{text}"""
model = 'gpt-4o-mini'
res = get_completion(prompt, model=model)

# %%
print(res)

# %%

# %%

# %%

# %% [markdown]
# ### Strategy 2

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
