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
from crawl4ai import AsyncWebCrawler
from crawl4ai.extraction_strategy import LLMExtractionStrategy


# %%
import nest_asyncio
nest_asyncio.apply()

# %%
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# %%
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


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


# %% [markdown]
# ### Load mongodb

# %%
# Connect to the MongoDB server
client = MongoClient("mongodb://localhost:27017/")  # Replace with your MongoDB connection string
db = client['crawler']  # Replace with your database name
cache_coll = db['httpcache']
batches_coll = db['batches']  # Replace with your collection name
urls_coll = db['urls_db'] 
content_coll = db['content'] 

# %%
query = {"fingerprint": '4f834bcc47a545cb55894d101761bf5ee80684e0'}
documents = list(content_coll.find(query))

# ids = [doc['batch_id'] for doc in documents]
# ids

len(documents)

# %%
body = documents[0]['body']
len(body)

# %%
# async def extract_tech_content(url):
#     async with AsyncWebCrawler(verbose=True) as crawler:
#         result = await crawler.arun(
#             # url="https://www.nbcnews.com/business",
#             url=url,
#             extraction_strategy=LLMExtractionStrategy(
#                 provider="openai/gpt-4o",
#                 api_token=os.getenv('OPENAI_API_KEY'),
#                 instruction="Extract only content related to technology"
#             ),
#             bypass_cache=True,
#         )

#     return result

# async def make_requests_in_parallel(url):
#     """Make requests in parallel and return the responses."""
#     return await asyncio.gather(extract_tech_content(url))

# %% [markdown]
# ### OpenAI scrape

# %%

# %%
from scrapy.selector import Selector

# %%
query = {"url": 'https://www.brat.ro/sati/rezultate/type/site/page/1/c/all'}
# query = {"url": 'https://www.brat.ro/sati/export-rezultate/export/xls/type/site/c/all/period_type/day/category/all/editor/all/regie/all/period_filter/2024-12-3/order_by/name/order/asc/'}
projection = {'url': 1, 'body': 1, '_id': 0}
documents = list(cache_coll.find(query, projection))

len(documents)

# %%
query = {"fingerprint": '4f834bcc47a545cb55894d101761bf5ee80684e0'}
documents = list(content_coll.find(query))

# %%
# documents
body = documents[0]['body']
type(body), len(body)

# %%
soup = BeautifulSoup(body, "html.parser")

# %%
# # Remove all text, leaving only tags
# for element in soup.find_all(string=True):
#     element.replace_with("")

print(len(soup.prettify()))

# # Remove all <script> tags
# for script in soup.find_all("script"):
#     script.decompose()

for tag in soup(['script', 'style', 'meta', 'head', 'nav', 'footer', 'aside']):
    tag.decompose()

print(len(soup.prettify()))

# %%
len(soup.get_text(separator=' ', strip=True))

# %%
text = soup.get_text(separator=' ', strip=True)
len(text), text

# %%
num_tokens_from_string(text, 'gpt-4o')

# %%

# %%
print(soup.prettify())

# %%
sel = Selector(text=body)

# %%
# foo = sel.xpath("//body//table").getall()
foo = sel.xpath("//body//tbody").getall()

len(foo), len(foo[0]), len(body)

# %%
foo


# %%
num_tokens_from_string(foo[0], 'gpt-4o'), num_tokens_from_string(foo[0], 'gpt-3.5-turbo')

# %%
num_tokens_from_string(soup.prettify(), 'gpt-4o'), num_tokens_from_string(soup.prettify(), 'gpt-3.5-turbo')

# %%
# prompt = f'you are webscraper. your job is to extract the data from the following html file and extract the table into json. the html follow here: {foo[0]}'
prompt = f"""the following text is a news article scraped off a news webpage and is polluted with bits of text that are not related to the subject matter of the article.
determine the relevant data and leave out any unrelated data.
return a json object with the article date, title and body
the text is bellow:
{text}"""
model = 'gpt-4o-mini'
res = get_completion(prompt, model=model)

# %%
res

# %%
print(res[7:])

# %%
foo = json.loads(res)

# %%
# prompt = f'you are webscraper. your job is to extract the data from the following html file and extract the table into json. the html follow here: {foo[0]}'
prompt = f'you are webscraper. your job is to extract the data from the following html file and extract the table into json. the html follow here: {soup.prettify()}'
model = 'gpt-4o-mini'
res = get_completion(prompt, model=model)

# %%
print(res)

# %%

# %%
body

# %%

# %%

# %%

# %%

# %%

# %%
