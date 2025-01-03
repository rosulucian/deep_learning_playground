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
doc['url']

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
# * low input token count per page ~3-4 tokens -> lowihs cost, fast
# * higher output token count similar to input bc it return the same text. could try asking to return indices
# * gpt-4o-mini appears to handle the job well (cheaper)
# * could batch a lot of articles in the same prompt considering 'gpt-4o-mini' 128k context window (untested!)
# * its actually about 4X as originally thought. output tokens have a 4x higher cost and output is just a slightly trimmed down version of the input
# * consider: 1) training a smaller (bert-like?) model instead with data generated from llms (start-stop indices) or 2) since the text would already go through proccesing request outputs of text processing tasks

# %%
prompt = f"""the following text is a news article scraped off a news webpage and is polluted with bits of text that are not related to the subject matter of the article.
determine the relevant data and leave out any unrelated data.
return a json object with the article date, title and body
the text is bellow:
{text}"""
model = 'gpt-4o-mini'
res = get_completion(prompt, model=model)

print(res)

# %% [markdown]
# ### Strategy 2

# %% [markdown]
# ### Automate xpath selectors

# %% [markdown]
#
# * clean html file of polutting elemnents, 'script', 'style', 'meta', 'head', 'nav', 'footer', 'aside'
# * extract all text into a list of numbered dicts with item, text, and css_classes keys
# * convert that to json
# * 
# * prompt llm to sieve out the garbage and keep only the relevant info
# * ask to return xpath selectors for relevant fields
# * run once a day for each website and use selectors in Scrapy spiders
#
# ##### Features
# * 

# %%
elements = []

def get_xpath(tag):
    path = []
    while tag is not None and tag.name != '[document]':
        siblings = tag.find_previous_siblings(tag.name)  # Get previous siblings of the same type
        index = len(siblings) + 1  # XPath indices are 1-based
        path.insert(0, f"{tag.name}[{index}]")
        tag = tag.parent
    return "/" + "/".join(path)

def get_xpath(tag):
    path = []
    while tag is not None and tag.name != '[document]':
        siblings = tag.find_previous_siblings(tag.name)
        index = len(siblings) + 1  # XPath indices are 1-based
        class_attr = ".".join(tag.get("class", []))  # Combine CSS classes
        if class_attr:
            path.insert(0, f"{tag.name}[{index}][class='{class_attr}']")
        else:
            path.insert(0, f"{tag.name}[{index}]")
        tag = tag.parent
    return "/" + "/".join(path)

def get_xpath(tag):
    path = []
    while tag is not None and tag.name != '[document]':
        classes = [c for c in tag.get("class", []) if len(c) < 20]
        class_attr = ".".join(classes)  # Combine CSS classes
        if class_attr:
            # if len(class_attr) < 25:
            path.insert(0, f"{tag.name}[class='{class_attr}']")
        else:
            path.insert(0, tag.name)
        tag = tag.parent
    return "/" + "/".join(path)

# Remove non-visible elements like <script> and <style>
for tag in soup(['script', 'style']):
    tag.decompose()

for idx, tag in enumerate(soup.find_all(), start=1):
    # Direct text only: subtract nested children's text
    direct_text = ''.join(tag.find_all(text=True, recursive=False)).strip()
    if direct_text:  # Only include tags with direct text
        elements.append({
            "id": idx,  # Add numbering
            "item": tag.name,  # Tag name
            "text": direct_text,  # Direct text only
            "css_classes": tag.get("class", []),  # CSS classes as a list
            "xpath": get_xpath(tag)  # Generate XPath-like selector
        })

text = json.dumps(elements, indent=4, ensure_ascii=False)

# %%
len(elements)

# %%
elements

# %%

# %%
prompt = f""" you are looking at a json of html elements scraped of a news website

the following text is a news article scraped off a news webpage and is polluted with bits of text that are not related to the subject matter of the article.
determine the relevant data and leave out any unrelated data.

return a json object with the article date, title, author and article body each with the list of element ids relevant for it as well as the actual text

return also a json with a simple, minimal, short, xpath selector that will extract the text for the categories described above
prefer using css classes over order selectors
end the xpath with text()


json is bellow:
{text}"""
model = 'gpt-4o-mini'
res = get_completion(prompt, model=model)

# for the article body return only indices for the first and last words but not the article body itself

print(res)

# %%
prompt = f""" you are looking at a json of html elements scraped of a news website

the following text is a news article scraped off a news webpage and is polluted with bits of text that are not related to the subject matter of the article.
determine the relevant data and leave out any unrelated data.

return a json object with the article date, title, author and article body each with the list of element ids relevant for it as well as the actual text

return also a json with a simple, minimal, short, xpath selector that will extract the text for the categories described above
prefer using css classes over order selectors
end the xpath with text()


json is bellow:
{text}"""
model = 'gpt-4o'
res = get_completion(prompt, model=model)

# for the article body return only indices for the first and last words but not the article body itself

print(res)

# %%
len(prompt), len(res)

# %%
from parsel import Selector

# %%
sel = Selector(doc['body'])

# %%
body = sel.xpath("//time[@class='entry-date published']").extract()[0]

# %%
sel.xpath("//a[@class='author vcard']//text()").getall()
sel.xpath("//body/div[@class='site']//a[contains(@class, 'author.vcard')]//text()").getall()

# %%
sel.xpath("//div[@class='entry-content']//text()").getall()
sel.xpath("//body/div[@class='site']//div[@class='entry-content']/text()").getall()

# %%
