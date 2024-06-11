# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np

# %%
df_path = "E:\data\BirdCLEF\submission.csv"
sub_path = 'E:\data\BirdCLEF\sample_submission.csv'

USE_MISSING_LABELS = True

# %%
df = pd.read_csv(df_path)
sample_submission = pd.read_csv(sub_path)
df.shape

# %%
sec_labels = ['lotshr1', 'orhthr1', 'magrob', 'indwhe1', 'bltmun1', 'asfblu1']

target_columns = sample_submission.columns[1:].tolist()
if USE_MISSING_LABELS:
    target_columns += sec_labels

num_classes = len(target_columns)
bird2id = {b: i for i, b in enumerate(target_columns)}

num_classes

# %%
# bird2id['magrob']

# %%
df['name'] = df.label.apply(lambda row: target_columns[row])

# %%
df.head()

# %% [markdown]
# ### Most detected labels

# %%
most_certain = df[df['score'] > 0.99]
most_certain.shape

# %%
most_certain.label.nunique(), most_certain.label.value_counts()

# %% [markdown]
# ### Find nocall

# %%
no_call = df[df['score'] < 0.1]
no_call.shape

# %%
no_call.head()

# %%

# %%
