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
# %load_ext autoreload
# %autoreload 2

# %%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import torchaudio
import soundfile as sf
import seaborn as sns

from pathlib import Path
import plotly.express as px
import matplotlib.pyplot as plt
from IPython.display import Audio

# %%
train_dir = Path('E:\data\BirdCLEF')
train_audio = train_dir / 'train_audio'
train_csv = train_dir / 'train_metadata.csv'
taxonomy_csv = train_dir / 'eBird_Taxonomy_v2021.csv'

# %% [markdown]
# ### Metadata

# %%
taxonomy_df = pd.read_csv(taxonomy_csv)
meta_df = pd.read_csv(train_csv)

meta_df.shape, taxonomy_df.shape

# %%
meta_df.head()

# %%
taxonomy_df.head()

# %%
taxonomy_df['SPECIES_CODE'].count()

# %%
taxonomy_df[taxonomy_df['SPECIES_CODE'] == 'integr']

# %%
meta_df.primary_label.value_counts()

# %%
meta_df.filename[0]

# %%
# import plotly.io as pio
# pio.renderers

# %%
data, rate = torchaudio.load(train_audio / meta_df.filename[0])
print("Audio data shape:", data.shape)
print("Sample rate:", rate)

# %%
display(Audio(data[0, :rate*5], rate=rate))
px.line(y=data[0, :rate*5], title=meta_df.common_name[0])
# fig.show()

# %%
plt.figure(figsize=(14, 5))
plt.plot(data[0, 0:])
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.title('Waveform')
plt.show()

# %%
# Assuming metadata_df contains the data
# Create scatter plot on a map
fig = px.scatter_mapbox(meta_df, lat='latitude', lon='longitude', color='primary_label', 
                        hover_name='primary_label', hover_data=['latitude', 'longitude'], 
                        title='Geographical Distribution of Bird Species',
                        zoom=1, height=600)
fig.update_layout(mapbox_style="open-street-map")
fig.show()

# %%
foo

# %%
