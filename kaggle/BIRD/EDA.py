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
class CFG:
    random_seed = 42


# %%
train_dir = Path('E:\data\BirdCLEF')
train_audio = train_dir / 'train_audio'
train_csv = train_dir / 'train_metadata.csv'
bird20223 = train_dir / 'bird2023.csv'
taxonomy_csv = train_dir / 'eBird_Taxonomy_v2021.csv'

# %% [markdown]
# ### Metadata

# %%
taxonomy_df = pd.read_csv(taxonomy_csv)
meta_df = pd.read_csv(train_csv)
df_23 = pd.read_csv(bird20223)
sample_submission = pd.read_csv(train_dir / 'sample_submission.csv')
ss_2023 = pd.read_csv(train_dir / 'sample_submission_2023.csv')

meta_df.shape, df_23.shape, taxonomy_df.shape

# %%
labels = sample_submission.columns[1:].to_list()
labels2023 = ss_2023.columns[1:].to_list()

len(labels), len(labels2023)

# %%
unique2024 = list(set(labels) - set(labels2023))
unique2023 = list(set(labels2023) - set(labels))

len(labels2023), len(labels)

# %%
df_23.sample(2)

# %%
meta_df[meta_df['filename'] == 'spemou2/XC368137.ogg']

# df_23[df_23['filename'] == 'asbfly/XC134896.ogg']

# %%
meta_df.head(2)

# %%
meta_df[meta_df['scientific_name'].str.contains('Alauda gulgula')]

# %%
taxonomy_df[taxonomy_df['SCI_NAME'].str.contains('Alauda gulgula')]

# %%

# %%

# %%
taxonomy_df.head(2)

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
                        zoom=1, height=1000)
fig.update_layout(mapbox_style="open-street-map")
fig.show()

# %% [markdown]
# ### Missing labels

# %%
primary = meta_df.primary_label.unique().tolist()
type(primary), len(primary)

# %%
secondary = meta_df[meta_df['secondary_labels'] != '[]']
secondary.shape

# %%
secondary = secondary.secondary_labels.tolist()
secondary = [eval(x) for x in secondary]
len(secondary)

# %%
sec_lab = [x for xs in secondary for x in xs]
print(len(sec_lab))
sec_lab = set(sec_lab)
len(sec_lab)

# %%
unique_primary = list(set(primary) - set(sec_lab))
unique_secondary = list(set(sec_lab) - set(primary))
len(unique_primary), len(unique_secondary)

# %%
unique_secondary


# %%

# %% [markdown]
# ### Distribution

# %%
def cat_feature_dist(data, feature):
    # Count unique values
    value_counts = data[feature].value_counts().sort_values(ascending=False)
    
    # Plot
    fig = px.bar(y=value_counts.index[::-1], x=value_counts[::-1], orientation='h')
    fig.update_yaxes(title='')
    fig.update_xaxes(title_text='Count')
    fig.update_layout(
        showlegend=False, 
        plot_bgcolor='#1C1D20', 
        paper_bgcolor='#1C1D20',
        font=dict(size=16, color='#E1B12D'),
        title_font=dict(size=20, color='#222'),
        barmode='group',  
        title=f"Distribution of '{feature}'"
    )
    fig.show()
    print(f"\nTotal unique values in '{feature}'are:",data[feature].nunique())
    print("\nTop 5 values:", value_counts.head())
    print("\nBottom 5 values:", value_counts.tail())


# %%
cat_feature_dist(meta_df, 'primary_label')


# %%
def upsample_data(df, thr=20):
    # get the class distribution
    class_dist = df['primary_label'].value_counts()

    # identify the classes that have less than the threshold number of samples
    down_classes = class_dist[class_dist < thr].index.tolist()

    # create an empty list to store the upsampled dataframes
    up_dfs = []

    # loop through the undersampled classes and upsample them
    for c in down_classes:
        # get the dataframe for the current class
        class_df = df.query("primary_label==@c")
        # find number of samples to add
        num_up = thr - class_df.shape[0]
        # upsample the dataframe
        class_df = class_df.sample(n=num_up, replace=True, random_state=CFG.random_seed)
        # append the upsampled dataframe to the list
        up_dfs.append(class_df)

    # concatenate the upsampled dataframes and the original dataframe
    up_df = pd.concat([df] + up_dfs, axis=0, ignore_index=True)
    
    return up_df


# %%
up_thr = 50
up_df = upsample_data(meta_df, thr=up_thr)

# %%
# cat_feature_dist(up_df, 'primary_label')

# %%
plt.figure(figsize=(12*2, 6))

# Upsample data
# up_thr = 50
# up_df = upsample_data(meta_df, thr=up_thr)
print("\n# BirdCLEF - 23")
print(f'> Before Upsample: {len(meta_df)}')
print(f'> After Upsample: {len(up_df)}')

ax1 = plt.subplot(1, 2, 1)
up_df.primary_label.value_counts()[:].plot.bar(color='green', label='w/ upsample')
meta_df.primary_label.value_counts()[:].plot.bar(color='blue', label='original')

# dn_df.primary_label.value_counts()[:].plot.bar(color='red', label='w/ dowsample')
plt.xticks([])
plt.axhline(y=up_thr, color='g', linestyle='--', label='up threshold')
plt.axhline(y=400, color='r', linestyle='--', label='down threshold')
plt.legend()
plt.title("Upsample for Pre-Training")

# Show effect of upsample
ax2 = plt.subplot(1, 2, 2, sharey=ax1)
up_df.primary_label.value_counts()[:].plot.bar(color='green', label='w/ upsample')
meta_df.primary_label.value_counts()[:].plot.bar(color='red', label='w/o upsample')
plt.xticks([])
plt.axhline(y=up_thr, color='g', linestyle='--', label='up threshold')
plt.legend()
plt.title("Upsample in BirdCLEF - 23")

# plt.tight_layout()
plt.show()

# %%
# up_df.head()

# %% [markdown]
# ### Secondary labels

# %%
meta_df[meta_df['secondary_labels'] != '[]'].shape, meta_df.shape

# %%
primary_df = meta_df[meta_df['secondary_labels'] == '[]']
primary_df.shape

# %%
# aparently not loosing any labels when filtering out secondary labels
len(meta_df.primary_label.unique()), len(primary_df.primary_label.unique())

# %%
meta_df[meta_df['secondary_labels'] == '[]'].primary_label.value_counts()

# %% [markdown]
# ### BirdNet

# %%

# %%

# %%

# %%
