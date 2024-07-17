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
import numpy as np 
import pandas as pd

from pathlib import Path

# %%
train_dir = Path('E:\data\RSNA2024')


# %%
class CFG:
    random_seed = 42
    
    ROOT_FOLDER = train_dir / 'original'
    DEST_FOLDER = train_dir
    IMAGES_DIR = ROOT_FOLDER / 'train_images'
    TRAIN_CSV = ROOT_FOLDER / 'train.csv'
    FILES_CSV = ROOT_FOLDER / 'train_files.csv'
    TRAIN_DESC_CSV = ROOT_FOLDER / 'train_series_descriptions.csv'
    COORDS_CSV = ROOT_FOLDER / 'train_label_coordinates.csv'


# %% [markdown]
# ### Rename train_df columns and values

# %%
train_df = pd.read_csv(CFG.TRAIN_CSV)
train_desc_df = pd.read_csv(CFG.TRAIN_DESC_CSV)

train_df.shape, train_desc_df.shape

# %%
train_df.head()

# %%
cols = train_df.columns[1:]
# first = [c.split('_')[:-2] for c in cols]
# last = [c.split('_')[-2:] for c in cols]

cols = [c.split('_') for c in cols]
cols = [''.join([i[0] if len(i) > 2 else i for i in c]).upper() for c in cols]

cols = ['study_id'] + cols

cols[:5]

# %%
dict(zip(train_df.columns, cols))

# %%
train_df.rename(columns=dict(zip(train_df.columns, cols)), inplace=True)
train_df.shape

# %%
train_df.iloc[1]

# %%
train_df.study_id.nunique()

# %%
vals = {'Normal/Mild': 'N', 'Moderate': 'M', 'Severe': 'S'}
vals

# %%
train_df[cols[1:]] = train_df[cols[1:]].replace(vals)

# %%
train_df.sample(2)

# %% [markdown]
# ### Coordinates

# %%
coords_df = pd.read_csv(CFG.COORDS_CSV)
files_df = pd.read_csv(CFG.FILES_CSV)

coords_df.shape, files_df.shape

# %%
coords_df.rename(columns={'instance_number': 'instance'}, inplace=True)

# %%
coords_df.study_id.nunique(), coords_df.condition.nunique(), coords_df.level.nunique()

# %%
coords_df.condition.unique(), coords_df.level.unique()

# %%
coords_df.series_id.nunique()

# %%
coords_df['id'] = coords_df.apply(lambda row: str(row['study_id']) + str(row['series_id']), axis=1)
train_desc_df['id'] = train_desc_df.apply(lambda row: str(row['study_id']) + str(row['series_id']), axis=1)

# %%
coords_df.sample(2)

# %%
# rename condition
coords_df['condition'] = coords_df.apply(lambda row: ''.join([w[0] for w in row['condition'].split(' ')]), axis=1)

# %%
# rename level
coords_df['level'] = coords_df.level.apply(lambda l: ''.join(l.split('/')))

# %%
coords_df.condition.nunique()

# %%
coords_df.head(10)

# %%
train_desc_df[train_desc_df['id'] == '4003253702807833'].series_description.values[0]

# %%
coords_df['plane'] = coords_df.apply(lambda row: train_desc_df[train_desc_df['id'] == row['id']].series_description.values[0], axis=1)

# %%
coords_df.sample(5)

# %%
# check canal stenosis is noy only in axial plane
coords_df[(coords_df.condition == 'SCS') & (coords_df.plane != 'Axial T2')].sample()

# %%
# get the positive slices
coords_df.groupby(['study_id','series_id']).instance.unique()

# %%
coords_df.id.nunique()

# %%

# %%

# %%
