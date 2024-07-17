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
    
    ROOT_FOLDER = train_dir
    IMAGES_DIR = ROOT_FOLDER / 'train_images'
    TRAIN_CSV = ROOT_FOLDER / 'train.csv'
    FILES_CSV = ROOT_FOLDER / 'train_files.csv'
    TRAIN_DESC_CSV = ROOT_FOLDER / 'train_series_descriptions.csv'
    COORDS_CSV = ROOT_FOLDER / 'train_label_coordinates.csv'


# %% [markdown]
# ### Train data

# %%
train_df = pd.read_csv(CFG.TRAIN_CSV)
train_desc_df = pd.read_csv(CFG.TRAIN_DESC_CSV)

train_df.shape, train_desc_df.shape

# %%
train_df.head()

# %%
train_df.iloc[1]

# %%
train_df.study_id.nunique()

# %% [markdown]
# ### Coordinates

# %%
coords_df = pd.read_csv(CFG.COORDS_CSV)
files_df = pd.read_csv(CFG.FILES_CSV)

coords_df.shape, files_df.shape

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
train_desc_df[train_desc_df['id'] == '4003253702807833'].series_description.values[0]

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
pos_slices = coords_df.groupby(['study_id','series_id']).instance_number.unique().apply(list).reset_index(name='slice').explode('slice')
pos_slices.shape

# %%
# coords_df[coords_df.instance_number > 100]

# %%
pos_slices

# %%
pos_slices.groupby('study_id').slice.nunique().mean()/3

# %%

# %% [markdown]
# ### train_df

# %%
# look at categories
for f in ['condition','level']:
    print(coords_df[f].value_counts())
    print('-'*50);print();

# %%
pd.crosstab(coords_df.condition, coords_df.level)

# %%

# %%

# %% [markdown]
# ### Files

# %%
files_df.head(3)

# %%
files_df.rows.min(), files_df.rows.max(), files_df['columns'].min(), files_df['columns'].max(), 

# %%
# files_df.image.max(), files_df.image.mean()

# %%
# file names do not correspond to file count
files_df[files_df.image == 5049]

# %%
# max/mean images per patient
files_df.groupby(['patient','series']).image.count().max(), files_df.groupby(['patient','series']).image.count().mean()

# %%
# mean positive imgs per series
coords_df.groupby(['study_id','series_id']).instance_number.nunique().mean()

# %%
files_df.groupby(['patient','series']).series.count()

# %% [markdown]
# ### Analyze one example

# %%
patient = 4003253
train_df[train_df['study_id'] == patient].iloc[0]

# %%
coords_df[coords_df['study_id'] == patient]

# %%

# %%

# %%

# %%
