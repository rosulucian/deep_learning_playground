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

    image_size = 256
    
    ROOT_FOLDER = train_dir / 'original'
    DEST_FOLDER = train_dir
    PNG_DIR = DEST_FOLDER / f'pngs_{image_size}'
    IMAGES_DIR = ROOT_FOLDER / 'train_images'
    TRAIN_CSV = ROOT_FOLDER / 'train.csv'
    FILES_CSV = ROOT_FOLDER / 'train_files.csv'
    TRAIN_DESC_CSV = ROOT_FOLDER / 'train_series_descriptions.csv'
    COORDS_CSV = ROOT_FOLDER / 'train_label_coordinates.csv'


# %% [markdown]
# ### Train_df

# %%
train_df = pd.read_csv(CFG.TRAIN_CSV)
train_desc_df = pd.read_csv(CFG.TRAIN_DESC_CSV)

train_df.shape, train_desc_df.shape

# %%
train_desc_df['ss_id'] = train_desc_df.apply(lambda row: f'{str(row["study_id"])}_{str(row["series_id"])}', axis=1)

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
train_df.iloc[0]

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
# ### Coordinates_df

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
coords_df['ss_id'] = coords_df.apply(lambda row: f'{str(row["study_id"])}_{str(row["series_id"])}', axis=1)
coords_df['instance_id'] = coords_df.apply(lambda row: f'{str(row["study_id"])}_{str(row["series_id"])}_{str(row["instance"])}', axis=1)

# %%
train_desc_df.sample()

# %%
coords_df.sample(2)

# %%
# rename condition
coords_df['condition'] = coords_df.apply(lambda row: ''.join([w[0] for w in row['condition'].split(' ')]), axis=1)

# %%
# rename level
coords_df['level'] = coords_df.level.apply(lambda l: ''.join(l.split('/')))

# %%
coords_df['cl'] = coords_df['condition'] + coords_df['level']

# %%
coords_df.condition.nunique()

# %%
coords_df.head(10)

# %%
train_desc_df.series_description.unique()

# %%
train_desc_df.series_description.isna().sum()

# %%
coords_df.shape

# %%
coords_df = pd.merge(coords_df, train_desc_df.loc[:, ['ss_id', 'series_description']],  how='inner', left_on=['ss_id'], right_on=['ss_id'])

coords_df.sample(2)

# %%
coords_df.shape

# %%
# coords_df['plane'] = coords_df.apply(lambda row: train_desc_df[train_desc_df['ss_id'] == row['ss_id']].series_description.values[0], axis=1)

# %%
# coords_df.sample(5)

# %%
# check canal stenosis is noy only in axial plane
coords_df[(coords_df.condition == 'SCS') & (coords_df.series_description != 'Axial T2')].sample()

# %%
# get the positive slices
coords_df.groupby(['study_id','series_id']).instance.unique()

# %%
coords_df.ss_id.nunique(), coords_df.instance_id.nunique()

# %% [markdown]
# ### Files_df

# %%
files_df = pd.read_csv(CFG.FILES_CSV)

# %%
files_df.sample(5)

# %%
# files_df.rename(columns={'patient': 'study_id', 'series': 'series_id', 'image': 'instance'}, inplace=True)

# %%
files_df.patientposition.value_counts(), files_df.patientposition.isna().sum()

# %%
files_df.groupby(['study_id']).patientposition.unique().value_counts()

# %%
files_df['ss_id'] = files_df.apply(lambda row: f'{str(row["study_id"])}_{str(row["series_id"])}', axis=1)
files_df['instance_id'] = files_df.apply(lambda row: f'{str(row["study_id"])}_{str(row["series_id"])}_{str(row["instancenumber"])}', axis=1)

# %%
source_dir = CFG.PNG_DIR
files_df['filename'] = files_df.apply(lambda row: f'{source_dir}\\{row.study_id}_{row.series_id}_{row.image}.png', axis=1)

# %%
files_df.sample()

# %%
train_desc_df.sample()

# %%
coords_df = pd.merge(coords_df, files_df[['instance_id', 'rows', 'columns', 'filename']], left_on='instance_id', right_on='instance_id')

# %%
coords_df.shape

# %%
coords_df.sample()

# %%
# TODO: make sure we match coords corectly
coords_df['x_perc'] = coords_df['x'] / coords_df['columns']
coords_df['y_perc'] = coords_df['y'] / coords_df['rows']

# %%
ax, non_ax = coords_df[coords_df['series_description'] == 'Axial T2'], coords_df[coords_df['series_description'] != 'Axial T2']
ax.shape, non_ax.shape, coords_df.shape

# %%
coords_df.y.min()

# %%
for c in [ax, non_ax]:
    print(c['x_perc'].min(), c['y_perc'].min())
    print(c['x_perc'].max(), c['y_perc'].max())
    print('/////////////')
    # print(c['x_perc'].mean(), c['y_perc'].mean())

# %%
files_df.shape

# %%
files_df = pd.merge(files_df, train_desc_df.loc[:, ['ss_id', 'series_description']],  how='inner', left_on=['ss_id'], right_on=['ss_id'])

files_df.sample(2)

# %%
files_df.shape

# %%
files_df['cl'] = 'H'

# %%
files_df.sample(5)

# %% [markdown]
# ### Save results

# %%
train_df.to_csv(CFG.DEST_FOLDER / 'train.csv', index=False)

# %%
coords_df.to_csv(CFG.DEST_FOLDER / 'train_label_coordinates.csv', index=False)

# %%
files_df.to_csv(CFG.DEST_FOLDER / 'train_files.csv', index=False)

# %%
