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

import pydicom as dicom
import matplotlib.pyplot as plt

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
coords_df.condition.nunique(), coords_df.level.nunique()

# %%
coords_df.condition.unique(), coords_df.level.unique()

# %%
coords_df.study_id.nunique(), coords_df.series_id.nunique()

# %%
# coords_df['id'] = coords_df.apply(lambda row: str(row['study_id']) + str(row['series_id']), axis=1)
train_desc_df['id'] = train_desc_df.apply(lambda row: str(row['study_id']) + str(row['series_id']), axis=1)

# %%
coords_df.sample(2)

# %%
train_desc_df[train_desc_df['id'] == '4003253702807833'].series_description.values[0]

# %%
# pd.crosstab(coords_df.plane, coords_df.cl)

# %%
pd.crosstab(coords_df.condition, coords_df.plane)

# %%
# get the positive slices
coords_df.groupby(['study_id','series_id']).instance.unique()

# %%
# total positive images 
coords_df.instance_id.nunique()

# %%
# mean pos images per patient
coords_df.instance_id.nunique()/coords_df.study_id.nunique(), coords_df.groupby(['study_id']).instance_id.nunique().mean()

# %%
# total labels
pos_slices = coords_df.groupby(['study_id','series_id']).instance.unique().apply(list).reset_index(name='slice').explode('slice')
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

# %% [markdown]
# #### Patient positions

# %%
files_df.patientposition.unique()

# %%
files_df.patientposition.value_counts()

# %%
files_df.groupby(['study_id']).patientposition.unique().value_counts()

# %%
# some studies have more series
grp = files_df.groupby('study_id').filter(lambda group: group.series_id.nunique() > 3).study_id
multi_studies = grp.unique().tolist()
grp.nunique(), len(multi_studies)

# %% [markdown]
# #### Positive labels

# %%
# max/mean images per patient
files_df.groupby(['study_id','series_id']).image.count().max(), files_df.groupby(['study_id','series_id']).image.count().mean()

# %%
# mean positive imgs per series
grp = coords_df.groupby(['study_id'])
grp.instance_id.nunique().mean(), grp.instance_id.nunique().max(), grp.instance_id.nunique().min()

# %%
# mean positive imgs per series
coords_df.groupby(['study_id','series_id']).instance_id.nunique().mean()

# %%
# img per series
coords_df.groupby(['study_id','series_id']).instance_id.count().mean()

# %%
files_df.groupby(['study_id','series_id']).series_id.count()

# %% [markdown]
# ### Analyze one example

# %%
patient = 4003253
train_df[train_df['study_id'] == patient].iloc[0]

# %%
coords_df[coords_df['study_id'] == patient].condition.unique()

# %%
coords_df[coords_df['study_id'] == patient][['instance', 'cl', 'condition']]

# %%

# %%

# %% [markdown]
# ### Visualisations

# %%
files_df.patientposition.isna().sum()

# %%
files_df.groupby(['study_id']).patientposition.unique().value_counts()

# %%
files_df.seriesdescription.unique()

# %%
files_df.seriesdescription.value_counts(), files_df.seriesdescription.isna().sum()

# %%
files_df.sample(1)

# %%
coords_df.plane.unique()

# %%
coords_df[(coords_df['study_id'] == 1883368654) & (coords_df['plane'] == 'Sagittal T1')].sample()


# %% [markdown]
# #### image orientations

# %%
def get_dcoms(study_id, source=CFG.IMAGES_DIR):
    files = []

    # for t in ['T1', 'T2']:
    for t in ['Sagittal T1', 'Axial T2']:
        # samp = files_df[(files_df['study_id'] == study_id) & (files_df['seriesdescription'] == t)].sample(1)
        samp = coords_df[(coords_df['study_id'] == study_id) & (coords_df['plane'] == t)].sample(1) 
        series_id = samp.series_id.values[0]
        
        samp = files_df[(files_df['study_id'] == study_id) & (files_df['series_id'] == series_id)].sample(1)
        image = samp.image.values[0]
    
        filename = source / str(study_id) / str(series_id) / f'{image}.dcm'

        files.append(filename)

    return files
    

def plot_dcom(files, title='title'):
    cols = 5
    rows = len(files) // cols +1
    if len(files) < cols:
        cols = len(files) % cols
        # rows += 1

    fig, axs = plt.subplots(rows, cols, figsize=(15, 3*rows))
    fig.suptitle(title)

    for idx, ax in enumerate(axs.flat):
        if idx +  1 > len(files):
            break

        ax.margins(0, 0)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        ds = dicom.dcmread(files[idx])
        img = ds.pixel_array

        # img = (img - img.min()) / (img.max() - img.min())

        if ds.PhotometricInterpretation == "MONOCHROME1":
            img = 1 - img
    
        ax.imshow(img, cmap="gray")

    plt.show()


# %%
hfs = files_df[files_df['patientposition'] == 'HFS'].sample(1).study_id.values[0]
ffs = files_df[files_df['patientposition'] == 'FFS'].sample(1).study_id.values[0]

hfs, ffs

# %%
files = get_dcoms(ffs)
plot_dcom(files, title='ffs')

# %%
files = get_dcoms(hfs)
plot_dcom(files, title='hfs')

# %%

# %% [markdown]
# #### Sorting

# %%
study_id = files_df.sample(1).study_id.values[0]

study_id

# %%
series_id = files_df[files_df['study_id'] == study_id].sample(1).series_id.values[0]

series_id

# %%
imgs = files_df[files_df['series_id'] == 1482718348].image.to_list()


# %%
def plot_series(study_id, source=CFG.IMAGES_DIR):
    series_id = files_df[files_df['study_id'] == study_id].sample(1).series_id.values[0]

    imgs = files_df[files_df['series_id'] == series_id].image.to_list()
    imgs.sort()

    files = [source / str(study_id) / str(series_id) / f'{image}.dcm' for image in imgs]

    plot_dcom(files, title=f'{series_id} {len(files)} images')


# %%
study_id = 29931867

print(files_df[files_df['study_id'] == study_id].sample(1).patientposition.values[0])

plot_series(study_id)

# %%

# %%

# %%

# %%
