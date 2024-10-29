# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ### Imports

# %%
import os
import shutil

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

# %% [markdown]
# #### Conditions/plane crosstab

# %%
pd.crosstab(coords_df.condition, coords_df.series_description)

# %%
coords_df.study_id.nunique()

# %%
coords_df.groupby('condition').instance_id.nunique()

# %%
# get the positive slices
coords_df.groupby(['study_id','series_id']).instance.unique()

# %%
# total positive images 
coords_df.instance_id.nunique()

# %%
# positive images per patient
grp = coords_df.groupby(['study_id']).instance_id.count()

grp.mean(), grp.min(), grp.max()

# %%

# %% [markdown]
# ### Train_df

# %%
# look at categories
for f in ['condition','level']:
    print(coords_df[f].value_counts())
    print('-'*50);print();

# %%
# pd.crosstab(coords_df.condition, coords_df.level)

# %% [markdown]
# ### Files

# %%
files_df.head(3)

# %%
files_df.groupby('study_id').instance_id.count().agg(['min', 'max', 'mean'])

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

# %% [markdown]
# #### Studies with multiple series

# %%
# some studies have more series
grp = files_df.groupby('study_id').filter(lambda group: group.series_id.nunique() > 3).study_id
multi_studies = grp.unique().tolist()
grp.nunique(), len(multi_studies), files_df.study_id.nunique()

# %%
files_df.series_description.value_counts()

# %%
study_id = multi_studies[1]
print(study_id)
files_df[files_df['study_id'] == study_id]

# %% [markdown]
# #### positive/total images

# %%
# study_id = multi_studies[1]
# study 1018005303 has multiple axial series
study_id = 1018005303
print(study_id)
files_df[files_df['study_id'] == study_id].groupby('series_id').series_description.value_counts()

# %%
coords_df[coords_df['study_id'] == study_id].groupby('series_id').instance.count()

# %%
coords_df[coords_df['study_id'] == study_id].groupby('series_id').cl.unique()

# %%
coords_df[coords_df['study_id'] == study_id].groupby('series_id').condition.unique()

# %% [markdown]
# #### Positive labels

# %%
coords_df[(coords_df['study_id'] == study_id) & (coords_df['series_description'] == 'Axial T2')]

# %%
# files_df[(files_df['study_id'] == study_id) & (files_df['series_description'] == 'Axial T2')].sort_values(['series_id', 'image'])[50:]

# %%
# max/mean images per patient
grp = files_df.groupby(['study_id','series_id']).image.count()
grp.min(), grp.max(), grp.mean()

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
# ### Order

# %%
files_df.sample(2)

# %%
study = 1018005303

df = files_df[files_df['study_id'] == study]

df.shape

# %%
df.series_description.unique()

# %%
df[df['series_description'] == 'Sagittal T2/STIR'].sort_values(['series_id', 'proj'], ascending=[True, False])[:60]

# %%
df[df['series_description'] == 'Axial T2'].sort_values(['series_id', 'proj'], ascending=[True, False])[:60]

# %%
df[df['series_description'] == 'Axial T2'].sort_values(['proj'], ascending=[False])[:60]

# %%
selection = df[df['series_description'] == 'Axial T2'].sort_values(['proj'], ascending=[False])

selection.study_id.unique()[0]

# %%
selection.shape

# %%
selection = df[df['series_description'] == 'Axial T2'].sort_values(['proj'], ascending=[False]).reset_index(drop=True)
# dcoms = selection.filename.to_list()


dest_dir = Path('E:\\data\\RSNA2024\\train_images') / str(selection.study_id.unique()[0])

# dest = ('\\').join(dcoms[0].split('\\')[:-1] + ['joined'])

dest_dir

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)


for i, row in selection.iterrows():
    src = dest_dir / f'{row["series_id"]}'/ f'{row["image"]}.dcm'
    dest = dest_dir / 'joined' / f'{i}.dcm'
    
    shutil.copy(src, dest)

# %%

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
coords_df.series_description.unique(), files_df.seriesdescription.unique()

# %%
coords_df[(coords_df['study_id'] == 1883368654) & (coords_df['series_description'] == 'Sagittal T1')].sample()


# %% [markdown]
# #### Patient position

# %%
def get_dcoms(study_id, source=CFG.IMAGES_DIR):
    files = []

    # for t in ['T1', 'T2']:
    for t in ['Sagittal T1', 'Axial T2']:
        # samp = files_df[(files_df['study_id'] == study_id) & (files_df['seriesdescription'] == t)].sample(1)
        samp = coords_df[(coords_df['study_id'] == study_id) & (coords_df['series_description'] == t)].sample(1) 
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
study = 665627263
files = get_dcoms(ffs)

plot_dcom(files)

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

def plot_df(df, source=CFG.IMAGES_DIR):
    files = [source / str(row.study_id) / str(row.series_id) / f'{row.image}.dcm' for _, row in df.iterrows()]

    plot_dcom(files, title=f'{series_id} {len(files)} images')


# %%
study = 1018005303
df = files_df[files_df['study_id'] == study]

df = df[df['series_description'] == 'Axial T2'].sort_values(['proj'], ascending=[False])[:60]

plot_df(df)

# %%
hfs = files_df[files_df['patientposition'] == 'HFS'].sample(1).study_id.values[0]
ffs = files_df[files_df['patientposition'] == 'FFS'].sample(1).study_id.values[0]

hfs, ffs

# %%
df = files_df[files_df['study_id'] == hfs]

df = df[df['series_description'] == 'Sagittal T1'].sort_values(['proj'], ascending=[False])[:60]

plot_df(df)

# %%
df = files_df[files_df['study_id'] == hfs]

df = df[df['series_description'] == 'Sagittal T1'].sort_values(['proj'], ascending=[True])[:60]

plot_df(df)

# %%
df = files_df[files_df['study_id'] == ffs]

df = df[df['series_description'] == 'Sagittal T1'].sort_values(['proj'], ascending=[False])[:60]

plot_df(df)

# %%
df = files_df[files_df['study_id'] == ffs]

df = df[df['series_description'] == 'Sagittal T1'].sort_values(['proj'], ascending=[True])[:60]

plot_df(df)

# %%
