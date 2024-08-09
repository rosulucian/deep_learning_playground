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

class CFG:
    random_seed = 42
    
    ROOT_FOLDER = train_dir
    IMAGES_DIR = ROOT_FOLDER / 'train_images'
    TRAIN_CSV = ROOT_FOLDER / 'train.csv'
    FILES_CSV = ROOT_FOLDER / 'train_files.csv'
    TRAIN_DESC_CSV = ROOT_FOLDER / 'train_series_descriptions.csv'
    COORDS_CSV = ROOT_FOLDER / 'train_label_coordinates.csv'



# %%

# %%
train_df = pd.read_csv(CFG.TRAIN_CSV)
train_desc_df = pd.read_csv(CFG.TRAIN_DESC_CSV)

coords_df = pd.read_csv(CFG.COORDS_CSV)
files_df = pd.read_csv(CFG.FILES_CSV)

coords_df.shape, files_df.shape, train_df.shape, train_desc_df.shape

# %%
coords_df.sample(2)

# %% [markdown]
# ### Filter bad labels

# %%
bad_labels = [3819260179, 2444340715] + coords_df[coords_df['x'] < 10].study_id.tolist()

print(bad_labels), len(bad_labels)

# %%
clean_coords_df = coords_df[~coords_df['study_id'].isin(bad_labels)]

coords_df.shape, clean_coords_df.shape

# %%

# %% [markdown]
# ### Condition positions on frame

# %%
clean_coords_df.groupby('series_description').agg({'x_perc': ['min', 'max'],'y_perc': ['min', 'max']})

# %%
# clean_coords_df.groupby('series_description').agg({'y_perc': ['min', 'max']})

# %%
coords_df.iloc[coords_df['y_perc'].argmin()]

# %% [markdown]
# ### Plot conditions

# %%
row = coords_df.iloc[coords_df['y_perc'].argmin()]

row['level']

# %%
coords_df.groupby(['study_id', 'condition']).instance_id.count().mean()


# %%

# %% jupyter={"source_hidden": true}
def plot(row, source=CFG.IMAGES_DIR):
    # filename = row['filename']

    filename = source / str(row['study_id']) / str(row['series_id']) / f'{row["instance"]}.dcm'

    conds = coords_df[coords_df['ss_id'] == row['ss_id']]

    print(row['study_id'])
    print(conds.x.to_list())
    print(conds.cl.to_list())
    print(conds.y.to_list())
    
    ds = dicom.dcmread(filename)
    img = ds.pixel_array

    # img = (img - img.min()) / (img.max() - img.min())

    if ds.PhotometricInterpretation == "MONOCHROME1":
        img = 1 - img

    plt.imshow(img, cmap="gray")
    
    # plt.scatter(row['x'], row['y'], marker="x", color="red", s=200)

    plt.title(row['series_description'] + f' - {row["instance"]}')

    plt.scatter(conds.x.to_list(), conds.y.to_list(), marker="x", color="red", s=200)
    
    plt.show()


# %%
coords_df.sample(2)


# %%
def plot_conditions(study_id, source=CFG.IMAGES_DIR):
    df = coords_df[coords_df['study_id'] == study_id].sort_values(by='series_description', ascending=False)

    ss_ids = df.instance_id.unique()
    imgs = dict(zip(ss_ids, [{'points':[], 'labels':[]} for i in ss_ids]))

    for i in df.instance_id.unique():
        sel = df[df['instance_id'] == i]

        row = sel.iloc[0]
        imgs[i]['filename'] = source / str(row['study_id']) / str(row['series_id']) / f'{row["instance"]}.dcm'
        imgs[i]['title'] = ('-').join([row.series_description] + sel.condition.unique().tolist())

        for index, row in sel.iterrows():
            imgs[i]['points'].append((row.x, row.y, row.condition))

    rows = len(imgs.keys()) // 4 + 1
    fig, axs = plt.subplots(rows, 4, figsize=(15, rows*3))

    fig.suptitle(study_id)

    axs = axs.flat

    for i, (key, value) in enumerate(imgs.items()):
        ax = axs[i]

        ax.margins(0, 0)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.title.set_text(value['title'])
        
        ds = dicom.dcmread(value['filename'])
        img = ds.pixel_array

        # img = (img - img.min()) / (img.max() - img.min())

        if ds.PhotometricInterpretation == "MONOCHROME1":
            img = 1 - img

        ax.imshow(img, cmap="gray")

        for p in value['points']:
            ax.scatter(p[0], p[1], label=p[2], marker="x", color="red", s=200)
    
    plt.show()


# %%
plot_conditions(1820866003)

# %% [markdown]
# #### Plot min x,y labels

# %%
# bad labels
# bad_labels = [1820866003, 665627263, 2151509334, 1901348744, 286903519, 1880970480, 2151467507, 2905025904]

bad_labels = coords_df[coords_df['x_perc'] < 0.15].study_id.tolist()

bad_labels = coords_df[coords_df['x'] < 10].study_id.tolist()

selection = coords_df[~coords_df['study_id'].isin(bad_labels)]
row = selection.iloc[selection['y_perc'].argmax()]

print(row.study_id)

# plot(row)
plot_conditions(row.study_id)

# %%
coords_df[coords_df['x'] < 10].shape, coords_df[coords_df['y'] < 10].shape

# %% [markdown]
# #### Plot max x,y labels

# %%
# bad labels
# bad_labels = [1820866003, 665627263, 2151509334, 1901348744, 286903519, 1880970480, 2151467507, 2905025904]

# bad_labels = [3819260179]

# bad_labels = coords_df[coords_df['x'] < 10].study_id.tolist()

selection = coords_df[~coords_df['study_id'].isin(bad_labels)]
row = selection.iloc[selection['y_perc'].argmax()]

print(row.study_id)

# plot(row)
plot_conditions(row.study_id)

# %%
# bad labels
# bad_labels = [1820866003, 665627263, 2151509334, 1901348744, 286903519, 1880970480, 2151467507, 2905025904]

bad_labels = [3819260179, 2444340715]

# bad_labels = coords_df[coords_df['x'] < 10].study_id.tolist()

selection = coords_df[~coords_df['study_id'].isin(bad_labels)]
row = selection.iloc[selection['y_perc'].argmax()]

# plot(row)
plot_conditions(row.study_id)

# %%

# %%

# %% [markdown]
# ### Right/left orientation

# %%
selection = coords_df[coords_df['condition'] == 'RSS']

selection.x_perc.hist(bins=200)

# %%
selection.patientposition.value_counts()

# %%
selection[selection['x_perc'] > 0.50].patientposition.value_counts()

# %%
selection[selection['x_perc'] < 0.45].patientposition.value_counts()

# %%
selection = coords_df[coords_df['condition'] == 'LSS']

selection.x_perc.hist(bins=200)

# %%
selection[selection['x_perc'] < 0.45].patientposition.value_counts()

# %%
selection[selection['x_perc'] < 0.45].study_id.unique()

# %%

# %% [markdown]
# ### Frame positions in series

# %%

# %%
coords_df.sample(2)

# %%

# %%
