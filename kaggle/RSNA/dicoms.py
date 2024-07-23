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
import os
import cv2
import shutil

import numpy as np
import pandas as pd
import pydicom as dicom
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm.notebook import tqdm
from joblib import Parallel, delayed

# %%
train_dir = Path('E:\data\RSNA2024')

class CFG:
    random_seed = 42

    size = 256
    
    ROOT_FOLDER = train_dir
    DEST_FOLDER = train_dir / 'original'
    IMAGES_DIR = ROOT_FOLDER / 'train_images'
    OUTPUT_DIR = ROOT_FOLDER / f'pngs_{size}'
    FILES_CSV = DEST_FOLDER / 'train_files.csv'
    TRAIN_CSV = ROOT_FOLDER / 'train.csv'
    TRAIN_DESC_CSV = ROOT_FOLDER / 'train_series_descriptions.csv'
    COORDS_CSV = ROOT_FOLDER / 'train_label_coordinates.csv'


# %%
CFG.OUTPUT_DIR

# %%
files = [str(item) for item in CFG.IMAGES_DIR.rglob('*.dcm') ]
len(files)

# %%
files[1000]

# %% [markdown]
# ### Get metadata

# %%
ds = dicom.dcmread(files[63])

# %%
ds.pixel_array.shape

# %%
print(ds)

# %%
ds.get_item('PhotometricInterpretation').keyword, ds.get_item('PhotometricInterpretation').tag

# %%
for i in ds.items():
    # print(i[1])
    if i[1].keyword == 'PixelData':
        continue
    print(i[1].name, i[1].keyword, i[1].value)

# %%
ds.get('SeriesDescription')

# %%
type(ds)

# %%
ds.pixel_array.shape, ds.Rows, ds.PhotometricInterpretation, ds.SeriesDescription

# %%
len(files)

# %%
# train_desc_df = pd.read_csv(CFG.TRAIN_DESC_CSV)
# # train_desc_df['folders'] = train_desc_df['study_id'] + train_desc_df['series_id']
# train_desc_df['filename'] = train_desc_df.apply(lambda row: f'{CFG.IMAGES_DIR}\\{row.study_id}\\{row.series_id}', axis=1)

# train_desc_df.shape

# %%
# train_desc_df.iloc[0]

# %%
# folders = train_desc_df.filename.to_list()

# len(folders)

# %%
# folder = Path(folders[0])
# files = [str(item) for item in folder.rglob('*.dcm') ]
# len(files)

# %%
# [f.split('\\')[-1][:-4] for f in files]

# %%
keys = ['InstanceNumber', 'Rows', 'Columns', 'SliceThickness', 'SpacingBetweenSlices', 'PatientPosition', 'SeriesDescription']

def process(f, size=CFG.size, keys=keys):
    series = f.split('\\')[-2]
    study = f.split('\\')[-3]
    image = f.split('\\')[-1][:-4]
    
    ds = dicom.dcmread(f)
    
    values = []

    for k in keys:
        values.append(ds.get(k))

    values = [study, series, image] + values
    
    return tuple(values)


# %%
data = Parallel(n_jobs=16)(
    delayed(process)(f)
    # for f in tqdm(files[:24])
    for f in tqdm(files)
)

# %%
len(files)

# %%
data[:2]

# %%
columns = [k.lower() for k in keys]
columns = ['study_id', 'series_id', 'image'] + columns

columns

# %%
files_df = pd.DataFrame(data, columns=columns)
files_df.shape

# %%
files_df.sample(5)

# %%
files_df.to_csv(CFG.FILES_CSV, index=False)

# %% [markdown]
# ### Convert files to png

# %%
f = files[0]
f.split('\\')

# %%
for f in tqdm(files[65:68]):
    series = f.split('\\')[-2]
    patient = f.split('\\')[-3]
    image = f.split('\\')[-1][:-4]
    print(patient, image)

    ds = dicom.dcmread(f)
    img = ds.pixel_array

    img = (img - img.min()) / (img.max() - img.min())

    if ds.PhotometricInterpretation == "MONOCHROME1":
        img = 1 - img
        
    plt.figure(figsize=(5, 5))
    plt.imshow(img[:100], cmap="gray")
    plt.title(f"{patient} {image}")
    plt.show()

# %%
shutil.rmtree(str(CFG.OUTPUT_DIR))
os.makedirs(str(CFG.OUTPUT_DIR), exist_ok=True)


# %%
def process(f, size=CFG.size, save_folder=str(CFG.OUTPUT_DIR), extension="png"):
    series = f.split('\\')[-2]
    study = f.split('\\')[-3]
    image = f.split('\\')[-1][:-4]

    ds = dicom.dcmread(f)
    img = ds.pixel_array

    img = (img - img.min()) / (img.max() - img.min())

    if ds.PhotometricInterpretation == "MONOCHROME1":
        img = 1 - img

    img = cv2.resize(img, (size, size))

    file_name =  f"{save_folder}\\{study}_{series}_{image}.{extension}"

    cv2.imwrite(file_name, (img * 255).astype(np.uint8))

    return study, series, image, ds.Rows, ds.Columns


# %%
# process(files[0])

# %%
data = Parallel(n_jobs=16)(
    delayed(process)(f)
    # for f in tqdm(files[:24])
    for f in tqdm(files)
)

# %%
data[:10]

# %%
# list(zip(*foo))

# %%
files_df = pd.DataFrame(data, columns=['study_id', 'series_id', 'instance', 'rows', 'columns'])
files_df.shape

# %%
# files_df.to_csv(CFG.FILES_CSV, index=False)

# %%

# %%
