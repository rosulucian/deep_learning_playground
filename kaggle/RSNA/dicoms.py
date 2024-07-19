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
    IMAGES_DIR = ROOT_FOLDER / 'train_images'
    OUTPUT_DIR = ROOT_FOLDER / f'pngs_{size}'
    FILES_CSV = ROOT_FOLDER / 'train_files.csv'
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

# %%
ds = dicom.dcmread(files[63])

# %%
ds.pixel_array.shape

# %%
print(ds)

# %%
ds.pixel_array.shape, ds.Rows, ds.PhotometricInterpretation, ds.SeriesDescription

# %%
f = files[0]
f.split('\\')

# %%
for f in tqdm(files[65:68]):
    study = f.split('\\')[-2]
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
    patient = f.split('\\')[-3]
    image = f.split('\\')[-1][:-4]

    ds = dicom.dcmread(f)
    img = ds.pixel_array

    img = (img - img.min()) / (img.max() - img.min())

    if ds.PhotometricInterpretation == "MONOCHROME1":
        img = 1 - img

    img = cv2.resize(img, (size, size))

    file_name =  f"{save_folder}\\{patient}_{series}_{image}.{extension}"

    cv2.imwrite(file_name, (img * 255).astype(np.uint8))

    return patient, series, image, ds.Rows, ds.Columns


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
files_df.to_csv(CFG.FILES_CSV, index=False)

# %%

# %%
