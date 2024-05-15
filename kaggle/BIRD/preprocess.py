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
import cv2

# %%
import os
import shutil
import librosa
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed

# %% [markdown]
# ### Config

# %%
# Input folder
train_dir = Path('E:\data\BirdCLEF')
train_csv = train_dir / 'train_metadata.csv'


# %%
class Config():
    # Horizontal melspectrogram resolution
    MELSPEC_H = 128
    # Competition Root Folder
    ROOT_FOLDER = train_dir
    AUDIO_FOLDER = train_dir / 'train_audio'
    OUTPUT_DIR = train_dir / 'spectros'
    # Maximum decibel to clip audio to
    TOP_DB = 100
    # Minimum rating
    MIN_RATING = 3.0
    # Sample rate as provided in competition description
    SR = 32000
    N_FFT = 2048
    HOP_LENGTH = 512
    
CONFIG = Config()

# %%
meta_df = pd.read_csv(train_csv)
meta_df.head(2)

# %%
sample_submission = pd.read_csv(train_dir / 'sample_submission.csv')

# Set labels
CONFIG.LABELS = sample_submission.columns[1:]
CONFIG.N_LABELS = len(CONFIG.LABELS)
print(f'# labels: {CONFIG.N_LABELS}')

display(sample_submission.head())

# %%
# Maps a class to corresponding integer label
CLASS2LABEL = dict(zip(CONFIG.LABELS, np.arange(CONFIG.N_LABELS)))
# Label to class mapping
LABEL2CLASS = dict([(v,k) for k, v in CLASS2LABEL.items()])

# %%
CONFIG.LABELS


# %% [markdown]
# ### Preprocess scripts

# %%
# A function for processing a single audio file and saving it
def ogg2npy(file_path, destination, sr=32000):
    # Audio loading
    y, sr = librosa.load(file_path, sr=sr)
    
    # Create Mel-Spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    log_mel_spectrogram = librosa.amplitude_to_db(mel_spectrogram)

    # Path to save
    npy_file_path = os.path.join(destination, f'{os.path.splitext(os.path.basename(file_path))[0]}.npy')

    # Save as NPY
    np.save(npy_file_path, log_mel_spectrogram)

def ogg2npy_parallel(files, destination):
    # Parallel using joblib and a progress bar using tqdm
    result = Parallel(n_jobs=8)(
        delayed(ogg2npy)
        (file, destination) 
        for file in tqdm(files)
    )


# %%
# https://www.kaggle.com/code/markwijkhuizen/birdclef-2024-eda-preprocessed-dataset/notebook
def ogg2png(file_name, source=Config.AUDIO_FOLDER):
    file_path = source / file_name
    
    # Load the audio file
    y, _ = librosa.load(file_path, sr=CONFIG.SR)
    # Normalize audio
    y = librosa.util.normalize(y)
    # Convert to mel spectrogram
    spec = librosa.feature.melspectrogram(
        y=y,
        sr=CONFIG.SR, # sample rate
        n_fft=CONFIG.N_FFT, # number of samples in window 
        hop_length=CONFIG.HOP_LENGTH, # step size of window
        n_mels=CONFIG.MELSPEC_H, # horizontal resolution from fmin→fmax in log scale
        fmin=40, # minimum frequency
        fmax=15000, # maximum frequency
        power=2.0, # intensity^power for log scale
    )
    # Convert to Db
    spec = librosa.power_to_db(spec, ref=CONFIG.TOP_DB)
    # Normalize 0-min
    spec = spec - spec.min()
    # Normalize 0-255
    spec = (spec / spec.max() * 255).astype(np.uint8)
    # Convert to PNG bytes
    _, spec_png_uint8 = cv2.imencode('.png', spec)
    spec_png_bytes = bytes(spec_png_uint8)
    
    return file_name, spec_png_bytes


# %%
def ogg2png_parallel(df):
    files = df['filename'].tolist()

    labels = df[['filename', 'primary_label']]

    # Create dataset
    X = {}
    y = {}

    for idx, row in meta_df[['filename', 'primary_label']].iterrows():
        key = row['filename']
        y[key] = CLASS2LABEL.get(row['primary_label'])
    
    # Parallel using joblib and a progress bar using tqdm
    results = Parallel(n_jobs=8)(
        delayed(ogg2png)
        (file) 
        for file in tqdm(files)
    )

    for item in results:
        X[item[0]] = item[1]

    return X, y


# %%

# %% [markdown]
# ### Preprocess

# %%
files = [item for item in Config.AUDIO_FOLDER.rglob('*.ogg') ]
len(files)

# %%
# %%time

# ogg2npy_parallel(files, Config.OUTPUT_DIR) # TOO LARGE
# create_spectros(files, ogg2png, output_folder)

# %%
# %%time
X, y = ogg2png_parallel(meta_df)

# %%

# %% [markdown]
# ### Test

# %%
import imageio.v3 as imageio


# %%
# Example Plots
def example_plots(X, y, N):
    np.random.seed(42)
    random_keys = np.random.choice(list(X.keys()), N)
    
    for k in random_keys:
        spec = imageio.imread(X[k])
        plt.figure(figsize=(12,5))
        plt.title(
                f'Label: {y[k]}, Class: {LABEL2CLASS[y[k]]}, shape: {spec.shape} '
            )
        # plt.imshow(spec)
        librosa.display.specshow(spec, y_axis="mel")
        plt.show()


# %%
example_plots(X, y, 8)

# %% [markdown]
# ### Save data

# %%
import pickle

# %%
# Write X
with open(Config.OUTPUT_DIR / 'X.pkl', 'wb') as f:
    pickle.dump(X, f)
    
# Write y
with open(Config.OUTPUT_DIR / 'y.pkl', 'wb') as f:
    pickle.dump(y, f)

# %%
