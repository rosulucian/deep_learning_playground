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

from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from birdnetlib.batch import DirectoryMultiProcessingAnalyzer

# %%
train_dir = Path('E:\data\BirdCLEF')

class CFG:
    random_seed = 42
    
    ROOT_FOLDER = train_dir
    AUDIO_FOLDER = train_dir / 'train_audio'
    DATA_DIR = train_dir / 'spectros'
    TRAIN_CSV = train_dir / 'train_metadata.csv'
    RESULTS_DIR = train_dir / 'results'
    CKPT_DIR = RESULTS_DIR / 'ckpt'
    bird20223 = train_dir / 'bird2023.csv'
    UNLABELED_FOLDER = train_dir / 'unlabeled_soundscapes'


# %%
meta_df = pd.read_csv(CFG.TRAIN_CSV)
df_23 = pd.read_csv(CFG.bird20223)
df_23.shape, df_23.shape

# %%
directories = meta_df.primary_label.unique().tolist()
directories = [str(CFG.AUDIO_FOLDER / d) for d in directories]
len(directories), directories[0]

# %%
meta_df['filename'] = f'{str(CFG.AUDIO_FOLDER)}\\' + meta_df['filename']

# %%
meta_df.head(2)

# %%
prim_df = meta_df[meta_df['secondary_labels'] == '[]']
prim_df.shape

# %%
meta_df[meta_df['primary_label'] == 'zitcis1']

# %% [markdown]
# ### Analyze

# %%
prim_df.iloc[0]

# %%
filename = prim_df.iloc[0].filename
filename = r"E:\data\BirdCLEF\unlabeled_soundscapes\1225526.ogg"
filename

# %%
# Load and initialize the BirdNET-Analyzer models.
analyzer = Analyzer()

recording = Recording(
    analyzer,
    filename,
    # lat=35.4244,
    # lon=-120.7463,
    # date=datetime(year=2022, month=5, day=10), # use date or week_48
    min_conf=0.6,
)
recording.analyze()
len(recording.detections)

# %%
recording.detections

# %%
recording.path, recording.path.split('/')[0].split('\\')[-1]

# %%
label = recording.path.split('/')[0].split('\\')[-1]

# for det in recording.detections:
#     print(det['start_time'])
data = [(label, x['start_time'], x['end_time']) for x in recording.detections]

data[0]

# %%
recording.duration


# %%
def on_analyze_directory_complete(recordings, file=train_dir / "bird_preds.csv"):
    preds = []
    
    for rec in recordings:
        if rec.error:
            print(f'{rec.error_message} in {rec.path}')
        else:
            filename= rec.path.split('\\')[-1]
            label = rec.path.split('\\')[-2]
            
            # print(filename, label)
            
            data = [(filename, label, x['scientific_name'], x['start_time'], x['end_time'], x['confidence']) for x in rec.detections]
            preds.append(pd.DataFrame(data, columns = ['filename', 'label', 'name', 'start', 'end', 'confidence']))

    print(len(preds))

    results_df = pd.concat(preds, axis=0)

    results_df.to_csv(file, index=False)    
    
    # return preds


# %%
directory = directories[1]
directory = CFG.AUDIO_FOLDER
print(directory)

batch = DirectoryMultiProcessingAnalyzer(
    directory,
    analyzers=[analyzer],
    patterns=["*/*.ogg"],
    # lon=-120.7463,
    # lat=35.4244,
    # # date=datetime(year=2022, month=5, day=10),
    min_conf=0.8,
)

batch.on_analyze_directory_complete = on_analyze_directory_complete

# %%
batch.process()


# %%

# %% [markdown]
# ### Predict unlabeled

# %%
def unlabeled_complete(recordings, file=train_dir / "unlabeled_preds.csv"):
    preds = []
    
    for rec in recordings:
        if rec.error:
            print(f'{rec.error_message} in {rec.path}')
        else:
            filename = rec.path.split('\\')[-1]
            
            # print(filename)
            print(len(rec.detections))
            
            data = [(filename, x['scientific_name'], x['start_time'], x['end_time'], x['confidence']) for x in rec.detections]
            preds.append(pd.DataFrame(data, columns = ['filename', 'name', 'start', 'end', 'confidence']))


    print(len(preds))

    results_df = pd.concat(preds, axis=0)

    results_df.to_csv(file, index=False)  

# %%
directory = CFG.UNLABELED_FOLDER

batch = DirectoryMultiProcessingAnalyzer(
    directory,
    analyzers=[analyzer],
    patterns=["*.ogg"],
    # patterns=["460830.ogg"],
    # lon=10.1667,
    # lat=77.0667,
    # # date=datetime(year=2022, month=5, day=10),
    min_conf=0.75,
)

batch.on_analyze_directory_complete = unlabeled_complete

# %%
batch.process()

# %%

# %%
