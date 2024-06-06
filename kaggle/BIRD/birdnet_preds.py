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

    bird_preds_csv = train_dir / 'bird_preds.csv'
    unlabeled_preds_csv = train_dir / 'unlabeled_preds.csv'
    taxonomy_csv = train_dir / 'eBird_Taxonomy_v2021.csv'


# %%
sec_labels = ['lotshr1', 'orhthr1', 'magrob', 'indwhe1', 'bltmun1', 'asfblu1']

sample_submission = pd.read_csv(train_dir / 'sample_submission.csv')

# Set labels
CFG.LABELS = sample_submission.columns[1:].tolist()
bird2id = {b: i for i, b in enumerate(CFG.LABELS)}

len(CFG.LABELS)

# %%
meta_df = pd.read_csv(CFG.TRAIN_CSV)
df_23 = pd.read_csv(CFG.bird20223)
bird_preds_df = pd.read_csv(CFG.bird_preds_csv)
unlabeled_preds_df = pd.read_csv(CFG.unlabeled_preds_csv)
taxonomy_df = pd.read_csv(CFG.taxonomy_csv)

df_23.shape, bird_preds_df.shape, unlabeled_preds_df.shape, taxonomy_df.shape

# %%
taxonomy_df.head(2)

# %%
all_birds = taxonomy_df.SCI_NAME.unique().tolist()
bird_codes = taxonomy_df.SPECIES_CODE.unique().tolist()

sci2code = {b: c for b, c in zip(all_birds, bird_codes)}


# %%
sci2code['Struthio camelus']

# %%

# %% [markdown]
# ### BirdNet train predictions

# %%
bird_preds_df.sample(4)

# %%
bird_preds_df['pred_code'] = bird_preds_df.apply(lambda row: sci2code[row['name']] if row['name'] in sci2code.keys() else '', axis=1)
bird_preds_df['ood'] = bird_preds_df.apply(lambda row: False if row['pred_code'] in bird2id.keys() else True, axis=1)

# %%
bird_preds_df.shape, bird_preds_df[bird_preds_df['pred_code'] == ''].shape

# %%
bird_preds_df[bird_preds_df['ood'] == True].shape

# %%
bird_preds_df.sample(2)

# %%
bird_preds_df.label.value_counts()[:15]

# %% [markdown]
# ### low freq classes

# %%
bird_preds_df.label.value_counts()[-10:]

# %%
meta_df[meta_df['secondary_labels'] == '[]'].primary_label.value_counts()[-10:]

# %%

# %%

# %%

# %%

# %%
