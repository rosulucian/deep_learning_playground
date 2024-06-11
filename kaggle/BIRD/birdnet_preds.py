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

# %% [markdown]
# ### Config

# %%
train_dir = Path('E:\data\BirdCLEF')

class CFG:
    random_seed = 42

    include_ul = True
    
    split_fraction = 0.95
    
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
print(meta_df.filename.duplicated().sum())
meta_df['file'] = meta_df.apply(lambda row: row.filename.split('/')[-1], axis=1)

# %%
all_files = meta_df.filename.unique().tolist()
len(all_files)

# %%
# durations = {}
# for f in all_files:
#     filename = CFG.AUDIO_FOLDER / f
#     info = torchaudio.info(filename)
#     duration = int(info.num_frames / info.sample_rate )

#     durations[f] = duration

# meta_df['duration'] = meta_df.apply(lambda row: durations[row['filename']], axis=1)

# meta_df.to_csv(CFG.TRAIN_CSV, index=False) 

# %%
meta_df[meta_df['duration'] > 60].shape, meta_df['duration'].max()

# %%
meta_df[meta_df['duration'] < 7].shape, meta_df[meta_df['duration'] < 7].primary_label.nunique()

# %%
meta_df[(meta_df['duration'] < 7) & (meta_df['secondary_labels'] == '[]')].primary_label.nunique()

# %%
meta_df[meta_df['duration'] < 7].sample(2)

# %%
meta_df.head(2)

# %%
taxonomy_df.head(2)

# %%
all_birds = taxonomy_df.SCI_NAME.unique().tolist()
bird_codes = taxonomy_df.SPECIES_CODE.unique().tolist()

sci2code = {b: c for b, c in zip(all_birds, bird_codes)}
code2sci = {c:b for b, c in zip(all_birds, bird_codes)}

# %%
sci2code['Struthio camelus'], code2sci['ostric2']

# %%

# %% [markdown]
# ### BirdNet train predictions

# %%
bird_preds_df.sample(2)

# %%
# bird_preds_df['path'] = fr'{str(CFG.AUDIO_FOLDER)}/' + bird_preds_df['label'] + '/' + bird_preds_df['filename']

# %%
# bird_preds_df[bird_preds_df['filename'] == 'XC496054.ogg']

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
ood_list = bird_preds_df[bird_preds_df['ood'] == True].pred_code.unique().tolist()
len(ood_list)

# %%
bird_preds_df[bird_preds_df['duration'] != -1]

# %%
bird_preds_df.sample(2)

# %%
bird_preds_df.label.value_counts()[:15]

# %% [markdown]
# ### Unlabeled soundscapes

# %%
unlabeled_preds_df.sample(2)

# %%
unlabeled_preds_df['pred_code'] = unlabeled_preds_df.apply(lambda row: sci2code[row['name']] if row['name'] in sci2code.keys() else '', axis=1)
unlabeled_preds_df['label'] = unlabeled_preds_df['pred_code']
unlabeled_preds_df['ood'] = unlabeled_preds_df.apply(lambda row: False if row['pred_code'] in bird2id.keys() else True, axis=1)

# %%
unlabeled_preds_df['path'] = fr'{str(CFG.UNLABELED_FOLDER)}/' + unlabeled_preds_df['filename']

# %%
unlabeled_preds_df.sample(2)

# %%
unlabeled_preds_df.shape, unlabeled_preds_df[unlabeled_preds_df['pred_code'] == ''].shape

# %%
unlabeled_preds_df[unlabeled_preds_df['ood'] == False].shape

# %%

# %%
unlabeled_preds_df[unlabeled_preds_df['ood'] == False].sample(4)

# %%

# %% [markdown]
# ### Low freq classes

# %%
labels_pred = bird_preds_df.pred_code.unique().tolist()

missing = list(set(CFG.LABELS) - set(labels_pred))
extra = list(set(labels_pred) - set(CFG.LABELS))

len(labels_pred), len(missing), len(extra)

# %%
labels_pred = bird_preds_df[bird_preds_df['ood'] == False].pred_code.unique().tolist()

extra = list(set(labels_pred) - set(CFG.LABELS))
len(extra)

# %%
missing

# %%
'darter2' in labels_pred

# %%
counts = bird_preds_df.label.value_counts()
# bird_preds_df.label.isin(counts[counts < 1000]).index.tolist()
counts[counts > 1000].count(), counts[counts < 50].count(), counts.count()

# %%
counts = bird_preds_df.pred_code.value_counts()
counts[counts > 1000].count(), counts[counts < 50].count(), counts[counts < 10].count(), len(CFG.LABELS)

# %%
ood_df = bird_preds_df[bird_preds_df['ood'] == False]
counts = ood_df.pred_code.value_counts()
counts[counts > 1000].count(), counts[counts < 50].count(), counts[counts < 10].count(), len(CFG.LABELS)

# %%
bird_preds_df.shape,  ood_df.shape

# %%
# bird_preds_df.label.value_counts()[-20:]

# %%
bird_preds_df[bird_preds_df['filename'] == 'XC49755.ogg']

# %%
bird_preds_df[bird_preds_df['label'] != bird_preds_df['pred_code']]

# %%
# meta_df[meta_df['secondary_labels'] == '[]'].primary_label.value_counts()

# %% [markdown]
# ### Post process predictions

# %%
from utils import cat_feature_dist, downsample_data, upsample_data

# %%
cat_feature_dist(bird_preds_df, 'label')

# %%
cat_feature_dist(bird_preds_df, 'pred_code')

# %%
preds = bird_preds_df[bird_preds_df['ood'] == False]
preds.shape, bird_preds_df.shape

# %%
# Upsample data
up_thr = 100
# pre downsample for graphical reasons, remove very high classes
pre_df = bird_preds_df[bird_preds_df['ood'] == False]
pre_df = downsample_data(pre_df, 'pred_code', thr=2000, seed=CFG.random_seed)
dn_df = downsample_data(pre_df, 'pred_code', thr=500, seed=CFG.random_seed)
up_df = upsample_data(dn_df, 'pred_code', thr=up_thr, seed=CFG.random_seed)

print("# Pretraing Dataset")
print(f'> Original: {len(pre_df)}')
print(f'> After Upsample: {len(up_df)}')
print(f'> After Downsample: {len(dn_df)}')

# Show effect of upsample
plt.figure(figsize=(12*2, 6))

ax1 = plt.subplot(1, 2, 1)
pre_df.pred_code.value_counts()[:].plot.bar(color='blue', label='original')
up_df.pred_code.value_counts()[:].plot.bar(color='green', label='w/ upsample')
dn_df.pred_code.value_counts()[:].plot.bar(color='red', label='w/ dowsample')
plt.xticks([])
plt.axhline(y=up_thr, color='g', linestyle='--', label='up threshold')
plt.axhline(y=400, color='r', linestyle='--', label='down threshold')
plt.legend()
plt.title("Upsample for Pre-Training")

# plt.tight_layout()
plt.show()

# %% [markdown]
# ### Preprocess training data

# %%
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

# %%
sss = StratifiedShuffleSplit(n_splits=1, test_size=1-CFG.split_fraction, random_state=CFG.random_seed)
train_idx, val_idx = next(sss.split(meta_df.filename, meta_df.primary_label))

t_df = meta_df.iloc[train_idx]
v_df = meta_df.iloc[val_idx]

t_df.shape, v_df.shape

# %%
files = t_df.filename.tolist()
v_df.apply(lambda row: row['filename'] in files, axis=1).sum()

# %%
# Use only classes that are in distribution (western ghats)
df = bird_preds_df[bird_preds_df['ood'] == False]
df.shape, bird_preds_df.shape

# %%
labels_pred = bird_preds_df.pred_code.unique().tolist()

missing = list(set(CFG.LABELS) - set(labels_pred))
extra = list(set(labels_pred) - set(CFG.LABELS))

len(labels_pred), len(missing), len(extra)

# %%
# missing labels
missing

# %%
# underrepressented labels
counts = df.pred_code.value_counts()
labels_under = counts[counts < 10]

counts[counts < 50].count(), labels_under.count(), labels_under.index.tolist()

# %%
missing += labels_under.index.tolist()
len(missing), missing

# %%
df.head(2)

# %%
# meta_df.head(2)

# %%
results = []
soft_label = 0.9
frame = 3 # seconds

for label in missing:
    files = meta_df[meta_df['primary_label'] == label].filename.tolist()
    
    for f in files:
        path = CFG.AUDIO_FOLDER / f
        wav, org_sr = torchaudio.load(path)

        duration = int(wav.shape[1] / org_sr)

        steps = list(range(0, duration, frame))

        filename = f.split('/')[-1] 

        intervals = [(x,x+frame) for x in steps[:-1]]
        
        data = [(filename, label, code2sci[label], i[0], i[1], soft_label, label, False) for i in intervals]
        results.append(pd.DataFrame(data, columns = ['filename', 'label', 'name', 'start', 'end', 'confidence', 'pred_code', 'ood']))

    # break
len(results), df.isnull().values.any()

# %%
# for r in results:
#    print (r.isnull().values.any())

# %%
# merge dfs and upsample
results_df = pd.concat(results, axis=0)
print(results_df.shape)

results_df.sample(2)

# %%
df.head(2)

# %%
df = pd.concat([df,results_df], axis=0)
df.shape

# %%
df['start'] = df['start'].astype(int)
df['end'] = df['end'].astype(int)

df['path'] = fr'{str(CFG.AUDIO_FOLDER)}/' + df['label'] + '/' + df['filename']

# %%
df.head(2)

# %%
labels_pred = bird_preds_df.pred_code.unique().tolist()

missing = list(set(CFG.LABELS) - set(labels_pred))
extra = list(set(labels_pred) - set(CFG.LABELS))

len(labels_pred), len(missing), len(extra)

# %%
# check missing again
labels_pred = df.pred_code.unique().tolist()

missing = list(set(CFG.LABELS) - set(labels_pred))
extra = list(set(labels_pred) - set(CFG.LABELS))

len(labels_pred), len(missing), len(extra)

# %% [markdown]
# ### Filter preds based on files from split

# %%
train_files = t_df.file.unique().tolist()
val_files = v_df.file.unique().tolist()

len(train_files), len(val_files)

# %%
train_df = df[df.filename.isin(train_files)]
val_df = df[df.filename.isin(val_files)]

train_df.shape, val_df.shape, df.shape

# %% [markdown]
# ### Add unlabeled

# %%
unl_df = unlabeled_preds_df[unlabeled_preds_df['ood'] == False]
unl_df.shape

# %%
# unl_df['label'] = unl_df['pred_code']

# %%
train_df.head(2)

# %%
unl_df.head(2)

# %%
print(train_df.shape)

if CFG.include_ul:
    train_df = pd.concat([train_df, unl_df], axis=0)

print(train_df.shape)

# %%

# %%

# %% [markdown]
# ### Up/down sample train dataset

# %%
# Upsample data

# label = 'label'
label = 'pred_code'

up_thr = 150
down_thr = 400
# pre downsample for graphical reasons, remove very high classes
# pre_df = bird_preds_df[bird_preds_df['ood'] == False]
pre_df = train_df
pre_df = downsample_data(pre_df, label, thr=2000, seed=CFG.random_seed)
dn_df = downsample_data(pre_df, label, thr=down_thr, seed=CFG.random_seed)
up_df = upsample_data(dn_df, label, thr=up_thr, seed=CFG.random_seed)

print("# Pretraing Dataset")
print(f'> Original: {len(pre_df)}')
print(f'> After Upsample: {len(up_df)}')
print(f'> After Downsample: {len(dn_df)}')

# Show effect of upsample
plt.figure(figsize=(12*2, 6))

ax1 = plt.subplot(1, 2, 1)
pre_df[label].value_counts()[:].plot.bar(color='blue', label='original')
up_df[label].value_counts()[:].plot.bar(color='green', label='w/ upsample')
dn_df[label].value_counts()[:].plot.bar(color='red', label='w/ dowsample')
plt.xticks([])
plt.axhline(y=up_thr, color='g', linestyle='--', label='up threshold')
plt.axhline(y=400, color='r', linestyle='--', label='down threshold')
plt.legend()
plt.title("Upsample for Pre-Training")

# plt.tight_layout()
plt.show()

# %%
up_df.shape, val_df.shape

# %%
# Save datasets
up_df.to_csv(train_dir / "train_set.csv", index=False) 
val_df.to_csv(train_dir / "val_set.csv", index=False) 

# %%

# %%
