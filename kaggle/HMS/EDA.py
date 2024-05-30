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

# %%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# %%
data_dir = Path('D:\data\HMS')

spectro_dir = data_dir / 'train_spectrograms'
eeg_dir = data_dir / 'train_eegs'

data_dir, spectro_dir, eeg_dir

# %%
# data_dir = 'D:\data\HMS'

# results_dir = f'{data_dir}/results'
# ckpt_path = f'{results_dir}/ckpt'

# spect_dir =  f'{data_dir}/train_spectrograms'

# ckpt_path, spect_dir

# %%
train_df = pd.read_csv(f'{data_dir}/train.csv')
train_df.columns

# %%
train_df.shape

# %%
train_df.sample(5)

# %% [markdown]
# ### Consensus

# %%
plt.figure(figsize=(10, 6))
sns.countplot(data=train_df, x='expert_consensus')
plt.title('Distribution of Expert Consensus')
plt.xlabel('Expert Consensus')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# %%
vote_cols = [x for x in train_df.columns if 'vote' in x]
vote_cols

# %%
train_df['total_votes'] = train_df.loc[:, vote_cols].sum(axis=1)
train_df['cons_votes'] = train_df.loc[:, vote_cols].max(axis=1)

# %%
train_df['consensus'] = train_df['cons_votes']/train_df['total_votes']

# %%
train_df['consensus'].hist()

# %%
train_df.loc[train_df['consensus'] > 0.9].shape[0]

# %%
hi = train_df.loc[train_df['consensus'] > 0.9]['eeg_id'].nunique()
lo = train_df.loc[train_df['consensus'] < 0.9]['eeg_id'].nunique()

rate = hi/(hi+lo)

hi, lo, rate

# %% [markdown]
# ### Check values

# %%
train_df.nunique()

# %%
train_df.sample(6)

# %%
train_df.loc[train_df['eeg_id'] == 226568387]

# %%
train_df.loc[train_df['eeg_label_offset_seconds'] <= 0].shape[0], train_df.loc[train_df['eeg_label_offset_seconds'] > 0].shape[0]

# %%
train_df.loc[(train_df['eeg_label_offset_seconds'] < 1000) & (train_df['eeg_label_offset_seconds'] > 0)].eeg_label_offset_seconds.hist(bins=25)

# %%
# train.eeg_label_offset_seconds.hist(bins=25)
train_df.loc[(train_df['eeg_label_offset_seconds'] < 100) & (train_df['eeg_label_offset_seconds'] > 0)].eeg_label_offset_seconds.hist(bins=25)

# %% [markdown]
# ### Plots: EEG&Spectros

# %%
eeg_id = 2444120992
spectrogram_id = train_df[train_df['eeg_id'] == eeg_id].iloc[0].spectrogram_id
eeg_id, spectrogram_id

# %%
eeg = pd.read_parquet(f'{eeg_dir}/{eeg_id}.parquet')
spectro = pd.read_parquet(f'{spectro_dir}/{spectrogram_id}.parquet')
# print(eeg.shape)
print(spectro.shape)

# %%
np.sort(eeg.columns)

# %%
eeg.iloc[:20]

# %%
spectro.iloc[:,:101]

# %%

# %%

# %%

# %%

# %%
