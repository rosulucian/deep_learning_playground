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
import librosa
import librosa.display

import seaborn as sns
import soundfile as sf
import plotly.express as px
import matplotlib.pyplot as plt

from pathlib import Path
from IPython.display import Audio

from utils import plot_energy, plot_spectrogram

# %%
train_dir = Path('E:\data\BirdCLEF')
train_audio = train_dir / 'train_audio'
train_csv = train_dir / 'train_metadata.csv'
taxonomy_csv = train_dir / 'eBird_Taxonomy_v2021.csv'

# %% [markdown]
# ### Metadata

# %%
taxonomy_df = pd.read_csv(taxonomy_csv)
meta_df = pd.read_csv(train_csv)

meta_df.shape, taxonomy_df.shape

# %%
meta_df.head()

# %%
taxonomy_df.head()

# %%
taxonomy_df['SPECIES_CODE'].count()

# %%
taxonomy_df[taxonomy_df['SPECIES_CODE'] == 'integr']

# %%
meta_df.primary_label.value_counts()

# %%
meta_df[meta_df['primary_label'] == 'nocall']

# %%
meta_df.filename[0]

# %%
# import plotly.io as pio
# pio.renderers

# %%
file_path = train_audio / meta_df.filename[0]

# %%
data, rate = torchaudio.load(file_path)
print("Audio data shape:", data.shape)
print("Sample rate:", rate)

# %%
display(Audio(data[0, :rate*5], rate=rate))
# display(Audio(data[0], rate=rate))
px.line(y=data[0, :rate*5], title=meta_df.common_name[0])
# fig.show()

# %%
file_path = train_audio / meta_df.filename[0]

# %%
data, rate = librosa.load(file_path, sr=32000)
energy = librosa.feature.rms(y=data)

# %%
data.shape, energy.shape, rate

# %%
stft = librosa.stft(data[:5*rate])
stft.shape

# %%
librosa.times_like(data, sr=rate)[-1], librosa.times_like(energy).shape

# %%
plt.figure(figsize=(14, 5))
# plt.plot(librosa.times_like(data, sr=rate), data)
# plt.plot(librosa.samples_like(data), data)
plt.plot(data)
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.title('Waveform')
plt.show()

# %%
plot_energy(data, rate)

# %% [markdown]
# ### Spectrograms

# %%
stft = librosa.stft(data)
stft.shape

# %%
stft = librosa.stft(data[:5*rate])
librosa.get_duration(S=stft, sr=rate)

# %%
D = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)
D.shape

# %%
plot_spectrogram(data[:5*rate], rate, feature='linear')

# %%
plot_spectrogram(data, rate, feature='mel')

# %%
S = librosa.feature.melspectrogram(
        y=data,
        sr=rate,
    )
    
D = librosa.power_to_db(S ** 2, ref=np.max)

# %%
fig, ax = plt.subplots()
img = librosa.display.specshow(D, x_axis='time',
                         y_axis='mel', sr=rate,
                         fmax=8000, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Mel-frequency spectrogram')
plt.show()

# %%
plot_spectrogram(data, rate)

# %%
for index, row in meta_df.sample(5).iterrows():
    data, rate = librosa.load(train_audio / row.filename)
    plot_energy(data, rate)
    print(row['primary_label'], train_audio / row.filename)

# %%

# %%

# %%
