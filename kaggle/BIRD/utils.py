import torch
import librosa
import torchaudio

import numpy as np
import pandas as pd

import plotly.express as px
import matplotlib.pyplot as plt

def read_wav(path, sr, frame_offset=0, num_frames=-1):
    wav, org_sr = torchaudio.load(path, normalize=True, frame_offset=frame_offset, num_frames=num_frames)
    
    wav = torchaudio.functional.resample(wav, orig_freq=org_sr, new_freq=sr)
    
    return wav

def crop_wav(wav, start, duration):
    while wav.size(-1) < duration:
        wav = torch.cat([wav, wav], dim=1)
    
    wav = wav[:, start:start+duration]

    return wav

def normalize_melspec(X, eps=1e-6):
    mean = X.mean((1, 2), keepdim=True)
    std = X.std((1, 2), keepdim=True)
    Xstd = (X - mean) / (std + eps)

    norm_min, norm_max = (
        Xstd.min(-1)[0].min(-1)[0],
        Xstd.max(-1)[0].max(-1)[0],
    )
    fix_ind = (norm_max - norm_min) > eps * torch.ones_like(
        (norm_max - norm_min)
    )
    V = torch.zeros_like(Xstd)
    if fix_ind.sum():
        V_fix = Xstd[fix_ind]
        norm_max_fix = norm_max[fix_ind, None, None]
        norm_min_fix = norm_min[fix_ind, None, None]
        V_fix = torch.max(
            torch.min(V_fix, norm_max_fix),
            norm_min_fix,
        )
        V_fix = (V_fix - norm_min_fix) / (norm_max_fix - norm_min_fix)
        V[fix_ind] = V_fix
    return V


def plot_energy(data, rate, figsize=(14,5)):
    energy = librosa.feature.rms(y=data)
    
    fig = plt.figure(figsize=figsize)
    
    plt.plot(librosa.samples_like(energy), energy[0] * 10)
    plt.plot(data)

    ax = fig.gca()
    ax.set_xticks(np.arange(0, data.shape[0], rate*5))
    
    plt.xlabel('samples')
    plt.ylabel('Energy')
    plt.title('Energy Plot')
    plt.grid()
    plt.show()

def plot_spectrogram(signal, sr, feature='linear', n_fft=2048, hop_length=512, win_length=None, window='hann', **kwargs):

    if feature == 'linear':
        S = librosa.amplitude_to_db(np.abs(librosa.stft(signal)), ref=np.max)
    elif feature == 'mel':
        S = librosa.feature.melspectrogram(
            y=signal,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            **kwargs,
        )
        
        S = librosa.power_to_db(S, ref=np.max)
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S, y_axis=feature)
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"{feature.capitalize()}-scaled spectrogram")
    plt.show()

def cat_feature_dist(data, feature):
    # Count unique values
    value_counts = data[feature].value_counts().sort_values(ascending=False)
    
    # Plot
    fig = px.bar(y=value_counts.index[::-1], x=value_counts[::-1], orientation='h')
    fig.update_yaxes(title='')
    fig.update_xaxes(title_text='Count')
    fig.update_layout(
        showlegend=False, 
        plot_bgcolor='#1C1D20', 
        paper_bgcolor='#1C1D20',
        font=dict(size=16, color='#E1B12D'),
        title_font=dict(size=20, color='#222'),
        barmode='group',  
        title=f"Distribution of '{feature}'"
    )
    fig.show()
    print(f"\nTotal unique values in '{feature}'are:",data[feature].nunique())
    print("\nTop 5 values:", value_counts.head())
    print("\nBottom 5 values:", value_counts.tail())

def upsample_data(df, label, thr=20, seed=42):
    # get the class distribution
    class_dist = df[label].value_counts()

    # identify the classes that have less than the threshold number of samples
    down_classes = class_dist[class_dist < thr].index.tolist()

    # create an empty list to store the upsampled dataframes
    up_dfs = []

    # loop through the undersampled classes and upsample them
    for c in down_classes:
        # get the dataframe for the current class
        class_df = df.query(f'{label}==@c')
        # find number of samples to add
        num_up = thr - class_df.shape[0]
        # upsample the dataframe
        class_df = class_df.sample(n=num_up, replace=True, random_state=seed)
        # append the upsampled dataframe to the list
        up_dfs.append(class_df)

    # concatenate the upsampled dataframes and the original dataframe
    up_df = pd.concat([df] + up_dfs, axis=0, ignore_index=True)
    
    return up_df

def downsample_data(df, label, thr=500, seed=42):
    # get the class distribution
    class_dist = df['pred_code'].value_counts()
    
    # identify the classes that have less than the threshold number of samples
    up_classes = class_dist[class_dist > thr].index.tolist()

    # create an empty list to store the upsampled dataframes
    down_dfs = []

    # loop through the undersampled classes and upsample them
    for c in up_classes:
        # get the dataframe for the current class
        class_df = df.query(f'{label}==@c')
        # Remove that class data
        df = df.query(f'{label}!=@c')
        # upsample the dataframe
        class_df = class_df.sample(n=thr, replace=False, random_state=seed)
        # append the upsampled dataframe to the list
        down_dfs.append(class_df)

    # concatenate the upsampled dataframes and the original dataframe
    down_df = pd.concat([df] + down_dfs, axis=0, ignore_index=True)
    
    return down_df

