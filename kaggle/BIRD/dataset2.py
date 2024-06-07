import torch
import imageio
import random
import torchaudio

import numpy as np
import pandas as pd
import albumentations as A

from pathlib import Path
from albumentations.pytorch import ToTensorV2

sample_submission = pd.read_csv('E:\data\BirdCLEF\sample_submission.csv')

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

# Set labels
sec_labels = ['lotshr1', 'orhthr1', 'magrob', 'indwhe1', 'bltmun1', 'asfblu1']
target_columns = sample_submission.columns[1:].tolist()
num_classes = len(target_columns)
bird2id = {b: i for i, b in enumerate(target_columns + sec_labels)}

def check_intersect(start, stop, s, e):
    intersect = (start >= s and start < e) or (stop > s and stop <= e)

    return intersect

class birdnet_dataset(torch.utils.data.Dataset):
    def __init__(self, df, bird_df, cfg, tfs=None, normalize=True, mode='train'):
        super().__init__()
        
        self.df = df
        self.bird_df = bird_df
        self.sr = cfg.SR
        self.mel_spec_params = cfg.mel_spec_params
        self.len = len(self.df)

        self.duration = 5
        self.mel_transform = torchaudio.transforms.MelSpectrogram(**cfg.mel_spec_params)
        self.db_transform = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=cfg.TOP_DB)

        self.tfs = tfs

        self.mode = mode

        self.use_missing = cfg.USE_MISSING_LABELS

        self.num_classes = num_classes
        if self.use_missing:
            self.num_classes += len(sec_labels)

        self.bird2id = bird2id

        self.normalize = normalize
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index: int):
        entry = self.df.iloc[index]
        filename = entry.filename

        wav = read_wav(filename, self.sr)
        start = 0 
        
        if self.mode == 'train' and wav.shape[1] > self.duration * self.sr:
            stop = int(wav.shape[1] / self.sr)
            stop = stop - self.duration
            
            start = random.randint(0, stop)

        stop = start + self.duration

        wav = crop_wav(wav, start * self.sr, self.duration * self.sr)

        mel_spectrogram = normalize_melspec(self.db_transform(self.mel_transform(wav)))
        mel_spectrogram = mel_spectrogram * 255
        
        spect = mel_spectrogram.squeeze(dim=0)
        spect = torch.stack([spect, spect, spect], dim = 0)

        if self.tfs is not None:
            img = spect.permute(1,2,0).numpy()
            spect = self.tfs(image=img)['image']
            spect = spect.transpose(2,0,1)

        # get labels
        target = np.zeros(self.num_classes, dtype=np.float32)

        if self.mode == 'train':
            interv = self.bird_df[self.bird_df['filename'] == entry.file]

            interv['intersect'] = interv.apply(lambda row: check_intersect(start, stop, row['start'], row['end']), axis=1)
            
            if interv.intersect.sum() > 0:
                label = entry.primary_label
                target[self.bird2id[label]] = 1
        else:
            label = entry.primary_label
            target[self.bird2id[label]] = 1

        target = torch.from_numpy(target).float()
        
        return spect, target
