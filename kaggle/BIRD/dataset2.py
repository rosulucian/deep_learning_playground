import torch
import imageio
import random
import torchaudio

import numpy as np
import pandas as pd
import albumentations as A

from pathlib import Path
from albumentations.pytorch import ToTensorV2

from utils import read_wav, crop_wav, normalize_melspec

sample_submission = pd.read_csv('E:\data\BirdCLEF\sample_submission.csv')

# Set labels
sec_labels = ['lotshr1', 'orhthr1', 'magrob', 'indwhe1', 'bltmun1', 'asfblu1']
target_columns = sample_submission.columns[1:].tolist()
num_classes = len(target_columns)
bird2id = {b: i for i, b in enumerate(target_columns + sec_labels)}

def check_intersect(start, stop, s, e):
    intersect = (start >= s and start < e) or (stop > s and stop <= e)

    return intersect

class birdnet_dataset(torch.utils.data.Dataset):
    def __init__(self, df, cfg, tfs=None, normalize=True, mode='train'):
        super().__init__()
        
        self.df = df
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

        start = entry.start
        stop = start + self.duration

        wav = read_wav(filename, self.sr, start * self.sr, stop * self.sr)
        wav = crop_wav(wav, start * self.sr, self.duration * self.sr)

        print(wav.shape)
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
        
        label = entry.pred_code

        target[self.bird2id[label]] = 1

        target = torch.from_numpy(target).float()
        
        return spect, target
