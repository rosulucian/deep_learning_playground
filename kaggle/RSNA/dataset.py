import torch
import imageio
import random
# import torchaudio

import numpy as np
import pandas as pd
import albumentations as A

from pathlib import Path
from albumentations.pytorch import ToTensorV2

class birdnet_dataset(torch.utils.data.Dataset):
    def __init__(self, files_df, coords_df, cfg, tfs=None, normalize=True, mode='train'):
        super().__init__()
        
        self.df = files_df
        self.coords_df = coords_df
        self.len = len(self.df)
        # self.dir = cfg.PNG_DIR
        
        self.tfs = tfs

        self.mode = mode

        self.normalize = normalize
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index: int):
        entry = self.df.iloc[index]
        filename = entry.filename


        ###############################
        
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
        