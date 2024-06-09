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

class bird_dataset(torch.utils.data.Dataset):
    def __init__(self, df, cfg, tfs=None, normalize=True, mode='train'):
        super().__init__()
        
        self.df = df
        self.sr = cfg.SR
        # self.dir = Path(cfg.AUDIO_FOLDER)
        self.mel_spec_params = cfg.mel_spec_params
        self.len = len(self.df)

        self.duration = self.sr * 5
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

        start = entry.range
        
        if start != 0:
            start *= self.sr
            start -= self.duration

            wav = read_wav(filename, self.sr, start, self.duration)

        else:
            wav = read_wav(filename, self.sr)
            if self.mode == 'train' and wav.shape[1] > self.duration:
                stop = wav.shape[1] - self.duration
                start = random.randint(0, stop)

            wav = crop_wav(wav, start, self.duration)

        mel_spectrogram = normalize_melspec(self.db_transform(self.mel_transform(wav)))
        # mel_spectrogram = self.db_transform(self.mel_transform(wav))

        mel_spectrogram = mel_spectrogram * 255

        # print(mel_spectrogram.shape)
        
        spect = mel_spectrogram.squeeze(dim=0)
        spect = torch.stack([spect, spect, spect], dim = 0)

        if self.tfs is not None:
            img = spect.permute(1,2,0).numpy()
            spect = self.tfs(image=img)['image']
            spect = spect.transpose(2,0,1)

        # ###### get labels
        target = np.zeros(self.num_classes, dtype=np.float32)
        
        if entry.range == 0:
            label = entry.primary_label
            target[self.bird2id[label]] = 1
    
            # add secondary labels
            for l in eval(entry.secondary_labels):
                if l != "" and (l in target_columns or self.use_missing):
                    target[self.bird2id[l]] = 1
        else:
            idx = entry[['top_1_idx', 'top_2_idx', 'top_3_idx']].astype(np.int32)
            vals = entry[['top_1', 'top_2', 'top_3']]

            target[idx.tolist()] = vals

        target = torch.from_numpy(target).float()
        
        return spect, target

class bird_dataset2(torch.utils.data.Dataset):
    def __init__(self, df, cfg, tfs=None, normalize=True, mode='train'):
        super().__init__()
        
        self.df = df
        self.sr = cfg.SR
        # self.dir = Path(cfg.AUDIO_FOLDER)
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

        info = torchaudio.info(filename)
        length = int(info.num_frames / self.sr)
        
        start = 0
        if self.mode == 'train' and length > self.duration:
            start = random.randint(0, length-5)

            offset = start * self.sr
            num_frames = self.duration * self.sr
            wav = read_wav(filename, self.sr, offset, num_frames)
        else:
            wav = read_wav(filename, self.sr)

        wav = crop_wav(wav, 0, self.duration * self.sr)

        mel_spectrogram = normalize_melspec(self.db_transform(self.mel_transform(wav)))
        # mel_spectrogram = self.db_transform(self.mel_transform(wav))

        mel_spectrogram = mel_spectrogram * 255

        # print(mel_spectrogram.shape)
        
        spect = mel_spectrogram.squeeze(dim=0)
        spect = torch.stack([spect, spect, spect], dim = 0)

        if self.tfs is not None:
            img = spect.permute(1,2,0).numpy()
            spect = self.tfs(image=img)['image']
            spect = spect.transpose(2,0,1)

        # ###### get labels
        target = np.zeros(self.num_classes, dtype=np.float32)

        label = entry.primary_label
        target[self.bird2id[label]] = 1

        target = torch.from_numpy(target).float()
        
        return spect, target

class bird_dataset_inference(torch.utils.data.Dataset):
    def __init__(self, files, directory, cfg, normalize=True):
        super().__init__()
        
        self.files = files
        self.sr = cfg.SR
        # self.dir = Path(cfg.UNLABELED_FOLDER)
        self.dir = directory
        self.mel_spec_params = cfg.mel_spec_params
        self.len = len(self.files)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(**cfg.mel_spec_params)
        self.db_transform = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=cfg.TOP_DB)

        self.tfs = A.Compose([
            A.Resize(cfg.image_size, cfg.image_size),
            A.Normalize(),
        ])
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index: int):
        file = self.files[index]
        filename = self.dir / file
        
        wav = read_wav(filename, self.sr)

        # check length and add padding
        length = wav.shape[1]
        limit = 4 * 60 * self.sr
        if length < 4 * 60 * self.sr:
            pad = torch.zeros(1, limit-length)
            wav = torch.cat([wav,pad], dim=1)

        mel_spectrogram = normalize_melspec(self.db_transform(self.mel_transform(wav)))
        mel_spectrogram = mel_spectrogram * 255
        
        spect = mel_spectrogram.squeeze(dim=0)
        spect = torch.stack([spect, spect, spect], dim = 0)

        remainder = spect.shape[-1] % 48
        spect = torch.split(spect[:,:,:-remainder], 312, dim=-1)
        
        transformed = []
        for img in spect:
            img = img.permute(1,2,0).numpy()
            img = self.tfs(image=img)['image']
            img = img.transpose(2,0,1)
            img = torch.from_numpy(img).float()

            transformed.append(img)

        spect = torch.stack(transformed)

        # print(spect.shape)
        
        return spect, file

class spectro_dataset(torch.utils.data.Dataset):
    def __init__(self, df, X, y, normalize=True):
        super().__init__()
        
        self.df = df
        self.X = X
        self.y = y
        self.len = len(self.df)

        self.tf = None

        self.num_classes = num_classes
        self.bird2id = bird2id

        self.normalize = normalize
        
        # self.norm_tf = norm_tf
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index: int):
        entry = self.df.iloc[index]
        filename = entry.filename
        # print(filename)

        spec = imageio.imread(self.X[filename])
        spec = spec[:,:312] # 5secs

        image = torch.from_numpy(spec).float()
        image = torch.stack([image, image, image], dim = 0)

        label = self.y[filename]
        target = np.zeros(self.num_classes, dtype=np.float32)
        
        # target[self.bird2id[label]] = 1
        target[label] = 1
        
        target = torch.from_numpy(target).float()
        
        return image, target

