import torch
import torchaudio
import imageio

import numpy as np
import pandas as pd
import albumentations as A

from pathlib import Path
from albumentations.pytorch import ToTensorV2

# Imagenet vals
# img_mean=[0.485, 0.456, 0.406]
# img_std=[0.229, 0.224, 0.225]

# norm_tf = A.Normalize(mean=img_mean, std=img_std, p=1.0)

sample_submission = pd.read_csv('E:\data\BirdCLEF\sample_submission.csv')

def read_wav(path, sr):
    wav, org_sr = torchaudio.load(path, normalize=True)
    wav = torchaudio.functional.resample(wav, orig_freq=org_sr, new_freq=sr)
    return wav

def crop_wav(wav, start, duration):
    while wav.size(-1) < duration:
        wav = torch.cat([wav, wav], dim=1)
    wav = wav[:, :duration]

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
target_columns = sample_submission.columns[1:]
num_classes = len(target_columns)
bird2id = {b: i for i, b in enumerate(target_columns)}

class bird_dataset(torch.utils.data.Dataset):
    def __init__(self, df, cfg, tfs=None, normalize=True):
        super().__init__()
        
        self.df = df
        self.sr = cfg.SR
        self.dir = Path(cfg.AUDIO_FOLDER)
        self.mel_spec_params = cfg.mel_spec_params
        self.len = len(self.df)

        self.duration = self.sr * 5
        self.mel_transform = torchaudio.transforms.MelSpectrogram(**cfg.mel_spec_params)
        self.db_transform = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=cfg.TOP_DB)

        self.tfs = tfs

        self.num_classes = num_classes
        self.bird2id = bird2id

        self.normalize = normalize
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index: int):
        entry = self.df.iloc[index]
        filename = self.dir / entry.filename
        
        wav = read_wav(filename, self.sr)
        wav = crop_wav(wav, 0, self.duration)

        mel_spectrogram = normalize_melspec(self.db_transform(self.mel_transform(wav)))
        # mel_spectrogram = self.db_transform(self.mel_transform(wav))

        mel_spectrogram = mel_spectrogram * 255
        
        spect = mel_spectrogram.squeeze(dim=0)
        spect = torch.stack([spect, spect, spect], dim = 0)

        if self.tfs is not None:
            img = spect.permute(1,2,0).numpy()
            spect = self.tfs(image=img)['image']
            spect = spect.transpose(2,0,1)
        
        label = entry.primary_label
        
        target = np.zeros(self.num_classes, dtype=np.float32)
        target[self.bird2id[label]] = 1
        target = torch.from_numpy(target).float()
        
        return spect, target

class bird_dataset_inference(torch.utils.data.Dataset):
    def __init__(self, files, cfg, normalize=True):
        super().__init__()
        
        self.files = files
        self.sr = cfg.SR
        self.dir = Path(cfg.UNLABELED_FOLDER)
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

