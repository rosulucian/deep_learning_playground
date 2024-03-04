import torch
import numpy as np
import pandas as pd
import albumentations as A

from albumentations.pytorch import ToTensorV2

# Imagenet vals
# img_mean=[0.485, 0.456, 0.406]
# img_std=[0.229, 0.224, 0.225]

mean = np.array([-0.22308692079693776, -0.23225270031972337, -0.2646080688103756, -0.27772951886156666], dtype=np.float32)
std = np.array([2.4021079842036257, 2.3784709900060506, 2.4214762588834593, 2.366489507911308], dtype=np.float32)

# norm_tf = A.Normalize(mean=img_mean, std=img_std, p=1.0)

class spectro_dataset(torch.utils.data.Dataset):
    def __init__(self, df, base_dir, cols, normalize=True):
        super().__init__()
        
        self.df = df
        self.cols = cols
        self.len = len(self.df)
        self.base_dir = base_dir

        self.normalize = normalize
        
        # self.norm_tf = norm_tf
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index: int):
        entry = self.df.iloc[index]
        entry_id = entry.spectrogram_id
        
        # step forward one second
        offset = int(entry.spectrogram_label_offset_seconds)
        if offset%2==0: offset += 1
        
        spectro = pd.read_parquet(f'{self.base_dir}/{entry_id}.parquet')
        # 300x400
        spectro = spectro.loc[(spectro.time >= offset)&(spectro.time <= offset+598)]
        spectro = spectro.iloc[:,1:]
        # 400x300
        spectro = spectro.to_numpy().T
#         spectro = spectro.reshape(4,100,-1)

        spectro = np.nan_to_num(spectro, copy=True, nan=0.0)
        np.clip(spectro,np.exp(-4),np.exp(8), out=spectro)
        np.log(spectro, out=spectro)

        # 4X100X300
        spectro = spectro.reshape(4,100,-1)
        
        if self.normalize is True:
            denom = np.reciprocal(std, dtype=np.float32)
            spectro = spectro.T - mean
            spectro *= denom
            
            spectro = spectro.T

        target = entry[self.cols].div(entry['total_votes'], axis=0).to_numpy(dtype=np.float32)
        
        return torch.from_numpy(spectro), target

