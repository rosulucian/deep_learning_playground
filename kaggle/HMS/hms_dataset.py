import torch
import numpy as np
import pandas as pd
import albumentations as A

from albumentations.pytorch import ToTensorV2

# Imagenet vals
# img_mean=[0.485, 0.456, 0.406]
# img_std=[0.229, 0.224, 0.225]

img_mean=[0.485]
img_std=[0.229]

norm_tf = A.Normalize(mean=img_mean, std=img_std, p=1.0)

class spectro_dataset(torch.utils.data.Dataset):
    def __init__(self, df, base_dir, cols):
        super().__init__()
        
        self.df = df
        self.cols = cols
        self.len = len(self.df)
        self.base_dir = base_dir
        
        self.norm_tf = norm_tf
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index: int):
        entry = self.df.iloc[index]
        entry_id = entry.spectrogram_id
        
        # step forward one second
        offset = int(entry.spectrogram_label_offset_seconds)
        if offset%2==0: offset += 1
        
        spectro = pd.read_parquet(f'{self.base_dir}/{entry_id}.parquet')
        spectro = spectro.loc[(spectro.time >= offset)&(spectro.time <= offset+598)]
        spectro = spectro.iloc[:,1:]

        spectro = spectro.to_numpy().T
#         spectro = spectro.reshape(4,100,-1)
        
        spectro = np.nan_to_num(spectro, copy=True, nan=0.0)
        # spectro = self.norm_tf(image=spectro)['image']
        
        spectro = ToTensorV2(p=1.0)(image=spectro)['image'].reshape(4,100,-1)
        # spectro = self.norm_tf(image=spectro)['image']

        target = entry[self.cols].div(entry['total_votes'], axis=0).to_numpy(dtype=np.float32)
        
        return spectro, target