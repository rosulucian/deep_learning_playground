import torch
import imageio
import random

import PIL as pil
import numpy as np
import pandas as pd
import albumentations as A

from pathlib import Path
from albumentations.pytorch import ToTensorV2

# TODO: determine these values
img_mean = (0.625, 0.448, 0.688)
img_std = (0.131, 0.177, 0.101)

# from eca_nfnet_l1
img_mean = (0.485, 0.456, 0.406)
img_std = (0.229, 0.224, 0.225)

# # TODO: maybe use condition and level for classes
# classes = ['SCS', 'RNFN', 'LNFN', 'LSS', 'RSS'] + ['H'] # add healthy class

# classes = ['SCSL1L2', 'SCSL2L3', 'SCSL3L4', 'SCSL4L5', 'SCSL5S1', 'RNFNL4L5',
#        'RNFNL5S1', 'RNFNL3L4', 'RNFNL1L2', 'RNFNL2L3', 'LNFNL1L2',
#        'LNFNL4L5', 'LNFNL5S1', 'LNFNL2L3', 'LNFNL3L4', 'LSSL1L2',
#        'RSSL1L2', 'LSSL2L3', 'RSSL2L3', 'LSSL3L4', 'RSSL3L4', 'LSSL4L5',
#        'RSSL4L5', 'LSSL5S1', 'RSSL5S1'] + ['H']

# num_classes = len(classes)
# class2id = {b: i for i, b in enumerate(classes)}

class rsna_dataset(torch.utils.data.Dataset):
    def __init__(self, coords_df, cfg, tfs=None, mode='train'):
        super().__init__()
        
        self.df = coords_df
        self.files = list(coords_df.filename.unique())
        self.len = len(self.files)

        self.classes = cfg.classes
        self.num_classes = len(self.classes)

        self.class2id = {b: i for i, b in enumerate(self.classes)}
        
        self.tfs = tfs

        self.mode = mode
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index: int):
        filename = self.files[index]

        img = np.array(pil.Image.open(filename), dtype=np.float32)
        img = np.stack([img, img, img], axis = 0)

        if self.tfs is not None:
            img = img.transpose(1,2,0)
            img = self.tfs(image=img)['image']
            img = img.transpose(2,0,1)

        target = np.zeros(self.num_classes, dtype=np.float32)

        # labels = self.df[self.df['filename'] ==  filename].cl.to_list()
        labels = self.df[self.df['filename'] ==  filename].condition.to_list()

        for label in labels:
            target[self.class2id[label]] = 1

        target = torch.from_numpy(target).float()
        
        return img, target
        