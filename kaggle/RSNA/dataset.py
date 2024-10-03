import torch
import imageio
import random

import PIL as pil
import numpy as np
import pandas as pd
import albumentations as A

import torch
import torch.nn.functional as F

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

from torch.nn.utils.rnn import pad_sequence

def collate_fn_padd(data):
    tensors, targets = zip(*data)
    
    features = pad_sequence(tensors, batch_first=True)
    targets = torch.stack(targets)
    
    return features, targets

class rsna_dataset(torch.utils.data.Dataset):
    def __init__(self, files_df, coords_df, cfg, tfs=None, mode='train'):
        super().__init__()

        self.files_df = files_df
        self.coords_df = coords_df
        self.files = list(files_df.filename.unique())
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

        labels = self.coords_df[self.coords_df['filename'] ==  filename].condition.to_list()

        for label in labels:
            target[self.class2id[label]] = 1

        # if self.files_df[self.files_df['filename'] ==  filename].healthy.values[0] == True:
        if len(labels) == 0:
            target[-1] = 1

        target = torch.from_numpy(target).float()
        
        return img, target

class rsna_inf_dataset(torch.utils.data.Dataset):
    def __init__(self, files_df, coords_df, cfg, tfs=None, mode='train'):
        super().__init__()
        
        self.coords_df = coords_df
        self.files_df = files_df
        
        self.files = list(files_df.filename.unique())
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
        labels = self.coords_df[self.coords_df['filename'] ==  filename].condition.to_list()

        for label in labels:
            target[self.class2id[label]] = 1

        # print(self.files_df[self.files_df['filename'] ==  filename].healthy.values[0])

        if self.files_df[self.files_df['filename'] ==  filename].healthy.values[0] == True:
            target[-1] = 1

        target = torch.from_numpy(target).float()

        instance_id = self.files_df[self.files_df['filename'] ==  filename].instance_id.values[0]
        
        return img, instance_id, target
       

class rsna_lstm_dataset(torch.utils.data.Dataset):
    def __init__(self, train_df, train_desc_df, embeds_path):
        super().__init__()

        # self.cfg = cfg
        self.embeds_path = embeds_path

        self.train_df = train_df
        self.studies = train_df.study_id.unique().tolist()
        self.len = train_df.study_id.nunique()

        self.train_desc_df = train_desc_df

    def __len__(self):
        return self.len

    def __getitem__(self, index: int):
        entry = self.train_df.iloc[index]
        study_id = entry.study_id

        df = self.train_desc_df[self.train_desc_df.study_id == study_id]
        df = df.sort_values('series_description', ascending=False)

        files = df.ss_id.tolist()

        files = [self.embeds_path / f'{f}.npy' for f in files]

        embeds = [np.load(f) for f in files]
        embeds = np.concatenate(embeds)
        
        targets = torch.tensor(entry.values.flatten().tolist()[1:])
        # targets = F.one_hot(targets).T

        return torch.from_numpy(embeds), targets

class rsna_lstm_dataset2(torch.utils.data.Dataset):
    def __init__(self, train_df, preds_df, embeds_path, cfg):
        super().__init__()

        # self.cfg = cfg
        self.embeds_path = embeds_path
        self.embeds = np.load(self.embeds_path / 'stacked.npy')

        self.train_df = train_df
        self.studies = train_df.study_id.unique().tolist()
        self.len = train_df.study_id.nunique()

        self.preds_df = preds_df

        self.healthy_frac = cfg.healthy_frac

    def __len__(self):
        return self.len

    def __getitem__(self, index: int):
        entry = self.train_df.iloc[index]
        study_id = entry.study_id

        df = self.preds_df[self.preds_df.study_id == study_id]

        # use random healthy slide
        if self.healthy_frac is not None:
            healthy = df[df['pred_H'] > 0.8]
            unhealthy = df[df['pred_H'] < 0.8]

            healthy = healthy.sample(frac=self.healthy_frac)

            df = pd.concat([healthy, unhealthy])

        # sort after concatenating
        # df = df.sort_values(['series_description', 'proj'], ascending=[False, True])
        # df = df.sort_values(['series_description', 'proj'], ascending=[False, False])
        df = df.sort_values(['series_id', 'instance'], ascending=[False, True])

        idx = df.index.to_list()
        embeds = self.embeds[idx]
        
        targets = torch.tensor(entry.values.flatten().tolist()[1:])
        # targets = F.one_hot(targets).T

        return torch.from_numpy(embeds), targets 
        