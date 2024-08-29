# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ### Imports

# %%
import os
import time
import wandb
import torch
import random
import torchvision

import numpy as np
import pandas as pd
import torchmetrics as tm 
# import plotly.express as px
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from torch import nn
from pathlib import Path, PurePath
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW, RMSprop # optmizers
from sklearn import preprocessing 
# from warmup_scheduler import GradualWarmupScheduler
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau # Learning rate schedulers

import albumentations as A
# from albumentations.pytorch import ToTensorV2

import torch.nn.functional as F

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback, LearningRateMonitor
from torchmetrics.wrappers import ClasswiseWrapper
from torchmetrics import MetricCollection
from torchmetrics.classification import MultilabelAccuracy, MultilabelPrecision, MultilabelRecall, MultilabelF1Score

import timm

# %%
print('timm version', timm.__version__)
print('torch version', torch.__version__)

# %%
wandb.login(key=os.getenv('wandb_api_key'))

# %%
# detect and define device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)


# %%
# for reproducibility
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# %% [markdown]
# ### Config

# %%
# TODO: maybe use condition and level for classes
classes = ['SCS', 'RNFN', 'LNFN', 'LSS', 'RSS'] + ['H'] # add healthy class

# classes = ['SCS', 'RNFN', 'LNFN'] + ['H'] # add healthy class

# classes = ['LSS', 'RSS'] + ['H'] # add healthy class

# classes = ['SCSL1L2', 'SCSL2L3', 'SCSL3L4', 'SCSL4L5', 'SCSL5S1', 'RNFNL4L5',
#        'RNFNL5S1', 'RNFNL3L4', 'RNFNL1L2', 'RNFNL2L3', 'LNFNL1L2',
#        'LNFNL4L5', 'LNFNL5S1', 'LNFNL2L3', 'LNFNL3L4', 'LSSL1L2',
#        'RSSL1L2', 'LSSL2L3', 'RSSL2L3', 'LSSL3L4', 'RSSL3L4', 'LSSL4L5',
#        'RSSL4L5', 'LSSL5S1', 'RSSL5S1'] + ['H']

num_classes = len(classes)
class2id = {b: i for i, b in enumerate(classes)}

# %%
train_dir = Path('E:\data\RSNA2024')

class CFG:

    project = 'rsna-2'
    comment = 'bottleneck'

    ### model
    model_name = 'eca_nfnet_l0' # 'resnet34', 'resnet200d', 'efficientnet_b1_pruned', 'efficientnetv2_m', efficientnet_b7 

    image_size = 256
    
    ROOT_FOLDER = train_dir
    IMAGES_DIR = ROOT_FOLDER / 'train_images'
    PNG_DIR = ROOT_FOLDER / f'pngs_{image_size}'
    FILES_CSV = ROOT_FOLDER / 'train_files.csv'
    TRAIN_CSV = ROOT_FOLDER / 'train.csv'
    TRAIN_DESC_CSV = ROOT_FOLDER / 'train_series_descriptions.csv'
    COORDS_CSV = ROOT_FOLDER / 'train_label_coordinates.csv'

    # ckpt_path = Path(r"E:\data\RSNA2024\results\ckpt\eca_nfnet_l0 5e-05 10 eps all-labels\ep_03_loss_0.15231.ckpt")
    embeds_path = Path(r"E:\data\RSNA2024\embeddings")

    RESULTS_DIR = train_dir / 'results'
    CKPT_DIR = RESULTS_DIR / 'ckpt'

    input_dim = 64
    hidden_dim = 64
    target_size = 64

    classes = classes

    split_fraction = 0.95

    MIXUP = False

    ### training
    BATCH_SIZE = 1
    
    ### Optimizer
    N_EPOCHS = 10
    USE_SCHD = False
    WARM_EPOCHS = 3
    COS_EPOCHS = N_EPOCHS - WARM_EPOCHS

    # LEARNING_RATE = 5*1e-5 # best
    LEARNING_RATE = 5e-5
    
    weight_decay = 1e-6 # for adamw

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ### split train and validation sets
    num_workers = 16

    random_seed = 42

CFG.N_LABELS = len(CFG.classes)

seed_torch(seed = CFG.random_seed)

# %%
CFG.N_LABELS 

# %% [markdown]
# ### Load data

# %%
train_df = pd.read_csv(CFG.TRAIN_CSV)
train_desc_df = pd.read_csv(CFG.TRAIN_DESC_CSV)
coords_df = pd.read_csv(CFG.COORDS_CSV)
files_df = pd.read_csv(CFG.FILES_CSV)

train_df.shape, train_desc_df.shape, coords_df.shape, files_df.shape

# %%
train_df.fillna('N', inplace=True)

# %%
train_df.head(2)

# %%
le = preprocessing.LabelEncoder() 
le.fit(train_df.iloc[:, 1])

le.classes_
# foo = le.fit_transform(train_df.iloc[:,1])

# %%
train_df.iloc[:,1:] = train_df.iloc[:,1:].apply(le.fit_transform)

# %%
train_df.head(2)

# %%
coords_df.sample(2)

# %%
coords_df.condition.unique()

# %%
coords_df.cl.unique()

# %%
coords_df.cl.nunique()

# %%
embed_files = os.listdir(CFG.embeds_path)

len(embed_files)

# %%
study_id = 838134337

# %%
selected_files = files_df[files_df.study_id == study_id]

selected_files.head(2)

# %%
# selected_files.groupby('series_description').sort(['proj'])

# %%
for name, group in selected_files.sort_values('proj').groupby('series_description'):
    print(name)
    print(group.image.count())

# %%
files = files_df[files_df.study_id == study_id].instance_id.to_list()
files = [CFG.embeds_path / f'{f}.npy' for f in files]

files[0]

# %%
np.load(files[2])

# %%
train_df[train_df.study_id == study_id].values.flatten().tolist()[1:]

# %% [markdown]
# ### Dataset

# %%
from dataset import rsna_lstm_dataset

# %%
dset = rsna_lstm_dataset(train_df, files_df, CFG)

print(dset.__len__())

seq, target = dset.__getitem__(1)
print(seq.shape, target.shape)
print(seq.dtype, target.dtype)

# %%
# seq dim: (bs, seq_len, 1, num_features)
# target dim: (N, d1)
target

# %%
seq[0]

# %% [markdown]
# ### Data Module

# %%
from dataset import rsna_lstm_dataset

# %%
from torch.nn.utils.rnn import pad_sequence

def collate_fn_padd(data):
    tensors, targets = zip(*data)
    features = pad_sequence(tensors, batch_first=True)
    targets = torch.stack(targets)
    return features, targets


# %%
class lstm_datamodule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, files_df, cfg=CFG):
        super().__init__()
        
        self.train_df = train_df
        self.val_df = val_df
        self.files_df = files_df
        
        self.train_bs = cfg.BATCH_SIZE
        self.val_bs = cfg.BATCH_SIZE

        self.cfg = cfg
        
        self.num_workers = cfg.num_workers
        
    def train_dataloader(self):
        train_ds = rsna_lstm_dataset(self.train_df, self.files_df, self.cfg, mode='train')
        
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=self.train_bs,
            collate_fn=collate_fn_padd,
            pin_memory=False,
            drop_last=False,
            shuffle=True,
            # persistent_workers=True,
            num_workers=self.num_workers,
        )
        
        return train_loader
        
    def val_dataloader(self):
        val_ds = rsna_lstm_dataset(self.val_df, self.files_df, self.cfg, mode='val')
        
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=self.val_bs,
            collate_fn=collate_fn_padd,
            pin_memory=False,
            drop_last=False,
            shuffle=False,
            persistent_workers=True,
            num_workers=2,
        )
        
        return val_loader


# %%
t_df = train_df[:-100]
# t_df = pd.concat([meta_df[:-100], ul_df[:-100]], ignore_index=True)
v_df = train_df[-100:]

CFG2 = CFG()
# CFG2 = copy.deepcopy(CFG)
CFG2.BATCH_SIZE = 2
CFG2.num_workers = 2

dm = lstm_datamodule(t_df, v_df, files_df, cfg=CFG2)

x, y = next(iter(dm.train_dataloader()))
x.shape, y.shape, x.dtype, y.dtype


# %%

# %% [markdown]
# ### Loss function

# %% jupyter={"source_hidden": true}
class FocalLossBCE(torch.nn.Module):
    def __init__(
            self,
            alpha: float = 0.25,
            gamma: float = 2,
            reduction: str = "mean",
            bce_weight: float = 1.0,
            focal_weight: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = torch.nn.BCEWithLogitsLoss(reduction=reduction)
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight

    def forward(self, logits, targets):
        focall_loss = torchvision.ops.focal_loss.sigmoid_focal_loss(
            inputs=logits,
            targets=targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )
        bce_loss = self.bce(logits, targets)
        return self.bce_weight * bce_loss + self.focal_weight * focall_loss

class GeM(torch.nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = torch.nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        bs, ch, h, w = x.shape
        x = torch.nn.functional.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(
            1.0 / self.p)
        x = x.view(bs, ch)
        return x


# %% [markdown]
# ### Model

# %%
class LSTMClassifier(pl.LightningModule):
    def __init__(self, cfg=CFG):
        super(LSTMClassifier, self).__init__()
        
        self.input_dim = cfg.input_dim
        self.hidden_dim = cfg.hidden_dim
        self.target_size = cfg.target_size

        self.criterion = torch.nn.CrossEntropyLoss()

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers=1, batch_first=True)

        self.fc = nn.Linear(self.hidden_dim, self.target_size)


        self.classifiers = [nn.Linear(self.target_size, 3) for i in range(25)]

    def forward(self, sequence):

        #  seq: (seq_len, bs, num_features)
        lstm_out, (h, c) = self.lstm(sequence)
        
        y = self.fc(h[-1])

        preds = [c(y).T for c in self.classifiers]
        preds = torch.stack(preds).T
        
        return preds

    def step(self, batch, batch_idx, mode='train'):
        x, y = batch

        preds = self(x)

        loss = self.criterion(preds, y)

        return loss

# %% [markdown]
# #### building blocks

# %%
seq.shape, seq.view(1, len(seq), -1).shape, target.shape

# %%
target_size = 64

lstm = nn.LSTM(target_size, target_size, num_layers=1, batch_first=True)
fc = nn.Linear(target_size, target_size)
classifiers = [nn.Linear(target_size, 3) for i in range(25)]

lstm_out, (h, c) = lstm(torch.randn(5,88,64))
print(lstm_out.shape, h.shape, c.shape)

y = fc(h[-1])
print(y.shape)

preds = [c(y) for c in classifiers]
print(preds[0].shape)
preds = torch.stack(preds)

preds.shape

# %%
preds.T.shape

# %%

# %% [markdown]
# #### Test out inputs/outputs

# %%
model = LSTMClassifier(CFG)

# %%
model.step((seq.view(1, len(seq), -1), target.view(1, len(target))), 0)

# %%

# %%
y = model(torch.randn(5,88,64))

y.shape, y.softmax(dim=0).shape

# %%
# y[0].softmax(dim=0)

# %%
y.softmax(dim=1)

# %%
# y.softmax(dim=1).sum(dim=1)

# %%

# %% [markdown]
# ### Split

# %%

# %%

# %%

# %% [markdown]
# ### Train

# %%

# %%
