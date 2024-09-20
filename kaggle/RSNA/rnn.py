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

import torchmetrics as tm

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback, LearningRateMonitor
from torchmetrics.wrappers import ClasswiseWrapper
from torchmetrics import MetricCollection
from torchmetrics.classification import MultilabelAccuracy, MultilabelPrecision, MultilabelRecall, MultilabelF1Score
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score

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

classes = ['SCSL1L2', 'SCSL2L3', 'SCSL3L4', 'SCSL4L5', 'SCSL5S1', 'RNFNL4L5',
       'RNFNL5S1', 'RNFNL3L4', 'RNFNL1L2', 'RNFNL2L3', 'LNFNL1L2',
       'LNFNL4L5', 'LNFNL5S1', 'LNFNL2L3', 'LNFNL3L4', 'LSSL1L2',
       'RSSL1L2', 'LSSL2L3', 'RSSL2L3', 'LSSL3L4', 'RSSL3L4', 'LSSL4L5',
       'RSSL4L5', 'LSSL5S1', 'RSSL5S1']

num_classes = len(classes)
class2id = {b: i for i, b in enumerate(classes)}

num_classes

# %%
train_dir = Path('E:\data\RSNA2024')

class CFG:

    project = 'lstm'
    comment = '16l128e'

    ### model
    model_name = 'eca_nfnet_l0' # 'resnet34', 'resnet200d', 'efficientnet_b1_pruned', 'efficientnetv2_m', efficientnet_b7 

    image_size = 256
    
    ROOT_FOLDER = train_dir
    IMAGES_DIR = ROOT_FOLDER / 'train_images'
    PNG_DIR = ROOT_FOLDER / f'pngs_{image_size}'
    FILES_CSV = ROOT_FOLDER / 'train_files.csv'
    PREDS_CSV = ROOT_FOLDER / 'predictions.csv'
    TRAIN_CSV = ROOT_FOLDER / 'train.csv'
    TRAIN_DESC_CSV = ROOT_FOLDER / 'train_series_descriptions.csv'
    COORDS_CSV = ROOT_FOLDER / 'train_label_coordinates.csv'

    # ckpt_path = Path(r"E:\data\RSNA2024\results\ckpt\eca_nfnet_l0 5e-05 10 eps all-labels\ep_03_loss_0.15231.ckpt")
    embeds_path = Path(r"E:\data\RSNA2024\embeddings")
    stacked_path = Path(r"E:\data\RSNA2024\embeddings_stacked")

    RESULTS_DIR = train_dir / 'results'
    CKPT_DIR = RESULTS_DIR / 'ckpt'

    input_dim = 128
    hidden_dim = 128
    target_size = 128

    classes = classes

    split_fraction = 0.95

    MIXUP = False

    ### training
    BATCH_SIZE = 16
    
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
    num_workers = 4

    random_seed = 42

# CFG.N_LABELS = len(CFG.classes)
CFG.N_LABELS = 3

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
preds_df = pd.read_csv(CFG.PREDS_CSV)

train_df.shape, train_desc_df.shape, coords_df.shape, files_df.shape, preds_df.shape

# %%
train_desc_df['ss_id'] = train_desc_df.apply(lambda row: f'{row.study_id}_{row.series_id}', axis=1)
preds_df['ss_id'] = preds_df.apply(lambda row: f'{row.study_id}_{row.series_id}', axis=1)

train_desc_df.head(2)

# %%
preds_df.head(2)

# %%
files_df.head(2)

# %%
preds_df = pd.merge(preds_df, train_desc_df.loc[:, ['ss_id', 'series_description']],  how='inner', left_on=['ss_id'], right_on=['ss_id'])

preds_df.sample(2)

# %%
preds_df = pd.merge(preds_df, files_df.loc[:, ['instance_id', 'proj']],  how='inner', left_on=['ids'], right_on=['instance_id'])

preds_df.sample(2)

# %%
# train_desc_df[train_desc_df['series_id'] == 3909740603]

# %%
train_df.fillna('N', inplace=True)
train_df.head(2)

# %%
le = preprocessing.LabelEncoder() 
le.fit(train_df.iloc[:, 1])

le.classes_
# foo = le.fit_transform(train_df.iloc[:,1])

# %%
from sklearn.utils.class_weight import compute_class_weight

# %%
foo = train_df.iloc[:, 1:].to_numpy()

foo.shape

# %%
foo.flatten().shape

# %%
compute_class_weight(class_weight='balanced', classes=np.unique(foo.flatten()), y=foo.flatten())

# %%

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

# %% [markdown]
# #### Prepare

# %%
embed_files = os.listdir(CFG.embeds_path)
stacked_files = os.listdir(CFG.stacked_path)

len(embed_files), len(stacked_files)

# %%
study_id = 838134337

# %%
selected_files = preds_df[preds_df.study_id == study_id]

selected_files.head(2)

# %%
# selected_files.head()

# %%
# selected_files.groupby('series_description').sort(['proj'])

# %%
selected_files.sort_values(['series_description', 'proj'], ascending=[False, False])

# %%
selected_files.study_id.unique()

# %%
idx = selected_files.sort_values(['series_description', 'proj'], ascending=[False, True]).index.to_list()
len(idx), selected_files.shape

# %%
stacked_embeds = np.load(CFG.embeds_path / 'stacked.npy')
stacked_embeds.shape

# %%
stacked_embeds[idx].shape

# %%
# for name, group in selected_files.sort_values('proj').groupby('series_description'):
#     print(name)
#     print(group.image.count())

# %%
files = files_df[files_df.study_id == study_id].instance_id.to_list()
files = [CFG.embeds_path / f'{f}.npy' for f in files]

files[0]

# %%
np.load(files[2])

# %%
files = files_df[files_df.study_id == study_id].ss_id.unique().tolist()
files = [CFG.stacked_path / f'{f}.npy' for f in files]

len(files)

# %%
np.load(files[0]).shape

# %%
train_df[train_df.study_id == study_id].values.flatten().tolist()[1:]

# %%
train_desc_df.head(5)

# %%
foo = train_desc_df[train_desc_df.study_id == study_id].sort_values('series_description', ascending=False)

foo

# %%
foo.ss_id.tolist()

# %% [markdown]
# ### Dataset

# %%
from dataset import rsna_lstm_dataset, rsna_lstm_dataset2

# %%
dset = rsna_lstm_dataset(train_df, train_desc_df, CFG.stacked_path)
dset = rsna_lstm_dataset2(train_df, preds_df, CFG.stacked_path)

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
from dataset import rsna_lstm_dataset, rsna_lstm_dataset2, collate_fn_padd


# %%
# from torch.nn.utils.rnn import pad_sequence

# def collate_fn_padd(data):
#     tensors, targets = zip(*data)
#     features = pad_sequence(tensors, batch_first=True)
#     targets = torch.stack(targets)
#     return features, targets
    
class lstm_datamodule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, train_desc_df, cfg):
        super().__init__()
        
        self.train_df = train_df
        self.val_df = val_df
        self.train_desc_df = train_desc_df
        
        self.train_bs = cfg.BATCH_SIZE
        self.val_bs = cfg.BATCH_SIZE

        self.cfg = cfg
        self.path = cfg.stacked_path
        
        self.num_workers = cfg.num_workers
        
    def train_dataloader(self):
        train_ds = rsna_lstm_dataset2(self.train_df, self.train_desc_df, self.path)
        
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=self.train_bs,
            collate_fn=collate_fn_padd,
            pin_memory=False,
            drop_last=False,
            shuffle=True,
            persistent_workers=True,
            num_workers=self.num_workers,
        )
        
        return train_loader
        
    def val_dataloader(self):
        val_ds = rsna_lstm_dataset2(self.val_df, self.train_desc_df, self.path)
        
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
CFG2.BATCH_SIZE = 8
CFG2.num_workers = 4

# dm = lstm_datamodule(t_df, v_df, train_desc_df, CFG2)
dm = lstm_datamodule(t_df, v_df, preds_df, CFG2)

x, y = next(iter(dm.train_dataloader()))
x.shape, y.shape, x.dtype, y.dtype

# %%

# %%
del dm


# %% [markdown]
# ### Loss function

# %%
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

# %% [markdown]
# #### Definition

# %%
# class LSTMClassifier(pl.LightningModule):
#     def __init__(self, cfg=CFG):
#         super(LSTMClassifier, self).__init__()

#         self.cfg = cfg
        
#         self.input_dim = cfg.input_dim
#         self.hidden_dim = cfg.hidden_dim
#         self.target_size = cfg.target_size

#         # https://discuss.pytorch.org/t/pytorchs-non-deterministic-cross-entropy-loss-and-the-problem-of-reproducibility/172180/9
#         # reduction is set to none
#         self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

#         self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers=16, batch_first=True)
#         self.fc = nn.Linear(self.hidden_dim, self.target_size)
#         self.classifiers = torch.nn.ModuleList([nn.Linear(self.target_size, cfg.N_LABELS) for i in range(25)])

#         # no average maccs
#         macc_serverity = ClasswiseWrapper(MulticlassAccuracy(
#             num_classes=self.cfg.N_LABELS,
#             average='none', 
#             multidim_average='global'
#             # label encoder classes
#         ), labels=le.classes_, prefix='multiacc_severity/')

#         macc_levels = ClasswiseWrapper(MulticlassAccuracy(
#             num_classes=self.cfg.N_LABELS,
#             average='none', 
#             multidim_average='global'
#         ), labels=classes, prefix='multiacc_levels/')
        
#         metrics = MetricCollection({
#             'macc': MulticlassAccuracy(num_classes=self.cfg.N_LABELS),
#             'macc_none': macc_serverity,
#             'mpr': MulticlassPrecision(num_classes=self.cfg.N_LABELS),
#             'mrec': MulticlassRecall(num_classes=self.cfg.N_LABELS),
#             'f1': MulticlassF1Score(num_classes=self.cfg.N_LABELS)
#         })

#         self.train_metrics = metrics.clone(prefix='train/')
#         self.valid_metrics = metrics.clone(prefix='val/')

#     def forward(self, sequence):
#         #  seq: (seq_len, bs, num_features)
#         lstm_out, (h, c) = self.lstm(sequence)
        
#         y = self.fc(h[-1])

#         preds = [c(y).T for c in self.classifiers]
#         preds = torch.stack(preds).T
        
#         return preds

#     def step(self, batch, batch_idx, mode='train'):
#         x, y = batch

#         preds = self(x)

#         loss = self.criterion(preds, y)

#         # https://discuss.pytorch.org/t/pytorchs-non-deterministic-cross-entropy-loss-and-the-problem-of-reproducibility/172180/9
#         loss = loss.mean()

#         # print(preds.shape, y.shape)

#         if mode == 'train':
#             output = self.train_metrics(preds, y)
#             self.log_dict(output)
#         else:
#             self.valid_metrics.update(preds, y)

#         self.log(f'{mode}/loss', loss, on_step=True, on_epoch=True)

#         return loss

#     def training_step(self, batch, batch_idx):
#         loss = self.step(batch, batch_idx, mode='train')
        
#         return loss
        
#     def validation_step(self, batch, batch_idx):
#         loss = self.step(batch, batch_idx, mode='val')
    
#         return loss

#     def on_train_epoch_end(self):
#         self.train_metrics.reset()

#     def on_validation_epoch_end(self):
#         output = self.valid_metrics.compute()
#         self.log_dict(output)

#         self.valid_metrics.reset()

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.LEARNING_RATE, weight_decay=CFG.weight_decay)
        
#         if self.cfg.USE_SCHD:
#             scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg.COS_EPOCHS)
#             scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=self.cfg.WARM_EPOCHS, after_scheduler=scheduler_cosine)

#             return [optimizer], [scheduler_warmup]
#         else:
#             return optimizer

# %%
CFG.N_LABELS


# %%
# torch.tensor([[[0, 1], [2, 1], [0, 2]], [[1, 1], [2, 0], [1, 2]]]).shape, torch.tensor([[[0, 2], [2, 0], [0, 1]], [[2, 2], [2, 1], [1, 0]]]).shape

# %%
class LSTMClassifier(pl.LightningModule):
    def __init__(self, cfg=CFG):
        super(LSTMClassifier, self).__init__()

        self.cfg = cfg
        
        self.input_dim = cfg.input_dim
        self.hidden_dim = cfg.hidden_dim
        
        self.levels = 25
        self.classes = cfg.N_LABELS

        # https://discuss.pytorch.org/t/pytorchs-non-deterministic-cross-entropy-loss-and-the-problem-of-reproducibility/172180/9
        # reduction is set to none
        # self.criterion = torch.nn.CrossEntropyLoss(reduction='none', weight=torch.tensor([1,0.1,2]))
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none', weight=torch.tensor([2.06762982, 0.42942998, 5.32804575]))

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers=16, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.levels * self.classes)
        # self.classifiers = torch.nn.ModuleList([nn.Linear(self.target_size, cfg.N_LABELS) for i in range(25)])

        # no average maccs
        macc_serverity = ClasswiseWrapper(MulticlassAccuracy(
            num_classes=self.cfg.N_LABELS,
            average='none', 
            multidim_average='global'
            # label encoder classes
        ), labels=le.classes_.tolist(), prefix='multiacc_severity/')

        macc_levels = ClasswiseWrapper(MulticlassAccuracy(
            num_classes=self.cfg.N_LABELS,
            average='none', 
            multidim_average='global'
        ), labels=classes, prefix='multiacc_levels/')
        
        metrics = MetricCollection({
            'macc': MulticlassAccuracy(num_classes=self.cfg.N_LABELS),
            'macc_none': macc_serverity,
            'mpr': MulticlassPrecision(num_classes=self.cfg.N_LABELS),
            'mrec': MulticlassRecall(num_classes=self.cfg.N_LABELS),
            'f1': MulticlassF1Score(num_classes=self.cfg.N_LABELS)
        })

        self.train_metrics = metrics.clone(prefix='train/')
        self.valid_metrics = metrics.clone(prefix='val/')

    def forward(self, sequence):
        #  seq: (seq_len, bs, num_features)
        lstm_out, (h, c) = self.lstm(sequence)
        
        y = self.fc(h[-1])

        # (N,C,d1) -> (N,3,25)
        return y.view(-1, self.classes, self.levels)

    def step(self, batch, batch_idx, mode='train'):
        x, y = batch

        preds = self(x)

        loss = self.criterion(preds, y)

        # https://discuss.pytorch.org/t/pytorchs-non-deterministic-cross-entropy-loss-and-the-problem-of-reproducibility/172180/9
        loss = loss.mean()

        # print(preds.shape, y.shape)

        if mode == 'train':
            output = self.train_metrics(preds, y)
            self.log_dict(output)
        else:
            self.valid_metrics.update(preds, y)

        self.log(f'{mode}/loss', loss, on_step=True, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, mode='train')
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, mode='val')
    
        return loss

    def on_train_epoch_end(self):
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        output = self.valid_metrics.compute()
        self.log_dict(output)

        self.valid_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.LEARNING_RATE, weight_decay=CFG.weight_decay)
        
        if self.cfg.USE_SCHD:
            scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg.COS_EPOCHS)
            scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=self.cfg.WARM_EPOCHS, after_scheduler=scheduler_cosine)

            return [optimizer], [scheduler_warmup]
        else:
            return optimizer


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
print('fc shape:', y.shape)

preds = [c(y).T for c in classifiers]
print('pred shape:', preds[0].shape)

preds = torch.stack(preds).T

print('preds shape:', preds.shape)

# %%
target_size = 64

lstm = nn.LSTM(target_size, target_size, num_layers=1, batch_first=True)
fc = nn.Linear(target_size, 75)

criterion = torch.nn.CrossEntropyLoss(reduction='none')

lstm_out, (h, c) = lstm(torch.randn(5,88,64))
print(lstm_out.shape, h.shape, c.shape)

pred = fc(h[-1])
print('fc shape:', y.shape)

pred = pred.view(-1, 3, 25)
print('preds shape:', pred.shape)

targets = torch.stack(5*[target])
loss = criterion(preds, targets)

print('loss:', loss.shape)

# %%
preds.argmax(dim=1).shape, targets.shape

# %%
preds.argmax(dim=1)

# %%
# preds.max(dim=1)

# %%
preds.max(dim=1)[1].shape, preds.argmax(1).shape

# %%
zeros = torch.zeros(preds.shape, dtype=preds.dtype)
ones = torch.ones(preds.shape, dtype=preds.dtype)

max_vals = preds.argmax(1)

foo = zeros.scatter(1, max_vals.unsqueeze(1), 1)

zeros.shape, ones.shape, foo.shape, preds.shape, foo.swapaxes(1,2).shape

# %%
a = torch.arange(15*25).reshape(5,3,25)
a.shape, a.swapaxes(1,2).shape

# %%
zeros.scatter(1, targets.unsqueeze(1), 1).swapaxes(1,2).shape

# %%
# acc = tm.functional.classification.multilabel_accuracy(foo, targets, 25, )
acc = tm.functional.classification.multilabel_accuracy(foo.swapaxes(1,2), zeros.scatter(1, targets.unsqueeze(1), 1).swapaxes(1,2), 25, average='none')
acc

# %%
torch.stack(5*[target])
preds.shape, targets.shape

# %%
metric = MulticlassAccuracy(num_classes=CFG.N_LABELS, average='none', multidim_average='samplewise')
macc = MulticlassAccuracy(num_classes=CFG.N_LABELS, average='none')

m = metric(preds, targets)

m.shape, m

# %%
macc(preds, targets)

# %%

# %%

# %%

# %% [markdown]
# #### Test out inputs/outputs

# %%
model = LSTMClassifier(CFG)

# %%
model.forward((seq.view(1, len(seq), -1))).shape

# %%
model.step((seq.view(1, len(seq), -1), target.view(1, len(target))), 0)

# %%

# %%
y = model(torch.randn(5,88,128))

y.shape, y.softmax(dim=0).shape

# %%
y.softmax(dim=1).shape

# %%
# y.softmax(dim=1).sum(dim=1)

# %%
# y[0].softmax(dim=0)

# %%

# %% [markdown]
# ### Split

# %%
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

# %%
train_df.shape

# %%
train_df.sample(2)

# %%
train_df.iloc[:,1:].shape

# %%
# TODO: data split

# %%
# sss = StratifiedShuffleSplit(n_splits=1, test_size=1-CFG.split_fraction, random_state=CFG.random_seed)
# train_idx, val_idx = next(sss.split(train_df.study_id, train_df.iloc[:,1:]))

# t_df = train_df.iloc[train_idx]
# v_df = train_df.iloc[val_idx]

# t_df.shape, v_df.shape

# %%

# %% [markdown]
# ### Train

# %%
CFG.BATCH_SIZE, CFG.device

# %%
train_desc_df.head()

# %%
# dm = lstm_datamodule(t_df, v_df, train_desc_df, cfg=CFG)
dm = lstm_datamodule(t_df, v_df, preds_df, cfg=CFG)

len(dm.train_dataloader()), len(dm.val_dataloader())

# %%
run_name = f'{CFG.model_name} {CFG.LEARNING_RATE} {CFG.N_EPOCHS} eps {CFG.comment}'
run_name

# %%
wandb_logger = WandbLogger(
    name=run_name,
    project=CFG.project,
    job_type='train',
    save_dir=CFG.RESULTS_DIR,
    # config=cfg,
)

loss_ckpt = pl.callbacks.ModelCheckpoint(
    monitor='val/loss',
    auto_insert_metric_name=False,
    dirpath=CFG.CKPT_DIR / run_name,
    filename='ep_{epoch:02d}_loss_{val/loss:.5f}',
    save_top_k=2,
    mode='min',
)

# acc_ckpt = pl.callbacks.ModelCheckpoint(
#     monitor='val/acc',
#     auto_insert_metric_name=False,
#     dirpath=CFG.CKPT_DIR / run_name,
#     filename='ep_{epoch:02d}_acc_{val/acc:.5f}',
#     save_top_k=2,
#     mode='max',
# )

lr_monitor = LearningRateMonitor(logging_interval='step')

# %%
trainer = pl.Trainer(
    max_epochs=CFG.N_EPOCHS,
    deterministic=True,
    accelerator=CFG.device,
    default_root_dir=CFG.RESULTS_DIR,
    gradient_clip_val=0.5, 
    # gradient_clip_algorithm="value",
    logger=wandb_logger,
    callbacks=[loss_ckpt, lr_monitor],
)

# %%

# %% [markdown]
# #### Fit

# %%
model = LSTMClassifier(CFG)

# %%
trainer.fit(model, dm)

# %%

# %% [markdown]
# ### Predict

# %%
x, y = next(iter(dm.train_dataloader()))

x.shape, y.shape

# %%
x.shape

# %%
# pred = model(x.to(CFG.device)).detach().cpu()
pred = model(x).detach().cpu()
pred.shape

# %%
y

# %%
train_df.head()

# %%
pred.softmax(dim=1)

# %%
pred.argmax(dim=1)

# %%

# %%
