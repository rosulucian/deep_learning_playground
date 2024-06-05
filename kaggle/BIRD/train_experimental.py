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

# %%
import os
import time
import wandb
import torch
import random
import pickle
import imageio
import librosa
import torchvision

import numpy as np
import pandas as pd
import torchmetrics as tm 
import plotly.express as px
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from torch import nn
from pathlib import Path, PurePath
from IPython.display import Audio
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW, RMSprop # optmizers
from warmup_scheduler import GradualWarmupScheduler
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau # Learning rate schedulers

import albumentations as A
# from albumentations.pytorch import ToTensorV2

import timm

# %%
print('timm version', timm.__version__)
print('torch version', torch.__version__)

# %%
# print(os.getenv('wandb_api_key'))

# %%
wandb.login(key=os.getenv('wandb_api_key'))

# %%
# detect and define device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

# %% [markdown]
# ### Config

# %%
train_dir = Path('E:\data\BirdCLEF')


# %%
class CFG:
    project = 'Bird-local-3'
    comment = 'better-labels'
    
    MIXUP = True
    USE_SCHD = False
    USE_UL = False
    # USE_MISSING_LABELS = False
    USE_SECONDARY = False
    USE_MISSING_LABELS = USE_SECONDARY
    USE_UPSAMPLE = False

    UL_THRESH = 0.05
    up_thr = 50
    
    # Competition Root Folder
    ROOT_FOLDER = train_dir
    birds_csv = train_dir / 'bird_preds.csv'
    AUDIO_FOLDER = train_dir / 'train_audio'
    DATA_DIR = train_dir / 'spectros'
    TRAIN_CSV = train_dir / 'train_metadata.csv'
    UNLABELED_CSV = train_dir / 'predictions.csv'
    RESULTS_DIR = train_dir / 'results'
    CKPT_DIR = RESULTS_DIR / 'ckpt'

    num_workers = 16
    # Maximum decibel to clip audio to
    # TOP_DB = 100
    TOP_DB = 80
    # Minimum rating
    MIN_RATING = 3.0
    # Sample rate as provided in competition description
    # SR = 32000
    SR = 32000

    image_size = 128
    
    ### split train and validation sets
    split_fraction = 0.95
    
    ### model
    model_name = 'eca_nfnet_l0' # 'resnet34', 'resnet200d', 'efficientnet_b1_pruned', 'efficientnetv2_m', efficientnet_b7 ...  
    
    ### training
    BATCH_SIZE = 128

    ### Optimizer
    N_EPOCHS = 30
    WARM_EPOCHS = 3
    COS_EPOCHS = N_EPOCHS - WARM_EPOCHS
    
    # LEARNING_RATE = 5*1e-5 # best
    LEARNING_RATE = 5e-5
    
    weight_decay = 1e-6 # for adamw

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    random_seed = 42

mel_spec_params = {
    "sample_rate": CFG.SR,
    "n_mels": 128,
    "f_min": 150,
    "f_max": CFG.SR / 2,
    "n_fft": 2048,
    "hop_length": 512,
    "normalized": True,
    "center" : True,
    "pad_mode" : "constant",
    "norm" : "slaney",
    "mel_scale" : "slaney"
}

CFG.mel_spec_params = mel_spec_params

sec_labels = ['lotshr1', 'orhthr1', 'magrob', 'indwhe1', 'bltmun1', 'asfblu1']

sample_submission = pd.read_csv(train_dir / 'sample_submission.csv')

# Set labels
CFG.LABELS = sample_submission.columns[1:].tolist()
if CFG.USE_MISSING_LABELS:
    CFG.LABELS += sec_labels
    
CFG.N_LABELS = len(CFG.LABELS)
print(f'# labels: {CFG.N_LABELS}')

bird2id = {b: i for i, b in enumerate(CFG.LABELS)}

display(sample_submission.head())

# %%
CFG.N_LABELS


# %%
# for reproducibility
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_torch(seed = CFG.random_seed)

# %%
meta_df = pd.read_csv(CFG.TRAIN_CSV)
meta_df.head(2)

# %%
# meta_df[meta_df['primary_label'] == 'magrob']

# %%
len(CFG.LABELS)

# %%
columns = ['primary_label', 'secondary_labels', 'filename', 'file']

# %%
meta_df['file'] = meta_df.apply(lambda row: row['filename'].split('/')[-1], axis=1)
meta_df['filename'] = fr'{str(CFG.AUDIO_FOLDER)}/' + meta_df['filename']

# %%
meta_df[columns].head(2)

# %%
meta_df.iloc[0].filename

# %%
meta_df = meta_df[columns]

# %%
meta_df['range'] = 0

# %%
cols = ['top_1', 'top_2', 'top_3', 'top_1_idx', 'top_2_idx', 'top_3_idx']


# %%
def get_scores(row):
    labels = [row.primary_label] + eval(row.secondary_labels)
    labels[:3]
    labels = [bird2id[l] for l in labels]

    labels = np.array(labels)

    scores = np.zeros(3)
    scores[labels] = 1
    
    return scores


# %%
get_scores(meta_df.iloc[0])

# %%
# meta_df[cols[:3]] = meta_df.apply(get_scores, axis='columns', result_type='expand')

# %%
meta_df.iloc[0].filename

# %%
meta_df.head(2)

# %%
ul_df = pd.read_csv(CFG.UNLABELED_CSV)
print(ul_df.shape)
ul_df = ul_df[ul_df['top_1'] < CFG.UL_THRESH]
print(ul_df.shape)

ul_df.head(2)

# %%
ul_df = ul_df.sample(1000)

# %%
# ul_df[ul_df['top_2'] > CFG.UL_THRESH].sample(5)

# %%
# meta_df = pd.concat([meta_df, ul_df.sample(1000)], ignore_index=True)

# %%
# meta_df.sample(10)

# %% [markdown]
# ### BIRDNet predictions df

# %%
birds_df = pd.read_csv(CFG.birds_csv)
birds_df.shape, birds_df.head(2)

# %%
birds_df['filename'].value_counts()

# %%

# %%

# %%

# %% [markdown]
# ### Load data

# %%
from dataset import birdnet_dataset

# %%
dset = birdnet_dataset(meta_df, birds_df, CFG)

print(dset.__len__())

spect, label, = dset.__getitem__(0)
print(spect.shape, label.shape)
print(spect.dtype, label.dtype)

# %%
# interv.intersect.sum()

# %%
librosa.display.specshow(spect[0].numpy(), y_axis="mel", x_axis='s', sr=CFG.SR, cmap='gray')
plt.show()

# %%
librosa.display.specshow(spect[1].numpy(), y_axis="mel", x_axis='s', sr=CFG.SR, cmap='gray')
plt.show()

# %% [markdown]
# ### Data Module

# %%
from dataset import birdnet_dataset


# %%
class wav_datamodule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, birds_df, cfg=CFG, train_tfs=None, val_tfs=None):
        super().__init__()
        
        self.train_df = train_df
        self.val_df = val_df
        self.birds_df = birds_df
        
        self.train_bs = cfg.BATCH_SIZE
        self.val_bs = cfg.BATCH_SIZE

        self.train_tfs = train_tfs
        self.val_tfs = val_tfs

        self.cfg = cfg
        
        self.num_workers = cfg.num_workers
        
    def train_dataloader(self):
        train_ds = birdnet_dataset(self.train_df, self.birds_df, self.cfg, tfs=self.train_tfs, mode='train')
        
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=self.train_bs,
            pin_memory=False,
            drop_last=False,
            shuffle=True,
            persistent_workers=True,
            num_workers=self.num_workers,
        )
        
        return train_loader
        
    def val_dataloader(self):
        val_ds = birdnet_dataset(self.val_df, self.birds_df, self.cfg, tfs=self.val_tfs, mode='val')
        
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=self.val_bs,
            pin_memory=False,
            drop_last=False,
            shuffle=False,
            persistent_workers=True,
            num_workers=2,
        )
        
        return val_loader


# %% jupyter={"source_hidden": true}
# class spectro_datamodule(pl.LightningDataModule):
#     def __init__(self, train_df, val_df, cfg=CFG):
#         super().__init__()
        
#         self.train_df = train_df
#         self.val_df = val_df
        
#         self.train_bs = cfg.BATCH_SIZE
#         self.val_bs = cfg.BATCH_SIZE
        
#         self.num_workers = cfg.num_workers
        
#     def train_dataloader(self):
#         train_ds = spectro_dataset(self.train_df, X, y)
        
#         train_loader = torch.utils.data.DataLoader(
#             train_ds,
#             batch_size=self.train_bs,
#             pin_memory=False,
#             drop_last=False,
#             shuffle=True,
#             num_workers=self.num_workers,
#         )
        
#         return train_loader
        
#     def val_dataloader(self):
#         val_ds = spectro_dataset(self.val_df, X, y)
        
#         val_loader = torch.utils.data.DataLoader(
#             val_ds,
#             batch_size=self.val_bs,
#             pin_memory=False,
#             drop_last=False,
#             shuffle=False,
#             num_workers=self.num_workers,
#         )
        
#         return val_loader

# %%
image_size = CFG.image_size

train_tfs = A.Compose([
    # A.HorizontalFlip(p=0.5),
    A.Resize(image_size, image_size),
    A.CoarseDropout(max_height=int(image_size * 0.375), max_width=int(image_size * 0.375), max_holes=1, p=0.7),
    # A.CoarseDropout(max_height=int(image_size * 0.17), max_width=int(image_size * 0.17), max_holes=2, p=0.7),
    A.Normalize()
])

val_tfs = A.Compose([
    A.Resize(image_size, image_size),
    A.Normalize()
])

# %%
t_df = meta_df[:-100]
# t_df = pd.concat([meta_df[:-100], ul_df[:-100]], ignore_index=True)
v_df = meta_df[-100:]

CFG2 = CFG
CFG2.BATCH_SIZE = 16
CFG2.num_workers = 2

dm = wav_datamodule(t_df, v_df, birds_df, cfg=CFG2)
# dm = wav_datamodule(t_df, v_df, cfg=CFG, train_tfs=train_tfs, val_tfs=val_tfs)

x, y = next(iter(dm.train_dataloader()))
x.shape, y.shape, x.dtype, y.dtype

# %%
# librosa.display.specshow(x[0].numpy()[0], y_axis="mel", x_axis='s', sr=CFG.SR)
# plt.show()

# %%
librosa.display.specshow(x[2].numpy()[0], y_axis="mel", x_axis='s', sr=CFG.SR, cmap='gray')
plt.show()

# %%
dm = wav_datamodule(t_df, v_df, birds_df, cfg=CFG, train_tfs=train_tfs, val_tfs=val_tfs)

x, y = next(iter(dm.train_dataloader()))
x.shape, y.shape, x.dtype, y.dtype

# %%
librosa.display.specshow(x[0].numpy()[0], y_axis="mel", x_axis='s', sr=CFG.SR)
plt.show()

# %%
# img = x[0]
# img.shape, img.unsqueeze(dim=0).numpy().shape, img.expand(3, -1, -1).shape

# %%
# img.expand(3, -1, -1).permute(1, 2, 0).shape, img.expand(3, -1, -1).permute(1, 2, 0).numpy().transpose(2,0,1).shape

# %%
# train_tfs(image=img.numpy())

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


# %%
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
# ### Optimizer

# %%
# Fix Warmup Bug
class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)
        
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                    
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
            
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]


# %% [markdown]
# ### Model

# %%
print('Number of models available: ', len(timm.list_models(pretrained=True)))
print('Number of models available: ', len(timm.list_models()))
print('\nDensenet models: ', timm.list_models('eca_nfnet_*'))

# %%
backbone = 'eca_nfnet_l1'
# backbone = 'efficientnet_b4'
out_indices = (3, 4)

model = timm.create_model(
    backbone,
    features_only=True,
    pretrained=False,
    in_chans=3,
    num_classes=5,
    # out_indices=out_indices,
    )

model.feature_info.channels(), np.sum(model.feature_info.channels())


# %%
def mixup(data, targets, alpha, device):
    indices = torch.randperm(data.size(0))
    data2 = data[indices]
    targets2 = targets[indices]

    lam = torch.FloatTensor([np.random.beta(alpha, alpha)]).to(device)
    data = data * lam + data2 * (1 - lam)
    
    targets = targets * lam + targets2 * (1 - lam)
    return data, targets

    # data += data2
    # targets += targets2
    # return data, targets.clip(max=1)


# %%
class GeMModel(pl.LightningModule):
    def __init__(self, cfg = CFG, pretrained = True):
        super().__init__()

        self.cfg = cfg
        
        out_indices = (3, 4)

        self.criterion = FocalLossBCE()

        self.train_acc = tm.classification.MulticlassAccuracy(num_classes=self.cfg.N_LABELS)
        self.val_acc = tm.classification.MulticlassAccuracy(num_classes=self.cfg.N_LABELS)

        # self.train_acc = tm.classification.MultilabelAccuracy(num_labels=self.cfg.N_LABELS)
        self.val_macc = tm.classification.MultilabelAccuracy(num_labels=self.cfg.N_LABELS)

        self.train_auroc = tm.classification.MulticlassAUROC(num_classes=self.cfg.N_LABELS)
        self.val_auroc = tm.classification.MulticlassAUROC(num_classes=self.cfg.N_LABELS)

        # self.model_name = self.cfg.model_name
        print(self.cfg.model_name)
        
        self.backbone = timm.create_model(
            self.cfg.model_name, 
            features_only=True,
            pretrained=pretrained,
            in_chans=3,
            num_classes=self.cfg.N_LABELS,
            out_indices=out_indices,
        )

        feature_dims = self.backbone.feature_info.channels()

        self.global_pools = torch.nn.ModuleList([GeM() for _ in out_indices])
        self.mid_features = np.sum(feature_dims)
        
        self.neck = torch.nn.BatchNorm1d(self.mid_features)
        self.head = torch.nn.Linear(self.mid_features, self.cfg.N_LABELS)

    def forward(self, x):
        ms = self.backbone(x)
        
        h = torch.cat([global_pool(m) for m, global_pool in zip(ms, self.global_pools)], dim=1)
        x = self.neck(h)
        x = self.head(x)
        
        return x
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.LEARNING_RATE, weight_decay=CFG.weight_decay)
        
        if self.cfg.USE_SCHD:
            scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg.COS_EPOCHS)
            scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=self.cfg.WARM_EPOCHS, after_scheduler=scheduler_cosine)

            return [optimizer], [scheduler_warmup]
        else:
            # LRscheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
            
            # return [optimizer], [LRscheduler]
            return optimizer

    def step(self, batch, batch_idx, mode='train'):
        x, y = batch

        if self.cfg.MIXUP and mode == 'train':
            x, y = mixup(x, y, 0.5, self.cfg.device)
        
        preds = self(x)
        
        loss = self.criterion(preds, y)
        
        if mode == 'train':
            self.train_acc(preds, y.argmax(1))
            # self.train_auroc(preds, y.argmax(1))
        else:
            self.val_acc(preds, y.argmax(1))
            self.val_macc(preds, y)
            # self.val_auroc(preds, y.argmax(1))
        
        self.log(f'{mode}/loss', loss, on_step=True, on_epoch=True)
        # self.log(f'{mode}/kl_loss', kl_loss, on_step=True, on_epoch=True)

        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, mode='train')
        self.log(f'train/acc', self.train_acc, on_step=True, on_epoch=True)
        # self.log(f'train/auroc', self.train_auroc, on_step=True, on_epoch=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, mode='val')
        self.log(f'val/acc', self.val_acc, on_step=True, on_epoch=True)
        self.log(f'val/macc', self.val_macc, on_step=True, on_epoch=True)
        # self.log(f'val/auroc', self.val_auroc, on_step=True, on_epoch=True)
    
        return loss
    
    def on_train_epoch_end(self):
        self.train_acc.reset()

    def on_validation_epoch_end(self):
        self.val_acc.reset()
        self.val_macc.reset()


# %%
model = GeMModel(CFG)

# %%
foo = model(x)

# %%
foo.shape

# %% [markdown]
# ### Split

# %%
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit


# %%
def upsample_data(df, thr=20):
    # get the class distribution
    class_dist = df['primary_label'].value_counts()

    # identify the classes that have less than the threshold number of samples
    down_classes = class_dist[class_dist < thr].index.tolist()

    # create an empty list to store the upsampled dataframes
    up_dfs = []

    # loop through the undersampled classes and upsample them
    for c in down_classes:
        # get the dataframe for the current class
        class_df = df.query("primary_label==@c")
        # find number of samples to add
        num_up = thr - class_df.shape[0]
        # upsample the dataframe
        class_df = class_df.sample(n=num_up, replace=True, random_state=CFG.random_seed)
        # append the upsampled dataframe to the list
        up_dfs.append(class_df)

    # concatenate the upsampled dataframes and the original dataframe
    up_df = pd.concat([df] + up_dfs, axis=0, ignore_index=True)
    
    return up_df


# %%
sss = StratifiedShuffleSplit(n_splits=1, test_size=1-CFG.split_fraction, random_state=CFG.random_seed)
train_idx, val_idx = next(sss.split(meta_df.filename, meta_df.primary_label))

t_df = meta_df.iloc[train_idx]
v_df = meta_df.iloc[val_idx]

print(t_df.shape)

if not CFG.USE_SECONDARY:
    t_df = t_df[t_df['secondary_labels'] == '[]']

if CFG.USE_UPSAMPLE:
    t_df = upsample_data(t_df, thr=CFG.up_thr)

if CFG.USE_UL:
    t_df = pd.concat([t_df, ul_df], ignore_index=True)

t_df.shape, v_df.shape

# %%
# t_df = t_df[t_df['rating'] > 1]
# t_df.shape

# %% [markdown]
# ### Train

# %%
# dm = wav_datamodule(t_df,v_df)
dm = wav_datamodule(t_df, v_df, birds_df, CFG, train_tfs=train_tfs, val_tfs=val_tfs) 

# %%
len(dm.train_dataloader()), len(dm.val_dataloader())

# %%
model = GeMModel(CFG)

# %%
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback, LearningRateMonitor

# %%
run_name = f'{CFG.model_name} {CFG.LEARNING_RATE} {CFG.N_EPOCHS} eps {CFG.comment}'

# %%
wandb_logger = WandbLogger(
    name=run_name,
    project=CFG.project,
    job_type='train',
    save_dir=CFG.RESULTS_DIR,
    # config=cfg,
)

# %%
loss_ckpt = pl.callbacks.ModelCheckpoint(
    monitor='val/loss',
    auto_insert_metric_name=False,
    dirpath=CFG.CKPT_DIR / run_name,
    filename='ep_{epoch:02d}_loss_{val/loss:.5f}',
    save_top_k=2,
    mode='min',
)

# %%
acc_ckpt = pl.callbacks.ModelCheckpoint(
    monitor='val/acc',
    auto_insert_metric_name=False,
    dirpath=CFG.CKPT_DIR / run_name,
    filename='ep_{epoch:02d}_acc_{val/acc:.5f}',
    save_top_k=2,
    mode='max',
)

# %%
lr_monitor = LearningRateMonitor(logging_interval='step')

# %%
CFG.device

# %%
trainer = pl.Trainer(
    max_epochs=CFG.N_EPOCHS,
    deterministic=True,
    accelerator=CFG.device,
    default_root_dir=CFG.RESULTS_DIR,
    gradient_clip_val=0.5, 
    # gradient_clip_algorithm="value",
    logger=wandb_logger,
    callbacks=[loss_ckpt, acc_ckpt, lr_monitor],
    
)

# %%
trainer.fit(model, dm)

# %%
wandb.finish()

# %% [markdown]
# ### Predict

# %%
x, y = next(iter(dm.train_dataloader()))

# %%
foo = model(x)
# foo = model(x.to(CFG.device))
foo.shape

# %%
y[1]

# %%
foo.sigmoid().topk(3,dim=-1)

# %%
topk = foo.sigmoid().topk(3,dim=-1)

# %%
vals = topk[0].detach().numpy()
idx = topk[1].detach().numpy()
vals.shape, idx.shape

# %%
# idx, vals

# %%
np.concatenate([vals,idx], axis=-1).shape

# %%
torch.nn.functional.softmax(foo[0], dim=-1)

# %%
torch.nn.functional.softmax(foo, dim=-1).max(dim=-1)

# %%
torch.nn.functional.softmax(foo, dim=-1).argmax(dim=-1)

# %%
y.argmax(dim=-1)

# %%

# %%

# %%

# %%

# %%

# %%
