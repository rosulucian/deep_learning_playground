# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %% _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5"
import os
import wandb
import torch
import random
import efficientnet_pytorch

import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import pytorch_lightning as pl

import torch.nn.functional as F

from albumentations.pytorch import ToTensorV2
from efficientnet_pytorch import EfficientNet

# %%
print(os.getenv('wandb_api_key'))

# %%
# os.environ

# %%
# print(key=os.environ['wandb_api_key'])

# %%
# wandb.login(key=os.environ['wandb_api_key'])
wandb.login(key='4015c794f679e3b16458b585c36e77213d391bc2')

# %%
accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'  # Check GPU
accelerator, torch.__version__, pl.__version__

# %% [markdown]
# ### Config

# %%
data_dir = 'D:\data\HMS'

cfg = {
    'encoder': 'efficientnet-b0',
    'seed': 42,
    'cls_num': 6,
    'spect_dir': f'{data_dir}/train_spectrograms',
    'train_bs': 32,
    'val_bs': 32,
    'num_workers': 4,
    'max_epochs': 8,
    'lr': 1e-4,
    'comment': '',
}

results_dir = f'{data_dir}/results'
ckpt_path = f'{results_dir}/ckpt'

ckpt_path

# %%
# type(cfg)

# %%
try:
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)
except Exception as e:
    print(e)
    print('dir already exists')


# %%
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(cfg['seed'])

# %% [markdown]
# ### Data

# %%
train_df = pd.read_csv(f'{data_dir}/train.csv')
train_df.columns

# %%
vote_cols = [x for x in train_df.columns if 'vote' in x]
# vote_cols

train_df['total_votes'] = train_df.loc[:, vote_cols].sum(axis=1)
train_df['cons_votes'] = train_df.loc[:, vote_cols].max(axis=1)

train_df['consensus'] = train_df['cons_votes']/train_df['total_votes']

# %%
vote_cols

# %%
train_df.sample(5)

# %%
train_df.iloc[:, 3:].head()

# %%
train_df[vote_cols].div(train_df['total_votes'], axis=0).sample(5)

# %% [markdown]
# ### Dataset

# %%
from hms_dataset import spectro_dataset

# %%
dset = spectro_dataset(df=train_df, base_dir=cfg['spect_dir'], cols=vote_cols)
print(dset.__len__())

spect, target, = dset.__getitem__(1)
print(spect.shape, target.shape)
print(spect.dtype, target.dtype)


# %%
def print_spectros(spectro, lg=True):
    plt.figure()
    f, axs = plt.subplots(4,1)
    f.set_figheight(4)
    f.set_figwidth(12)
    
    # spectro = np.clip(spectro,np.exp(-4),np.exp(8))
    # if lg == True:
    #     spectro = np.log(spectro)
    
    for i in range(4):
        start = i*100+1
        
        img = spectro[i]
        
        # img = spectro.iloc[:,start:start+100]
            
        axs[i].imshow(img, origin='lower', cmap='jet')
        
    plt.show()

print_spectros(spect)


# %%
# print_spectros(spect, lg=False)

# %% [markdown]
# ### Datamodule

# %%
class spectro_datamodule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, train_bs=cfg['train_bs'], val_bs=cfg['val_bs'], num_workers=cfg['num_workers'],):
        super().__init__()
        
        self.train_df = train_df
        self.val_df = val_df
        
        self.train_bs = train_bs
        self.val_bs = val_bs
        
        self.num_workers = num_workers
        
    def train_dataloader(self):
        train_ds = spectro_dataset(self.train_df, base_dir=cfg['spect_dir'], cols=vote_cols)
        
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=self.train_bs,
            pin_memory=False,
            drop_last=False,
            shuffle=True,
            num_workers=self.num_workers,
        )
        
        return train_loader
        
    def val_dataloader(self):
        val_ds = spectro_dataset(self.val_df, base_dir=cfg['spect_dir'], cols=vote_cols)
        
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=self.val_bs,
            pin_memory=False,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
        )
        
        return val_loader


# %%
t_df = train_df[:-400]
v_df = train_df[-400:]

bs = 64

dm = spectro_datamodule(
    t_df,
    v_df, 
    train_bs=bs,
)

x, y = next(iter(dm.train_dataloader()))
x.shape, y.shape, x.dtype, y.dtype

# %% [markdown]
# ### Model

# %%
# from torchmetrics.classification import Accuracy as Accuracy
import torchmetrics as tm 


# %%
class spectro_model(pl.LightningModule):
    def __init__(self, cfg,):
        super().__init__()
        
        self.cfg = cfg
        self.encoder=cfg['encoder']
        
#         self.criterion = nn.NLLLoss()
        self.train_acc = tm.classification.MulticlassAccuracy(num_classes=cfg['cls_num'])
        self.val_acc = tm.classification.MulticlassAccuracy(num_classes=cfg['cls_num'])
        
        self.criterion = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        
        self.model = EfficientNet.from_pretrained(
            self.encoder, 
            in_channels=4, 
            num_classes=cfg['cls_num'],
        )
        
    def configure_optimizers(self):
        return torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=0.015)
        
    def forward(self, x):
        return self.model(x)
    
    def step(self, batch, batch_idx, mode='train'):
        x, y = batch
        
        preds = self(x)
        
        # loss = self.criterion(preds, y)

        kl_loss = self.kl_loss(F.log_softmax(preds, dim=1), y)
        loss = kl_loss
        
        if mode == 'train':
            self.train_acc(preds, y.argmax(1))
        else:
            self.val_acc(preds, y.argmax(1))
        
#         values = {f'{mode}/loss': loss, }
#         self.log_dict(values)
        
        self.log(f'{mode}/loss', loss, on_step=True, on_epoch=True)
        self.log(f'{mode}/kl_loss', kl_loss, on_step=True, on_epoch=True)

        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, mode='train')
        self.log(f'train/acc', self.train_acc, on_step=True, on_epoch=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        loss =  self.step(batch, batch_idx, mode='val')
        self.log(f'val/acc', self.val_acc, on_step=True, on_epoch=True)
    
        return loss
    
    def on_train_epoch_end(self):
        self.train_acc.reset()
        self.val_acc.reset()


# %%
model = spectro_model(cfg)

# %%
# foo = model(x.nan_to_num(0))
foo = model(x)
foo.shape

# %%
y.shape, foo.shape

# %%
y.argmax(1)

# %%
y[:5], foo[:5]

# %%
model.training_step((x, y), 0)

# %% [markdown]
# ### Split

# %%
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

# %%
# rs = ShuffleSplit(n_splits=5, test_size=.1, random_state=seed)
# train_idx, val_idx = next(rs.split(sequences_df))

# %%
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=cfg['seed'])
train_idx, val_idx = next(sss.split(train_df.eeg_id, train_df.expert_consensus))

t_df = train_df.iloc[train_idx]
v_df = train_df.iloc[val_idx]

t_df.shape, v_df.shape

# %% [markdown]
# ### Train

# %%
dm = spectro_datamodule(t_df,v_df)

# %%
len(dm.train_dataloader()), len(dm.val_dataloader())

# %%
model = spectro_model(cfg)

# %%
from pytorch_lightning.loggers import WandbLogger

# %%
wandb_logger = WandbLogger(
    name=f'{cfg["encoder"][-2:]} {cfg["max_epochs"]} eps {cfg["comment"]}',
    project='HMS-spectro-local',
    job_type='train',
    save_dir=results_dir,
    config=cfg,
)

# %%
# wandb.init()

# %%
from pytorch_lightning.callbacks import Callback, LearningRateMonitor

# %%
loss_ckpt = pl.callbacks.ModelCheckpoint(
    monitor='val/loss',
    dirpath=ckpt_path,
    filename='loss-{epoch:02d}-{val/loss:.2f}',
    save_top_k=1,
    mode='min',
)

# %%
trainer = pl.Trainer(
    max_epochs=cfg['max_epochs'],
    deterministic=True,
    accelerator=accelerator,
    default_root_dir=results_dir,
    logger=wandb_logger,
)

# %%
trainer.fit(model, dm)

# %%
wandb.finish()

# %% [markdown]
# ### Predict

# %%
# foo = model(x.to(accelerator))
foo = model(x)
foo.shape

# %%
foo[5:10]

# %%
y[5:10]

# %%
torch.nn.functional.softmax(foo, dim=-1)[:5]

# %%
torch.nn.functional.softmax(foo, dim=-1).argmax(dim=-1)

# %%
y.argmax(dim=-1)

# %%
(torch.nn.functional.softmax(foo, dim=-1).argmax(dim=-1) == y.argmax(dim=-1)).sum()

# %%
# model.save_hyperparameters()

# %%
