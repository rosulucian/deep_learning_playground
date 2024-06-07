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
import torchaudio
import torchvision

import numpy as np
import pandas as pd
import torchmetrics as tm 
import plotly.express as px
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from torch import nn
from pathlib import Path
from IPython.display import Audio
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW, RMSprop # optmizers
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau # Learning rate schedulers

import albumentations as A
# from albumentations.pytorch import ToTensorV2

import timm

# %%
print('timm version', timm.__version__)
print('torch version', torch.__version__)

# %%
# detect and define device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

# %%
train_dir = Path('E:\data\BirdCLEF')

class CFG:
    DEBUG = False # True False

    USE_MISSING_LABELS = False

    # Competition Root Folder
    ROOT_FOLDER = train_dir
    AUDIO_FOLDER = train_dir / 'train_audio'
    UNLABELED_FOLDER = train_dir / 'unlabeled_soundscapes'
    DATA_DIR = train_dir / 'spectros'
    TRAIN_CSV = train_dir / 'train_metadata.csv'
    RESULTS_DIR = train_dir / 'results'
    CKPT_DIR = train_dir / 'ckpt'

    num_workers = 8
    # Maximum decibel to clip audio to
    TOP_DB = 100
    # Minimum rating
    MIN_RATING = 3.0
    # Sample rate as provided in competition description
    SR = 32000

    image_size = 128
    
    ### split train and validation sets
    split_fraction = 0.95
    
    ### model
    model_name = 'eca_nfnet_l0' # 'resnet34', 'resnet200d', 'efficientnet_b1_pruned', 'efficientnetv2_m', efficientnet_b7 ...  
    
    ### training
    BATCH_SIZE = 4
    # N_EPOCHS = 3 if DEBUG else 40
    N_EPOCHS = 20
    LEARNING_RATE = 5*1e-6
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    random_seed = 42
    comment = 'first'

mel_spec_params = {
    "sample_rate": CFG.SR,
    "n_mels": 128,
    "f_min": 20,
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

display(sample_submission.head())


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

# %% [markdown]
# ### Prepare dataframe

# %%
files = [f for f in sorted(os.listdir(CFG.UNLABELED_FOLDER))]
len(files)

# %%
file_path = CFG.UNLABELED_FOLDER / files[0]
data, rate = torchaudio.load(file_path)
print("Audio data shape:", data.shape)
print("Sample rate:", rate)

# %%
step = 5
ranges = np.arange(0+step, 240+step, step)
ranges

# %%
results_df = pd.DataFrame(files, columns = ['file'])
results_df['range'] = [ranges] * len(results_df)
# results_df.reset_index(drop=True, inplace=True)
results_df = results_df.explode('range', ignore_index=True)
# results_df.reset_index(drop=True, inplace=True)

# %%
pd.set_option('max_colwidth', 40)

# %%
results_df.head()

# %% [markdown]
# ### Prepare dataset

# %%
from dataset import bird_dataset_inference, read_wav

# %%
files = [f for f in sorted(os.listdir(CFG.UNLABELED_FOLDER))]
len(files)

# %%
files[0]

# %%
dset = bird_dataset_inference(files, CFG.UNLABELED_FOLDER, CFG)
len(dset)

# %%
spect, filename = dset.__getitem__(0)

spect.dtype, spect.shape, filename

# %%
plt.figure(figsize=(14, 4))
librosa.display.specshow(spect[0,0].numpy(), y_axis="mel", x_axis='s', sr=CFG.SR)
plt.show()

# %%
spect = spect[0]
spect.shape

# %%
remainder = spect.shape[-1] % 48
spect.shape[-1] % 48, spect.shape[-1] // 48

# %%
splits = torch.split(spect[:,:,:-remainder], 312, dim=-1)
len(splits), splits[0].shape, splits[-1].shape

# %%
splits = torch.stack(splits)
splits.shape

# %%
librosa.display.specshow(splits[0,0].numpy(), y_axis="mel", x_axis='s', sr=CFG.SR)
plt.show()

# %% [markdown]
# ### Dataloader

# %%
from dataset import bird_dataset_inference, bird_dataset


# %%
def collate_fn(batch):
   return  torch.stack([x[0] for x in batch]), torch.stack([x[1] for x in batch])


# %%
class inference_datamodule(pl.LightningDataModule):
    def __init__(self, files, directory, cfg=CFG, tfs=None, resize_tf=None):
        super().__init__()
        
        self.files = files
        self.dir = directory

        self.tfs = tfs
        self.resize_tf = resize_tf

        self.cfg = cfg

        self.bs = self.cfg.BATCH_SIZE
        self.num_workers = cfg.num_workers
        
    def predict_dataloader(self):
        ds = bird_dataset_inference(self.files, self.dir, self.cfg)
        
        train_loader = torch.utils.data.DataLoader(
            ds,
            batch_size=self.bs,
            pin_memory=False,
            drop_last=False,
            # shuffle=True,
            # collate_fn=collate_fn,
            persistent_workers=True,
            num_workers=self.num_workers,
        )
        
        return train_loader


# %%
dm = inference_datamodule(files, CFG.UNLABELED_FOLDER)

# %%
x, filenames = next(iter(dm.predict_dataloader()))
x.shape, x.dtype

# %%
torch.flatten(x, start_dim=0, end_dim=1).shape

# %%
filenames


# %% [markdown]
# ### Load model

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

class GeMModel(pl.LightningModule):
    def __init__(self, cfg = CFG, pretrained = True):
        super().__init__()

        self.cfg = cfg
        
        out_indices = (3, 4)

        print(self.cfg.model_name)
        
        self.backbone = timm.create_model(
            self.cfg.model_name, 
            features_only=True,
            pretrained=False,
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

    def predict_step(self, batch):
        spects, files = batch
        spects = torch.flatten(spects, start_dim=0, end_dim=1)

        preds = self(spects)

        results_df = pd.DataFrame(files, columns = ['file'])
        results_df['range'] = [ranges] * len(results_df)
        results_df = results_df.explode('range', ignore_index=True)
        results_df['row_id'] = results_df.apply(lambda row: row['file'] + '_' + str(row['range']), axis=1)

        topk = preds.sigmoid().topk(3, dim=-1)
        
        vals = topk[0].cpu().detach().numpy()
        idx = topk[1].cpu().detach().numpy()

        cols = [f'top_{k+1}' for k in range(3)] + [f'top_{k+1}_idx' for k in range(3)]
        vals_df = pd.DataFrame(vals, columns=cols[:3])
        idx_df = pd.DataFrame(idx, columns=cols[3:])
        
        # results = torch.cat(results[0].tolist(), results[1].tolist())

        
        # preds_df = pd.DataFrame(results, columns=cols)

        results_df = pd.concat([results_df, vals_df, idx_df], axis=1)
        
        # return results_df, preds
        return results_df

# %%
# model_path = Path("E:\\data\\BirdCLEF\\results\\Bird-local\\g5aw82o5\\checkpoints")
model_path = Path("E:\\data\\BirdCLEF\\results\\ckpt\\eca_nfnet_l0 5e-05 30 eps mixup-plain\\ep_29_acc_0.60490.ckpt")

# model_path = model_path / os.listdir(model_path)[0]
model_path

# %%
([f'top_{k+1}' for k in range(3)] + [f'top_{k+1}_idx' for k in range(3)])

# %%
model = GeMModel.load_from_checkpoint(model_path)

# %% [markdown]
# ### Validation set

# %%
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

# %%
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=CFG.random_seed)
train_idx, val_idx = next(sss.split(meta_df.filename, meta_df.primary_label))

v_df = meta_df.iloc[val_idx]

v_df.shape

# %%
columns = ['primary_label', 'secondary_labels', 'filename']
v_df['filename'] = f'{str(CFG.AUDIO_FOLDER)}\\' + v_df['filename']
v_df = v_df[columns]
v_df['range'] = 0

# %%
v_df.head()

# %%
image_size = CFG.image_size

val_tfs = A.Compose([
    A.Resize(image_size, image_size),
    A.Normalize()
])

# %%
val_ds = bird_dataset(v_df, CFG, tfs=val_tfs, mode='val')
        
val_loader = torch.utils.data.DataLoader(
    val_ds,
    batch_size=64,
    pin_memory=False,
    drop_last=False,
    shuffle=False,
    persistent_workers=True,
    num_workers=2,
)

# %%
preds = [ ]
for x, y in val_loader:
    preds.append(model(x.to(CFG.device)).sigmoid().detach().cpu())

len(preds)

# %%
preds = [p.numpy() for p in preds]
preds = np.concatenate(preds)
preds.shape

# %%
preds.max(axis=-1)

# %%
v_df['label'] = preds.argmax(axis=-1)
v_df['label_name'] = v_df.apply(lambda row: CFG.LABELS[row['label']], axis=1)

# %%
wrong_df = v_df[v_df['primary_label'] != v_df['label_name']]
wrong_df.shape

# %%
wrong_df['primary_label'].value_counts()

# %%
v_df[v_df['primary_label'] != v_df['label_name']].sample(5)

# %%
meta_df[meta_df['primary_label'] == 'commoo3'].shape, meta_df[meta_df['primary_label'] == 'litgre1'].shape

# %%

# %% [markdown]
# ### Predict

# %%
files = [f for f in sorted(os.listdir(CFG.UNLABELED_FOLDER))]
len(files)

# %%
files[0]

# %%
# dm = inference_datamodule(files[:10], CFG.UNLABELED_FOLDER)
dm = inference_datamodule(files, CFG.UNLABELED_FOLDER)

# %%
trainer = pl.Trainer()
predictions = trainer.predict(model, dataloaders=dm)
len(predictions)

# %%
# data = predictions[0][1]
# data.shape

# %%
# data_df = pd.DataFrame(data.softmax(dim=-1).numpy(), columns=CFG.LABELS)
# data_df.head()

# %%
len(predictions)

# %%
predictions = pd.concat(predictions, ignore_index=True)
predictions.shape

# %%
predictions.shape

# %%
predictions.sample(5)

# %%
predictions[predictions['top_1'] > 0.9].shape

# %%
predictions[predictions['top_1'] < 0.05].shape

# %%
predictions[predictions['top_1'] < 0.1].sample(4)

# %%

# %%
# predictions['row_id'] = predictions.apply(lambda row: row['file'] + '_' + str(row['range']), axis=1)

# %%
# predictions[predictions['score'] > 0.95].shape

# %%
# predictions['score'].hist()

# %% [markdown]
# ### Save

# %%
predictions.drop('row_id', axis=1, inplace=True)

# %%
predictions.rename(columns={'file': 'filename'}, inplace=True)


# %%
predictions['filename']  = f'{CFG.UNLABELED_FOLDER}\\' + predictions['filename']

# %%
predictions.to_csv(train_dir / "predictions.csv", index=False)

# %%

# %%

# %%

# %%
