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
# from warmup_scheduler import GradualWarmupScheduler
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau # Learning rate schedulers

import albumentations as A
# from albumentations.pytorch import ToTensorV2

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
    comment = '128neck'

    ### model
    model_name = 'eca_nfnet_l0' # 'resnet34', 'resnet200d', 'efficientnet_b1_pruned', 'efficientnetv2_m', efficientnet_b7 

    image_size = 256
    bottleneck_dim = 128
    
    ROOT_FOLDER = train_dir
    IMAGES_DIR = ROOT_FOLDER / 'train_images'
    PNG_DIR = ROOT_FOLDER / f'pngs_{image_size}'
    FILES_CSV = ROOT_FOLDER / 'train_files.csv'
    TRAIN_CSV = ROOT_FOLDER / 'train.csv'
    TRAIN_DESC_CSV = ROOT_FOLDER / 'train_series_descriptions.csv'
    COORDS_CSV = ROOT_FOLDER / 'train_label_coordinates.csv'

    RESULTS_DIR = train_dir / 'results'
    CKPT_DIR = RESULTS_DIR / 'ckpt'

    classes = classes

    split_fraction = 0.95

    MIXUP = False

    ### training
    BATCH_SIZE = 32
    
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
train_df.sample(5)

# %%
coords_df.sample(2)

# %%
coords_df.condition.unique()

# %%
coords_df.cl.unique()

# %%
# coords_df.groupby(['study_id','series_id']).instance.count()

# %%
train_desc_df.sample(5)

# %%
files_df.sample(3)

# %%
files_df.shape

# %%
# labels, potive imgs, total imgs
coords_df.instance_id.nunique(), coords_df.ss_id.nunique(), files_df.shape[0]

# %%
coords_df.sample(3)

# %%
files_df.shape, coords_df.shape

# %%
coords_df.instance_id.nunique(), coords_df.filename.nunique()

# %%
# conditions per file
grp = coords_df.groupby('instance_id').condition
grp.count().min(), grp.count().max(), grp.count().mean()

# %%
inst_id = '2509953825_3594374345_13'

coords_df[coords_df['instance_id'] == inst_id]

# %%
coords_df[coords_df['instance_id'] == inst_id].cl.to_list()

# %%
# list(coords_df.filename.unique())

# %%

# %% [markdown]
# ### Dataset

# %%
from dataset import rsna_dataset

# %%
selection = files_df[files_df['healthy'] == True]
selection.shape

# %%

# %%
dset = rsna_dataset(selection, coords_df, CFG)

print(dset.__len__())

img, label, = dset.__getitem__(2)
print(img.shape, label.shape)
print(img.dtype, label.dtype)

# %%
label

# %%
selection = files_df[files_df['healthy'] == False]
selection.shape

# %%
dset = rsna_dataset(selection, coords_df, CFG)

print(dset.__len__())

img, label, = dset.__getitem__(8)
print(img.shape, label.shape)
print(img.dtype, label.dtype)
label

# %%
img.mean(), img.std(), img.min(), img.max()

# %%
plt.imshow(img[0], cmap='gray')

# %%

# %% [markdown]
# ### Datamodule

# %%
from dataset import rsna_dataset


# %%
class rsna_datamodule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, coords_df, cfg=CFG, train_tfs=None, val_tfs=None):
        super().__init__()
        
        self.train_df = train_df
        self.val_df = val_df
        self.coords_df = coords_df
        
        self.train_bs = cfg.BATCH_SIZE
        self.val_bs = cfg.BATCH_SIZE

        self.train_tfs = train_tfs
        self.val_tfs = val_tfs

        self.cfg = cfg
        
        self.num_workers = cfg.num_workers
        
    def train_dataloader(self):
        train_ds = rsna_dataset(self.train_df, self.coords_df, self.cfg, tfs=self.train_tfs, mode='train')
        
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
        val_ds = rsna_dataset(self.val_df, self.coords_df, self.cfg, tfs=self.val_tfs, mode='val')
        
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

# %%


t_df = selection[:-100]
# t_df = pd.concat([meta_df[:-100], ul_df[:-100]], ignore_index=True)
v_df = selection[-100:]

CFG2 = CFG()
# CFG2 = copy.deepcopy(CFG)
CFG2.BATCH_SIZE = 16
CFG2.num_workers = 2

dm = rsna_datamodule(t_df, v_df, coords_df, cfg=CFG2)
# dm = wav_datamodule(t_df, v_df, cfg=CFG, train_tfs=train_tfs, val_tfs=val_tfs)

x, y = next(iter(dm.train_dataloader()))
x.shape, y.shape, x.dtype, y.dtype

# %%
plt.imshow(x[10][0], cmap='gray')

# %% [markdown]
# #### Check Transforms

# %%
image_size = CFG.image_size

img_mean = (0.485, 0.456, 0.406)
img_std = (0.229, 0.224, 0.225)

train_tfs = A.Compose([
    # A.HorizontalFlip(p=0.5),
    A.Resize(image_size, image_size),
    A.CoarseDropout(max_height=int(image_size * 0.2), max_width=int(image_size * 0.2), max_holes=4, p=0.7),
    A.Normalize(mean=img_mean, std=img_std)
])

val_tfs = A.Compose([
    A.Resize(image_size, image_size),
    A.Normalize(mean=img_mean, std=img_std)
])

# %%
dm = rsna_datamodule(t_df, v_df, coords_df, cfg=CFG2, train_tfs=train_tfs, val_tfs=val_tfs)
# dm = wav_datamodule(t_df, v_df, cfg=CFG, train_tfs=train_tfs, val_tfs=val_tfs)

x, y = next(iter(dm.train_dataloader()))
x.shape, y.shape, x.dtype, y.dtype

# %%
y[0]

# %%
plt.imshow(x[2][0], cmap='gray')

# %%
del dm


# %%

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


# %%

# %% [markdown]
# ### Model

# %%
backbone = 'eca_nfnet_l1'
# backbone = 'efficientnet_b4'
out_indices = (3, 4)

model = timm.create_model(
    backbone,
    features_only=True,
    pretrained=False,
    in_chans=3,
    num_classes=dset.num_classes,
    # out_indices=out_indices,
    )

model.feature_info.channels(), np.sum(model.feature_info.channels())

# %%
data_config = timm.data.resolve_model_data_config(model)

# %%
data_config


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

        self.dev = cfg.device
        
        out_indices = (3, 4)

        self.bottleneck_dim = cfg.bottleneck_dim

        self.criterion = FocalLossBCE()

        wrapped_acc = ClasswiseWrapper(MultilabelAccuracy(num_labels=self.cfg.N_LABELS, average='none'), labels=classes, prefix='multiacc/')
        wrapped_f1 = ClasswiseWrapper(MultilabelF1Score(num_labels=self.cfg.N_LABELS, average='none'), labels=classes, prefix='multif1/')
        
        metrics = MetricCollection({
            # 'macc': MultilabelAccuracy(num_labels=self.cfg.N_LABELS),
            'none_acc': wrapped_acc,
            'mpr': MultilabelPrecision(num_labels=self.cfg.N_LABELS),
            'mrec': MultilabelRecall(num_labels=self.cfg.N_LABELS),
            'f1': MultilabelF1Score(num_labels=self.cfg.N_LABELS),
            'none_f1': wrapped_f1,
        })

        self.train_metrics = metrics.clone(prefix='train/')
        self.valid_metrics = metrics.clone(prefix='val/')
        
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
        self.bottleneck = torch.nn.Linear(self.mid_features, self.bottleneck_dim)
        self.bottleneck_bn = torch.nn.BatchNorm1d(self.bottleneck_dim)
        self.head = torch.nn.Linear(self.bottleneck_dim, self.cfg.N_LABELS)

    def pre_forward(self, x):
        ms = self.backbone(x)
        
        h = torch.cat([global_pool(m) for m, global_pool in zip(ms, self.global_pools)], dim=1)
        x = self.neck(h)
        x = self.bottleneck(x)
        x = self.bottleneck_bn(x)

        return x
    
    def forward(self, x):
        x = self.pre_forward(x)

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


# %%
model = GeMModel(CFG)

# %%
foo = model(x)

# %%
x.shape, foo.shape

# %%

# %% [markdown]
# ### Split

# %%
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

# %%
train_cols = ['filename', 'instance_id', 'series_description', 'condition']

# %%
healthy_df = files_df[files_df.healthy == True]
healthy_df['condition'] = 'H'

healthy_df.shape

# %%
healthy_df = healthy_df.sample(frac=.3)
healthy_df.shape

# %%
healthy_df.head(2)

# %%
coords_df.head(2)

# %%
coords_df.loc[:, train_cols].sample(5)

# %%
healthy_df.loc[:, train_cols].sample(5)

# %%
train_df = pd.concat([coords_df.loc[:, train_cols], healthy_df.loc[:, train_cols]], ignore_index=True)
train_df.shape

# %%
train_df.filename.nunique()

# %%
sss = StratifiedShuffleSplit(n_splits=1, test_size=1-CFG.split_fraction, random_state=CFG.random_seed)
train_idx, val_idx = next(sss.split(train_df.filename, train_df.condition))

t_df = train_df.iloc[train_idx]
v_df = train_df.iloc[val_idx]

t_df.shape, v_df.shape

# %%
bool(set(t_df.instance_id.tolist()) & set(v_df.instance_id.tolist()))

# %%
intersection = list(set(t_df.instance_id.tolist()) & set(v_df.instance_id.tolist()))
len(intersection)

# %%
intersection[0]

# %%
coords_df[coords_df.instance_id == '3922074884_1280331258_30']

# %%
v_df[v_df.instance_id == '3922074884_1280331258_30']

# %%

# %% [markdown]
# #### Filter classes

# %%
CFG.classes

# %%
t_df[t_df['condition'].isin(CFG.classes)].shape, v_df[v_df['condition'].isin(CFG.classes)].shape

# %%
t_df = t_df[t_df['condition'].isin(CFG.classes)]
v_df = v_df[v_df['condition'].isin(CFG.classes)]

# %%
t_df.shape, v_df.shape

# %%
69576/128, 3662/128

# %% [markdown]
# ### Train

# %%
CFG.BATCH_SIZE, CFG.device

# %%
dm = rsna_datamodule(t_df, v_df, coords_df, cfg=CFG, train_tfs=train_tfs, val_tfs=val_tfs)
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
model = GeMModel(CFG)

# %% [markdown]
# #### Fit

# %%
trainer.fit(model, dm)

# %%
# np.array([0.0625, 0.9375, 0.8125, 0.0000, 1.0000, 1.0000, 0.0938, 0.8125, 0.7500,
#         0.8750, 0.2500, 0.0625, 0.9688, 0.9062, 0.9688, 0.1875, 1.0000, 0.6562,
#         0.7500, 0.7812, 0.6875, 0.8438, 0.8750, 0.9688, 0.4062, 0.2188]).mean()

# %%
wandb.finish()

# %% [markdown]
# ### Predict

# %%
x, y = next(iter(dm.train_dataloader()))

# %%
foo = model(x)
# foo = model(x.to(CFG.device)).detach().cpu()
foo.shape

# %%
foo[0]

# %%
foo[0].sigmoid()

# %%
macc = tm.classification.MultilabelAccuracy(num_labels=26)
mapp = tm.classification.MultilabelPrecision(num_labels=26)
marr = tm.classification.MultilabelRecall(num_labels=26)
maff = tm.classification.MultilabelF1Score(num_labels=26)

# %%
macc(foo, y), mapp(foo, y), marr(foo, y), maff(foo, y)

# %%
((foo.sigmoid() > 0.5) == y).sum()/16/26

# %%
((foo.sigmoid() > 0.5) != y).sum()

# %%
(1037.3 - 1026)/1026 * 100

# %%
torch.argwhere(y > 0).T

# %%
bar = foo.sigmoid().numpy()
np.argwhere(bar > 0.5).T

# %%
bar[12]

# %% [markdown]
# # 

# %%
torch.nn.functional.sigmoid(foo[0])

# %%
# foo.sigmoid().topk(1, dim=-1)

# %%
