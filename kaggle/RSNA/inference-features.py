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
    comment = 'all-labels'

    ckpt_path = Path(r"E:\data\RSNA2024\results\ckpt\eca_nfnet_l0 5e-05 10 eps 128neck\ep_04_loss_0.17137.ckpt")
    embeds_path = Path(r"E:\data\RSNA2024\embeddings")
    stacked_path = Path(r"E:\data\RSNA2024\embeddings_stacked")

    ### model
    model_name = 'eca_nfnet_l0' # 'resnet34', 'resnet200d', 'efficientnet_b1_pruned', 'efficientnetv2_m', efficientnet_b7 

    image_size = 256
    bottleneck_dim = 128
    
    ROOT_FOLDER = train_dir
    DEST_FOLDER = train_dir
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
    BATCH_SIZE = 128
    
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
labels = CFG.classes
pred_labels = [f'pred_{l}' for l in labels]

labels, pred_labels

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

# %%
files_df.shape, coords_df.shape

# %%
# files_df.cl.unique()

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
from dataset import rsna_inf_dataset

# %%
# test healthy
dset = rsna_inf_dataset(files_df[files_df.healthy != True], coords_df, CFG)

print(dset.__len__())

img, ids, label, = dset.__getitem__(2)
print(img.shape)
print(img.dtype, label)

# %%
# test unhealthy
dset = rsna_inf_dataset(files_df, coords_df, CFG)

print(dset.__len__())

img, ids, label, = dset.__getitem__(2)
print(img.shape)
print(img.dtype, label)

# %%
img.mean(), img.std(), img.min(), img.max()

# %%
plt.imshow(img[0], cmap='gray')

# %%

# %% [markdown]
# ### Datamodule

# %%
from dataset import rsna_inf_dataset


# %%
class inference_datamodule(pl.LightningDataModule):
    def __init__(self, files_df, coords_df, cfg=CFG, tfs=None):
        super().__init__()
        
        self.files_df = files_df
        self.coords_df = coords_df
        
        self.bs = cfg.BATCH_SIZE
        self.tfs = tfs
        self.cfg = cfg
        
        self.num_workers = cfg.num_workers
        
    def predict_dataloader(self):
        ds = rsna_inf_dataset(self.files_df, self.coords_df, self.cfg, tfs=self.tfs, mode='train')
        
        train_loader = torch.utils.data.DataLoader(
            ds,
            batch_size=self.bs,
            pin_memory=False,
            drop_last=False,
            # shuffle=True,
            persistent_workers=True,
            num_workers=self.num_workers,
        )
        
        return train_loader


# %%
t_df = selection[:-100]


CFG2 = CFG()
# CFG2 = copy.deepcopy(CFG)
CFG2.BATCH_SIZE = 16
CFG2.num_workers = 2

dm = inference_datamodule(files_df, t_df, cfg=CFG2)

x, ids, y = next(iter(dm.predict_dataloader()))
x.shape, len(y), x.dtype,

# %%
y[0]

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
dm = inference_datamodule(files_df, t_df, cfg=CFG2, tfs=val_tfs)
# dm = wav_datamodule(t_df, v_df, cfg=CFG, train_tfs=train_tfs, val_tfs=val_tfs)

x, ids, y = next(iter(dm.predict_dataloader()))
x.shape, x.dtype,

# %%
y[0], ids[0]

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
class featureModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()

        self.model = model

    def forward(self, x):
        embeds = self.model.pre_forward(x)
        preds = self.model.head(embeds)
        
        return preds, embeds

    def predict_step(self, batch):
        imgs, ids, targets = batch

        preds, embeds = self(imgs)

        t_df = pd.DataFrame(targets.detach().cpu().numpy(), columns=CFG.classes)
        p_df = pd.DataFrame(preds.sigmoid().detach().cpu().numpy(), columns=pred_labels)

        results_df = pd.DataFrame(ids, columns = ['ids'])
        results_df = pd.concat([results_df, t_df, p_df], axis=1)
    
        return embeds.detach().cpu().numpy(), results_df, ids


# %%
model = GeMModel(CFG)
fmodel = featureModel(model)

# %%
preds, results, ids = fmodel.predict_step((x, ids, y))

# %%
preds.shape

# %%
ids[:4]

# %%
preds[0]

# %%
results.shape

# %%
results

# %% [markdown]
# ### Inference

# %%
CFG.ckpt_path

# %%
model = GeMModel.load_from_checkpoint(checkpoint_path=CFG.ckpt_path, cfg=CFG)
model = featureModel(model)

accelerator = CFG.device

trainer = pl.Trainer(
    accelerator=accelerator,
)

print('model loaded')

# %%
model.to(accelerator)
model.eval()
model.freeze()

accelerator

# %%
files_df.shape, files_df.filename.nunique(), coords_df.filename.nunique()

# %%
train_cols = ['filename', 'healthy', 'series_description', 'instance_id']

# %%
files_df.loc[:, train_cols].head(2)

# %%
# files_df.condition.unique()

# %%

# %% [markdown]
# #### Predict

# %%
CFG.BATCH_SIZE, CFG.device

# %%
# make sure files_df has the correct condition
# dm = inference_datamodule(healthy_df[:248], tfs=val_tfs)
dm = inference_datamodule(files_df, coords_df, tfs=val_tfs)
# dm = inference_datamodule(coords_df, tfs=val_tfs)

# %%
trainer = pl.Trainer(accelerator=CFG.device)
predictions = trainer.predict(model, dataloaders=dm)

# %%
len(predictions)

# %%
predictions[0][0].shape, predictions[0][1].shape, len(predictions[0][2])

# %%
predictions[0][2][:4]

# %%
predictions[0][1].head(2)

# %%
(preds, results, ids) = list(map(list, zip(*predictions)))

# %% [markdown]
# ### Save predictions

# %% [markdown]
# #### Save results

# %%
results = pd.concat(results, ignore_index=True)
results.shape

# %%
results.sample(2)

# %%
results['study_id'] = results['ids'].apply(lambda x: x.split('_')[0])
results['series_id'] = results['ids'].apply(lambda x: (x.split('_')[1]))
results['instance'] = results['ids'].apply(lambda x: int(x.split('_')[-1]))

# %%
results.head(2)

# %%
results.study_id.nunique()

# %%
#  dont sort
# table and embeding order must match

# %%
# results = results.sort_values(by=['study_id', 'series_id', 'instance'], ascending=[True, True, True], ignore_index=True)

# results.shape

# %%
results.study_id.nunique()

# %%
# results.head(5)

# %%
results.index[results['study_id'] == '1002894806']

# %%
results[results['study_id'] == '1002894806'].head()

# %%
results[results.pred_H > 0.8].shape, results[results.pred_H < 0.8].shape

# %%
results[results.H < 1].shape

# %%
results[results.H < 1].sample(2)

# %%
results.to_csv(CFG.DEST_FOLDER / 'predictions.csv', index=False)

# %% [markdown]
# #### Save embeddings

# %%
stacked = np.vstack(preds)
stacked.shape

# %%
np.save(CFG.stacked_path / f'stacked.npy', stacked)

# %%

# %%
# for (embeds, ids) in zip(preds, ids):
#     for emb, file in zip(embeds, ids):
#         np.save(CFG.embeds_path / f'{file}.npy', emb)

# %%
# for x, y in results.groupby(['study_id', 'series_id']):   
#     filename = CFG.stacked_path / f'{x[0]}_{x[1]}.npy'

#     files = [CFG.embeds_path / f'{file}.npy' for file in y.ids.tolist()]

#     files = [np.load(f) for f in files]

#     stacked = np.vstack(files)

#     np.save(filename, stacked)

# %%

# %%
