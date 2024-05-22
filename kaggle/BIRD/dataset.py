import torch
import imageio

import numpy as np
import pandas as pd
import albumentations as A

from albumentations.pytorch import ToTensorV2

# Imagenet vals
# img_mean=[0.485, 0.456, 0.406]
# img_std=[0.229, 0.224, 0.225]

mean = np.array([-0.22308692079693776, -0.23225270031972337, -0.2646080688103756, -0.27772951886156666], dtype=np.float32)
std = np.array([2.4021079842036257, 2.3784709900060506, 2.4214762588834593, 2.366489507911308], dtype=np.float32)

# norm_tf = A.Normalize(mean=img_mean, std=img_std, p=1.0)

sample_submission = pd.read_csv('E:\data\BirdCLEF\sample_submission.csv')

# Set labels
target_columns = sample_submission.columns[1:]
num_classes = len(target_columns)
bird2id = {b: i for i, b in enumerate(target_columns)}

class spectro_dataset(torch.utils.data.Dataset):
    # def __init__(self, df, base_dir, cols, normalize=True):
    def __init__(self, df, X, y, normalize=True):
        super().__init__()
        
        self.df = df
        self.X = X
        self.y = y
        self.len = len(self.df)

        self.tf = None

        self.num_classes = num_classes
        self.bird2id = bird2id

        self.normalize = normalize
        
        # self.norm_tf = norm_tf
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index: int):
        entry = self.df.iloc[index]
        filename = entry.filename
        # print(filename)

        spec = imageio.imread(self.X[filename])
        spec = spec[:,:313] # 5secs

        image = torch.from_numpy(spec).float()
        image = torch.stack([image, image, image], dim = 0)

        label = self.y[filename]
        target = np.zeros(self.num_classes, dtype=np.float32)
        
        # target[self.bird2id[label]] = 1
        target[label] = 1
        
        target = torch.from_numpy(target).float()
        
        return image, target

