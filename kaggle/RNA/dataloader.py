import numpy as np
import torch

class RNA_Dataset(torch.utils.data.Dataset):
    def __init__(self, df, Lmax, v, seed,  mode='train',
                 mask_only=False, **kwargs):
        
        self.seq_map = v
        self.Lmax = Lmax
        
        self.seq = df['sequence'].values
        self.L = df['L'].values
        
        self.mask_only = mask_only
        
    def __len__(self):
        return len(self.seq)  
    
    def __getitem__(self, idx):
        seq = self.seq[idx]
        
        mask = torch.zeros(self.Lmax, dtype=torch.bool)
        mask[:len(seq)] = True
        
        if self.mask_only:
            return {'mask':mask}
        
        seq = [self.seq_map[s] for s in seq]
        seq = np.array(seq)        
        seq = np.pad(seq, (0, self.Lmax - len(seq)))
        seq = torch.from_numpy(seq)
        
        rand = torch.rand(mask.sum())
        mask_arr = rand < 0.15
        
        # print(rand.shape)
        
        selection = torch.flatten((mask_arr).nonzero()).tolist()
        
        mlm = seq.detach().clone()
        mlm[selection] = self.seq_map['M']
        
        # true when token is masked 
        token_mask = mlm == self.seq_map['M']
        
        #        
        mlm_target = seq.masked_fill(~token_mask, 0)
        
        
        return {'seq': mlm, 'att_mask': mask}, {'labels': seq, 'token_mask': token_mask, 'mlm_target': mlm_target.long()}

def create_dataloader(train_ds, batch_size=2, num_workers=0, persistent_workers=True, drop_last=True):
    
    loader =  torch.utils.data.DataLoader(
        train_ds, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        persistent_workers=persistent_workers, 
        drop_last=drop_last
    )
    
    return loader

class DataloaderWrapper:
    def __init__(self, train_ds, batch_size=2, num_workers=0, persistent_workers=True, drop_last=True, shuffle=False):
        self.dataloader = torch.utils.data.DataLoader(dataset=train_ds, batch_size=batch_size, persistent_workers=persistent_workers, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)