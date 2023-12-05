import torch
import numpy as np

class RNA_Dataset(torch.utils.data.Dataset):
    def __init__(self, df, Lmax, v, seed,  mode='train',
                 nsp=False, sep='S', kmers=False, **kwargs):
        
        self.seq_map = v
        self.Lmax = Lmax + 1
        
        self.seq = df['sequence'].values
        self.L = df['L'].values
        
        self.nsp = nsp
        self.sep = sep
        self.kmers = kmers
        
    def __len__(self):
        return len(self.seq)
    
    
    def get_nsp(self, seq):
        sep_idx = len(seq) // 2
        
        # isnext = True
        nsp_target = [1, 0]
        
        
        if torch.rand(1) < 0.5:
            # isnext = False
            nsp_target = [0, 1]
            
            # get another rand seq
            idx = np.random.choice(len(self.seq))
            other = self.seq[idx]

            max_len = max(len(seq), len(other))
            sep_idx = max_len // 2
            
            nsp = seq[:sep_idx] + other[sep_idx:]
        else:
            nsp = seq
            
        #     nsp = seq[:sep_idx] + self.sep + other[sep_idx:]
        # else:
        #     nsp = seq[:sep_idx] + self.sep + seq[sep_idx:]
        
        return nsp, sep_idx, torch.Tensor(nsp_target)
        
    def __getitem__(self, idx):
        seq = self.seq[idx]
        
        if self.nsp:
            seq, sep_idx, nsp_target = self.get_nsp(seq)
        
        if self.kmers:
            seq = [seq[i:i+3] for i in range(len(seq)-2)]
        
        seq = [self.seq_map[s] for s in seq]
        if self.nsp:
            seq.insert(sep_idx, self.seq_map[self.sep])
        
        seq = np.array(seq)        
        # seq = np.pad(seq, (0, self.Lmax - len(seq)))
        
        mask = torch.zeros(self.Lmax, dtype=torch.bool)
        mask[:len(seq)] = True
        
        seq = np.pad(seq, (0, self.Lmax - len(seq)))
        
        rand = torch.rand(mask.sum())
        mask_arr = rand < 0.15
        
        selection = torch.flatten((mask_arr).nonzero()).tolist()
        
        seq = torch.from_numpy(seq)
        mlm = seq.detach().clone()
        mlm[selection] = self.seq_map['M']
        
        # true when token is masked 
        token_mask = mlm == self.seq_map['M']
        
        # set all to 0 except mask
        mlm_target = seq.masked_fill(~token_mask, 0)
        
        inputs = {'seq': mlm, 'att_mask': mask}
        targets = {'labels': seq, 'token_mask': token_mask, 'mlm_target': mlm_target.long()}
        
        if self.nsp:
            # mlm[sep_idx] = self.seq_map['S']
            targets["nsp_target"] = nsp_target
        
        return inputs, targets

def create_dataloader(train_ds, batch_size=2, num_workers=0, persistent_workers=True, drop_last=True):
    
    loader =  torch.utils.data.DataLoader(
        train_ds, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        persistent_workers=persistent_workers, 
        drop_last=drop_last,
    )
    
    return loader

class DataloaderWrapper:
    def __init__(self, train_ds, batch_size=2, num_workers=0, persistent_workers=True, drop_last=True, shuffle=False):
        self.dataloader = torch.utils.data.DataLoader(
            dataset=train_ds, 
            batch_size=batch_size, 
            persistent_workers=persistent_workers, 
            shuffle=shuffle, 
            num_workers=num_workers, 
            drop_last=drop_last,
        )