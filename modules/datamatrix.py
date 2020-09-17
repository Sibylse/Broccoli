
import torch
from torch.utils.data import Dataset

class DataMatrix(Dataset):
        
    def __init__(self, D):
        self.data_ = torch.from_numpy(D).float()
        self.m_ = D.shape[0]
        self.n_ = D.shape[1]

    def __len__(self):
        return self.m_*self.n_
    
    def __getitem__(self, index):
        j = int(index/self.n_)
        i = index%self.n_       
        return j,i, self.data_[j,i]
