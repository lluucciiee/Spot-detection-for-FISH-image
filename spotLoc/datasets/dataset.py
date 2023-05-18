import torch
import numpy as np
import pandas as pd
import glob
import os
from .augment import *

class folderDataset(torch.utils.data.Dataset):
    def __init__(self,path,valid=False,aug=True,index=None): 
        self.path=path
        if index is not None:
            self.path=[self.path[i] for i in index]
        self.length=len(self.path)
        self.valid=valid
        self.aug=aug

    def __len__(self):
        return self.length

    def __getitem__(self,idx):
        data=np.load(self.path[idx])
        x=data['x']
        y=data['y']
        
        if self.valid:
            return torch.Tensor(x).unsqueeze(0),torch.Tensor(y).unsqueeze(0),data['coord']
        else:
            if not self.aug:
                return torch.Tensor(x).unsqueeze(0),torch.Tensor(y).unsqueeze(0)
            else:
                n=len(data['coord'])
                #print(n)
                if n<50 and np.random.rand()>0.5:
                    x,y=augment2(x,y)
                        
                return torch.Tensor(x).unsqueeze(0),torch.Tensor(y).unsqueeze(0)

class fileDataset(torch.utils.data.Dataset):
    def __init__(self,path,phase='train',aug=True,index=None): 
        self.data=np.load(path,allow_pickle=True)

        if phase=='train':
            self.x=self.data['x_train']
            self.y=self.data['y_train']
            self.seg=self.data['seg_train']
        elif phase=='valid':
            self.x=self.data['x_valid']
            self.y=self.data['y_valid']
            self.seg=self.data['seg_valid']
        else:
            self.x=self.data['x_test']
            self.y=self.data['y_test']

        if index is not None:
            self.x=self.x[index]
            self.y=self.y[index]
            self.seg=self.seg[index]

        self.phase=phase
        self.aug=aug
        self.length=len(self.x)
        

    def __len__(self):
        return self.length

    def __getitem__(self,idx):
        x=self.x[idx]
        y=self.seg[idx]
        coord=self.y[idx]
        if self.phase=='valid':
            return torch.Tensor(x).unsqueeze(0),torch.Tensor(y).unsqueeze(0),coord
        else:
            if not self.aug:
                return torch.Tensor(x).unsqueeze(0),torch.Tensor(y).unsqueeze(0)
            else:
                n=len(coord)
                #print(n)
                if n<50 and np.random.rand()>0.5:
                    x,y=augment2(x,y)
                        
                return torch.Tensor(x).unsqueeze(0),torch.Tensor(y).unsqueeze(0)


