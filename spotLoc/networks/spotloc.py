import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
import numpy as np
from .cd import *

class Stack(object):
    def __init__(self):
        self.items = []

    def is_empty(self):
        return self.items == []

    def put(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

class DownBlock(nn.Module):

    def __init__(self, inplanes, planes,decode=None,device='cuda'):
        super(DownBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.down1=nn.MaxPool2d(kernel_size=2,stride=2)
        self.decode=decode
        self.device=device
    
    def forward(self, x):
        if x.shape[0]==3:
            self.down1.return_indices=False
            out, relevant, irrelevant=x[0],x[1],x[2]
            relevant, irrelevant=propagate_conv_linear(relevant, irrelevant, self.conv1,device=self.device)
            relevant, irrelevant=propagate_conv_linear(relevant, irrelevant, self.bn1,device=self.device)
            relevant, irrelevant=propagate_relu(relevant, irrelevant, self.relu1)
            relevant, irrelevant=propagate_maxpooling(relevant, irrelevant,self.down1,device=self.device)

            out = self.conv1(out)
            out = self.bn1(out)
            out = self.relu1(out)
            out = self.down1(out)
            out=torch.cat((out.unsqueeze(0),relevant.unsqueeze(0),irrelevant.unsqueeze(0)),dim=0)
            return out
        
        else:
            self.down1.return_indices=True
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x,indices = self.down1(x)
            self.decode.put(indices)
            return x

class MidBlock(nn.Module):

    def __init__(self, inplanes, planes,decode=None,device='cuda'):
        super(MidBlock, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1,padding=0)
        self.activate = nn.Sigmoid()
        self.decode=decode
        self.device=device
    
    def forward(self, x):
        if x.shape[0]==3:
            out, relevant, irrelevant=x[0],x[1],x[2]
            relevant, irrelevant=propagate_conv_linear(relevant, irrelevant, self.conv,device=self.device)
            #relevant, irrelevant=propagate_sigmoid(relevant, irrelevant, self.activate,device=self.device)
            relevant, irrelevant=F.relu(relevant),F.relu(irrelevant)
            out = self.conv(out)
            out = self.activate(out)
            out=torch.cat((out.unsqueeze(0),relevant.unsqueeze(0),irrelevant.unsqueeze(0)),dim=0)
            return out

        else:
            out = self.conv(x)
            out = self.activate(out)
            return out


class UpBlock(nn.Module):

    def __init__(self, inplanes, planes,decode):
        super(UpBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes,planes, kernel_size=3, stride=1,padding=1)
        #self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.up1=nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decode=decode
    
    def forward(self, x):
        x = self.conv1(x)
        #x = self.bn1(x)
        x = self.relu1(x)
        x = self.up1(x,self.decode.pop())
        
        return x
      


class SpotLoc(nn.Module):

    def __init__(self,in_size=1,out_size=1,down_time=2,device='cuda'):
        super(SpotLoc, self).__init__()
        out_channels=[in_size,16,32,1]
        assert(down_time>=2)
        for i in range(down_time-2):
            out_channels.insert(3,32)
        
        self.indices=Stack()
        layers=[]
        for i in range(down_time):
            layers.append(DownBlock(out_channels[i],out_channels[i+1],decode=self.indices,device=device))
        
        self.encoder = nn.Sequential(*layers)
        self.middle = nn.Sequential(MidBlock(out_channels[-2],out_channels[-1],decode=self.indices,device=device))
        
        layers=[]
        for i in range(down_time):
            layers.append(UpBlock(out_channels[-1-i],out_channels[-2-i],decode=self.indices))
        self.decoder = nn.Sequential(*layers)
        
        self.logit=nn.Sequential(nn.Conv2d(out_channels[1],out_size,kernel_size=3, stride=1,padding=1),
            nn.Sigmoid())
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, x):
        x = self.encoder(x)
        e = self.middle(x)
        if e.shape[0]==3:
            return e
        else:
            x = self.decoder(e)
            x = self.logit(x)
            return x
     
