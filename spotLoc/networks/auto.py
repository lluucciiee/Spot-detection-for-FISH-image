import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
import numpy as np
import cd as CD

class DownBlock(nn.Module):

    def __init__(self, inplanes, planes, device='cuda'):
        super(DownBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(planes,momentum=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(planes,momentum=1)
        self.relu2 = nn.ReLU()
        #self.dropout=nn.Dropout(0.3)
        self.device=device
    
    def forward(self, x):
        if x.size(0)!=3:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu1(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu2(out)
            return out
        else:
            out,relevant, irrelevant=x[0],x[1],x[2]
            relevant, irrelevant=CD.propagate_conv_linear(relevant, irrelevant, self.conv1,device=self.device)
            relevant, irrelevant=CD.propagate_batch_norm(relevant, irrelevant, self.bn1,device=self.device)
            relevant, irrelevant=CD.propagate_relu(relevant, irrelevant, self.relu1,device=self.device)
            relevant, irrelevant=CD.propagate_conv_linear(relevant, irrelevant, self.conv2,device=self.device)
            #relevant, irrelevant=CD.propagate_dropout(relevant, irrelevant, self.dropout)
            relevant, irrelevant=CD.propagate_batch_norm(relevant, irrelevant, self.bn2,device=self.device)
            relevant, irrelevant=CD.propagate_relu(relevant, irrelevant, self.relu2,device=self.device)
            out = self.conv1(out)
            out = self.bn1(out)
            out = self.relu1(out)
            out = self.conv2(out)
            #out=self.dropout(out)
            out = self.bn2(out)
            out = self.relu2(out)
            out=torch.cat((out.unsqueeze(0),relevant.unsqueeze(0),irrelevant.unsqueeze(0)),dim=0)
            return out

class MidBlock(nn.Module):

    def __init__(self, inplanes, planes, device='cuda'):
        super(MidBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,padding=1)
        self.relu1 = nn.Sigmoid()
        self.device=device
    
    def forward(self, x):
        if x.size(0)!=3:
            out = self.conv1(x)
            out = self.relu1(out)
            return out
        else:
            out, relevant, irrelevant=x[0],x[1],x[2]
            relevant, irrelevant=CD.propagate_conv_linear(relevant, irrelevant, self.conv1,device=self.device)
            relevant, irrelevant=CD.propagate_relu(relevant, irrelevant, self.relu1,device=self.device)
            out = self.conv1(out)
            out = self.relu1(out)
            out=torch.cat((out.unsqueeze(0),relevant.unsqueeze(0),irrelevant.unsqueeze(0)),dim=0)
            return out


class UpBlock(nn.Module):

    def __init__(self, inplanes, planes):
        super(UpBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes,planes, kernel_size=3, stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()
    
    def forward(self, x):
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        return out

out_channels=[32,64,1]       
class Net(nn.Module):

    def __init__(self,in_size,out_size,enforce=False,pretrain=False,device='cuda'):
        
        super(Net, self).__init__()
        self.layer1 = DownBlock(in_size,out_channels[0],device=device)
        self.layer2 = DownBlock(out_channels[0],out_channels[1],device=device)
        self.layer3 = MidBlock(out_channels[1],1)
        self.down1=nn.MaxPool2d(kernel_size=2,stride=2)
        self.down2=nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.device=device
        self.enforce=enforce
        self.pretrain=pretrain
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    
    def forward(self, x):
        if x.size(0)==3:
            self.down1.return_indices=False
            self.down2.return_indices=False
            
            x = self.layer1(x)
            out, relevant, irrelevant=x[0],x[1],x[2]
            out = self.down1(out)
            relevant, irrelevant=CD.propagate_pooling(relevant, irrelevant, self.down1,device=self.device)
            out=torch.cat((out.unsqueeze(0),relevant.unsqueeze(0),irrelevant.unsqueeze(0)),dim=0)
            
            x = self.layer2(out)
            out, relevant, irrelevant=x[0],x[1],x[2]
            out = self.down2(out)
            relevant, irrelevant=CD.propagate_pooling(relevant, irrelevant, self.down2,device=self.device)
            out=torch.cat((out.unsqueeze(0),relevant.unsqueeze(0),irrelevant.unsqueeze(0)),dim=0)
            
            x = self.layer3(out)
            return x
        else:
            self.down1.return_indices=True
            self.down2.return_indices=True
            x = self.layer1(x)
            x,indices1 = self.down1(x)
            x = self.layer2(x)
            x,indices2 = self.down2(x)
            x = self.layer3(x)
            return x,indices1,indices2

class Decoder(nn.Module):

    def __init__(self,out_size,device='cuda'):
        
        super(Decoder, self).__init__()
        self.layer4=nn.Sequential(UpBlock(out_channels[2],out_channels[1]))
        self.layer5=nn.Sequential(UpBlock(out_channels[1],out_channels[0]))

        self.up1=nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.up2=nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.logit=nn.Sequential(nn.Conv2d(out_channels[0],out_size,kernel_size=3, stride=1,padding=1),nn.Conv2d(out_size,out_size,kernel_size=3, stride=1,padding=1),nn.Sigmoid())
        
        self.device=device
        self.indices1=0
        self.indices2=0
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    
    def forward(self, x):
        x = self.layer4(x)
        x = self.up1(x,self.indices2)
        x = self.layer5(x)
        x = self.up2(x,self.indices1)
        x=self.logit(x)
        return x
        
         
