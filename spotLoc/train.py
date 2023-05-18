import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import glob
import os

from .networks import *
from .datasets import *
from .utils import *


def train(args,name_model,train_index=None,valid_index=None):
    #model
    if args.archit=='SpotLoc':
        model=SpotLoc(device=args.device)
    elif args.archit=='UNet':
        model=UNet()
    model=model.to(args.device)

    #dataset
    path=glob.glob(args.train_root+'*.npz')
    train_data=folderDataset(path,aug=args.aug,index=train_index)
    path=glob.glob(args.valid_root+'*.npz')
    valid_data=folderDataset(path,valid=True,index=valid_index)
    
    dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=1,
        shuffle=True,
        drop_last=False
    )
    validDataloader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=1,
        shuffle=True,
        drop_last=False
    )
    print('#sample train valid:',len(dataloader),len(validDataloader))
    
    
    # Optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=10)
    optimizer.zero_grad()

    s=2**args.down_time
    area_spot=calculate_area(args.r)
    half_area=min(area_spot/4,s**2-4)

    current_best_loss=np.inf
    save_model=False
    # Iterations
    for epoch in range(args.n_epochs):
        eval_values=np.zeros(2)
        eval_values_valid=np.zeros(2)
        
        if epoch>=args.epoch_all:
            for (i,(x,y)) in enumerate(dataloader):

                #calculate gradient    
                x=x.to(args.device)
                y=y.to(args.device)
                
                out=model(x)
                #CE
                loss=-torch.mean(y*torch.log(out+EPS)+(1-y)*torch.log(1-out+EPS))
                loss.backward()


                if (i+1)%args.batch==0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                eval_values[0]+=loss.item()
        else:
            for (i,(x,y)) in enumerate(dataloader):

                #calculate gradient    
                relevant=torch.mul(x,y)
                x=torch.cat((x.unsqueeze(0),relevant.unsqueeze(0),(x-relevant).unsqueeze(0)),dim=0)
                x=x.to(args.device)

                n,c,l,w=y.size()
                y=y.view(n,1,l//s,s,w//s,s).sum((3,5))
                y=(y>=half_area).float()
                y=y.to(args.device)
                
                embed=model(x)
                #CE
                loss=-torch.mean(y*torch.log(embed[0]+EPS)+(1-y)*torch.log(1-embed[0]+EPS))
                
                penalty=-args.lambd*torch.sum(embed[1])/(torch.sum(embed[1:])+EPS)
                
                if args.penal:
                    (loss+penalty).backward()
                else:
                    loss.backward()


                if (i+1)%args.batch==0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                eval_values[0]+=loss.item()
                eval_values[1]+=-penalty.item()/args.lambd
        eval_values/=len(dataloader)
        scheduler.step()
        
        if epoch>args.n_epochs/2:
            
            #match_res=np.zeros(3)
            for (i,(x,y,true)) in enumerate(validDataloader): 
                with torch.no_grad():
                    #calculate gradient    
                    x=x.to(args.device)
                    y=y.to(args.device)
                    
                    out=model(x)
                    #CE
                    loss=-torch.mean(y*torch.log(out+EPS)+(1-y)*torch.log(1-out+EPS))
                    
                    eval_values_valid[0]+=loss.item()
                    
                    if args.debug and i==0:
                        plt.figure()
                        out=out[0,0].cpu().detach().numpy()
                        plt.imshow(out>0.5)
                        #plt.scatter(pred[:,1],pred[:,0],s=1,c='r')
                        plt.savefig('check.png')  
                    

            eval_values_valid/=len(validDataloader)
            
            if eval_values_valid[0]<current_best_loss:
                current_best_loss=eval_values_valid[0]
                print("save model")
                torch.save(model.state_dict(),args.model_root+name_model+'.pth')
                save_model=True
            if epoch==args.n_epochs-1 and not save_model:
                torch.save(model.state_dict(),args.model_root+name_model+'.pth')

        info="[Epoch %d/%d] [Train loss: %f] [Focus spot: %f] [Test loss: %f] [Focus spot: %f] \n"\
                % (epoch, args.n_epochs, eval_values[0],eval_values[1],eval_values_valid[0],eval_values_valid[1])
        print(info)
        log=open('log.txt',mode='a')
        log.write("%s\n"%info)
        log.close()
          
    

    
def run(args,name_model):
    log=open('log.txt',mode='a')
    log.write("%s\n"%args)
    log.close()

    os.makedirs(args.model_root, exist_ok=True)
    os.makedirs(args.index_root, exist_ok=True)
    
    if args.percent is None:
        train(args,name_model)
        return

    #train on patial train set
    N_train=len(glob.glob(args.train_root+'*.npz'))
    N_valid=len(glob.glob(args.valid_root+'*.npz'))
    
    length=int(N_train*args.percent/100)
    
    os.makedirs(args.model_root+str(length)+'/', exist_ok=True)

    for i in range(args.repeat):
        train_index_path=args.index_root+'train_index_%d_%d'%(length,i)
        valid_index_path=args.index_root+'valid_index_%d_%d'%(length,i)
        
        if not os.path.isfile(train_index_path+'.npy'):
            print("creating training and valid subset")
            train_index=np.random.choice(np.arange(N_train),length)
            np.save(train_index_path,train_index)

            valid_index=np.random.choice(np.arange(N_valid),int(N_valid*args.percent/100))
            np.save(valid_index_path,valid_index)

        train_index=np.load(train_index_path+'.npy')
        valid_index=np.load(valid_index_path+'.npy')
        
        print(args.repeat,i)
        train(args,'%d/%s_%d'%(length,name_model,i),train_index=train_index,valid_index=valid_index)
            

    