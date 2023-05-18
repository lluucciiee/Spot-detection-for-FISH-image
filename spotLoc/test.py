import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
import torchvision
import numpy as np
import pandas as pd
import glob
import re
import os
import cv2
import tifffile as tiff
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from mpl_toolkits import mplot3d

from spotLoc import *
from spotLoc.networks import *
from spotLoc.utils import *

os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu") 
#device = "cpu"


def predict(root,name,color,r,thred,name_model,archit='SpotLoc',cal_score=True,debug=False,image_idx=None,file=None,save=True):
    #model
    if archit=='SpotLoc':
        model=SpotLoc(device=device)
    elif archit=='UNet':
        model=UNet()
    
    
    match_score=np.zeros(4)
    #predict smfish
    if file is not None:
        if name=='smfish':
            MEAN=892.7769
            STD=942.78345
        elif name=='suntag':
            MEAN=1437.6512
            STD=992.32117
            
        model.load_state_dict(torch.load('%s%s.pth'%(root+'model/',name_model)))
        model=model.to(device)

        data=np.load(file,allow_pickle=True)
        x_test=data['x_test']
        y_test=data['y_test']
        n,n_z=x_test.shape[0],x_test.shape[1]
        res=[]
        for i in range(len(x_test)):
            x=(x_test[i]-MEAN)/STD
            x=torch.Tensor(x).unsqueeze(0).unsqueeze(0).to(device) 
            out=model(x)
            out=out[0,0].cpu().detach().numpy()
            pred=count(out,r,cutoff=thred)

            if debug and i<5:
                plt.figure(figsize=(16,9))
                plt.imshow(out>0.5)
                plt.scatter(pred[:,1],pred[:,0],s=0.1,c='r')
                plt.savefig('%d.png'%i)
            
            res.append(pred)
            true=y_test[i]
            match_score = match_score+compute_metrics(pred=pred,true=true,mdist=mdist)
        score=evaluate(match_score)
        if save:
            np.save(root+'predALL',res)
        return score
    #predict others
    F=load_io(name)
    os.makedirs(F.spotPredPath, exist_ok=True)
    
    if image_idx is None:
        image_idx=F.test_idx
    elif image_idx == 'all':
        image_idx=[i for i in range(F.length)]
    
    for i in image_idx:
        fs=glob.glob(F.rawPath[i]+color+'*.tif')
        if len(fs)==0:
            continue
            
        name=F.fname[i]
        print(name)

        model.load_state_dict(torch.load('%s%s.pth'%(F.modelPath,name_model)))
        model=model.to(device)

        #load data
        img=standardize(F.load_raw_image(i,color))
        mask=F.load_mas_image(i)
        n_z=img.shape[0]

        pred=[]
        for z in range(n_z):
            x=img[max(z-1,0):min(z+2,n_z),...].max(0)
            x=torch.Tensor(x).unsqueeze(0).unsqueeze(0).to(device) 
            if unet:
                out=model(x)
            else:
                out=model(x)
                out=decoder(out)
            out=out[0,0].cpu().detach().numpy()
            pred_z=count(out,r,cutoff=thred)
            
            pred_z=np.array(pred_z)
            pred.append(pred_z)
            
            if debug:
                plt.figure(figsize=(16,9))
                plt.imshow(out>0.5)
                plt.scatter(pred_z[:,1],pred_z[:,0],s=0.1,c='r')
                plt.savefig('%d.png'%z)
            
        if save:
            pred=adapt(pred,n_z,'%s%s_%s'%(F.spotPredPath,name,color))
        else:
            pred=adapt(pred,n_z,None)
        print("done predict")
        
        if cal_score:
            true=F.load_spot_txt2array(i,color)
            print('#true, #pred',len(true),len(pred))
            match_score = match_score+compute_metrics(pred=pred,true=true,mdist=mdist)
    
    score=evaluate(match_score)
    return score

def evaluation(root,name,color):  
    F=load_io(name)
    match_score=np.zeros(4)
    for i in F.test_idx:
        name=F.fname[i]
        print(name)
        #load data
        img=F.load_raw_image(i,color)
        mask=F.load_mas_image(i)
        pred=F.load_spot_array(i,color) 
        true=F.load_spot_txt2array(i,color)
        print('#true, #pred',len(true),len(pred))  
        match_score = match_score+compute_metrics(pred=pred,true=true,mdist=mdist)
    
    evaluate(match_score)

def view(root,name,color,image_idx=None,n_cell=20,min_spot=20,file=None):
    #smfish
    if file is not None:
        os.makedirs(root+'view/', exist_ok=True)
        data=np.load(file,allow_pickle=True)
        x_test=data['x_test']
        coord_t=data['y_test']
        coord_p=np.load(root+'predALL.npy',allow_pickle=True)
        if image_idx is None:
            image_idx=[i for i in range(len(coord_t))]
        for i in image_idx:
            plt.figure(figsize=(16,9))
            plt.subplot(1,2,1)
            plt.axis('off')    
            plt.imshow(x_test[i])
            plt.scatter(coord_t[:,1]-c,coord_t[:,0]-a,s=2,c='r')
            plt.subplot(1,2,2)
            plt.axis('off')    
            plt.imshow(x)
            plt.scatter(coord_p[:,1]-c,coord_p[:,0]-a,s=2,c='k')
            plt.title('true count: '+str(len(coord_t))+'     '+'pred count: '+str(len(coord_p)))  
            
            plt.savefig('%s%s%d.png'%(root,'view/',i))  

        return 
    F=load_io(name)
    os.makedirs(F.viewPath+'pred/', exist_ok=True)
    if image_idx is None:
        image_idx=F.test_idx

    for i in image_idx:
        name=F.fname[i]
        print(name)
        #load data
        img=F.load_raw_image(i,color)
        mask=F.load_mas_image(i)
        mask_nuc=F.load_mas_image(i,cell=False)

        dic_t=F.load_spot_txt2dict(i,color,index='m')
        dic_p=F.load_spot_array2dict(i,color,index='m')
        # choose #cell to view by n_cell
        if n_cell is None:
            n_cell=mask.max()+1
        
        for m in range(1,n_cell+1):
            if str(m) not in dic_t.keys() and str(m) not in dic_p.keys():
                continue
            a,b,c,d=find_min_wind(mask==m)

            if str(m) in dic_t.keys():
                coord_t=np.array(dic_t[str(m)])
            else:
                coord_t=np.zeros((0,3))

            if str(m) in dic_p.keys():
                coord_p=np.array(dic_p[str(m)])
            else:
                coord_p=np.zeros((0,3))
            
            # only view cells that are abundant of spots
            if len(coord_t)<min_spot:
                continue   
            
            #2d vision
            x1=standardize(img.max(0))
            x=(x1[a:b,c:d]-x1[a:b,c:d].min())*(mask==m)[a:b,c:d]
            
            plt.figure(figsize=(16,9))
            plt.subplot(1,2,1)
            plt.axis('off')    
            plt.imshow(x)
            plt.scatter(coord_t[:,1]-c,coord_t[:,0]-a,s=2,c='r')
            plt.subplot(1,2,2)
            plt.axis('off')    
            plt.imshow(x)
            plt.scatter(coord_p[:,1]-c,coord_p[:,0]-a,s=2,c='k')
            plt.title('true count: '+str(len(coord_t))+'     '+'pred count: '+str(len(coord_p)))  
            
            plt.savefig('%s%s%s_%d.png'%(F.viewPath,'pred/',name,m))  

            #3d vision
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            x=coord_t[:,0]-a
            y=coord_t[:,1]-c
            z=coord_t[:,2]
            ax.scatter3D(z, x, y, c=z, cmap='magma')

            x = [0,b-a,b-a,0,0]
            y = [d-c,d-c,0,0,d-c]
            ax.plot3D(np.zeros(5), x, y,  'gray')
            ax.plot3D(np.ones(5)*30, x, y,  'gray')
            
            ax.set_title('')
            plt.xticks([]) 
            #plt.yticks([]) 
            plt.savefig('%s%s%s_%d_3dt.png'%(F.viewPath,'pred/',name,m))

            fig = plt.figure()
            ax = plt.axes(projection='3d')
            x=coord_p[:,0]-a
            y=coord_p[:,1]-c
            z=coord_p[:,2]
            ax.scatter3D(z,x, y, c=z, cmap='magma')
            
            x = [0,b-a,b-a,0,0]
            y = [d-c,d-c,0,0,d-c]
            ax.plot3D(np.zeros(5), x, y,  'gray')
            ax.plot3D(np.ones(5)*30, x, y,  'gray')
            
            ax.set_title('')
            plt.xticks([]) 
            #plt.yticks([]) 
            plt.savefig('%s%s%s_%d_3dp.png'%(F.viewPath,'pred/',name,m))

            fig = plt.figure()
            plt.axis('off')
            plt.imshow(2-(mask==m).astype('int')[a:b,c:d]-(mask_nuc==m).astype('int')[a:b,c:d],cmap='Greys_r')
            plt.savefig('%s%s%s_%d_mask.png'%(F.viewPath,'pred/',name,m))
        

            
def view_by_layer(root,name,color,name_model,archit,i,m):
    F=load_io(name)
    idx=F.test_idx[i]
    name=F.fname[idx]
    #model
    if archit=='SpotLoc':
        model=SpotLoc(1,1,2)
    elif archit=='Unet':
        model=UNet()
    
    model.load_state_dict(torch.load('%s%s.pth'%(F.modelPath,name_model)))
    model=model.to(device)
    
    #load data
    img=standardize(F.load_raw_image(idx,color))
    mask=F.load_mas_image(idx)==m
    
    dic_t=F.load_spot_txt2dict(idx,color,index='m')
    coord_t=np.array(dic_t[str(m)])
    
    dic_p=F.load_spot_array2dict(idx,color,index='m')
    coord_p=np.array(dic_p[str(m)])
    
    print(len(coord_t),len(coord_p))
    
    a,b,c,d=find_min_wind(mask)
    print(b-a,d-c)
    
    out_2d=[]
    for z in range(F.n_z):
        #process_image
        x=img[max(z-1,0):min(z+2,F.n_z)].max(0)
        x=torch.Tensor(x).unsqueeze(0).unsqueeze(0).to(device) 
        out=model(x)
        out_2d.append(out[0,0].cpu().detach().numpy()[a:b,c:d])
        
    plt.figure(figsize=(16,9))
    plt.subplot(1,2,1) 
    plt.axis('off')
    plt.imshow(img.max(0)[a:b,c:d]*mask[a:b,c:d])
    plt.scatter(coord_t[:,1]-c,coord_t[:,0]-a,c='r',s=2)
                
    plt.subplot(1,2,2)
    plt.axis('off')
    plt.imshow(np.array(out_2d).max(0)*mask[a:b,c:d])
    plt.scatter(coord_p[:,1]-c,coord_p[:,0]-a,c='r',s=2)
    
    plt.savefig('%s_%d.png'%(name,m))
    
    np.savez_compressed('%s_%d'%(name,m),out=np.array(out_2d),x=img[:,a:b,c:d],mask=mask[a:b,c:d]) 

def test(root,name,color,thred,archit,n_cell=None,min_spot=50):
    name_model='model_%s_%s_%s'%(archit,name,color)
    predict(root,color,thred,name_model,archit='SpotLoc')
    view(root,color,n_cell=n_cell,min_spot=min_spot)
'''    
if __name__ == "__main__" :
    model=SpotLoc(device='cpu')
    modelsummary=summary(model,(1, 512,512),device='cpu')
    print(modelsummary)
'''