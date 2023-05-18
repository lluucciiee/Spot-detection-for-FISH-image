import glob
import os
import re
import pandas as pd
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from spotLoc.utils import *
from get_meta import load_io

def prepare_input_bycell(root,name,r,color,n_minimum=0,del_zero_percent=0.5,debug=False):
    
    F=load_io(name)
    F.select_todo_raw()
    F.split_train_valid_test()
    print('train fov_num,valid fov_num,test fov_num,')
    print(len(F.train_idx),len(F.valid_idx),len(F.test_idx))     
    
    os.makedirs(F.trainPath, exist_ok=True)
    os.makedirs(F.validPath, exist_ok=True)
    print(list(F.train_idx)+list(F.valid_idx))
    
    cell_num=0
    spot_num=0
    for i in list(F.train_idx)+list(F.valid_idx):
        name=F.fname[i]
        print(name)
        outPath=F.trainPath if i in F.train_idx else F.validPath
        print(outPath)
        #load data
        img=standardize(F.load_raw_image(i,color))
        mask=F.load_mas_image(i)
        spot=F.load_spot_txt2dict(i,color,index='mz')
        spot_count=F.load_spot_txt2dict(i,color,index='m')
        
        #create input by cell
        for m in range(1,mask.max()+1):
            if str(m) not in spot.keys():
                continue
            #drop some low dense cell
            if len(spot_count[str(m)])<n_minimum:
                continue
            cell_num+=1
            spot_num+=len(spot_count[str(m)])
            
            a,b,c,d=find_min_wind(mask==m)
            l,w=b-a,d-c
            #drop abnormal
            if l==0 or w==0:
                continue
            for z in range(1,F.n_z-1):
                if str(z) not in spot[str(m)].keys():
                    coord=np.zeros((0,2))
                else:
                    coord=np.array(spot[str(m)][str(z)])
                
                if str(z-1) in spot[str(m)].keys():
                    coord=np.concatenate([spot[str(m)][str(z-1)],coord],0)
                if str(z+1) in spot[str(m)].keys():
                    coord=np.concatenate([spot[str(m)][str(z+1)],coord],0)

                if len(coord)==0 and np.random.rand()<del_zero_percent:
                    continue
                #make input
                img_proj=img[z-1:z+2,a:b,c:d].max(0)
                x=np.ones(((l//4)*4+4,(w//4)*4+4))*np.median(img_proj)
                x[:l,:w]=img_proj*(mask==m)[a:b,c:d]+np.median(img_proj)*(mask!=m)[a:b,c:d]
                if len(coord)>0:
                    coord[:,0]-=a
                    coord[:,1]-=c
                    y=toMap(coord,x.shape,r)
                else:
                    y=np.zeros(x.shape)

                np.savez_compressed('%s%s_%d_%d'%(outPath,F.fname[i],m,z),x=x,y=y,coord=coord)
                
                if m<5 and debug:
                    #make view
                    plt.figure(figsize=(16,9))
                    figure, axes = plt.subplots()
                    plt.imshow(x)
                    for (pa,pb) in coord:
                        draw_circle = plt.Circle((pb,pa),r+0.5,fill=False,color='r')
                        axes.add_artist(draw_circle)
                    plt.savefig('%s%s%s_%d_%d.png'%(F.viewPath,'spot/',F.fname[i],m,z))
    print('total cell_num',cell_num)
    print('total spot_num',spot_num)     

def prepare_input_byimage(root,name,r,color,img_size=1024,del_zero_percent=0.5,debug=False):
    F=load_io(name)
    F.select_todo_raw()
    F.split_train_valid_test()

    x_res=[]
    y_res=[]
    seg_res=[]
    for idx_list in [F.train_idx,F.valid_idx]:
        n=len(idx_list)
        x_train=np.zeros((n*F.n_z,img_size,img_size))
        y_train=[]
        n_img=0
        for i in range(n):
            print(F.fname[idx_list[i]])
            #load data
            img=standardize(F.load_raw_image(idx_list[i],color))
            spot=F.load_spot_txt2dict(idx_list[i],color,index='z')
                
            for z in range(1,F.n_z-1):
                if str(z) not in spot.keys():
                    coord=np.zeros((0,2))
                else:
                    coord=np.array(spot[str(z)])
                    
                if str(z-1) in spot.keys():
                    coord=np.concatenate([spot[str(z-1)],coord],0)
                if str(z+1) in spot.keys():
                    coord=np.concatenate([spot[str(z+1)],coord],0)

                if len(coord)==0 and np.random.rand()<del_zero_percent:
                    continue
                x=img[z-1:z+2,...].max(0)
                x_train[n_img,:,:]=x
                n_img+=1
                y_train.append(coord)
                
                    
            
        x_res.append(x_train[:n_img,...])
        y_res.append(y_train)
        print(n_img)

    idx_list=F.test_idx
    n=len(idx_list)
    if d3:
        x_test=np.zeros((n,F.n_z,img_size,img_size))
    else:
        x_test=np.zeros((n,img_size,img_size))
    
    y_test=[]
    for i in range(n):
        print(F.fname[idx_list[i]])
        #load data
        img=standardize(F.load_raw_image(idx_list[i],color))
        spot=F.load_spot_txt2array(idx_list[i],color)
        y_test.append(spot)
        for z in range(F.n_z):
            x=img[max(z-1,0):min(z+2,F.n_z),...].max(0)
            x_test[i,z,:,:]=x
                    
        
    savename=root+name+'_'+color
    np.savez_compressed(savename,x_train=x_res[0],y_train=y_res[0],x_valid=x_res[1],y_valid=y_res[1],x_test=x_test,y_test=y_test)
                                
def prepare_input_smfish(root,name,file,r):
    os.makedirs(root+'train_set/', exist_ok=True)
    os.makedirs(root+'valid_set/', exist_ok=True)
    data=np.load(file,allow_pickle=True)
    x_train=data['x_train']
    y_train=data['y_train']
    x_valid=data['x_valid']
    y_valid=data['y_valid']

    if name=='smfish':
        MEAN=892.7769
        STD=942.78345
    elif name=='suntag':
        MEAN=1437.6512
        STD=992.32117

    if MEAN==0 or STD==0:
        MEAN=np.mean(x_train)
        STD=np.std(x_train)
    print(MEAN,STD)

    x_train=(x_train-MEAN)/STD
    x_valid=(x_valid-MEAN)/STD

    for i in range(len(x_train)):
        print(i)
        y=toMap(y_train[i],(512,512),r)
        np.savez_compressed('%s%d'%(root+'train_set/',i),x=x_train[i],y=y,coord=y_train[i])
        
    for i in range(len(x_valid)):
        print(i)
        y=toMap(y_valid[i],(512,512),r)
        np.savez_compressed('%s%d'%(root+'/valid_set/',i),x=x_valid[i],y=y,coord=y_valid[i])
         
      

if __name__ == "__main__" :
    '''
    root='../yeast/'
    name='yeast'
    r=2
    color='tmr'
    prepare_input_bycell(root,name,r,color,n_minimum=15,del_zero_percent=0.5)
    prepare_input_byimage(root,name,r,color,n_minimum=15,del_zero_percent=0.5)
    '''
    root='../lnc/'
    name='lnc'
    r=1
    color='alexa'
    
    prepare_input_bycell(root,name,r,color,n_minimum=1,del_zero_percent=0.2)
    #prepare_input_byimage(root,name,r,color,n_minimum=1,del_zero_percent=0.2)
    '''
    root='../syn_data/smfish/'
    name='smfish'
    r=3
    prepare_input_smfish(root,name,r)
    '''
    