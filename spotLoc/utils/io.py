import glob
import os
import re
import pandas as pd
import numpy as np
import tifffile as tiff
import torch
from torch.utils.data import random_split

class IO(object):
    
    def __init__(self,root,datafolder=None,dapiPath=None):
        dapiPath=glob.glob(root+datafolder+dapiPath)#'raw/*...*/dapi*.tif' directory depth can be adapted here
        self.root=root
        self.n_z,self.n_rc,_=tiff.imread(dapiPath[0]).shape#assume all channels are of same size, so that dimension info is extracted here
        self.length=len(dapiPath)
        self.rawPath=[p.split('dapi')[0] for p in dapiPath]#raw fov image folder path
        self.fname=[p.split('/')[-2] for p in self.rawPath]#all files afterwards named after the folder name
        self.masPath=root+'seg/'#create by segmentation section
        self.spotPath=root+'spot_detect_matlab/'#create by spot labeling section
        self.spotPredPath=root+'spot_pred/'#create by test
        self.trainPath=root+'train_set/'#create by prepare_input(this and below)
        self.validPath=root+'valid_set/'
        self.viewPath=root+'view/'
        f=self.root+'split_train_valid_test'
        if os.path.isfile(f+'.npz'):
            x=np.load(f+'.npz')
            self.train_idx,self.valid_idx,self.test_idx=x['train_idx'],x['valid_idx'],x['test_idx']
        os.makedirs(self.viewPath, exist_ok=True)
        self.modelPath=root+'model/'
        
    def select_todo_raw(self):
        df=pd.read_csv(self.root+'meta.csv')
        #print(list(df['todo']))
        self.todo_idx=np.arange(self.length)[np.array(df['todo'],dtype='bool')]
        self.length=len(self.todo_idx)
        print(self.todo_idx)

    def split_train_valid_test(self,ratio_train=0.3,ratio_valid=0.3):
        f=self.root+'split_train_valid_test'
        if os.path.isfile(f+'.npz'):
            x=np.load(f+'.npz')
            self.train_idx,self.valid_idx,self.test_idx=x['train_idx'],x['valid_idx'],x['test_idx']
        else:
            n_tra,n_val=int(self.length*ratio_train),int(self.length*ratio_valid)
            self.train_idx,self.valid_idx,self.test_idx=random_split(
                self.todo_idx, 
                lengths=[n_tra,n_val,self.length-n_tra-n_val],
                generator=torch.Generator().manual_seed(0))
            
            self.train_idx,self.valid_idx,self.test_idx=[6,7],[8,9],[22,23,24,25]
            np.savez_compressed(f,train_idx=self.train_idx,valid_idx=self.valid_idx,test_idx=self.test_idx)
            
            
        print(list(self.train_idx),list(self.valid_idx),list(self.test_idx))

    def load_raw_image(self,i,channel):
        fs=glob.glob(self.rawPath[i]+channel+'*.tif')
        if len(fs)==0:
            print(channel+' not found for '+self.fname[i])
            return None
        elif len(fs)>1:
            print('more than one images of '+channel+' are found for '+self.fname[i])
            return None
        else:
            return(tiff.imread(fs[0]))

    def load_mas_image(self,i,cell=True):
        f=self.masPath+self.fname[i]
        if cell:
            f+='_cell.npy'
        else:
            f+='_nuc.npy'
        if os.path.isfile(f):
            return(np.load(f))
        else:
            raise ValueError('mask not found for '+self.fname[i])


    def load_spot_txt2dict(self,i,color,index='m'):
        f=self.spotPath+self.fname[i]+'_'+color+'.txt'
        if os.path.isfile(f):
            dic={}
            with open(f,'r') as file:
                for line in file:
                    cx,cy,cz,m=line.split(',')
                    cx,cy,cz,m=int(cx)-2,int(cy)-2,int(cz)-1,int(m)
                    if index=='z':
                        key=str(cz)
                        if key not in dic.keys():
                            dic[key]=[]
                        dic[key].append([cx,cy])

                    elif index=='m':
                        key=str(m)
                        if key not in dic.keys():
                            dic[key]=[]
                        dic[key].append([cx,cy,cz])

                    elif index=='mz':
                        key=str(m)
                        if key not in dic.keys():
                            dic[key]={}
                        if str(cz) not in dic[key].keys():
                            dic[key][str(cz)]=[]
                        dic[key][str(cz)].append([cx,cy])
            return dic
        else:
            raise ValueError('true spot not found for '+self.fname[i])

    
    
    def load_spot_array2dict(self,i,color,index='m'):
        f=self.spotPredPath+self.fname[i]+'_'+color+'.npy'
        if os.path.isfile(f):
            coord=np.load(f)
            mask=self.load_mas_image(i)
            dic={}
            for (x,y,z) in coord:
                x,y,z=int(x),int(y),int(z)
                m=mask[x,y]
                if str(m) not in dic.keys():
                    dic[str(m)]=[]
                dic[str(m)].append([x,y,z])

            return dic
            
        else:
            raise ValueError('pred spot not found for '+self.fname[i])


    def load_spot_txt2array(self,i,color,only_masked=False):
        f=self.spotPath+self.fname[i]+'_'+color+'.txt'
        mask=self.load_mas_image(i)
        if os.path.isfile(f):
            res=[]
            exclude=0
            with open(f,'r') as file:
                for line in file:
                    cx,cy,cz,_=line.split(',')
                    cx,cy,cz=int(cx)-2,int(cy)-2,int(cz)-1
                    if not only_masked:
                        res.append([cx,cy,cz])
                    else:
                        m=mask[cx,cy]
                        if m>0:
                            res.append([x,y,z])
                        else:
                            exclude+=1
            res=np.array(res)
            print(len(res)-exclude,len(res))
            return res
        else:
            raise ValueError('true spot not found for '+self.fname[i])

    def load_spot_array(self,i,color,only_masked=False):
        f=self.spotPredPath+self.fname[i]+'_'+color+'.npy'
        if os.path.isfile(f):
            coord=np.load(f)
            if not only_masked:
                return coord
            else:
                mask=self.load_mas_image(i)
                res=[]
                for (x,y,z) in coord:
                    x,y,z=int(x),int(y),int(z)
                    m=mask[x,y]
                    if m>0:
                        res.append([x,y,z])
                print(len(res),len(coord))
                return np.array(res)
        else:
            raise ValueError('pred spot not found for '+self.fname[i])

