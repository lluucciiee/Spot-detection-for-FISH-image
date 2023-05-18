import glob
import os
import re
import pandas as pd
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from get_meta import load_io
from hpacellseg.cellsegmentator import CellSegmentator
from hpacellseg.utils import label_cell


def normalize(x):
    mmax,mmin=np.max(x),np.min(x)
    return (x-mmin)/(mmax-mmin+1e-6)

def cut(x,a,b):
    mean,std=np.mean(x),np.std(x)
    x=x.clip(mean+std*a,mean+std*b)
    return x


def run(root,name):
    cell_names=['CRL2097','Hela','Imr90','IMR90']

    
    channel_priority=['gfp','cy']
    c1,c2=channel_priority


    F=load_io(name)
    assert(F.n_rc%512==0)#model with fixed input image size 512*512, simple solution for other images: if size 512n*512n, cut into n*n blocs of 512*512
    n_b=F.n_rc//512
    os.makedirs(F.masPath, exist_ok=True)
    os.makedirs(F.viewPath+'seg/', exist_ok=True)
    
    #segemntor from github
    cellsegmentor = CellSegmentator()

    #Iterations
    for i in range(F.length):
        name=F.fname[i]
        print(name)
        #load data
        dapi=F.load_raw_image(i,'dapi').max(0)
        trans = F.load_raw_image(i,'trans').mean(0)

        imc1 =F.load_raw_image(i,c1).min(0)
        imc2 =F.load_raw_image(i,c2)
        if imc2 is not None:
            imc2=imc2.max(0)
        else:
            imc2=F.load_raw_image(i,c1).min(0)
        
        cell=''
        for cell_name in cell_names:
            if name.find(cell_name)>=0:
                cell=cell_name
        
        dapi=normalize(cut(dapi,-1,10))
        if cell=='Hela':
            a=0
        elif cell=='CRL2097':
            a=1
        else:
            a=1

        trans=normalize(cut(trans,a,10))
        imc1=normalize(cut(imc1,a,10))
        imc2=normalize(cut(imc2,a,10))
            
        cell = np.stack((imc1,imc2,trans), axis=2).reshape(n_b,512,n_b,512,3).transpose(0,2,1,3,4).reshape(n_b**2,512,512,3)
        dapi = dapi.reshape(n_b,512,n_b,512).transpose(0,2,1,3).reshape(n_b**2,512,512)
            
        nuc_segmentations = cellsegmentor.pred_nuclei(dapi)
        cell_segmentations = cellsegmentor.pred_cells(cell,precombined=True)

        cell_res=np.zeros((F.n_rc,F.n_rc,3))
        nuc_res=np.zeros((F.n_rc,F.n_rc,3))
        
        #predict by bloc
        for i in range(n_b):
            for j in range(n_b):
                nuc_seg=nuc_segmentations[n_b*i+j]
                cell_seg=cell_segmentations[n_b*i+j]
                cell_res[512*i:512*(i+1),512*j:512*(j+1),:]=cell_seg
                nuc_res[512*i:512*(i+1),512*j:512*(j+1),:]=nuc_seg

        nuc, cell = label_cell(nuc_res, cell_res)
        
        plt.figure(figsize=(16,9))
        plt.subplot(1,2,1)
        plt.imshow(cell)
        plt.subplot(1,2,2)
        plt.imshow(nuc)
        plt.savefig('%s%s%s.png'%(F.viewPath,'seg/',name))
        np.save('%s%s_cell.npy'%(F.masPath,name), cell)
        np.save('%s%s_nuc.npy'%(F.masPath,name), nuc)

if __name__ == "__main__" :
    root='../lnc/'
    name='lnc'
    run(root,name)




