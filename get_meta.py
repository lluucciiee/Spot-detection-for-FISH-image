import pandas as pd
import glob
from spotloc.util import IO

def load_io(name):
    #####################
    #Modify the path before running
    #Class IO needs 3 paths to initiate: absolute path of name_project/, 
    #local path of image data in name_project/,
    #local path of an image in name_project/ 
    #####################

    if name=='lnc':
        return IO('../lnc/','raw/','*/*/*/*/dapi*.tif')  
    elif name=='yeast':
        return IO('../yeast/','raw/','*/dapi*.tif') 

def make_meta(name):
    F=File(name)
    res=[]
    for i in range(F.length):       
        path=F.rawPath[i]
        fnum=glob.glob(root+path+'dapi*.tif')[0].split('.')[-2][-1]
        #redo for new dataset
        is_tmr=len(glob.glob(root+path+'tmr*.tif'))>0
        is_cy=len(glob.glob(root+path+'cy*.tif'))>0
        #name the segmentation mask to be created in run_seg.py 
        mas=F.masPath+F.fname[i]+'_cell.npy'
        #name the true label to be created raj's matlab package
        out=F.spotPath+F.fname[i]
        #the last column stands for if the image needs true label(will be used in train/valid/test)
        res.append([path, fnum, mas,out,is_tmr,is_cy,0])
        

    meta=pd.DataFrame(res,columns=['rawPath','fnum','masPath','outPath','is_tmr','is_cy','todo'])
    meta.to_csv(root+'meta.csv')

if __name__ == '__main__':
    name='lnc'
    make_meta(name)
