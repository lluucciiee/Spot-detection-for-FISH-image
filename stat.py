import numpy as np
import pandas as pd
import re
import os
import glob
import matplotlib.pyplot as plt
plt.switch_backend('agg')
'''
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import pearsonr
'''
from spotLoc.utils import *
from get_meta import load_io

cell_names=['CRL2097','Hela','Imr90','IMR90']
lnc_name_subst={"Meg3oddsTmrEvensA594":"Meg3","lincFOXF1":"FENDRR","lincMKLN1_A1new":"MKLN1-AS","GAS5":"GAS5",  "NRON":"NRON" ,"HOTAIR":"HOTAIR","NM_139078":"MAPKAPK5-AS1","ANCR":"ANCR", "HOXA13_ORF_3UTR":"HOXA13_ORF_3UTR","Anril":"Anril","MIATnew":"MIAT","Terc":"Terc","TINCR": "TINCR"}
seq=["NRON" , "lincFOXF1","HOTAIR","lincMKLN1_A1new","Anril","GAS5",  "NM_139078","HOXA13_ORF_3UTR" ]


def search_intersection(mask,xc,yc,alpha):
    l,w=mask.shape
    x=xc
    y=yc
    if (alpha>45 and alpha<135) or (alpha>225 and alpha<315):
        dy=1
        dx=1/np.tan(np.pi*alpha/180)*dy
    else:
        dx=1
        dy=np.tan(np.pi*alpha/180)*dx
    while x>0 and y>0 and x<l and y<w and mask[int(x),int(y)]:
        x+=dx
        y+=dy
    return int(x),int(y)


def search_oneside(matrix,xc,yc,left):
    l,w=matrix.shape
    best_score=0
    best_coeff=[]
    if left:
        start=0
        end=90
    else:
        start=-90
        end=0
    for alpha in np.arange(start,end):
        x,y=search_intersection(matrix,xc,yc,alpha)
        la=np.sqrt((xc-x)**2+(yc-y)**2)
        x,y=search_intersection(matrix,xc,yc,alpha+90)
        lb=np.sqrt((xc-x)**2+(yc-y)**2)
        if la==0 or lb==0:
            continue
        #print(la,lb)
        score=0
        cos_a=np.cos(np.pi*alpha/180)
        sin_a=np.sin(np.pi*alpha/180)
        for i in range(l):
            for j in range(w):
                val=((i-xc)*cos_a-(j-yc)*sin_a)**2/la**2+((j-yc)*cos_a+(i-xc)*sin_a)**2/lb**2
                if val<=1:
                    if matrix[i,j]:
                        score+=1
                    else:
                        score-=1
        if score>best_score:
            best_score=score
            best_coeff=[la,lb,alpha]
        if score-best_score<-10:
            break
    return best_score,best_coeff

def eclipse(mask,m):
    a,b,c,d=find_min_wind(mask==m)
    xc,yc=(a+b)/2-a,(c+d)/2-c
    matrix=(mask==m)[a:b,c:d]
    best_score1,best_coeff1=search_oneside(matrix,xc,yc,True)
    best_score2,best_coeff2=search_oneside(matrix,xc,yc,False)
    if best_score1>best_score2:
        return best_score1,best_coeff1,xc+a,yc+c
    else:
        return best_score2,best_coeff2,xc+a,yc+c
    

def show_image(show_mat,coord,title,xc,yc,name):
    plt.figure()
    plt.imshow(show_mat)
    plt.scatter(coord[:,1],coord[:,0],c='k',s=1)
    plt.scatter([yc],[xc],c='r')
    plt.title(title)
    plt.savefig('img/'+name)



def distribution_draw(dic,savename,seq=None):

    figure=plt.figure(figsize=(20,20))
    #figure=plt.figure(figsize=(20,30))
    n_gene=len(list(dic.keys()))
    print(n_gene)
    colors=['b','g','y']
    for j_cell in range(3):
        plt.subplot(n_gene+1,4,j_cell+2)
        plt.axis('off')
        plt.text(0,0,cell_names[j_cell],c=colors[j_cell],fontsize=15)
    
    mean_list=[]       
    i_gene=1
    if seq is None:
        seq=dic.keys()
    for key in seq:
        plt.subplot(n_gene+1,4,i_gene*4+1)
        plt.axis('off')
        gene=key
        if key in lnc_name_subst.keys():
            gene=lnc_name_subst[key]
        plt.text(0.4,0.5,gene,fontsize=15)
        for key2 in dic[key].keys():
            j_cell=cell_names.index(key2)
            axes=figure.add_subplot(n_gene+1,4,i_gene*4+j_cell+2)
            y=axes.hist(np.array(dic[key][key2]),range=(0,1),bins=20,rwidth=0.9,label='Empirical',color=colors[j_cell])
            mean=np.array(dic[key][key2]).mean()
            mean_list.append(mean)
            plt.vlines(x =mean,ymin=0,ymax=y[0].max(),color='r')
            axes.set_aspect(aspect=1/y[0].max()/10)
            
        i_gene+=1
    plt.savefig(savename)
    return mean_list


def stat(root,name,color,debug=False,view=False):
    if not os.path.isfile(root+'count_'+color+'.csv'):
        if debug:
            os.makedirs(root+'debug/', exist_ok=True)
        if debug:
            os.makedirs(root+'view/', exist_ok=True)
        F=load_io(name)
        tbl=[]
        tbl_2d=[]
        for i in range(F.length):
            fs=glob.glob(F.rawPath[i]+color+'*.tif')
            if len(fs)==0:
                continue
            fname=F.fname[i]
            print(fname)
            
            for cell_name in cell_names:
                if fname.find(cell_name)>=0:
                    cell=cell_name
            gene=fname[:fname.find(cell)-1]
            if cell=='IMR90':
                cell='Imr90'
            
            #load data
            img=F.load_raw_image(i,color)
            img_show=standardize(img.max(0))
            mask=F.load_mas_image(i)
            mask_nuc=F.load_mas_image(i,cell=False)
            dic=F.load_spot_array2dict(i,color,index='m')
            n_z=img.shape[0]
            
            #in nuclei 3d
            for key in dic.keys():
                m=int(key)
                n_cell=len(dic[key])
                if m==0 or n_cell<5:
                    continue

                a,b,c,d=find_min_wind(mask==m)
                if b-a<=0 or d-c<=0:
                    continue

                score,coeff,xc,yc=eclipse(mask_nuc[a:b,c:d],m)
                print(score)
                if score<2000 or len(coeff)==0:
                    if debug:
                        #debug
                        plt.figure()
                        plt.imshow(mask_nuc[a:b,c:d]==m)
                        plt.savefig(root+'debug/'+str(m)+'-'+str(n_cell)+'.png')
                    continue

                zc=int(len(img)/2)
                la,lb,alpha=coeff
                if la==0 or lb==0:
                    continue
                lc=zc
                cos_a=np.cos(np.pi*alpha/180)
                sin_a=np.sin(np.pi*alpha/180)

                n_nuc=0
                n_nucmembrane=0
                n_nuc_2d=0
                n_nuc_2d_eclip=0
                n_nucmembrane_2d=0
                coord=np.array(dic[key])
                coord[:,0]=coord[:,0]-a
                coord[:,1]=coord[:,1]-c
                for (x,y,z) in coord:
                    val=((x-xc)*cos_a-(y-yc)*sin_a)**2/la**2+((y-yc)*cos_a+(x-xc)*sin_a)**2/lb**2+(z-zc)**2/lc**2
                    val2=((x-xc)*cos_a-(y-yc)*sin_a)**2/la**2+((y-yc)*cos_a+(x-xc)*sin_a)**2/lb**2
                    if val<=1:
                        n_nuc+=1
                    if abs(val-1)<0.3:
                        n_nucmembrane+=1
                    
                    if mask_nuc[x+a,y+c]==m:
                        n_nuc_2d+=1
                    if val2<=1:
                        n_nuc_2d_eclip+=1
                    if abs(val2-1)<0.3:
                        n_nucmembrane_2d+=1
                    
                print('n_nuc ',n_nuc)
                print('n_nucmebrane ',n_nucmembrane)
                print('n_nuc_2d ',n_nuc_2d)
                print('n_nuc_2d_eclip ',n_nuc_2d_eclip)
                print('n_nucmebrane_2d ',n_nucmembrane_2d)
                
                if view and gene in seq:
                    plt.figure()
                    plt.subplot(1,2,1)
                    plt.imshow((mask_nuc[a:b,c:d]==m).astype('int')+(mask[a:b,c:d]==m).astype('int'))
                    plt.scatter(coord[:,1],coord[:,0],c='r',s=2)
                    plt.subplot(1,2,2)
                    plt.imshow(img_show[a:b,c:d])
                    plt.title('total: '+str(n_cell)+'     '+'n_nuc: '+str(n_nuc_2d)+'     '+'n_nucmembrane: '+str(n_nucmembrane_2d)) 
                    plt.savefig(root+'img/'+fname+'-'+str(m)+'.png')
                
                tbl.append([gene,cell,n_cell,n_nuc,n_nucmembrane])
                tbl_2d.append([gene,cell,n_cell,n_nuc_2d,n_nucmembrane_2d])
            
            
        data=pd.DataFrame(np.array(tbl),columns=['gene','cell','n_cell','n_nuc','n_nucmembrane'])
        data.to_csv(root+'count_'+color+'.csv')
        data=pd.DataFrame(np.array(tbl_2d),columns=['gene','cell','n_cell','n_nuc','n_nucmembrane'])
        data.to_csv(root+'count_2d_'+color+'.csv')
            

def draw(file,mode="nuc"):
    df=pd.read_csv(file)
    dic={}
    
    for i in range(len(df)):
        _,gene,cell,n_cell,n_nuc,n_nucmembrane=df.iloc[i]
        if gene not in dic.keys():
            dic[gene]={}
        if cell not in dic[gene].keys():
            dic[gene][cell]=[]

        if mode=="nuc":
            ratio=n_nuc/n_cell
        elif mode=="mem":
            ratio=n_nucmembrane/n_cell
        dic[gene][cell].append(ratio)
    
    gs=list(dic.keys())
    for gene in gs:
        cs=list(dic[gene].keys())
        for cell in cs:
            if len(dic[gene][cell])<10:
                del dic[gene][cell]
        if len(dic[gene].keys())==0:
            del dic[gene]
               
    distribution_draw(dic,'distribution_%s_%s.png'%(mode,color))

def shift(file,file_2d,mode="nuc"):
    dic={}
    dic_2d={}
    for gene in seq:
        dic[gene]=[]
        dic_2d[gene]=[]
    
    df=pd.read_csv(file)
    for i in range(len(df)):
        _,gene,cell,n_cell,n_nuc,n_nucmembrane=df.iloc[i]
        if gene in dic.keys() and cell=='Hela':
            dic[gene].append(n_nuc/n_cell)
    
    df=pd.read_csv(file_2d)
    for i in range(len(df)):
        _,gene,cell,n_cell,n_nuc,n_nucmembrane=df.iloc[i]
        if gene in dic_2d.keys() and cell=='Hela':
            dic_2d[gene].append(n_nuc/n_cell)
    
    colors=['b','g']
    n_gene=len(seq)

    figure=plt.figure(figsize=(13,12))
    plt.subplot(n_gene+1,3,2)
    plt.axis('off')
    plt.text(0,0,"before",c=colors[0],fontsize=15)
    plt.subplot(n_gene+1,3,3)
    plt.axis('off')
    plt.text(0,0,"after",c=colors[1],fontsize=15)
    
    mean_list=[]       
    i_gene=1
    for key in seq:
        plt.subplot(n_gene+1,3,i_gene*3+1)
        plt.axis('off')
        plt.text(0.4,0.5,lnc_name_subst[key],fontsize=15)
        
        axes=figure.add_subplot(n_gene+1,3,i_gene*3+3)
        y=axes.hist(np.array(dic[key]),range=(0,1),bins=5,rwidth=0.9,label='Empirical',color=colors[1])
        mean=np.array(dic[key]).mean()
        mean_list.append(mean)
        plt.vlines(x =mean,ymin=0,ymax=y[0].max(),color='r')
        axes.set_aspect(aspect=1/y[0].max()/10)

        axes=figure.add_subplot(n_gene+1,3,i_gene*3+2)
        y=axes.hist(np.array(dic_2d[key]),range=(0,1),bins=5,rwidth=0.9,label='Empirical',color=colors[0])
        mean=np.array(dic_2d[key]).mean()
        mean_list.append(mean)
        plt.vlines(x =mean,ymin=0,ymax=y[0].max(),color='r')
        axes.set_aspect(aspect=1/y[0].max()/10)
            
        i_gene+=1
    plt.savefig('shift.png')
    

if __name__ == "__main__" :
    root='../lnc/'
    name='lnc'
    color='alexa'
    
    #stat(root,name,color)
    file=root+'count_'+color+'.csv'
    file_2d=root+'count_2d_'+color+'.csv'
    #draw(file,mode="nuc")
    #draw(file,mode="mem")
    shift(file,file_2d,mode="nuc")

