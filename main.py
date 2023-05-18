from spotLoc import *
from spotLoc.utils import *
from preprocess import *
from get_meta import load_io
import os
import torch

def load_io(name):
    if name=='lnc':
        return IO('../lnc/','raw/','*/*/*/*/dapi*.tif')  
    elif name=='yeast':
        return IO('../yeast/','raw/','*/dapi*.tif')  

def run_on_subset(args,root,sub_model_root=''):
    
    percent_list=[1.25,2.5,5.0,10.0,20.0,40.0]
    file_name=[5,10,20,41,82,164]
    n_epochs_list=[300,200,150,100,80,60]
    rep_list=[5,5,5,5,5,5]
    score_list=[]
    
    #train
    for i in range(len(percent_list)):
        args_curr=parse(root,args.name,args.r,args.device,n_epochs=n_epochs_list[i],sub_model_root=sub_model_root,
            percent=percent_list[i],rep=rep_list[i],penal=args.penal,aug=args.aug)
        name_model='model_%s_%s'%(args.archit,args.name)
        #run(args_curr,name_model)
        for j in range(rep_list[i]):
            score=predict(root,args.name,None,args.r,0.4,sub_model_root+'%d/'%file_name[i]+name_model+'_%d'%j,archit=args.archit,file=root+'../smfish.npz',save=False)
            score_list.append(score['f1 score'])    
    print(score_list)

def main():
    os.environ["CUDA_VISIBLE_DEVICES"]="2"
    device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu") 


    root='../lnc/'
    name='lnc'
    color='alexa'
    r=1
    archit='SpotLoc'
    
    args=parse(root,name,r,device,n_epochs=40,epoch_all=20,archit=archit)
    name_model='model_%s_%s_%s'%(args.archit,args.name,color)
    #prepare_input_bycell(root,name,r,color,n_minimum=1,del_zero_percent=0.2)
    
    #run(args,name_model)
    #predict(root,name,color,r,0.2,name_model,archit=args.archit,save=True,image_idx='all',cal_score=False)
    #predict(root,name,color,r,0.1,name_model,archit=args.archit,save=True)
    #evaluation(root,name,color)
    #view(root,name,color,n_cell=None,min_spot=50)
    #view_by_layer(root,name,color,name_model,archit,1,6)
    
    args=parse(root,name,r,device,n_epochs=40,epoch_all=20,archit=archit,penal=False)
    name_model='model_%s_%s_%s_nopenal'%(args.archit,args.name,color)
    #run(args,name_model)
    #predict(root,name,color,r,0.1,name_model,archit=args.archit,save=False)
    
    
    root='../syn_data/smfish/'
    name='smfish'
    r=3
    
    #train
    args=parse(root,name,r,device,n_epochs=40,sub_model_root='standard/')
    name_model='model_%s_%s'%(args.archit,args.name)
    #run(args,name_model)
    #predict(root,name,None,args.r,0.5,'standard/'+name_model,archit=args.archit,file=root+'../smfish.npz',save=False,debug=True)
    #run_on_subset(args,root,sub_model_root='standard/')
    
    args=parse(root,name,r,device,n_epochs=40,sub_model_root='no_penal/',penal=False)
    name_model='model_%s_%s'%(args.archit,args.name)
    #run(args,name_model)
    #predict(root,name,None,args.r,0.4,'no_penal/'+name_model,archit=args.archit,file=root+'../smfish.npz',save=False)
    
    args=parse(root,name,r,device,n_epochs=40,sub_model_root='no_aug/',aug=False)
    name_model='model_%s_%s'%(args.archit,args.name)
    #run(args,name_model)
    #predict(root,name,None,args.r,0.4,'no_aug/'+name_model,archit=args.archit,file=root+'../smfish.npz',save=False)
    #run_on_subset(args,root,sub_model_root='no_aug/')

    args=parse(root,name,r,device,n_epochs=40,sub_model_root='no_penal_aug/',aug=False,penal=False)
    name_model='model_%s_%s'%(args.archit,args.name)
    #run(args,name_model)
    #predict(root,name,None,args.r,0.4,'no_penal_aug/'+name_model,archit=args.archit,file=root+'../smfish.npz',save=False)
    

if __name__ == '__main__':
    main()

