import argparse
from .io import load_io

mdist=3.0
EPS = 1e-12

def parse(root,name,r,device,n_epochs=60,epoch_all=30,sub_model_root='',percent=None,rep=None,penal=True,aug=True,archit='SpotLoc'):
    if name=='smfish' or name=='suntag':
        trainPath=root+'train_set/'
        validPath=root+'valid_set/'
        modelPath=root+'model/'
    else:
        F=load_io(name)
        trainPath=F.trainPath
        validPath=F.validPath
        modelPath=F.modelPath

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--name', type=str, default=name, help='')
    parser.add_argument('--device', type=str, default='cuda', help='')
    parser.add_argument('--train_root', type=str, default=trainPath, help='')
    parser.add_argument('--valid_root', type=str, default=validPath, help='')
    parser.add_argument('--model_root', type=str, default=modelPath+sub_model_root, help='')
    parser.add_argument('--index_root', type=str, default=modelPath+'index/', help='')
    
    parser.add_argument('--archit', type=str, default=archit, help='')
    parser.add_argument('--lambd', type=float, default=1e-2, help='')
    parser.add_argument('--down_time', type=int, default=2, help='')
    parser.add_argument('--r', type=float, default=r, help='')
    parser.add_argument('--aug', type=bool, default=aug, help='')
    parser.add_argument('--penal', type=bool, default=penal, help='')
        
    parser.add_argument('--n_epochs', type=int, default=n_epochs, help='')
    parser.add_argument('--epoch_all', type=int, default=epoch_all, help='')
    parser.add_argument('--batch', type=int, default=1, help='')
    parser.add_argument('--lr', type=float, default=1e-4, help='')
    parser.add_argument('--step_size', type=int, default=n_epochs//2, help='')

    
    parser.add_argument('--percent', type=float, default=percent, help='')
    parser.add_argument('--repeat', type=int, default=rep, help='')
    parser.add_argument('--debug', type=bool, default=False, help='')
    
    args = parser.parse_args()
    return args


    