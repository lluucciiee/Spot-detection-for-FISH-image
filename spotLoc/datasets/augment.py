import numpy as np
a=0.5
def rand(a):
    return 1+np.random.rand()*a
def augment(x,y):
    x_new=np.expand_dims(x,0)
    y_new=np.expand_dims(y,0)
    if np.random.rand()>0.5:
        x_new=np.concatenate([x_new,np.expand_dims(x[:,::-1],0)],0)
        y_new=np.concatenate([y_new,np.expand_dims(y[:,::-1],0)],0)
    if np.random.rand()>0.5:
        x_new=np.concatenate([x_new,np.expand_dims(x[::-1,:],0)],0)
        y_new=np.concatenate([y_new,np.expand_dims(y[::-1,:],0)],0)
    if np.random.rand()>0.5:
        x_new=np.concatenate([x_new,np.expand_dims(x[::-1,::-1],0)],0)
        y_new=np.concatenate([y_new,np.expand_dims(y[::-1,::-1],0)],0)
    return x_new.max(0),y_new.max(0)

def augment2(x,y):
    if np.random.rand()>0.5:
        m=1
    else:
        m=2
    return shift_x(x,m),shift_y(y,m)

def shift_x(x,m):
    n=8*m
    l,w=x.shape
    xx=np.repeat(x,n).reshape(l,w,n)
    s=0
    for i in np.linspace(0,1,m+1)[1:]:
        l_shift=int(l/2*i)
        w_shift=int(w/2*i)
        xx[l_shift:,:,s]=x[:-l_shift,:]
        xx[:-l_shift,:,s+1]=x[l_shift:,:]
        xx[:,w_shift:,s+2]=x[:,:-w_shift]
        xx[:,:-w_shift,s+3]=x[:,w_shift:]
        xx[l_shift:,w_shift:,s+4]=x[:-l_shift,:-w_shift]
        xx[:-l_shift,w_shift:,s+5]=x[l_shift:,:-w_shift]
        xx[l_shift:,:-w_shift,s+6]=x[:-l_shift,w_shift:]
        xx[:-l_shift,:-w_shift,s+7]=x[l_shift:,w_shift:]
        s+=8
    return(xx.max(2))

def shift_y(x,m):
    n=8*m
    l,w=x.shape
    xx=np.repeat(x,n).reshape(l,w,n)
    s=0
    for i in np.linspace(0,1,m+1)[1:]:
        l_shift=int(l/2*i)
        w_shift=int(w/2*i)
        xx[l_shift:,:,s]=x[:-l_shift,:]
        xx[:-l_shift,:,s+1]=x[l_shift:,:]
        xx[:,w_shift:,s+2]=x[:,:-w_shift]
        xx[:,:-w_shift,s+3]=x[:,w_shift:]
        xx[l_shift:,w_shift:,s+4]=x[:-l_shift,:-w_shift]
        xx[:-l_shift,w_shift:,s+5]=x[l_shift:,:-w_shift]
        xx[l_shift:,:-w_shift,s+6]=x[:-l_shift,w_shift:]
        xx[:-l_shift,:-w_shift,s+7]=x[l_shift:,w_shift:]
        s+=8
    return(xx.max(2))
