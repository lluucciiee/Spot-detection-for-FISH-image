from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import scipy.optimize
import warnings
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects,remove_small_holes
from .config import *
#image process
def standardize(x):
    mean=np.mean(x)
    std=np.std(x)
    return (x-mean)/std

#generate quasi_segmentation
def find_min_wind(mask):
    boolmap=np.argwhere(mask==True)
    if boolmap.sum()==0:
        return 0,0,0,0
    else:
        return np.min(boolmap[:,0]),np.max(boolmap[:,0])+1,np.min(boolmap[:,1]),np.max(boolmap[:,1]+1)

def target(l,w,r):
    c1=(l-1)/2
    c2=(w-1)/2
    res=np.array([[np.sqrt((i-c1)**2+(j-c2)**2)<r+0.5 for j in range(w)] for i in range(l)],dtype='int')
    return res
  
def toMap(coord,shape,r):
    (l,w)=shape
    mol=target(r*2+1,r*2+1,r)
    res=np.zeros((l,w))
    for (a,b) in list(coord):
        a,b=int(a),int(b)
        c1=max(a-r,0)
        c2=min(a+r+1,l)
        c3=max(b-r,0)
        c4=min(b+r+1,w)
                
        res[c1:c2,c3:c4]+=mol[c1-(a-r):c2-(a+r+1)+r*2+1,c3-(b-r):c4-(b+r+1)+r*2+1]
    return res.clip(None,1)

def calculate_area(r):
    l=2*r+1
    return np.sum(target(l,l,r))
    
#post analysis
def count(out,r,cutoff=0.5):
    mask_img = remove_small_holes(out>cutoff,3*r**2/10)
    mask_img = remove_small_objects(mask_img,3*r**2/2)
    labels= watershed(-(out>cutoff).astype('int'),mask=mask_img,watershed_line=False) 
    N=labels.max()
    res=[]
    for i in range(1,N):
        coord=np.argwhere(labels==i)
        res.append(coord.mean(0))
    return np.array(res).reshape((-1,2))

def adapt(pred,n,save_root):
    cutoff=3.0
    match=[{} for i in range(n)]
    repete=[[] for i in range(n)]
    for i in range(n-1):
        l1,l2=len(pred[i]),len(pred[i+1])
        if l1==0 or l2==0:
            continue
        matrix = scipy.spatial.distance.cdist(pred[i],pred[i+1], metric="euclidean")
        matrix = np.where(matrix >= cutoff, matrix.max(), matrix)
        row, col = scipy.optimize.linear_sum_assignment(matrix)
        for r, c in zip(row, col):
            if matrix[r, c] <= cutoff:
                match[i][str(r)]=c
                repete[i].append(r)
                repete[i+1].append(c)  
    
    pred_res=[]
    for i in range(n):
        for j in range(len(pred[i])):
            if j not in repete[i]:
                pred_res.append([pred[i][j,0],pred[i][j,1],i])   
        
        for key in match[i].keys():
            r=int(key)
            c=match[i][key]
            xy_list=[pred[i][r],pred[i+1][c]]
            z_list=[i,i+1]
            while str(c) in match[z_list[-1]].keys():
                c_old=c
                c=match[z_list[-1]][str(c_old)]
                del match[z_list[-1]][str(c_old)]
                xy_list.append(pred[z_list[-1]+1][c])
                z_list.append(z_list[-1]+1)
            x,y=np.array(xy_list).mean(0)
            z=np.array(z_list).mean()
            pred_res.append([x,y,z])
        
    if save_root is not None:
        np.save(save_root,np.array(pred_res))
    return np.array(pred_res)


#evaluate
def evaluate(match_res):
    tp,n_t,pt,n_p=match_res
    recall = tp / (n_t + EPS)
    precision = pt / (n_p + EPS)
    f1_value = (2 * precision * recall) / (precision + recall + EPS)
    values = {
        "f1 score": f1_value,
        "recall": recall,
        "precision": precision
        }
    print(values)
    return values

def linear_sum_assignment(
    matrix: np.ndarray, cutoff: float = None
) -> Tuple[list, list]:
    """Solve the linear sum assignment problem with a cutoff.
    A problem instance is described by matrix matrix where each matrix[i, j]
    is the cost of matching i (worker) with j (job). The goal is to find the
    most optimal assignment of j to i if the given cost is below the cutoff.
    Args:
        matrix: Matrix containing cost/distance to assign cols to rows.
        cutoff: Maximum cost/distance value assignments can have.
    Returns:
        (rows, columns) corresponding to the matching assignment.
    """
    # Handle zero-sized matrices (occurs if true or pred has no items)
    if matrix.size == 0:
        return [], []

    # Prevent scipy to optimize on values above the cutoff
    if cutoff is not None and cutoff != 0:
        matrix = np.where(matrix >= cutoff, matrix.max(), matrix)

    row, col = scipy.optimize.linear_sum_assignment(matrix)

    if cutoff is None:
        return list(row), list(col)

    # As scipy will still assign all columns to rows
    # We here remove assigned values falling below the cutoff
    nrow = []
    ncol = []
    for r, c in zip(row, col):
        if matrix[r, c] <= cutoff:
            nrow.append(r)
            ncol.append(c)
    return nrow, ncol

def compute_metrics(
    pred: np.ndarray, true: np.ndarray, mdist: float = 3.0
) -> pd.DataFrame:
    """Calculate metric scores across cutoffs.
    Args:
        pred: Predicted set of coordinates.
        true: Ground truth set of coordinates.
        mdist: Maximum euclidean distance in px to which F1 scores will be calculated.
    Returns:
        DataFrame with one row per cutoff containing columns for:
            * f1_score: Harmonic mean of precision and recall based on the number of coordinates
                found at different distance cutoffs (around ground truth).
            * abs_euclidean: Average euclidean distance at each cutoff.
            * offset: List of (r, c) coordinates denoting offset in pixels.
            * f1_integral: Area under curve f1_score vs. cutoffs.
            * mean_euclidean: Normalized average euclidean distance based on the total number of assignments.
    """
    if pred.size == 0 or true.size == 0:
        warnings.warn(
            f"Pred ({pred.shape}) and true ({true.shape}) must have size != 0.",
            RuntimeWarning,
        )
        return np.zeros(4)

    matrix = scipy.spatial.distance.cdist(pred, true, metric="euclidean")

    # Assignment of pred<-true and true<-pred
    pred_true_r, _ = linear_sum_assignment(matrix, mdist)
    #true_pred_r, true_pred_c = linear_sum_assignment(matrix.T, mdist)

    # Calculation of tp/fn/fp based on number of assignments
    tp = len(pred_true_r)
    #pt = len(true_pred_r)
    fn = len(true) - tp
    fp = len(pred) - tp

    return np.array([tp,len(true),tp,len(pred)])

