from scipy import stats
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

clamp_possibility=0.5

def get_confus_matrix(pred,gt,**kwargs):
    """
    Calculate the L1 loss for the score.
    Order of input did not effect the result.
    """

    levels=np.argmax(pred,axis=1)
    # print()
    lent,num_levels=pred.shape
    confus_matrix=np.zeros([num_levels,num_levels])
    for ind in range(0,len(gt)):
      confus_matrix[gt[ind],levels[ind]]+=1

    return confus_matrix


def calculate_acc(pred,gt,**kwargs):
    """
    Calculate the L1 loss for the score.
    Order of input did not effect the result.
    """
    pred=pred.flatten()
    gt=gt.flatten()
    levels=np.where(pred<clamp_possibility,0,1)

    return accuracy_score(gt,levels)

def calculate_p(pred,gt,**kwargs):
    """
    Calculate the L1 loss for the score.
    Order of input did not effect the result.
    """

    pred=pred.flatten()
    gt=gt.flatten()
    levels=np.where(pred<clamp_possibility,0,1)

    # return [precision_score(gt,levels,average='macro'),precision_score(gt,levels,average='micro'),precision_score(gt,levels,average='weighted')]
    return precision_score(gt,levels,average='macro')

def calculate_r(pred,gt,**kwargs):
    """
    Calculate the L1 loss for the score.
    Order of input did not effect the result.
    """

    pred=pred.flatten()
    gt=gt.flatten()
    levels=np.where(pred<clamp_possibility,0,1)

    # return [recall_score(gt,levels,average='macro'),recall_score(gt,levels,average='micro'),recall_score(gt,levels,average='weighted')]
    return recall_score(gt,levels,average='macro')

def calculate_f1(pred,gt,**kwargs):
    """
    Calculate the L1 loss for the score.
    Order of input did not effect the result.
    """

    pred=pred.flatten()
    gt=gt.flatten()
    levels=np.where(pred<clamp_possibility,0,1)

    # return [f1_score(gt,levels,average='macro'),f1_score(gt,levels,average='micro'),f1_score(gt,levels,average='weighted')]
    return f1_score(gt,levels,average='macro')