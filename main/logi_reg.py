import torch
import torch.nn as nn
import numpy as np
import pandas as pd

import scipy
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import gan_tools as gt
import importlib
importlib.reload(gt)

import pdb

def k_fold_cv(x_train, y_train, n_splits = 5):
    """K_Fold Cross Validation"""
    
    kf        = KFold(n_splits=n_splits, shuffle=True)
    criterion = nn.BCELoss(reduction = 'mean')

    train_ls, valid_ls = [], []

    for k ,(train, valid) in enumerate (kf.split(x_train, y_train)):
        
        #y_lable = y_train[train]
        #g1_id = y_label[y_label==1]
        #len(g1_id)
        model   =  LogisticRegression()
        model   =  model.fit( x_train[train, :],  y_train[train])
        
        y_pred_t   =  model.predict_proba( x_train[train, :])
        y_pred_v   =  model.predict_proba( x_train[valid, :])
        
        t_ls = criterion(torch.Tensor(y_pred_t[:, -1]), 
                         torch.Tensor(y_train[train]))
        v_ls = criterion(torch.Tensor(y_pred_v[:, -1]),
                         torch.Tensor(y_train[valid]))

        print('K_FOLD:[{}]\tT_loss:{}\tV_Loss: {}'.format( k,  t_ls,  v_ls))

        train_ls.append(t_ls)
        valid_ls.append(v_ls)

    return train_ls, valid_ls 

def main(pars = [175, 100, 3000, 20, 15],
         prefix = "../Model_Data/simu_"):
    ## read data    
    simu_data      = gt.DG_LoadData(pars = pars, prefix = prefix)
    x_load, y_load = simu_data.data_cov, simu_data.data_lab
    #x_train, x_test, y_train,  y_test = train_test_split(x_load, y_load, test_size= 0)
    
    train_ls, valid_ls = k_fold_cv(x_load, y_load.squeeze(1), n_splits = 5)



if __name__ == '__main__':
    main()













