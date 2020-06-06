"""GAN-PS Module

Functions and classes for GAN-PS computation
"""

## ------------------------------------------------------------------
##          IMPORT
## ------------------------------------------------------------------

import argparse
import os

import numpy as np
import pandas as pd
import random
from   datetime import datetime
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as Data

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn import metrics

import pdb

## ------------------------------------------------------------------
##          PROPERTIES
## ------------------------------------------------------------------
all_pars  = np.array([(rep, sce, size)
                      for rep in range(500)
                      for sce in (175, 176)
                      for size in (100, 1000)])

n0_size   = 3000

## ------------------------------------------------------------------
##          TOOL FUNCTIONS
## ------------------------------------------------------------------
def as_list(x):
    """Convert x to list"""
    if type(x) is list:
        return x
    else:
        return [x]

def get_setting():
    g_nodes, g_inx, g_simu_path, g_rst_path  = read_args().values()
    n_each   = math.ceil(all_pars.shape[0] / g_nodes)
    cur_pars = range((g_inx - 1) * n_each, g_inx * n_each)

    print(cur_pars)
    return cur_pars

def read_args(argv = None):
    """read argument from command line"""

    parser = argparse.ArgumentParser(description = "GAN-PS")
    parser.add_argument("-n", "--nodes",
                        type = int, default = 1,
                        help = "total number of nodes")
    parser.add_argument("-i", "--inx",
                        type = int, default = 1,
                        help = "index of the current node")
    parser.add_argument("-s", "--simu_path",
                        default = "../SimuData/",
                        help = "path to simulation data")
    parser.add_argument("-r", "--rst_path",
                        default = "../Results/",
                        help = "path to results")
    parser.add_argument("-p", "--pre_path",
                        default = "../PRE/",
                        help = "path to pretrained model")
    parser.add_argument("-l", "--log_path",
                        default = "../Log/",
                        help = "path to log file")
    
    args = vars(parser.parse_args())

    ## print out
    prompt = """Arguments:
    Total number of nodes      :{nodes}
    Current node               :{inx}
    Path to simulation data    :{simu_path}
    Path to results            :{rst_path}
    Path to log file           :{log_path}
    Path to pretrained model   :{pre_path}
    """

    print(prompt.format(**args))

    ## return
    return args

def weights_init(m):
    classname = m.__class__.__name__
    """initialize parameters"""
    if classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight, 0, 0.1)
        torch.nn.init.zeros_(m.bias)

## ------------------------------------------------------------------
##          CLASSES
## ------------------------------------------------------------------
class DG_LoadData():
    """Load simulation data"""
    def __init__(self, fname = None, pars = None,
                 prefix = "./SimuData/simu_",
                 postfix = ".csv",
                 cov_grep = "^V[0-9]+$", grp = "group"):

        if fname is None:
            fname =  '_'.join(str(x) for x in pars)
        
        fname      = prefix + fname + postfix
        simu_data  = pd.read_csv(fname, index_col = 0, iterator = False)
        str_cov    = simu_data.columns.str.contains(cov_grep, regex = True)
        simu_cov   = np.asarray(simu_data.loc[:, str_cov])
        simu_label = np.asarray(simu_data[[grp]])

        self.data_fname = fname
        self.data_all   = simu_data
        self.data_cov   = simu_cov
        self.data_lab   = simu_label
        self.nx         = simu_cov.shape[1]
        self.batch_size = {}

    def get_group(self, group = 0):
        inx = group == self.data_lab
        dta = self.data_cov[inx[:, 0], :]
        return dta

    def get_loader_g(self, group, n_batch, grp_index = None):
        t_g  = self.get_group(group)
        if grp_index is not None:
            t_g = t_g[grp_index, :]
        b_s  = math.ceil(t_g.shape[0] / n_batch)
        self.batch_size[str(group)] = b_s
        d_g  = Data.TensorDataset(torch.Tensor(t_g))
        l_g  = Data.DataLoader(dataset     = d_g,
                               batch_size  = b_s)
        return l_g

    def get_loader_kf(self, t_id_0, t_id_1, v_id_0, v_id_1, n_batch =5):
        t_g0  = self.get_loader_g(0,  n_batch,   t_id_0 )
        t_g1  = self.get_loader_g(1,  n_batch,   t_id_1 )
        v_g0  = self.get_loader_g(0,  1,         v_id_0 )
        v_g1  = self.get_loader_g(0,  1,         v_id_1 )
        return t_g0, t_g1, v_g0,  v_g1

    def get_loader(self, n_batch = 5):
        l_g0    = self.get_loader_g(0, n_batch)
        l_g1    = self.get_loader_g(1, n_batch)
        return l_g0, l_g1


class DG_Net(nn.Module):
    """Neural network for both G and D-network"""
    def __init__(self, input_size, hidden_size, is_d = True):
        super(DG_Net, self).__init__()

        ## create all connected linear layers based on
        ## input- and hidden-sizes
        all_s  = as_list(input_size) + as_list(hidden_size) + [1]
        layers = []
        for i in range(len(all_s) - 1):
            layers.append(nn.Linear(all_s[i], all_s[i+1]))
            layers.append(nn.ReLU())

        ## remove the last activation function
        layers.pop()
        if is_d:
            layers.append(nn.Sigmoid())

        ## define D- or G-network
        self.is_d = is_d

        ## define network
        self.nn   = nn.Sequential(*layers)

        ## log file
        self.log_fname  = ""

    def init_pars(self):
        """initialize network parameter"""
        self.apply(weights_init)

    def get_optimizer(self, lr = 0.001):
        """optimizer"""
        #opti = optim.SGD(self.parameters(), lr = lr)
        opti = optim.Adam(self.parameters(), lr=lr,  weight_decay=1e-4)
        return opti

    def get_loss(self):
        """loss function"""
        criterion = nn.BCELoss(reduction = 'none')
        return criterion

    def print_pars(self):
        """print current parameters"""
        for name, param in self.nn.named_parameters():
            if param.requires_grad:
                print(name, param.data)

    def forward(self, input):
        y_score = self.nn(input)
        ## G-network needs exponential to make the results positive
        if not self.is_d:
            y_score = torch.exp(y_score)
        return y_score.squeeze(1)

    def set_log(self, log_fname = None, cols = ("epoch", "loss")):
        """initialize log file name"""

        if log_fname is None:
            log_fname = "log_" + datetime.now().strftime("%H_%M_%S") + ".csv"

        self.log_fname = log_fname
        self.log_cols  = cols

        fconv_d = open(log_fname, 'w')
        fconv_d.write(", ".join(cols))
        fconv_d.write("\n")
        fconv_d.close()

    def write_log(self, vals):
        """write to log file name"""

        if "" == self.log_fname:
            self.set_log()
        fconv_d = open(self.log_fname, 'a+')
        fconv_d.write(", ".join(str(x) for x in vals))
        fconv_d.write("\n")
        fconv_d.close()

## ------------------------------------------------------------------
##          UPDATE NETWORK
## ------------------------------------------------------------------
def forward_loss_D(g0_scores, g1_scores, f_loss,
                   g0_weights = None, g1_weights = None):

    """Calculate discriminator network loss

    Arguments:
        g0_scores: D-net sigmoid values for group 0 (fake) inputs
        g1_scores: D-net sigmoid values for group 1 (real) inputs
    """

    g0_n      = g0_scores.shape[0]
    g0_labels = torch.zeros(g0_n).cuda()
    g0_loss   = f_loss(g0_scores, g0_labels)

    if g0_weights is None:
        g0_weights = torch.ones(g0_n).cuda()

    g1_n      = g1_scores.shape[0]
    g1_labels = torch.ones(g1_n).cuda()
    g1_loss   = f_loss(g1_scores, g1_labels)

    if g1_weights is None:
        g1_weights = torch.ones(g1_n).cuda()

    loss =  torch.dot(g0_weights, g0_loss)
    loss += torch.dot(g1_weights, g1_loss)

    return loss

def update_D(net_D, loader_g0, loader_g1, d_optimizer, **kwargs):
    """Update discriminator"""

    net_D.train()

    i = 0
    running_loss = []
    for (g0, g1) in zip(loader_g0, loader_g1):
        
        g1_batch  = g1[0].cuda()
        g0_batch  = g0[0].cuda()
        g1_scores = net_D(g1_batch)
        g0_scores = net_D(g0_batch)
        
        d_optimizer.zero_grad()
        d_loss = forward_loss_D(g0_scores, g1_scores, **kwargs)
        d_loss.backward()
        d_optimizer.step()

        running_loss.append(d_loss.item())

    return sum(running_loss)/(len(loader_g0.dataset)+ len(loader_g1.dataset))

def infer_D(net_D, loader_g0, loader_g1, **kwargs):

    net_D.eval()

    running_loss = []
    with torch.no_grad():
        for (g0, g1) in zip(loader_g0, loader_g1):

            g1_batch  = g1[0].cuda()
            g0_batch  = g0[0].cuda()
            g1_scores = net_D(g1_batch)
            g0_scores = net_D(g0_batch)

            d_loss = forward_loss_D(g0_scores, g1_scores, **kwargs)

            running_loss.append(d_loss.item())
    
    return sum(running_loss)/(len(loader_g0.dataset)+ len(loader_g1.dataset))

## ------------------------------------------------------------------
##          AUC REPORT
## ------------------------------------------------------------------
def forward_auc_D(g0_scores, g1_scores ):

    g0_n      = g0_scores.shape[0]
    g0_labels = torch.zeros(g0_n).cuda()

    g1_n      = g1_scores.shape[0]
    g1_labels = torch.ones(g1_n).cuda()
    
    #y_scores = np.r_[g0_scores, g1_scores]
    #y_labels = np.r_[g0_labels, g1_labels]
    y_scores = torch.cat((g0_scores, g1_scores), 0).cpu().numpy()
    y_labels = torch.cat((g0_labels, g1_labels), 0).cpu().numpy()
    #y_labels = np.r_[torch.zeros(100), torch.ones(100)]
    
    fpr, tpr, thresholds = metrics.roc_curve(y_labels, y_scores, pos_label=1)

    return fpr, tpr, thresholds 

def auc_scores_D(net_D, loader_g0, loader_g1):

    net_D.eval()

    with torch.no_grad():
        for (g0, g1) in zip(loader_g0, loader_g1):

            g1_batch  = g1[0].cuda()
            g0_batch  = g0[0].cuda()
            g1_scores = net_D(g1_batch)
            g0_scores = net_D(g0_batch)
            
            fpr, tpr, thresholds = forward_auc_D(g0_scores, g1_scores)
            roc_auc = metrics.auc(fpr,tpr)
    
    return  {'auc': roc_auc, 'fpr': fpr, 'tpr':tpr}

## ------------------------------------------------------------------
##          MAIN
## ------------------------------------------------------------------

def main():
    """main function"""
    print("This is a toolbox file for GAN-PS Estimation.")

if __name__ == '__main__':
    main()
