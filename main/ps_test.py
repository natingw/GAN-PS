##
##
##   NN for PS ESTIMATION
##
##

import gan_tools as gt
import importlib
importlib.reload(gt)

import numpy as np
import pandas as pd
import torch
import pdb

## set parameters
## cur_pars = gt.get_setting()

def get_results(simu_data, net_d, 
                 prefix = "../Results/", 
                 postfix = "_net_20_2_1_output.csv",
                 pars = None):
    
    if pars is not None:
        fname =  '_'.join(str(x) for x in pars)

    fname    = prefix + fname + postfix

    data_all = simu_data.data_all
    data_cov = simu_data.data_cov

    net_d.eval()

    with torch.no_grad():
        d_scores   =  net_d(torch.Tensor(data_cov)).numpy()
    
    column_names = data_all.columns.tolist()
    column_names.append('D_scores')

    results = np.c_[data_all, d_scores]
    results = pd.DataFrame(results, columns=column_names)
    results.to_csv(fname, header=True) 
    

def main(pars = [175, 100, 3000, 20, 15],
         prefix = "../model_data/simu_"):

    """main function for using neural network to estimate PS"""

    ## read data
    simu_data = gt.DG_LoadData(pars = pars, prefix = prefix)
    loader_g0, loader_g1 = simu_data.get_loader(n_batch = 1)

    ## define model
    net_d       = gt.DG_Net(simu_data.nx, [2], is_d = True)
    d_optimizer = net_d.get_optimizer(lr = 0.001)
    d_loss      = net_d.get_loss()

    ## load pretrained model
    ch = torch.load("../pre/checkpoint_net_20_2_1_2.pth")
    net_d.load_state_dict(ch)
    
    ## get results
    results = get_results(simu_data, net_d, pars= pars)


if __name__ == '__main__':
    main()

