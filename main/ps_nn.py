##
##
##   NN for PS ESTIMATION
##
##

import gan_tools as gt
import importlib
importlib.reload(gt)

import torch
import pdb
import time

from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import ReduceLROnPlateau
#from torchsample.callbacks import EarlyStopping


## set parameters
## cur_pars = gt.get_setting()
def GD_Train(net_D, g0_train, g1_train, g0_valid, g1_valid,
             num_epoch = 1000, lr = 0.1, k = 1 ):
    """ Training """

    """Results: 
            'Train_Loss', 
            'Valid_Loss', 
            'AUC'   
       saved as .csv file
    '"""
    
    ## log file
    net_D.set_log(log_fname = PARS['log_path'] + '_'.join(['loss', str(k)]) + '.csv',
                  cols      = ('Epoch', 'Train_loss', 'Valid_loss', 'AUC'))
    
    D_optimizer = net_D.get_optimizer(lr = lr)
    #scheduler   = ReduceLROnPlateau(D_optimizer, 'min', factor=0.5, patience=10, verbose=True)
    
    D_loss      = net_D.get_loss().cuda()

    # initialize the early_stopping object
    #early_stopping = EarlyStopping(patience=10, verbose=True)

    for epoch in range(num_epoch):
        
        t_ls = gt.update_D(net_D, 
                           g0_train, g1_train, 
                           D_optimizer, 
                           f_loss =D_loss)

        v_ls = gt.infer_D(net_D, 
                          g0_valid, g1_valid, 
                          f_loss = D_loss)
        
        if (epoch+1) % 100 == 0:
            print('Epoch: [{}/{}]\tTRAIN_LOSS: {}\tVALID_LOSS: {}'.format( epoch+1, num_epoch, t_ls, v_ls))

        if (epoch+1) % 10  == 0:
            auc_scores = gt.auc_scores_D(net_D, g0_valid, g1_valid)
            net_D.write_log((epoch+1, t_ls, v_ls, auc_scores['auc']))
        
        #scheduler.step(v_ls)
        #early_stopping(v_ls, net_D)
        #if early_stopping.early_stop:
        #    print("Early stopping")
        #    break

    print('----------------------------------')
    print('Epoch: [{}/{}]\t**AUC_SCORE: {}'.format( epoch+1, num_epoch, auc_scores['auc']))
    net_D.print_pars()
    print('------------------------------------------------------------------')

    torch.save(net_D.state_dict(), 
               PARS['pre_path'] + '_'.join(['checkpoint_k', str(k), '.pth']))

    return auc_scores


def GD_KFold(net_D, simu_data, num_epoch, lr =0.1, n_batch=20, k_fold =5 ):
    """ k_fold cross validation"""

    data_g0, data_g1 = simu_data.get_group(0), simu_data.get_group(1)

    kf = KFold(n_splits=k_fold)
    
    auc_obj = {}
    for k, (train_id_0, valid_id_0), (train_id_1, valid_id_1) in zip(range(k_fold), kf.split(data_g0), kf.split(data_g1)):

        print('K_FOLD:[{}]************************************\t' .format(k))
        
        data = simu_data.get_loader_kf(train_id_0, train_id_1,
                                       valid_id_0, valid_id_1, 
                                       n_batch=n_batch)
        
        auc_scores = GD_Train(net_D, 
                              *data, 
                              num_epoch = num_epoch, 
                              lr = lr, k= k)

        auc_obj[k] = auc_scores

    torch.save(auc_obj, PARS['rst_path'] + '_'.join(['auc', str(k_fold)]) + '.obj')

    return auc_obj

    
def main(num_epoch = 1000, 
         n_batch   = 20, 
         k_fold    = 5,
         pars      = ['rand', 100, 3000],
         cov_grep  = "^x[0-9]+$",
         prefix    = "../Model_Data/simu_",
         grp       = "label"):

    """main function for using neural network to estimate PS"""
    global PARS
    PARS = gt.read_args()

    ## read data
    simu_data = gt.DG_LoadData(pars     = pars, 
                               prefix   = prefix, 
                               cov_grep = cov_grep,
                               grp      = grp,  )
    
    ## define model
    net_D = gt.DG_Net(simu_data.nx, [], is_d = True).cuda()
    
    ## initialize parameters
    net_D.init_pars()

    ## k_fold cross validation
    auc_obj = GD_KFold(net_D, simu_data, 
                       num_epoch = num_epoch, 
                       lr = 0.1, n_batch=20, k_fold =5 )
    

if __name__ == '__main__':
    starttime = time.time()

    main()

    endtime = time.time()
    dtime   = endtime - starttime
    print("Program Execution Time：%.8s s" % dtime)


