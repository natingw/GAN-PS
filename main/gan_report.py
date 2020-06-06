import matplotlib
matplotlib.use('Agg')

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

import torch 
import pdb

def auc_curve(auc_obj, colors):

    pdf = PdfPages('roc_figure.pdf')    
    plt.figure(figsize=(7,5.5))

    lw = 1
    for k in auc_obj:
        roc_auc = auc_obj[k]['auc']
        fpr     = auc_obj[k]['fpr']
        tpr     = auc_obj[k]['tpr']
        plt.plot(fpr, tpr, color=colors[k], lw=lw, label='ROC fold{0:n} (area = {1:.2f})'.format(k+1, roc_auc)) 
    
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.])
    plt.ylim([0.0, 1.05])

    #plt.plot([0, 0, 1], [0, 1, 1], linestyle=':', color='black', label='Perfect performance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC CURVE')
    plt.legend(loc="lower right")

    pdf.savefig() 
    plt.close()
    pdf.close()

def main():
    auc_obj = torch.load("auc_5_.obj")
    colors = ['red', 'blue', 'darkgreen', 'darkorange', 'purple']
    auc_curve(auc_obj, colors)


if __name__ == '__main__':
    main()

