from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.autograd import Variable
import collections
import itertools, random
import os,sys
import argparse
from modules import *
torch.set_printoptions(sci_mode=False,precision=2,linewidth=300)

cuda = torch.cuda.is_available()
dev = torch.device("cuda") if cuda else torch.device("cpu")

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--data',  help='data file directory')
parser.add_argument('--r', default=3, type=int, help='rank - number of clusters')
parser.add_argument('--lam_C_perc', type=float, default=99.9, help='L2 regularization weight for C')
parser.add_argument('--run', default='_', help='run specification, the results are saved in the path results_run/...')
parser.add_argument('--res_path', default='_', help='path to the folder where the result is to be stored')
args = parser.parse_args()

D = np.loadtxt(open(args.data, "rb"), delimiter=" ")
m,n = D.shape
r = args.r

i_loader = DataLoader(TensorDataset(torch.arange(0,n)),batch_size=int(n/10),shuffle=True)
j_loader = DataLoader(TensorDataset(torch.arange(0,m)),batch_size=int(m/10),shuffle=True)

#
# Helpers
#
def select_batch_J(J,I):
  return J, None
def select_batch_I(J,I):
  return None, I
def select_batch_none(J,I):
  return None, None
def get_target(J,I):
  if I is None and J is None:
    return D_full
  elif J is None:
    return D_full[:,I]
  else:
    return D_full[J,:]

#
# the binary penalizing function phi
#
def phi(x):
    if x.max() >1 or x.min() <0:
      raise Exception("Someone did not prox properly")
    return 1-torch.abs(1-2*x)

#
# train batch-wise
#
def train(epoch):
  batches = itertools.zip_longest(j_loader,i_loader)
  for batch_idx, (J,I) in enumerate(batches):
      J,I =J[0],I[0]
      lr_mean = 0
      for group in param_list:
        J_, I_ = group["batch"](J,I)
        optimizer = group["optimizer"]
        optParam = optimizer.param_groups[0]
        optimizer.zero_grad()
        target = get_target(J_,I_)
        output = model(J_,I_)
        loss = loss_func(output, target)
        loss.backward()
        L = max(group["L"](J_,I_), min_L )
        optParam['lr'] =1/L/4
        lr_mean += optParam['lr']
        optimizer.step()
        if "prox" in group:
            prox = group["prox"]
            prox(optParam['lr'], None, None)
  if epoch % 50 == 0:
    with torch.no_grad():
      mse = loss_func(D_full,model(None,None))
      print('==Train Epoch:\t\t\t {} \tMSE%: {:.2f}\t lambda: {:.3e}\t lr: {:.3f}'.format(
        epoch, mse/mse0*100,(torch.mean(model.lambdasX)+torch.mean(model.lambdasY))/2,lr_mean/4))
      return loss
  return 0.


#
# Write out
#
def writeMF():
    if not os.path.exists(args.res_path):
        os.makedirs(args.res_path)
    Y_alg = model.Y.weight.detach().cpu().numpy()
    C_alg = model.C.weight.detach().cpu().numpy()
    X_alg = model.X.weight.detach().cpu().numpy()
    keep_idx_t= X_alg.sum(0)>0
    keep_idx_s= Y_alg.sum(0)>0
    keep_idx_t= keep_idx_t*C_alg.sum(0)>0
    keep_idx_s= keep_idx_s*C_alg.sum(1)>0
    X_alg = X_alg[:,keep_idx_t]
    C_alg = C_alg[keep_idx_s,:][:,keep_idx_t]
    Y_alg = Y_alg[:,keep_idx_s]
    np.savetxt(args.res_path + 'Y.csv',Y_alg,delimiter=',',fmt='%.0f')
    np.savetxt(args.res_path + 'X.csv',X_alg,delimiter=',',fmt='%.0f')
    np.savetxt(args.res_path + 'C.csv',C_alg,delimiter=',',fmt='%.5f')


#
# Do the stuff
#
alpha = -1e-9
lam_C = 1/r**2/np.percentile(D[D>0],args.lam_C_perc)
model = MatrixFactorization(m, n, r=r, alphaX=alpha, alphaY=alpha, max_C=np.max(D), lam_C=lam_C)
model.to(dev)
model.train()

D_full = torch.from_numpy(D).float()
D_full = D_full.to(dev)
mse0 =(D_full**2).mean()
print('The mse  of the zero model is {:.3f}'.format(mse0.item())) 

optimizerY = torch.optim.SGD([model.Y.weight], lr=0.1) # learning rate
optimizerX = torch.optim.SGD([model.X.weight], lr=0.1) # learning rate
optimizerC = torch.optim.SGD([model.C.weight], lr=0.1, weight_decay = lam_C)
param_list = [{'optimizer': optimizerC, 'L': model.L_C, 'prox':model.prox_pos_C,    'batch' : select_batch_J},
              {'optimizer': optimizerX, 'L': model.L_X, 'prox':model.prox_binary_X, 'batch' : select_batch_J},
              {'optimizer': optimizerC, 'L': model.L_C, 'prox':model.prox_pos_C,    'batch' : select_batch_I},
              {'optimizer': optimizerY, 'L': model.L_Y, 'prox':model.prox_binary_Y, 'batch' : select_batch_I}
              ]
min_L = 1e-3
loss_func = torch.nn.MSELoss()
epoch = 1
phiX=phiY=1
loss, best_loss = 0, 10000000
candidate=0
while phiX+phiY or candidate<5:
    train(epoch)
    if epoch % 200 == 0 or epoch==100:
      with torch.no_grad():
        phiX, phiY = torch.mean(phi(model.X.weight)), torch.mean(phi(model.Y.weight))
        print('--\t\t\tphi(X):\t {:.3f} \tphi(Y) {:.3f}\n||X||: {} \n||Y||: {} \n C: {}'.format(phiX,phiY,(model.X.weight**2).sum(0),(model.Y.weight**2).sum(0),model.C.weight.data))
        if not phiX+phiY:
            candidate+=1
            loss=loss_func(D_full,model(None,None))
            print("CANDIDATE")
            if loss<best_loss:
                best_loss = loss
                writeMF()
                print("WRITE (the loss is {:.5f})".format(loss))
    if epoch % 400 == 0:
        sys.stdout.flush()

    if epoch >=2000 and epoch%500==0:
        model.alphaX*=2
        model.alphaY*=2
        #with torch.no_grad():
        #    model.X.weight.round_()
        #    model.Y.weight.round_()
        #    param_list = [{'optimizer': optimizerC, 'step': model.stepsizeC, 'prox':model.prox_pos_C,    'batch' : select_batch_J},
        #                  {'optimizer': optimizerC, 'step': model.stepsizeC, 'prox':model.prox_pos_C,    'batch' : select_batch_I}]
    epoch+=1


