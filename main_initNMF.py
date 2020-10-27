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

cuda = torch.cuda.is_available()
dev = torch.device("cuda") if cuda else torch.device("cpu")

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--data',  help='data file directory')
parser.add_argument('--X',  help='directory to ground truth X')
parser.add_argument('--C',  help='directory to ground truth C')
parser.add_argument('--Y',  help='directory to ground truth Y')
parser.add_argument('--r', default=3, type=int, help='rank - number of clusters')
parser.add_argument('--run', default='_', help='run specification, the results are saved in the path results_run/...')
args = parser.parse_args()

D = np.loadtxt(open(args.data, "rb"), delimiter=",")
Y = np.loadtxt(open(args.Y, "rb"), delimiter=",")
C = np.loadtxt(open(args.C, "rb"), delimiter=",")
X = np.loadtxt(open(args.X, "rb"), delimiter=",")
m,n = D.shape
r = args.r

print('The loss of the ground truth is {:.5f}'.format(np.mean((D-Y@C@X.T)**2)))
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
        stepsize = group["step"]
        optimizer.zero_grad()
        target = get_target(J_,I_)
        output = model(J_,I_)
        loss = loss_func(output, target)
        loss.backward()
        optParam['lr'] =stepsize(J_,I_)
        lr_mean += optParam['lr']
        optimizer.step()
        if "prox" in group:
            prox = group["prox"]
            prox(optParam['lr'], None, None)
  if epoch % 50 == 0:
    with torch.no_grad():
      loss = loss_func(D_full,model(None,None))
      print('==Train Epoch:\t\t\t {} \tLoss: {:.6f}\t lambda: {:.3e}\t lr: {:.3f}'.format(
        epoch, loss,(torch.mean(model.lambdasX)+torch.mean(model.lambdasY))/2,lr_mean/4))
      return loss
  return 0.


#
# Test
#
def test():
    with torch.no_grad():
      print('\nTest set: Average loss: {:.4f}\n'.format(loss_func(model(None,None),D_true).item()))

#
# Write out
#
def writeMF():
    res_path = 'results/'+args.Y.split('/',1)[1].rsplit('/',1)[0]+'/'+args.run+'/'
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    np.savetxt(res_path + args.Y.rsplit('/',1)[1],model.Y.weight.data.cpu().numpy(),delimiter=',',fmt='%.0f')
    np.savetxt(res_path + args.X.rsplit('/',1)[1],model.X.weight.data.cpu().numpy(),delimiter=',',fmt='%.0f')
    np.savetxt(res_path + args.C.rsplit('/',1)[1],model.C.weight.data.cpu().numpy(),delimiter=',',fmt='%.5f')


#
# Do the stuff
#
alphaY = -1e-9
alphaX = -1e-9
model = MatrixFactorization(m, n, r=r, alphaX=alphaX, alphaY=alphaY, max_C=np.percentile(D,99))
model.to(dev)
model.train()

D_full = torch.from_numpy(D).float()
D_full = D_full.to(dev)
D_true = torch.from_numpy(Y@C@X.T).float()
D_true = D_true.to(dev)

optimizerY = torch.optim.SGD([model.Y.weight], lr=0.1) # learning rate
optimizerX = torch.optim.SGD([model.X.weight], lr=0.1) # learning rate
optimizerC = torch.optim.SGD([model.C.weight], lr=0.1)
param_list = [{'optimizer': optimizerX, 'step': model.stepsizeX, 'prox':model.prox_pos_X, 'batch' : select_batch_J},
              {'optimizer': optimizerY, 'step': model.stepsizeY, 'prox':model.prox_pos_Y, 'batch' : select_batch_I}
              ]
loss_func = torch.nn.MSELoss()
epoch = 1
test()
phiX=phiY=1
loss, best_loss = 0, 10000000
candidate=0
while phiX+phiY or candidate<5:
    train(epoch)
    if epoch == 100: # start binary penalizer
        param_list = [{'optimizer': optimizerC, 'step': model.stepsizeC, 'prox':model.prox_pos_C,    'batch' : select_batch_J},
              {'optimizer': optimizerX, 'step': model.stepsizeX, 'prox':model.prox_binary_X, 'batch' : select_batch_J},
              {'optimizer': optimizerC, 'step': model.stepsizeC, 'prox':model.prox_pos_C,    'batch' : select_batch_I},
              {'optimizer': optimizerY, 'step': model.stepsizeY, 'prox':model.prox_binary_Y, 'batch' : select_batch_I}
              ]
        with torch.no_grad():
          diag_X = torch.from_numpy(np.percentile(model.X.weight.detach().cpu(),80,axis=0)).float().to(dev)
          diag_Y = torch.from_numpy(np.percentile(model.Y.weight.detach().cpu(),80,axis=0)).float().to(dev)
          diag_X[diag_X==0] = 0.01
          diag_Y[diag_Y==0] = 0.01
          model.X.weight.data = model.X.weight.data.matmul(torch.diag(diag_X**(-1)))
          model.Y.weight.data = model.Y.weight.data.matmul(torch.diag(diag_Y**(-1)))
          model.C.weight.mul_(torch.diag(diag_X*diag_Y))
          model.C.weight.clamp_(max=model.max_C/10.)
          model.X.weight.clamp_(max=1)
          model.Y.weight.clamp_(max=1)
    if epoch % 200 == 0 or epoch==100:
      with torch.no_grad():
        phiX, phiY = torch.mean(phi(model.X.weight)), torch.mean(phi(model.Y.weight))
        print('--\t\t\tphi(X):\t {:.3f} \tphi(Y): {:.3f}'.format(phiX,phiY))
        if not phiX+phiY:
            candidate+=1
            loss=loss_func(D_full,model(None,None))
            print("CANDIDATE")
            if loss<best_loss:
                best_loss = loss
                writeMF()
                print("WRITE (the loss is {:.5f})".format(loss))
                test()
    if epoch % 400 == 0:
        test()
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


