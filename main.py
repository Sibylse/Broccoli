import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import collections

cuda = torch.cuda.is_available()
dev = torch.device("cuda") if cuda else torch.device("cpu")


#
# prox operators
#
def phi(x):
    return 1-torch.abs(1-2*x)
def prox_binary_(x,lambdas,lr,alpha=-1e-8):
  with torch.no_grad():
    idx_up = x>0.5
    idx_down = x<=0.5
    x[idx_up] += 2*lr*lambdas[idx_up]
    x[idx_down] -= 2*lr*lambdas[idx_down]
    x[x>1] = 1
    x[x<0] = 0
    lambdas.add_(phi(x)-1,alpha=alpha)
def prox_pos_(X,weight,lr,alpha=0):
  with torch.no_grad():
    X[X<0] = 0
#
# train batch
#
def train(epoch,alpha):
  model.train()
  model_prev.train()
  cum_loss = 0.
  for batch_idx, (data, target) in enumerate(train_loader):
      lr_mean, lambda_mean =0,0
      if cuda:
        data, target = data.cuda(), target.cuda()
      data, target = Variable(data), Variable(target.double())
      
      for group in param_list:
        optimizer = group["optimizer"]
        optParam = optimizer.param_groups[0]
        stepsize = group["step"]
        optimizer.zero_grad()
        output = model(data)
        output_prev = model_prev(data)
        loss = loss_func(output, target)
        loss_prev = loss_func(output_prev, target)
        loss.backward()
        loss_prev.backward()
        #print("grad:",optParam['params'][0].grad)
        #print("grad nonzero:",(optParam['params'][0].grad !=0)*reg_weight*optParam['lr'])
        optParam['lr'] =stepsize(model,data)#/2
        lr_mean += optParam['lr']
        optimizer.step()
        if "prox" in group:
            prox = group["prox"]
            #The whole factor matrix is proxed for one batch? 
            #Most of the time, this is ok because not often there is no tupel for one row/column.
            prox(optParam['params'][0].data,group["lambda"],optParam['lr'], alpha=alpha) 
            lambda_mean += torch.mean(group["lambda"])
        cum_loss+=loss.item()/3
        if batch_idx % 2 +1 == 0:
            print('Train Epoch:\t\t\t {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader),
                100. * batch_idx / len(train_loader), loss.item()))
  if epoch % 3 == 0:
    print('==Train Epoch:\t\t\t {} \tLoss: {:.6f}\t lambda: {:.3e}\t lr: {:.3f}'.format(
        epoch, cum_loss/len(train_loader),lambda_mean/2,lr_mean/3))

#
# Train full grad
#
def train_full_grad(epoch,alpha):
  model.train()
  cum_loss = 0.
  lambda_mean, lr_mean=0,0
  #for every factor matrix - optimizer
  for group in param_list: 
      optimizer = group["optimizer"]
      optimizer.grad_buff =None #Sign that we do a full grad update
      optimizer.zero_grad()
      optParam = optimizer.param_groups[0]
      stepsize = group["step"]
      for batch_idx, (data, target) in enumerate(train_loader):
          if cuda:
            data, target = data.cuda(), target.cuda()
          data, target = Variable(data), Variable(target.double())
            
          output = model(data)
          # loss is mean squared error over a batch 
          loss = loss_func_full(output, target)/n/m
          loss.backward()
          cum_loss+=loss.item()
      # gamma = 1/(2L), L is normalized with bs but this is a full grad update  
      optParam['lr'] =stepsize(model,None)#/2 #TODO stepsize is computed for one batch! 
      lr_mean+= optParam['lr']
      optimizer.step()
      if "prox" in group:
          prox = group["prox"]
          prox(optParam['params'][0].data,group["lambda"],optParam['lr'], alpha=alpha) 
          lambda_mean += torch.mean(group["lambda"])
      if batch_idx % 2 +1 == 0:
          print('Train Full Grad Batch:\t {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader),
                100. * batch_idx / len(train_loader), loss.item()))
  if epoch % 3 == 0:
    print('==Train Full Grad Epoch:\t {} \tLoss: {:.6f}\t lambda: {:.3e}\t lr: {:.3f}'.format(
        epoch, cum_loss/3,lambda_mean/2,lr_mean/3))
#
# Test
#
def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        output = model(data)
        # sum up batch loss
        test_loss += loss_func(output, target).item()

    test_loss /= len(test_loader)
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))


#
# Do the stuff
#
model = MatrixFactorization(m, n, r=r)
model.to(dev)
model_prev = MatrixFactorization(m, n, r=r)
model_prev.to(dev)
optimizerY = SARAH([model.Y.weight],[model_prev.Y.weight], lr=0.1) # learning rate
optimizerX = SARAH([model.X.weight],[model_prev.X.weight], lr=0.1) # learning rate
optimizerC = SARAH([model.C.weight],[model_prev.C.weight], lr=0.1)
lambdasY = torch.zeros_like(model.Y.weight) #TODO is prev_model also proxed?
lambdasX = torch.zeros_like(model.X.weight)
param_list = [{'optimizer': optimizerX, 'step': stepsizeX, 'prox':prox_binary_, 'lambda':lambdasX},
              {'optimizer': optimizerY, 'step': stepsizeY, 'prox':prox_binary_, 'lambda':lambdasY},
              {'optimizer': optimizerC, 'step': stepsizeC, 'prox':prox_pos_, 'lambda':torch.tensor([0.0])}]
loss_func = torch.nn.MSELoss()
loss_func_full = torch.nn.MSELoss(reduction='sum')
epoch = 1
test()
alpha=-1e-8
#alpha = -1/n/m/len(train_loader)/10
thresh=0.1
full_grad_prob=0.3
#while not is_binary(model.Y.fact.weight.data) or not is_binary(model.X.fact.weight.data):
while not is_binary(model.Y.weight.data) or not is_binary(model.X.weight.data):
    full_batch_grad = np.random.binomial(1,full_grad_prob)
    if full_batch_grad:
      train_full_grad(epoch,alpha)
    else:
      train(epoch,alpha)
    if epoch % 6 == 0:
      #phiX, phiY = torch.mean(phi(model.X.fact.weight.data)), torch.mean(phi(model.Y.fact.weight.data))
      phiX, phiY = torch.mean(phi(model.X.weight.data)), torch.mean(phi(model.Y.weight.data))
      print('--\t\t\tphi(X):\t {:.3f} \tphi(Y): {:.3f}'.format(phiX,phiY))
      if max(phiX,phiY) < thresh:
        alpha*=10
        thresh/=2
    epoch+=1
    if epoch % 50 == 0:
      test()

#
# Write out
#
