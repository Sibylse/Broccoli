import torch

class MatrixFactorization(torch.nn.Module):

    def __init__(self, m, n, r=20, alphaX =-1e-8, alphaY=-1e-8, max_C=100):
        super().__init__()
        self.X = torch.nn.Embedding(n, r)
        self.Y = torch.nn.Embedding(m, r)
        self.C = torch.nn.Embedding(r, r)
        torch.nn.init.uniform_(self.Y.weight)
        torch.nn.init.uniform_(self.X.weight)
        torch.nn.init.constant_(self.C.weight,0)
        with torch.no_grad():
          self.X.weight.data = self.X.weight.softmax(1)
          self.Y.weight.data = self.Y.weight.softmax(1)
        self.register_buffer("lambdasY",torch.zeros_like(self.Y.weight))
        self.register_buffer("lambdasX",torch.zeros_like(self.X.weight))
        self.alphaX = alphaX
        self.alphaY = alphaY
        self.max_C = max_C
        self.n=n
        self.m=m

    def forward(self, J,I):
        if J is None and I is None:
            return self.Y.weight@self.C.weight@torch.transpose(self.X.weight,0,1)
        elif J is not None:
            return self.Y.weight[J,:]@self.C.weight@torch.transpose(self.X.weight,0,1)
        else:
            return self.Y.weight@self.C.weight@torch.transpose(self.X.weight[I,:],0,1)

    def phi(self,x):
      if x.max() >1 or x.min() <0:
        raise Exception("Someone did not prox properly")
      return 1-torch.abs(1-2*x)

    def stepsizeX(self,J,I):
      with torch.no_grad():
        if J is None:
          YC= self.Y.weight@self.C.weight
          L = 2*torch.sqrt(((torch.transpose(YC,0,1)@YC)**2).sum())/self.n/self.m
        else:
          YC = self.Y.weight[J,:]@self.C.weight
          L= 2*torch.sqrt((torch.transpose(YC,0,1)@YC)**2).sum()/J.shape[0]/self.n
      return 1/4/max(L,0.001)

    def stepsizeY(self,J,I):
      with torch.no_grad():
        if I is None:
          XCt = self.X.weight@torch.transpose(self.C.weight,0,1)
          L = 2*torch.sqrt(((torch.transpose(XCt,0,1)@XCt)**2).sum())/self.n/self.m
        else:
          XCt = self.X.weight[I,:]@torch.transpose(self.C.weight,0,1)
          L= 2*torch.sqrt(((torch.transpose(XCt,0,1)@XCt)**2).sum())/self.m/I.shape[0]
        return 1/4/max(L,0.001)

    def stepsizeC(self,J,I):
      with torch.no_grad():
        if J is None and I is None:
          L = 2*torch.sqrt((self.X.weight**2).sum()*(self.Y.weight**2).sum())/self.n/self.m
        elif I is None:
          L = 2*torch.sqrt((self.X.weight**2).sum()*(self.Y.weight[J,:]**2)).sum()/self.m/I.shape[0]
        else:
          L = 2*torch.sqrt((self.X.weight[I,:]**2).sum()*(self.Y.weight**2)).sum()/J.shape[0]/self.n
        return 1/16/max(L,0.001)

    def prox_binary(self, A, lambdas, lr, alpha):
      with torch.no_grad():
        idx_up = (A>0.5).float()
        A.add_(2*lambdas*idx_up-lambdas, alpha=2*lr)
        A.clamp_(0,1)
        return A

    def prox_binary_(self, A, lambdas, lr, alpha):
      with torch.no_grad():
        idx_up = (A>0.5).float()
        A.add_(2*lambdas*idx_up-lambdas, alpha=2*lr)
        A.clamp_(0,1)
        lambdas.add_(self.phi(A)-1,alpha=alpha)

    def prox_binary_X(self, lr, J,I):
      with torch.no_grad():
        if J is None:
          self.prox_binary_(self.X.weight,self.lambdasX,lr,self.alphaX)
        else:
          self.X.weight[I] = self.prox_binary(self.X.weight[I],self.lambdasX[I],lr,self.alphaX)
          self.lambdasX[I] = self.lambdasX[I].add(self.phi(self.X.weight[I])-1,alpha=self.alphaX)

    def prox_binary_Y(self, lr, J,I):
      with torch.no_grad():
        if J is None:
          self.prox_binary_(self.Y.weight,self.lambdasY,lr,self.alphaY)
        else:
          self.Y.weight[J] = self.prox_binary(self.Y.weight[J],self.lambdasY[J],lr,self.alphaX)
          self.lambdasY[J] = self.lambdasY[J].add(self.phi(self.Y.weight[J])-1,alpha=self.alphaY)

    def prox_pos_C(self, lr, J,I):
      with torch.no_grad():
        self.C.weight.clamp_(0,self.max_C)
