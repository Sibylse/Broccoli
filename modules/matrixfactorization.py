class MatrixFactorization(torch.nn.Module):
    
    def __init__(self, m, n, r=20):
        super().__init__()
        self.Y = torch.nn.Embedding(m, r).double()
        self.X = torch.nn.Embedding(n, r).double()
        self.C = torch.nn.Embedding(r, r).double()
        torch.nn.init.uniform_(self.Y.weight)
        torch.nn.init.uniform_(self.X.weight)
        torch.nn.init.uniform_(self.C.weight)
        
    def forward(self, idx):
      #j and i are torch tensors, denoting a set of indices i and j
      YC =torch.matmul(self.Y(idx[:,0]),self.C.weight) 
      YCX_batch = (YC * self.X(idx[:,1])).sum(1,keepdim=True)
      return YCX_batch.squeeze() #reduces every 1x dimension for tensor
