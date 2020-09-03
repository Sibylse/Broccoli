import torch
from torch.optim.optimizer import Optimizer,required

class SARAH(Optimizer):
    r"""Implements SARAH
    """
    def __init__(self, params, params_prev, lr=required, weight_decay=0):

        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(SARAH, self).__init__(params, defaults)
        self.grad_buff = None
        self.params_prev = params_prev

    def __setstate__(self, state):
        super(SARAH, self).__setstate__(state)
    
    def zero_grad(self):
        super(SARAH, self).zero_grad()
        for p in self.params_prev:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        """
        loss = None

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            for x, prev_x in zip(group['params'],self.params_prev):
                if x.grad is None:
                    continue
                if weight_decay != 0:
                    x.grad.add_(x, alpha=weight_decay)
                # x.grad gradient has same shape like the parameter but 
                # zeros at the positions which are not updated.
                # grad_p has same size as batch times r 
                #param_state = self.state[x]
                if self.grad_buff is None: #do full gradient update
                    self.grad_buff = torch.clone(x.grad).detach() 
                else:
                    self.grad_buff.add_(x.grad, alpha=1)
                    self.grad_buff.add_(prev_x.grad, alpha=-1) #g_t = g_t-1  +  grad_t - grad_t-1
                prev_x.mul_(0).add_(torch.clone(x),alpha=1)  
                x.add_(x.grad, alpha=-group['lr'])

        return loss
