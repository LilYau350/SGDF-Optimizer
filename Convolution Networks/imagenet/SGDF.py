import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer

class SGDF(Optimizer):
    def __init__(self, params, lr=0.5, betas=(0.9, 0.999), eps=1e-8, gamma=0.5, weight_decay=0.0, 
                                weight_decouple=False,  use_sign=False):
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if betas[0] < 0.0 or betas[1] < 0.0:
            raise ValueError('Invalid beta value: {}'.format(betas))   
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 < gamma:
            raise ValueError("Invalid gamma value: {}".format(gamma))

        defaults = dict(lr=lr, betas=betas, eps=eps, gamma=gamma, weight_decay=weight_decay, 
                        weight_decouple=weight_decouple, use_sign=use_sign)
        super(SGDF, self).__init__(params, defaults)
        
    @torch.no_grad()    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            eps = group['eps']
            gamma = group['gamma']
            beta1, beta2 = group['betas']
            weight_decay = group['weight_decay']
            weight_decouple = group['weight_decouple'] 
            use_sign = group['use_sign']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.detach() 
                # grad = p.grad.data
                if weight_decay != 0.0:
                    # Apply weight decay, optional decoupling
                    if weight_decouple:
                        p.mul_(1 - lr * weight_decay)          # Decoupled weight decay
                    else:
                        grad.add_(p, alpha=weight_decay)       # Regular weight decay

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(grad)
                    state['exp_var'] = torch.zeros_like(grad)

                exp_avg = state['exp_avg']
                exp_var = state['exp_var']

                # Compute gradient 1st and 2nd 
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                grad_residual = (grad - exp_avg)
                
                exp_var.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2)

                state['step'] += 1

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = (1 + beta1) * (1 - beta2**state['step']) / ((1 - beta1) * (1 - beta1**(2*state['step'])))
                
                exp_avg_corr = exp_avg / bias_correction1
                # exp_var_corr is NOT computed explicitly to save memory and stability
                
                # Residual for update
                grad_hat_residual = grad - exp_avg_corr
                
                # Estimation gain
                denom = grad_hat_residual.pow(2).add_(eps).mul_(bias_correction2).add_(exp_var)
                K = exp_var / denom
                
                # denom <- K^gamma
                if gamma == 1.0:
                    pass
                elif gamma == 0.5:
                    K.sqrt_()
                else:
                    K.pow_(gamma)
                    
                # grad_hat_residual = grad - exp_avg_corr
                grad_hat = exp_avg_corr + K * grad_hat_residual
                
                if use_sign:
                    p.add_(grad_hat.sign_(), alpha=-lr)
                else:
                    p.add_(grad_hat, alpha=-lr)
                                
        return loss