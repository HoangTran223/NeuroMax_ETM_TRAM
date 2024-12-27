import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import random
from scipy.optimize import minimize

class IMTL:
    def __init__(self, task_num, device='cuda'):
        self.task_num = task_num
        self.loss_scale = nn.Parameter(torch.zeros(task_num, device=device))
    
    def apply(self, components):
        task_num = len(components)
        grads = torch.stack(components) 
        
        scaled_grads = self.loss_scale.exp().unsqueeze(1) * grads 
        scaled_grads = scaled_grads - self.loss_scale.unsqueeze(1)  
        
        grads_unit = grads / (grads.norm(p=2, dim=1, keepdim=True) + 1e-8)  
        
        D = grads[0:1].repeat(self.task_num - 1, 1) - grads[1:] 
        U = grads_unit[0:1].repeat(self.task_num - 1, 1) - grads_unit[1:] 
        
        D_U = torch.matmul(D, U.t())  # shape: (task_num-1, task_num-1)
        try:
            D_U_inv = torch.inverse(D_U)  # shape: (task_num-1, task_num-1)
        except RuntimeError:
            D_U_inv = torch.pinverse(D_U)
        
        alpha_partial = torch.matmul(torch.matmul(grads[0], U.t()), D_U_inv)  # shape: (task_num-1,)
        alpha = torch.cat((1 - alpha_partial.sum().unsqueeze(0), alpha_partial), dim=0)  # shape: (task_num,)
        
        alpha = torch.clamp(alpha, min=0.0)
        
        adjusted_grad = torch.matmul(alpha, grads)  # shape: (param_size,)
        
        batch_weight = alpha.detach().cpu().numpy()
        
        return adjusted_grad, batch_weight
    
