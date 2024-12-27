import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import random
from scipy.optimize import minimize

class C1:
    def __init__(self, model, device, buffer_size=3):
        self.model = model
        self.device = device
        self.buffer_size = buffer_size
        self.grad_buffer = []  
        self.grad_dim = self._get_grad_dim()  
    
    def _get_grad_dim(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def _get_total_grad(self, total_loss):
        self.model.zero_grad()
        total_loss.backward(retain_graph=True)
        total_grad_list = []
        for p in self.model.parameters():
            if p.requires_grad:
                if p.grad is not None:
                    total_grad_list.append(p.grad.flatten())
                else:
                    # Append zeros if p.grad is None
                    total_grad_list.append(torch.zeros_like(p).flatten())
        total_grad = torch.cat(total_grad_list)
        return total_grad.detach()
    