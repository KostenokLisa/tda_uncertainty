import torch
import torch.nn as nn


class ConfidenceLoss:
    def __init__(self, scaling_factor=0.3):
        self.scaling_factor=scaling_factor
    
    def __call__(self, p_batch, y_batch, c_batch):
      '''
      p_batch is softmax output for the BERT classifier
      y_batch is target probalitity distribution: 1 for the true class, 0 otherwise
      c_batch is predicted confidence score
      '''
      p_biased = c_batch * p_batch + (1 - c_batch) * y_batch
      # task loss
      L_t = -torch.sum(torch.log(p_biased) * y_batch)
      # regularization loss
      L_c = -torch.sum(torch.log(c_batch))
      L = L_t + self.scaling_factor * L_c
      return L


class TemperatureLoss:
    def __call__(self, logits_batch, y_batch, c_batch):
        norm_term = torch.sum(torch.exp(c_batch * logits_batch), axis=1, keepdim=True)
        L = -torch.mean((c_batch * logits_batch - torch.log(norm_term)) * y_batch) 
        return L
