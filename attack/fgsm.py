import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"


class FGSM:
    def __init__(self, epsilon, data_min, data_max):
        self.epsilon = epsilon
        self.criterion = nn.CrossEntropyLoss()
        self.data_min = data_min
        self.data_max = data_max

    def __call__(self, model, data, target, num_steps=50, alpha=0.001):
        dt = data.clone().detach().requires_grad_(True)
        for _ in range(num_steps):
            output = model(dt)
            model.zero_grad()
            loss = self.criterion(output, target)
            loss.backward()
            data_grad_sign = dt.grad.data.sign()
            adv_data = dt + alpha * data_grad_sign
            total_grad = adv_data - data
            total_grad = torch.clamp(
                total_grad, -self.epsilon/255, self.epsilon/255)
            dt.data = torch.clamp(
                data + total_grad, self.data_min, self.data_max)
        adv_pred = model(dt).argmax(-1)
        success = adv_pred != target
        return dt, success, adv_pred
