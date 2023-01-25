import torch.nn as nn
import torch
import torch.nn.functional as F

class FGSMGrad:
    def __init__(self, epsilon, data_min, data_max):
        self.epsilon = epsilon
        self.criterion = nn.CrossEntropyLoss()
        self.data_min = data_min
        self.data_max = data_max
        
    def __call__(self, model, data, target, num_steps=50, alpha=0.001, early_stop=True,use_sign=False,use_softmax=False):
        dt = data.clone().detach().requires_grad_(True)
        hats = [data.clone()]
        grads = list()
        for _ in range(num_steps):
            output = model(dt)
            model.zero_grad()
            loss = self.criterion(output, target)
            loss.backward()
            if use_softmax:
                tgt_out = F.softmax(output,dim=-1)[:, target]
            else:
                tgt_out = output[:, target]
            grad = torch.autograd.grad(tgt_out, dt)[0]
            grads.append(grad.clone())
            if use_sign:
                data_grad = dt.grad.detach().sign()
                adv_data = dt + alpha * data_grad
                total_grad = adv_data - data
                total_grad = torch.clamp(
                    total_grad, -self.epsilon/255, self.epsilon/255)
                dt.data = torch.clamp(
                    data + total_grad, self.data_min, self.data_max)
                hats.append(dt.data.clone())    
            else:
                data_grad = grad / grad.norm()
                adv_data = dt - alpha * data_grad * 100
                dt.data = torch.clamp(
                    adv_data, self.data_min, self.data_max)
                hats.append(dt.data.clone())
            if early_stop:
                adv_pred = model(dt).argmax(-1)
                if adv_pred != target:
                    break
        adv_pred = model(dt)
        model.zero_grad()
        loss = self.criterion(adv_pred, target)
        loss.backward(retain_graph=True)
        if use_softmax:
            tgt_out = F.softmax(adv_pred,dim=-1)[:, target]
        else:
            tgt_out = adv_pred[:, target]
        grad = torch.autograd.grad(tgt_out, dt)[0]
        grads.append(grad.clone())
        hats = torch.cat(hats, dim=0)
        grads = torch.cat(grads, dim=0)
        success = adv_pred.argmax(-1) != target
        return dt, success, adv_pred, hats, grads
