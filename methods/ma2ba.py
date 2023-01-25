import torch
import torch.nn.functional as F
import numpy as np
from attack.fgsm import FGSM
from attack.fgsm_single import FGSMGrad as FGSMGradSingle
from attack.fgsm_multi import FGSMGrad as FGSMGradMulti
from .utils import get_time


class IntegratedGradientRemove:
    def __init__(self, model):
        self.model = model

    def __call__(self, data, target, baseline, gradient_steps=50):
        alphas = torch.linspace(0, 1, gradient_steps)
        scaled_features_tpl = torch.cat(
            [baseline + alpha * (data - baseline) for alpha in alphas], dim=0
        ).requires_grad_()
        with torch.autograd.set_grad_enabled(True):
            outputs = self.model(scaled_features_tpl)
            ces = F.softmax(outputs, dim=-1)[:, target]
            outputs = outputs[:, target]
            grads = torch.autograd.grad(
                torch.unbind(outputs), scaled_features_tpl)[0]
        idxes = list()
        for ce in ces:
            if ce > ces[-1]:
                idxes.append(False)
            else:
                idxes.append(True)
        grads = grads[idxes]
        alphas = alphas[idxes]
        n = len(alphas)
        step_sizes = [1 / n] * n
        step_sizes[0] = step_sizes[0] / 2
        step_sizes[-1] = step_sizes[-1] / 2
        scaled_grads = grads.contiguous().view(
            n, -1) * torch.tensor(step_sizes).view(n, 1).to(grads.device)
        total_grads = torch.sum(scaled_grads.reshape(
            n, *data.shape[1:]), dim=0).unsqueeze(0)
        attribution_map = total_grads * (data - baseline)
        return attribution_map.detach().cpu().numpy(), idxes


class IntegratedGradientMinus:
    def __init__(self, model) -> None:
        self.model = model

    def __call__(self, data, target, baseline, gradient_steps=50):
        alphas = torch.linspace(0, 1, gradient_steps)
        scaled_features_tpl = torch.cat(
            [baseline + alpha * (data - baseline) for alpha in alphas], dim=0
        ).requires_grad_()
        with torch.autograd.set_grad_enabled(True):
            outputs = self.model(scaled_features_tpl)
            ces = F.softmax(outputs, dim=-1)[:, target]
            outputs = outputs[:, target]
            grads = torch.autograd.grad(
                torch.unbind(outputs), scaled_features_tpl)[0]
        idxes = list()
        for ce in ces:
            if ce > ces[-1]:
                idxes.append(True)
            else:
                idxes.append(False)
        idxes[0] = True
        idxes[-1] = True

        grads = grads[idxes]
        alphas = alphas[idxes]
        n = len(alphas)
        step_sizes = [1 / n] * n
        step_sizes[0] = step_sizes[0] / 2
        step_sizes[-1] = step_sizes[-1] / 2
        scaled_grads = grads.contiguous().view(
            n, -1) * torch.tensor(step_sizes).view(n, 1).to(grads.device)
        total_grads = torch.sum(scaled_grads.reshape(
            n, *data.shape[1:]), dim=0).unsqueeze(0)
        attribution_map = total_grads * (data - baseline)
        return attribution_map.detach().cpu().numpy()


class Ma2Cos:
    def __init__(self, model):
        self.model = model

    def __call__(self, data, baseline, hats, grads):
        # input_clone = x_hats[0].clone().cpu().detach().numpy()
        # baseline_clone = baseline.clone().cpu().detach().numpy()
        # baseline_input = baseline_clone - input_clone
        # ts = []
        # for x_hat in x_hats:
        #     x_hat = x_hat.detach().cpu().numpy()
        #     adv_input = x_hat - input_clone
        #     t = np.sum(adv_input * baseline_input) / \
        #         (np.linalg.norm(baseline_input) ** 2)
        #     ts.append(t)
        # ts = np.array(ts)
        # ts_max = np.max(ts)
        # ts = ts / ts_max

        # n = len(grads)
        # ts = (ts[1:n] - ts[0:n-1])
        # total_grads = (grads[0:n-1] + grads[1:n]) / 2
        # n = len(ts)
        # scaled_grads = total_grads.contiguous().view(
        #     n, -1) * torch.tensor(ts).float().view(n, 1).to(grads.device)

        # total_grads = torch.sum(scaled_grads.reshape(
        #     n, *input.shape[1:]), dim=0).unsqueeze(0)

        # attr = total_grads * (input - baseline)
        # return attr.detach().cpu().numpy()
        
        input_clone = hats[0].clone().cpu().detach().numpy()
        baseline_clone = baseline.clone().cpu().detach().numpy()
        baseline_input = baseline_clone - input_clone
        t_list = list()
        for hat in hats:
            hat = hat.detach().cpu().numpy()
            hat_input = hat - input_clone
            t = np.sum(hat_input * baseline_input) / \
                (np.linalg.norm(baseline_input) ** 2)
            t_list.append(t)
        t_list = np.array(t_list)
        t_max = np.max(t_list)
        t_list = t_list / t_max

        n = len(grads)
        t_list = t_list[1:n] - t_list[0:n-1]
        total_grads = (grads[0:n-1] + grads[1:n]) / 2
        n = len(t_list)
        scaled_grads = total_grads.contiguous().view(
            n, -1) * torch.tensor(t_list).float().view(n, 1).to(grads.device)

        total_grads = torch.sum(scaled_grads.reshape(
            n, *data.shape[1:]), dim=0).unsqueeze(0)

        attribution_map = total_grads * (data - baseline)
        return attribution_map.detach().cpu().numpy()


class Ma2Ba:
    def __init__(self, model, type="1"):
        self.model = model
        self.type = type

    def __call__(self, hats, grads):
        t_list = hats[1:] - hats[:-1]
        if self.type == "1":
            grads = grads[:-1]
        else:
            grads = (grads[:-1] + grads[1:]) / 2
        total_grads = -torch.sum(t_list * grads, dim=0)
        attribution_map = total_grads.unsqueeze(0)
        return attribution_map.detach().cpu().numpy()


class Ma2Norm:
    def __init__(self, model):
        self.model = model

    def __call__(self, data, baseline, hats, grads):
        input_clone = hats[0].clone().cpu().detach().numpy()
        t_list = list()
        for hat in hats:
            hat = hat.detach().cpu().numpy()
            hat_input = hat - input_clone
            t = np.linalg.norm(hat_input)
            t_list.append(t)
        t_list = np.array(t_list)
        t_max = np.max(t_list)
        t_list = t_list / t_max
        n = len(grads)
        t_list = t_list[1:n] - t_list[0:n-1]
        total_grads = (grads[:n-1] + grads[1:]) / 2
        n = len(t_list)
        scaled_grads = total_grads.contiguous().view(
            n, -1) * torch.tensor(t_list).float().view(n, 1).to(grads.device)
        total_grads = torch.sum(scaled_grads.reshape(
            n, *data.shape[1:]), dim=0).unsqueeze(0)
        attribution_map = total_grads * (data - baseline)
        return attribution_map.detach().cpu().numpy()


@get_time("Remove")
def remove_pipeline(model, data, target, data_min, data_max, epsilon=0.3 * 255, gradient_steps=50):
    remove = IntegratedGradientRemove(model)
    attack = FGSM(epsilon=epsilon, data_min=data_min, data_max=data_max)
    dt, success, _ = attack(model, data, target)
    attribution_map, idxes = remove(
        data, target, dt, gradient_steps=gradient_steps)
    return attribution_map, idxes, success


@get_time("Minus")
def minus_pipeline(model, data, target, data_min, data_max, epsilon=0.3 * 255, gradient_steps=50):
    minus = IntegratedGradientMinus(model)
    attack = FGSM(epsilon=epsilon, data_min=data_min, data_max=data_max)
    dt, success, _ = attack(model, data, target)
    attribution_map = minus(data, target, dt, gradient_steps=gradient_steps)
    return attribution_map, success


@get_time("Ma2NormB4Softmax")
def ma2norm_b4_softmax_pipeline(model, data, target, data_min, data_max, epsilon=0.3 * 255):
    ma2norm = Ma2Norm(model)
    attack = FGSMGradSingle(
        epsilon=epsilon, data_min=data_min, data_max=data_max)
    dt, success, _, hats, grads = attack(
        model, data, target, use_sign=True, use_softmax=False)
    attribution_map = ma2norm(data, dt, hats, grads)
    return attribution_map, success


@get_time("Ma2NormAfterSoftmax")
def ma2norm_after_softmax_pipeline(model, data, target, data_min, data_max, epsilon=0.3 * 255):
    ma2norm = Ma2Norm(model)
    attack = FGSMGradSingle(
        epsilon=epsilon, data_min=data_min, data_max=data_max)
    dt, success, _, hats, grads = attack(
        model, data, target, use_sign=True, use_softmax=True)
    attribution_map = ma2norm(data, dt, hats, grads)
    return attribution_map, success


@get_time("Ma2CosSignB4Softmax")
def ma2cos_sign_b4_softmax_pipeline(model, data, target, data_min, data_max, epsilon=0.3 * 255):
    ma2cos = Ma2Cos(model)
    attack = FGSMGradSingle(
        epsilon=epsilon, data_min=data_min, data_max=data_max)
    dt, success, _, hats, grads = attack(
        model, data, target, use_sign=True, use_softmax=False)
    attribution_map = ma2cos(data, dt, hats, grads)
    return attribution_map, success


@get_time("Ma2CosSignAfterSoftmax")
def ma2cos_sign_after_softmax_pipeline(model, data, target, data_min, data_max, epsilon=0.3 * 255):
    ma2cos = Ma2Cos(model)
    attack = FGSMGradSingle(
        epsilon=epsilon, data_min=data_min, data_max=data_max)
    dt, success, _, hats, grads = attack(
        model, data, target, use_sign=True, use_softmax=True)
    attribution_map = ma2cos(data, dt, hats, grads)
    return attribution_map, success


@get_time("Ma2CosWithoutSignB4Softmax")
def ma2cos_without_sign_b4_softmax_pipeline(model, data, target, data_min, data_max, epsilon=0.3 * 255):
    ma2cos = Ma2Cos(model)
    attack = FGSMGradSingle(
        epsilon=epsilon, data_min=data_min, data_max=data_max)
    dt, success, _, hats, grads = attack(
        model, data, target, use_sign=False, use_softmax=False)
    attribution_map = ma2cos(data, dt, hats, grads)
    return attribution_map, success


@get_time("Ma2CosWithoutSignAfterSoftmax")
def ma2cos_without_sign_after_softmax_pipeline(model, data, target, data_min, data_max, epsilon=0.3 * 255):
    ma2cos = Ma2Cos(model)
    attack = FGSMGradSingle(
        epsilon=epsilon, data_min=data_min, data_max=data_max)
    dt, success, _, hats, grads = attack(
        model, data, target, use_sign=False, use_softmax=True)
    attribution_map = ma2cos(data, dt, hats, grads)
    return attribution_map, success


@get_time("Ma2BaSignB4Softmax")
def ma2ba_sign_b4_softmax_pipeline(model, data, target, data_min, data_max, epsilon=0.3 * 255):
    ma2ba = Ma2Ba(model)
    attack = FGSMGradMulti(
        epsilon=epsilon, data_min=data_min, data_max=data_max)
    _, success, _, hats, grads = attack(
        model, data, target, use_sign=True, use_softmax=False)
    attribution_map = list()
    for i in range(len(hats)):
        attribution_map.append(ma2ba(hats[i], grads[i]))
    attribution_map = np.concatenate(attribution_map, axis=0)
    return attribution_map, success


@get_time("Ma2BaSignAfterSoftmax")
def ma2ba_sign_after_softmax_pipeline(model, data, target, data_min, data_max, epsilon=0.3 * 255):
    ma2ba = Ma2Ba(model)
    attack = FGSMGradMulti(
        epsilon=epsilon, data_min=data_min, data_max=data_max)
    _, success, _, hats, grads = attack(
        model, data, target, use_sign=True, use_softmax=True)
    attribution_map = list()
    for i in range(len(hats)):
        attribution_map.append(ma2ba(hats[i], grads[i]))
    attribution_map = np.concatenate(attribution_map, axis=0)
    return attribution_map, success


@get_time("Ma2BaWithoutSignB4Softmax")
def ma2ba_without_sign_b4_softmax_pipeline(model, data, target, data_min, data_max, epsilon=0.3 * 255):
    ma2ba = Ma2Ba(model)
    attack = FGSMGradMulti(
        epsilon=epsilon, data_min=data_min, data_max=data_max)
    _, success, _, hats, grads = attack(
        model, data, target, use_sign=False, use_softmax=False)
    attribution_map = list()
    for i in range(len(hats)):
        attribution_map.append(ma2ba(hats[i], grads[i]))
    attribution_map = np.concatenate(attribution_map, axis=0)
    return attribution_map, success


@get_time("Ma2BaWithoutSignAfterSoftmax")
def ma2ba_without_sign_after_softmax_pipeline(model, data, target, data_min, data_max, epsilon=0.3 * 255):
    ma2ba = Ma2Ba(model)
    attack = FGSMGradMulti(
        epsilon=epsilon, data_min=data_min, data_max=data_max)
    _, success, _, hats, grads = attack(
        model, data, target, use_sign=False, use_softmax=True)
    attribution_map = list()
    for i in range(len(hats)):
        attribution_map.append(ma2ba(hats[i], grads[i]))
    attribution_map = np.concatenate(attribution_map, axis=0)
    return attribution_map, success
