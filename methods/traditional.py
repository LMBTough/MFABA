from captum.attr import Saliency, IntegratedGradients, DeepLift, NoiseTunnel
from .utils import get_time

class IntegratedGradient:
    """
    IG
    """
    def __init__(self, model):
        self.model = model
        self.saliency = IntegratedGradients(model)

    def __call__(self, data, target, gradient_steps=50):
        attribution_map = self.saliency.attribute(data,
                                                  target=target,
                                                  baselines=None,
                                                  n_steps=gradient_steps,
                                                  method="riemann_trapezoid")
        return attribution_map.detach().cpu().numpy()
    
class SaliencyGradient:
    """
    SM
    """
    def __init__(self, model):
        self.model = model
        self.saliency = Saliency(model)

    def __call__(self, data, target):
        attribution_map = self.saliency.attribute(data, target=target, abs=False)
        return attribution_map.detach().cpu().numpy()
    
class SmoothGradient:
    """
    SG
    """
    def __init__(self, model, stdevs=0.15):
        self.model = model
        self.saliency = NoiseTunnel(Saliency(model))
        self.stdevs = stdevs

    def __call__(self, data, target, gradient_steps=50):
        attribution_map = self.saliency.attribute(data,
                                                  target=target,
                                                  nt_samples = gradient_steps,
                                                  stdevs=self.stdevs,
                                                  abs=False)
        return attribution_map.detach().cpu().numpy()

class DL:
    """
    DeepLift
    """
    def __init__(self, model):
        self.model = model
        self.saliency = DeepLift(model)

    def __call__(self, data, target):
        attribution_map = self.saliency.attribute(data, target=target, baselines=None)
        return attribution_map.detach().cpu().numpy()
    
@get_time("IG")
def ig_pipeline(model, data, target, gradient_steps=50):
    ig = IntegratedGradient(model)
    return ig(data, target, gradient_steps=gradient_steps)

@get_time("SM")
def sm_pipeline(model, data, target):
    sm = SaliencyGradient(model)
    return sm(data, target)

@get_time("SG")
def sg_pipeline(model, data, target, gradient_steps=50):
    sg = SmoothGradient(model)
    return sg(data, target, gradient_steps=gradient_steps)

@get_time("DL")
def dl_pipeline(model, data, target):    
    dl = DL(model)
    return dl(data, target)