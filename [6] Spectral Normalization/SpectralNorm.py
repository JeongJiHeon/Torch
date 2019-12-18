import torch
import torch.nn as nn
from torch.nn import Parameter

def L2Normalization(v, eps = 1e-8):
    return v / (v.norm() + eps)


class SpectralNorm2d(nn.Module):
    def __init__(self, module, name = 'weight'):
        super(SpectralNorm2d, self).__init__()
        self.module = module
        self.name = name
        if not self._made_uv():
            self._make_uv()
        
    def _made_uv(self):
        try:
            getattr(self.module, '_u')
            getattr(self.module, '_v')
            getattr(self.module, '_wbar')
            return True
        except AttributeError:
            return False
        
    def _make_uv(self):
        w = Parameter(getattr(self.module, self.name))
        hei = w.shape[0]
        wid = w.view(w.shape[0],-1).shape[1]
        u = Parameter(torch.randn(hei), requires_grad = False)
        u.data = L2Normalization(u)
        v = Parameter(torch.randn(wid), requires_grad = False)
        v.data = L2Normalization(v)
        del self.module._parameters[self.name]
        
        self.module.register_parameter(self.name + '_u', u)
        self.module.register_parameter(self.name + '_v', v)
        self.module.register_parameter(self.name + '_bar', w)

        
        
    def _update_uv(self):
        u = getattr(self.module, self.name + '_u')
        v = getattr(self.module, self.name + '_v')
        w_bar = getattr(self.module, self.name + '_bar')
        
        v.data = L2Normalization(torch.mv(w_bar.view(w_bar.shape[0], -1).t(), u))
        u.data = L2Normalization(torch.mv(w_bar.view(w_bar.shape[0], -1), v))
        
        sigma = u.dot(w_bar.view(w_bar.shape[0], -1).mv(v))
        setattr(self.module, self.name, w_bar/sigma)
        
    def forward(self, *args):
        self._update_uv()
        return self.module.forward(*args)



