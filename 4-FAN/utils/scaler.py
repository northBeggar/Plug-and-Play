import torch
from timm.utils import ApexScaler, NativeScaler
try:
    from apex import amp
    has_apex = True
except ImportError:
    amp = None
    has_apex = False

class ApexScaler_SAM(ApexScaler):

    def __call__(self, loss, optimizer, clip_grad=None, clip_mode='norm', parameters=None, create_graph=False, step=0, rho=0.05):
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(create_graph=create_graph)
        if step==0 or step==2:
            if clip_grad is not None:
                dispatch_clip_grad(amp.master_params(optimizer), clip_grad, mode=clip_mode)
            optimizer.step()
        elif step==1:
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), rho, norm_type=2.0)
            optimizer.step()
