import torch
from torch import nn
from contextlib import contextmanager


class LitEma(nn.Module):
    def __init__(self, model: nn.Module, decay: float = 0.9999, use_num_updates: bool = True):
        super().__init__()
        self.decay = decay
        self.use_num_updates = use_num_updates
        self.m_name2s_name = {}
        self.collected_params = []

        if use_num_updates:
            self.register_buffer("num_updates", torch.tensor(0, dtype=torch.long))

        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue

            shadow_name = name.replace(".", "_")
            self.m_name2s_name[name] = shadow_name
            self.register_buffer(shadow_name, parameter.detach().clone())

    def forward(self, model: nn.Module):
        decay = self.decay
        if self.use_num_updates:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates.item()) / (10 + self.num_updates.item()))

        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            model_params = dict(model.named_parameters())
            for name, shadow_name in self.m_name2s_name.items():
                parameter = model_params[name]
                shadow = getattr(self, shadow_name)
                shadow.sub_(one_minus_decay * (shadow - parameter.detach()))

    def copy_to(self, model: nn.Module):
        model_params = dict(model.named_parameters())
        for name, shadow_name in self.m_name2s_name.items():
            model_params[name].data.copy_(getattr(self, shadow_name).data)

    def store(self, parameters):
        self.collected_params = [parameter.detach().clone() for parameter in parameters]

    def restore(self, parameters):
        for collected, parameter in zip(self.collected_params, parameters):
            parameter.data.copy_(collected.data)
        self.collected_params = []

    @contextmanager
    def ema_scope(self, model: nn.Module):
        self.store(model.parameters())
        self.copy_to(model)
        try:
            yield
        finally:
            self.restore(model.parameters())
