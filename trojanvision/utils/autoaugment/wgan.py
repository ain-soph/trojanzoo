#!/usr/bin/env python3

from .policy import Policy
from trojanvision.models import ImageModel

import torch
import torch.nn as nn
import torch.optim as optim


class WGAN(nn.Module):
    def __init__(self, model: ImageModel,
                 policy: Policy,
                 eps: float = 0.1,
                 gp_factor: float = 10.0):
        super().__init__()
        num_feats: int = model._model.classifier[0].in_features

        self.policy = policy
        self.model = model._model
        self.discriminator = nn.Sequential(
            nn.Linear(num_feats, num_feats),
            nn.ReLU(),
            nn.Linear(num_feats, 1))
        self.criterion = model.criterion

        self.eps = eps
        self.gp_factor = gp_factor

        self.model_optimizer = optim.Adam(model.parameters(),
                                          lr=1e-3, betas=[0.0, 0.999])
        self.policy_optimizer = optim.Adam(policy.parameters(),
                                           lr=1e-3, betas=[0.0, 0.999])

    def update(self, _input: torch.Tensor,
               _label: torch.Tensor) -> dict[str, float]:
        a_input, n_input = _input.chunk(2)
        a_label, n_label = _label.chunk(2)

        # Discriminator Update
        with torch.no_grad():
            aug_input = self.policy(a_input)
        n_feats = self.model.get_final_fm(n_input)
        n_output = self.model.classifier(n_feats)
        cls_n_loss: torch.Tensor = self.criterion(n_output, n_label)
        d_n_output: torch.Tensor = self.discriminator(n_feats)
        d_n_loss = d_n_output.mean()

        a_feats = self.model.get_final_fm(aug_input)
        d_a_output: torch.Tensor = self.discriminator(a_feats)
        d_a_loss = d_a_output.mean()

        gp_loss = self.gradient_penalty(n_input, aug_input)

        d_loss = self.eps * cls_n_loss - d_n_loss + d_a_loss \
            + self.gp_factor * gp_loss
        self.model_optimizer.zero_grad()
        d_loss.backward()
        self.model_optimizer.step()

        # Generator Update
        aug2_input = self.policy(a_input)
        a2_feats = self.model.get_final_fm(aug2_input)
        a2_output = self.model.classifier(a2_feats)
        cls_a2_loss: torch.Tensor = self.criterion(a2_output, a_label)

        d_a2_output: torch.Tensor = self.discriminator(a2_feats)
        d_a2_loss = d_a2_output.mean()

        g_loss = self.eps * cls_a2_loss - d_a2_loss
        self.policy_optimizer.zero_grad()
        g_loss.backward()
        self.policy_optimizer.step()
        for param in self.policy.parameters():
            param.data.nan_to_num_(0.5)

        loss_dict: dict[str, float] = {
            'discriminator': d_loss.item(),
            'generator': g_loss.item(),
            'classification': (cls_n_loss + cls_a2_loss).item(),
            'gradient_penalty': gp_loss.item()
        }
        return loss_dict

    def gradient_penalty(self, real: torch.Tensor,
                         fake: torch.Tensor) -> torch.Tensor:
        alpha = torch.rand(real.size(0), device=real.device)
        alpha = alpha.view([-1] + [1] * (real.dim() - 1))
        interpolated: torch.Tensor = alpha * real + (1 - alpha) * fake
        interpolated = interpolated.detach()
        interpolated.requires_grad_()

        _feats = self.model.get_final_fm(interpolated)
        d_output: torch.Tensor = self.discriminator(_feats)
        grad = torch.autograd.grad(d_output.sum(), interpolated,
                                   create_graph=True)[0]
        result: torch.Tensor = grad.norm(2, dim=1) - 1
        return result.square().mean()
