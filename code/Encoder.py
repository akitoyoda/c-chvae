import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BaseEncoder(nn.Module):
    def __init__(self, input_dims, z_dim, s_dim, include_condition=False, condition_dim=0):
        super().__init__()
        self.z_dim = z_dim
        self.s_dim = s_dim
        self.include_condition = include_condition
        self.condition_dim = condition_dim

        total_dim = sum(input_dims) + (condition_dim if include_condition else 0)
        self.s_layer = nn.Linear(total_dim, s_dim)

        self.z_mean_layers = nn.ModuleList()
        self.z_logvar_layers = nn.ModuleList()
        for dim in input_dims:
            in_dim = dim + s_dim + (condition_dim if include_condition else 0)
            self.z_mean_layers.append(nn.Linear(in_dim, z_dim))
            self.z_logvar_layers.append(nn.Linear(in_dim, z_dim))

        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            self.s_layer.weight.normal_(0.0, 0.05)
            if self.s_layer.bias is not None:
                self.s_layer.bias.zero_()

            for layer in self.z_mean_layers:
                layer.weight.normal_(0.0, 0.05)
                if layer.bias is not None:
                    layer.bias.zero_()

            for layer in self.z_logvar_layers:
                layer.weight.normal_(0.0, 0.05)
                if layer.bias is not None:
                    layer.bias.zero_()

    def forward(self, x_list, tau, x_cond_list=None, x_list_c=None, deterministic_s=False):
        # Support both `x_cond_list` and legacy `x_list_c` keyword names used elsewhere
        if x_cond_list is None and x_list_c is not None:
            x_cond_list = x_list_c
        """
        Args:
            x_list: list of tensors for each variable.
            tau: temperature for Gumbel-Softmax during training.
            x_cond_list: optional conditioning inputs.
            deterministic_s: when True, sample s as one-hot argmax(log_pi) for inference/visualization.
        """
        device = x_list[0].device
        batch_size = x_list[0].shape[0]

        x = torch.cat(x_list, dim=1)
        if self.include_condition and x_cond_list is not None:
            x_cond = torch.cat(x_cond_list, dim=1)
            x = torch.cat([x, x_cond], dim=1)
        else:
            x_cond = None

        log_pi = self.s_layer(x)
        if deterministic_s:
            indices = torch.argmax(log_pi, dim=1)
            samples_s = F.one_hot(indices, num_classes=log_pi.size(1)).to(log_pi.dtype)
        else:
            samples_s = F.gumbel_softmax(log_pi, tau=tau, hard=False, dim=1)

        mean_qz = []
        log_var_qz = []
        for idx, d in enumerate(x_list):
            pieces = [d, samples_s]
            if self.include_condition and x_cond is not None:
                pieces.append(x_cond)
            inp = torch.cat(pieces, dim=1)

            mean_qz.append(self.z_mean_layers[idx](inp))
            log_var_qz.append(self.z_logvar_layers[idx](inp))

        prior_log_var = torch.zeros(batch_size, self.z_dim, device=device, dtype=x.dtype)
        prior_mean = torch.zeros_like(prior_log_var)
        log_var_qz.append(prior_log_var)
        mean_qz.append(prior_mean)

        # Sanitize mean/logvar tensors to prevent NaN/Inf propagation
        sanitized_mean_qz = []
        sanitized_log_var_qz = []
        for mm, lv in zip(mean_qz, log_var_qz):
            # replace NaN/Inf in means; clip extreme values
            mm_clean = torch.nan_to_num(mm, nan=0.0, posinf=1e6, neginf=-1e6)
            mm_clean = torch.clamp(mm_clean, min=-1e6, max=1e6)
            # replace NaN/Inf in logvars; clamp to reasonable range
            lv_clean = torch.nan_to_num(lv, nan=0.0, posinf=50.0, neginf=-50.0)
            lv_clean = torch.clamp(lv_clean, min=-50.0, max=50.0)
            sanitized_mean_qz.append(mm_clean)
            sanitized_log_var_qz.append(lv_clean)

        mean_qz = sanitized_mean_qz
        log_var_qz = sanitized_log_var_qz

        log_var_stack = torch.stack(log_var_qz, dim=0)
        mean_stack = torch.stack(mean_qz, dim=0)

        # Numerically stable computation of mixture mean/variance using normalized weights
        # weights = softmax(-log_var_stack) across mixture components
        neg = -log_var_stack
        denom = torch.logsumexp(neg, dim=0)
        weights = torch.exp(neg - denom)  # stable, sums to 1 across components
        mean_joint = torch.sum(mean_stack * weights, dim=0)
        log_var_joint = -denom

        eps = torch.randn_like(mean_joint)
        samples_z = mean_joint + torch.exp(0.5 * log_var_joint) * eps

        samples = {'s': samples_s, 'z': samples_z, 'y': None, 'x': None}
        q_params = {'s': log_pi, 'z': (mean_joint, log_var_joint)}

        return samples, q_params


class Encoder(BaseEncoder):
    def __init__(self, input_dims, z_dim, s_dim):
        super().__init__(input_dims, z_dim, s_dim, include_condition=False)


class ConditionalEncoder(BaseEncoder):
    def __init__(self, input_dims, condition_dims, z_dim, s_dim):
        super().__init__(input_dims, z_dim, s_dim, include_condition=True, condition_dim=sum(condition_dims))


def z_distribution_GMM(samples_s, z_dim, mean_dec_z=None, log_var_param=None):
    """
    Compute the Gaussian parameters for p(z|s).

    Args:
        samples_s: one-hot or soft samples from q(s|x).
        z_dim: latent dimensionality.
        mean_dec_z: optional nn.Linear mapping s -> mean; if None, defaults to zeros.
        log_var_param: optional learnable log-variance Parameter expanded to batch shape.
    """
    if mean_dec_z is not None:
        mean_pz = mean_dec_z(samples_s)
    else:
        mean_pz = torch.zeros(samples_s.shape[0], z_dim, device=samples_s.device, dtype=samples_s.dtype)

    if log_var_param is not None:
        log_var_pz = log_var_param.expand_as(mean_pz)
    else:
        log_var_pz = torch.zeros_like(mean_pz)

    log_var_pz = torch.clamp(log_var_pz, -15.0, 15.0)
    return mean_pz, log_var_pz
