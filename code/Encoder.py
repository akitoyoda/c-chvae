import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, x_list, tau, x_cond_list=None):
        device = x_list[0].device
        batch_size = x_list[0].shape[0]

        x = torch.cat(x_list, dim=1)
        if self.include_condition and x_cond_list is not None:
            x_cond = torch.cat(x_cond_list, dim=1)
            x = torch.cat([x, x_cond], dim=1)
        else:
            x_cond = None

        log_pi = self.s_layer(x)
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

        log_var_stack = torch.stack(log_var_qz, dim=0)
        mean_stack = torch.stack(mean_qz, dim=0)

        log_var_joint = -torch.logsumexp(-log_var_stack, dim=0)
        mean_joint = torch.exp(log_var_joint) * torch.sum(mean_stack * torch.exp(-log_var_stack), dim=0)

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


def z_distribution_GMM(samples_s, z_dim):
    mean_pz = nn.Linear(samples_s.shape[1], z_dim, bias=True)(samples_s)
    log_var_pz = torch.zeros(samples_s.shape[0], z_dim, device=samples_s.device, dtype=samples_s.dtype)
    log_var_pz = torch.clamp(log_var_pz, -15.0, 15.0)
    return mean_pz, log_var_pz
