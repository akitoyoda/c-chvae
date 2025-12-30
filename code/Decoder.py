import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def y_partition(samples_y, types_list, y_dim_partition):
    grouped = []
    if len(y_dim_partition) != len(types_list):
        raise Exception("The length of the partition vector must match the number of variables in the data")

    partition_vector_cumsum = np.insert(np.cumsum(y_dim_partition), 0, 0)
    for i in range(len(types_list)):
        grouped.append(samples_y[:, partition_vector_cumsum[i]:partition_vector_cumsum[i + 1]])
    return grouped


def _normalize_type_label(value):
    return str(value).strip()


class Decoder(nn.Module):
    def __init__(self, types_list, y_dim_partition, z_dim, seed=None):
        super().__init__()
        generator = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
        self.types_list = types_list
        self.y_dim_partition = y_dim_partition
        self.y_layer = nn.Linear(z_dim, int(np.sum(y_dim_partition)))

        layers = []
        for idx, t in enumerate(types_list):
            dim = int(t['dim'])
            t_type = _normalize_type_label(t['type'])  # normalize potential CSV whitespace
            if t_type in ['real', 'pos']:
                layers.append(nn.ModuleDict({
                    'mean': nn.Linear(y_dim_partition[idx], dim),
                    'sigma': nn.Linear(y_dim_partition[idx], dim)
                }))
            elif t_type == 'count':
                layers.append(nn.ModuleDict({
                    'lambda': nn.Linear(y_dim_partition[idx], dim)
                }))
            elif t_type == 'cat':
                layers.append(nn.ModuleDict({
                    'log_pi': nn.Linear(y_dim_partition[idx], dim - 1)
                }))
            elif t_type == 'ordinal':
                layers.append(nn.ModuleDict({
                    'theta': nn.Linear(y_dim_partition[idx], dim - 1),
                    'mean': nn.Linear(y_dim_partition[idx], 1)
                }))
            else:
                raise ValueError(f"Unknown type {t_type}")
        self.type_layers = nn.ModuleList(layers)

        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    module.weight.normal_(0.0, 0.05, generator=generator)
                    if module.bias is not None:
                        module.bias.zero_()

    def forward(self, samples_z):
        samples = {'z': samples_z, 'y': None, 'x': None, 's': None}
        gradients = {'g1': None, 'g2': None, 'g3': None}

        samples['y'] = self.y_layer(samples_z)
        grouped_samples_y = y_partition(samples['y'], self.types_list, self.y_dim_partition)

        theta = []
        for idx, t in enumerate(self.types_list):
            out_layers = self.type_layers[idx]
            if t['type'] == 'real':
                theta.append([out_layers['mean'](grouped_samples_y[idx]),
                              out_layers['sigma'](grouped_samples_y[idx])])
            elif t['type'] == 'pos':
                theta.append([out_layers['mean'](grouped_samples_y[idx]),
                              out_layers['sigma'](grouped_samples_y[idx])])
            elif t['type'] == 'count':
                theta.append(out_layers['lambda'](grouped_samples_y[idx]))
            elif t['type'] == 'cat':
                logits = out_layers['log_pi'](grouped_samples_y[idx])
                theta.append(torch.cat([torch.zeros(samples_z.size(0), 1, device=samples_z.device, dtype=samples_z.dtype), logits], dim=1))
            elif t['type'] == 'ordinal':
                theta.append([out_layers['theta'](grouped_samples_y[idx]),
                              out_layers['mean'](grouped_samples_y[idx])])

        return theta, samples, gradients

    def decode_only(self, samples_z):
        samples = {'z': samples_z, 'y': None, 'x': None, 's': None}
        samples['y'] = self.y_layer(samples_z)
        grouped_samples_y = y_partition(samples['y'], self.types_list, self.y_dim_partition)

        theta = []
        for idx, t in enumerate(self.types_list):
            out_layers = self.type_layers[idx]
            if t['type'] == 'real':
                theta.append([out_layers['mean'](grouped_samples_y[idx]),
                              out_layers['sigma'](grouped_samples_y[idx])])
            elif t['type'] == 'pos':
                theta.append([out_layers['mean'](grouped_samples_y[idx]),
                              out_layers['sigma'](grouped_samples_y[idx])])
            elif t['type'] == 'count':
                theta.append(out_layers['lambda'](grouped_samples_y[idx]))
            elif t['type'] == 'cat':
                logits = out_layers['log_pi'](grouped_samples_y[idx])
                theta.append(torch.cat([torch.zeros(samples_z.size(0), 1, device=samples_z.device, dtype=samples_z.dtype), logits], dim=1))
            elif t['type'] == 'ordinal':
                theta.append([out_layers['theta'](grouped_samples_y[idx]),
                              out_layers['mean'](grouped_samples_y[idx])])
        return theta, samples


def decoder_test_time(decoder, samples_z):
    decoder.eval()
    with torch.no_grad():
        return decoder.decode_only(samples_z)
