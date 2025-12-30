import numpy as np
import torch
import torch.nn as nn
import Helpers
import Encoder
import Decoder
import Evaluation


class CHVAEModel(nn.Module):
    def __init__(self, types_list, types_list_c=None, z_dim=1, y_dim=1, s_dim=1, y_dim_partition=None):
        super().__init__()
        self.types_list = types_list
        self.types_list_c = types_list_c
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.s_dim = s_dim

        if y_dim_partition:
            self.y_dim_partition = y_dim_partition
        else:
            self.y_dim_partition = list(y_dim * np.ones(len(types_list), dtype=int))

        input_dims = [int(t['dim']) for t in types_list]
        if types_list_c:
            cond_dims = [int(t['dim']) for t in types_list_c]
            self.encoder = Encoder.ConditionalEncoder(input_dims, cond_dims, z_dim, s_dim)
        else:
            self.encoder = Encoder.Encoder(input_dims, z_dim, s_dim)

        self.decoder = Decoder.Decoder(types_list, self.y_dim_partition, z_dim)
        self.mean_dec_z = nn.Linear(s_dim, z_dim)
        self.prior_logvar = nn.Parameter(torch.zeros(1, z_dim))  # start from unit variance prior as in TF baseline
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            self.mean_dec_z.weight.normal_(0.0, 0.05)
            if self.mean_dec_z.bias is not None:
                self.mean_dec_z.bias.zero_()

    def forward(self, x_list, tau, x_list_c=None):
        normalized_data, normalization_params, noisy_data = Helpers.batch_normalization(x_list, self.types_list, len(x_list[0]))
        samples, q_params = self.encoder(noisy_data, tau, x_list_c)
        samples_s = samples['s']
        mean_pz, log_var_pz = Encoder.z_distribution_GMM(samples_s, self.z_dim, self.mean_dec_z, self.prior_logvar)
        p_params = {'z': (mean_pz, log_var_pz)}

        theta, decoder_samples, gradient_decoder = self.decoder(samples['z'])
        decoder_samples['s'] = samples_s

        log_p_x, samples_x, params_x = Evaluation.loglik_evaluation(normalized_data, self.types_list, theta, normalization_params)
        p_params['x'] = params_x

        ELBO, loss_reconstruction, KL_z, KL_s = Evaluation.cost_function(
            log_p_x, p_params, q_params, self.types_list, self.z_dim, np.sum(self.y_dim_partition), self.s_dim)

        outputs = {
            'samples': decoder_samples,
            'q_params': q_params,
            'p_params': p_params,
            'log_p_x': log_p_x,
            'loss_reconstruction': loss_reconstruction,
            'ELBO': ELBO,
            'KL_z': KL_z,
            'KL_s': KL_s,
            'theta': theta,
            'normalization_params': normalization_params
        }
        return outputs

    @torch.no_grad()
    def generate(self, x_list, tau=1e-3, x_list_c=None):
        self.eval()
        normalized_data, normalization_params, noisy_data = Helpers.batch_normalization(x_list, self.types_list, len(x_list[0]))
        samples, q_params = self.encoder(noisy_data, tau, x_list_c, deterministic_s=True)
        mean_pz, log_var_pz = Encoder.z_distribution_GMM(samples['s'], self.z_dim, self.mean_dec_z, self.prior_logvar)
        p_params = {'z': (mean_pz, log_var_pz)}
        theta, decoder_samples = self.decoder.decode_only(samples['z'])
        log_p_x, samples_x, params_x = Evaluation.loglik_evaluation(normalized_data, self.types_list, theta, normalization_params)
        return {
            'samples': {**decoder_samples, 's': samples['s'], 'x': samples_x},
            'q_params': q_params,
            'p_params': p_params,
            'theta': theta,
            'log_p_x': log_p_x,
            'params_x': params_x,
            'normalization_params': normalization_params
        }


def C_HVAE_graph(types_file, learning_rate=1e-4, z_dim=1, y_dim=1, s_dim=1, y_dim_partition=None, nsamples=1000, p=2):
    _, types_list = Helpers.place_holder_types(types_file, None)
    model = CHVAEModel(types_list, None, z_dim, y_dim, s_dim, y_dim_partition)
    return model


def C_CHVAE_graph(types_file, types_file_c, learning_rate=1e-3, z_dim=1, y_dim=1, s_dim=1, y_dim_partition=None, nsamples=1000, p=2, degree_active=0.95):
    _, types_list = Helpers.place_holder_types(types_file, None)
    _, types_list_c = Helpers.place_holder_types(types_file_c, None)
    model = CHVAEModel(types_list, types_list_c, z_dim, y_dim, s_dim, y_dim_partition)
    return model
