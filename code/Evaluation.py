import torch
import torch.nn.functional as F
import Loglik


def loglik_evaluation(batch_data_list, types_list, theta, normalization_params):
    log_p_x = []
    samples_x = []
    params_x = []

    for i, d in enumerate(batch_data_list):
        loglik_function = getattr(Loglik, 'loglik_' + types_list[i]['type'])
        out = loglik_function(d, types_list[i], theta[i], normalization_params[i])
        log_p_x.append(out['log_p_x'])
        samples_x.append(out['samples'])
        params_x.append(out['params'])

    return log_p_x, samples_x, params_x


def loglik_evaluation_test(batch_data_list, theta, normalization_params, list_type):
    samples_x_perturbed = []
    params_x_perturbed = []

    for i, _ in enumerate(batch_data_list):
        loglik_function = getattr(Loglik, 'loglik_test_' + list_type[i]['type'])
        out = loglik_function(theta[i], normalization_params[i], list_type[i])
        samples_x_perturbed.append(out['samples'])
        params_x_perturbed.append(out['params'])

    return samples_x_perturbed, params_x_perturbed


def cost_function(log_p_x, p_params, q_params, types_list, z_dim, y_dim, s_dim):
    eps = 1e-8
    log_pi = q_params['s']
    pi_param = F.softmax(log_pi, dim=1)
    KL_s = torch.sum(pi_param * (torch.log(pi_param + eps) - torch.log(torch.tensor(1.0 / s_dim, device=log_pi.device))), dim=1)

    mean_pz, log_var_pz = p_params['z']
    mean_qz, log_var_qz = q_params['z']
    KL_z = -0.5 * z_dim + 0.5 * torch.sum(
        torch.exp(log_var_qz - log_var_pz) + (mean_pz - mean_qz) ** 2 / torch.exp(log_var_pz) - log_var_qz + log_var_pz,
        dim=1)

    log_p_x_tensor = torch.stack(log_p_x, dim=0)
    loss_reconstruction = torch.sum(log_p_x_tensor, dim=0)
    ELBO = torch.mean(1.20 * loss_reconstruction - (KL_z + KL_s), dim=0)

    return ELBO, loss_reconstruction, KL_z, KL_s


def kl_z_diff(p_params, q_params, degree_active, batch_size, z_dim):
    mean_pz, log_var_pz = p_params['z']
    mean_qz, log_var_qz = q_params['z']

    ones = torch.ones(batch_size, z_dim, device=mean_pz.device)
    index = torch.gt(degree_active * ones, torch.mean(torch.exp(log_var_qz), dim=0))

    mean_qz_approx = mean_qz[:, index]
    mean_pz_approx = mean_pz[:, index]
    log_var_qz_approx = log_var_qz[:, index]
    log_var_pz_approx = log_var_pz[:, index]

    kl_approx = torch.mean(torch.sum(torch.exp(log_var_qz_approx - log_var_pz_approx) +
                                     (mean_pz_approx - mean_qz_approx) ** 2 / torch.exp(log_var_pz_approx) -
                                     log_var_qz_approx + log_var_pz_approx, dim=1), dim=0)
    kl = torch.mean(torch.sum(torch.exp(log_var_qz - log_var_pz) +
                              (mean_pz - mean_qz) ** 2 / torch.exp(log_var_pz) -
                              log_var_qz + log_var_pz, dim=1), dim=0)

    delta_kl = torch.abs(kl_approx - kl) / (kl + 1e-8)

    return [delta_kl, kl_approx, kl, index]
