#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F
import Helpers
from scipy.special import expit


def loglik_real(batch_data, list_type, theta, normalization_params):
    output = dict()
    epsilon = 1e-6

    data = batch_data
    data_mean, data_var = normalization_params
    data_var = torch.clamp(data_var, min=epsilon)

    est_mean, est_var = theta
    est_var = torch.clamp(F.softplus(est_var), min=epsilon, max=1.0)

    est_mean = torch.sqrt(data_var) * est_mean + data_mean
    est_var = data_var * est_var

    log_p_x = -0.5 * torch.sum((data - est_mean) ** 2 / est_var, dim=1) \
              - int(list_type['dim']) * 0.5 * torch.log(torch.tensor(2 * np.pi, device=data.device)) \
              - 0.5 * torch.sum(torch.log(est_var), dim=1)

    output['log_p_x'] = log_p_x
    output['params'] = [est_mean, est_var]
    normal = torch.distributions.Normal(est_mean, torch.sqrt(est_var))
    output['samples'] = normal.rsample()
    return output


def loglik_pos(batch_data, list_type, theta, normalization_params):
    output = dict()
    epsilon = 1e-6

    data_mean_log, data_var_log = normalization_params
    data_var_log = torch.clamp(data_var_log, min=epsilon)

    data = batch_data
    data_log = torch.log1p(data)

    est_mean, est_var = theta
    est_var = torch.clamp(F.softplus(est_var), min=epsilon, max=1.0)

    est_mean = torch.sqrt(data_var_log) * est_mean + data_mean_log
    est_var = data_var_log * est_var

    log_p_x = -0.5 * torch.sum((data_log - est_mean) ** 2 / est_var, dim=1) \
              - 0.5 * torch.sum(torch.log(2 * np.pi * est_var), dim=1) - torch.sum(data_log, dim=1)

    output['log_p_x'] = log_p_x
    output['params'] = [est_mean, est_var]
    normal = torch.distributions.Normal(est_mean, torch.sqrt(est_var))
    output['samples'] = torch.exp(normal.rsample()) - 1.0
    return output


def loglik_cat(batch_data, list_type, theta, normalization_params):
    output = dict()
    data = batch_data
    log_pi = theta
    log_p_x = -F.cross_entropy(log_pi, torch.argmax(data, dim=1), reduction='none')

    output['log_p_x'] = log_p_x
    output['params'] = log_pi
    cat = torch.distributions.Categorical(probs=F.softmax(log_pi, dim=1))
    output['samples'] = F.one_hot(cat.sample(), num_classes=int(list_type['dim'])).float()
    return output


def loglik_ordinal(batch_data, list_type, theta, normalization_params):
    output = dict()
    epsilon = 1e-6

    data = batch_data
    batch_size = data.shape[0]

    partition_param, mean_param = theta
    mean_value = mean_param.view(-1, 1)
    theta_values = torch.cumsum(torch.clamp(F.softplus(partition_param), min=epsilon), dim=1)
    sigmoid_est_mean = torch.sigmoid(theta_values - mean_value)
    mean_probs = torch.cat([sigmoid_est_mean, torch.ones([batch_size, 1], device=data.device)], dim=1) - \
                 torch.cat([torch.zeros([batch_size, 1], device=data.device), sigmoid_est_mean], dim=1)

    true_values = F.one_hot(torch.sum(data.long(), dim=1) - 1, int(list_type['dim'])).float()
    log_p_x = torch.log(torch.clamp(torch.sum(mean_probs * true_values, dim=1), min=epsilon))

    output['log_p_x'] = log_p_x
    output['params'] = mean_probs
    cat = torch.distributions.Categorical(probs=torch.clamp(mean_probs, min=epsilon))
    samples_idx = cat.sample()
    output['samples'] = F.one_hot(samples_idx + 1, int(list_type['dim'])).cumsum(dim=1)
    return output


def loglik_count(batch_data, list_type, theta, normalization_params):
    output = dict()
    epsilon = 1e-6

    data = batch_data
    est_lambda = torch.clamp(F.softplus(theta), min=epsilon)
    log_p_x = -torch.sum(F.poisson_nll_loss(est_lambda, data, reduction='none', log_input=False, full=True), dim=1)

    output['log_p_x'] = log_p_x
    output['params'] = est_lambda
    pois = torch.distributions.Poisson(est_lambda)
    output['samples'] = pois.sample()
    return output


def loglik_test_real(theta, normalization_params, list_type):
    output = dict()
    epsilon = 1e-6

    data_mean, data_var = normalization_params
    data_var = np.clip(data_var, epsilon, np.inf)

    est_mean, est_var = theta
    soft_plus_est_var = np.log(1 + np.exp(-np.abs(est_var))) + np.maximum(est_var, 0)
    est_var = np.clip(soft_plus_est_var, epsilon, 1.0)

    est_mean = np.sqrt(data_var) * est_mean + data_mean
    est_var = data_var * est_var

    output['samples'] = np.random.normal(est_mean, np.sqrt(est_var))
    output['params'] = [est_mean, est_var]
    return output


def loglik_test_pos(theta, normalization_params, list_type):
    output = dict()
    epsilon = 1e-6

    data_mean_log, data_var_log = normalization_params
    data_var_log = np.clip(data_var_log, epsilon, np.inf)

    est_mean, est_var = theta
    soft_plus_est_var = np.log(1 + np.exp(-np.abs(est_var))) + np.maximum(est_var, 0)
    est_var = np.clip(soft_plus_est_var, epsilon, 1.0)

    est_mean = np.sqrt(data_var_log) * est_mean + data_mean_log
    est_var = data_var_log * est_var

    output['samples'] = np.exp(np.random.normal(est_mean, np.sqrt(est_var))) - 1.0
    output['params'] = [est_mean, est_var]
    return output


def loglik_test_cat(theta, normalization_params, list_type):
    output = dict()
    log_pi = theta

    est_cat = Helpers.cat_sample(log_pi)
    estimated_samples = Helpers.indices_to_one_hot(est_cat, int(list_type['dim']))

    output['samples'] = estimated_samples
    output['params'] = log_pi
    return output


def loglik_test_ordinal(theta, normalization_params, list_type):
    output = dict()
    epsilon = 1e-6

    partition_param, mean_param = theta
    batch_size = mean_param.shape[0]

    mean_value = mean_param.reshape(-1, 1)
    soft_plus_partition_param = np.log(1 + np.exp(-np.abs(partition_param))) + np.maximum(partition_param, 0)

    theta_values = np.cumsum(np.clip(soft_plus_partition_param, epsilon, 1e20), axis=1)
    sigmoid_est_mean = expit(theta_values - mean_value)
    mean_probs = np.c_[sigmoid_est_mean, np.ones(batch_size)] - np.c_[np.zeros(batch_size), sigmoid_est_mean]
    mean_probs = np.clip(mean_probs, epsilon, 1e20)

    mean_logits = np.log(mean_probs / (1 - mean_probs))

    pseudo_cat = 1 + Helpers.cat_sample(mean_logits)

    output['samples'] = Helpers.sequence_mask(pseudo_cat, batch_size, int(list_type['dim']))
    output['params'] = mean_probs
    return output
