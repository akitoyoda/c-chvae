import torch
import Evaluation


def samples_generator(model, batch_data_list, X_list, types_list, tau, normalization_params):
    model.eval()
    with torch.no_grad():
        out = model.generate(X_list, tau=tau)
        samples_test = out['samples']
        test_params = {'x': out['params_x']}
        log_p_x = out['log_p_x']
        theta = out['theta']
    return samples_test, test_params, log_p_x, theta


def samples_generator_c(model, batch_data_list, X_list, X_list_c, types_list, tau, normalization_params):
    model.eval()
    with torch.no_grad():
        out = model.generate(X_list, tau=tau, x_list_c=X_list_c)
        samples_test = out['samples']
        test_params = {'x': out['params_x']}
        log_p_x = out['log_p_x']
        theta = out['theta']
    return samples_test, test_params, log_p_x, theta
