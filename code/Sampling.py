import numpy as np
import torch
import time
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

import Helpers
import Evaluation
import Graph
from Decoder import decoder_test_time


np.random.seed(619)


def print_loss(epoch, start_time, avg_loss, avg_KL_s, avg_KL_z):
    print("Epoch: [%2d]  time: %4.4f, train_loglik: %.8f, KL_z: %.8f, KL_s: %.8f, ELBO: %.8f"
          % (epoch, time.time() - start_time, avg_loss, avg_KL_z, avg_KL_s, avg_loss - avg_KL_z - avg_KL_s))


def _split_tensor(batch_tensor, types_dict):
    output = []
    idx = 0
    for t in types_dict:
        dim = int(t['dim'])
        output.append(batch_tensor[:, idx:idx + dim])
        idx += dim
    return output


def sampling(settings, types_dict, types_dict_c, out, ncounterfactuals, clf, n_batches_train, n_samples_train, k, n_input, degree_active):
    argvals = settings.split()
    args = Helpers.getArgs(argvals)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Graph.C_CHVAE_graph(args.types_file, args.types_file_c,
                                learning_rate=1e-3, z_dim=args.dim_latent_z,
                                y_dim=args.dim_latent_y, s_dim=args.dim_latent_s,
                                y_dim_partition=args.dim_latent_y_partition, nsamples=1000, p=2).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_data = out['training'][1]
    train_data_c = out['training'][2]

    start_time = time.time()
    for epoch in tqdm(range(args.epochs)):
        avg_loss = 0.0
        avg_KL_s = 0.0
        avg_KL_z = 0.0

        tau = max(1.0 - 0.001 * epoch, 1e-3)
        random_perm = np.random.permutation(range(np.shape(train_data)[0]))
        train_data_aux = train_data[random_perm, :]
        train_data_aux_c = train_data_c[random_perm, :]

        for i in range(n_batches_train):
            batch_np = Helpers.next_batch(train_data_aux, types_dict, args.batch_size, index_batch=i)
            batch_np_c = Helpers.next_batch(train_data_aux_c, types_dict_c, args.batch_size, index_batch=i)

            batch = torch.from_numpy(np.concatenate(batch_np, axis=1)).to(device=device, dtype=torch.float32)
            batch_c = torch.from_numpy(np.concatenate(batch_np_c, axis=1)).to(device=device, dtype=torch.float32)

            x_list = _split_tensor(batch, types_dict)
            x_list_c = _split_tensor(batch_c, types_dict_c)

            model.train()
            optimizer.zero_grad()
            outputs = model(x_list, tau, x_list_c)
            loss = -outputs['ELBO']
            loss.backward()
            optimizer.step()

            avg_loss += outputs['loss_reconstruction'].detach().cpu().numpy().mean()
            avg_KL_s += outputs['KL_s'].detach().cpu().numpy().mean()
            avg_KL_z += outputs['KL_z'].detach().cpu().numpy().mean()

        if epoch % args.display == 0:
            print_loss(epoch, start_time, avg_loss / n_batches_train, avg_KL_s / n_batches_train,
                       avg_KL_z / n_batches_train)

    model.eval()

    test_data_counter = out['test_counter'][1]
    test_data_c_counter = out['test_counter'][2]
    y_test_counter = out['test_counter'][3]
    n_samples_test = test_data_counter.shape[0]

    data_list = Helpers.next_batch(test_data_counter, types_dict, n_samples_test, index_batch=0)
    data_list_c = Helpers.next_batch(test_data_c_counter, types_dict_c, n_samples_test, index_batch=0)
    tau = 1e-3

    test_batch = torch.tensor(np.concatenate(data_list, axis=1), dtype=torch.float32, device=device)
    test_batch_c = torch.tensor(np.concatenate(data_list_c, axis=1), dtype=torch.float32, device=device)
    X_list = _split_tensor(test_batch, types_dict)
    X_list_c = _split_tensor(test_batch_c, types_dict_c)

    with torch.no_grad():
        normalized_data, normalization_params, noisy_data = Helpers.batch_normalization(X_list, types_dict, test_batch.shape[0])
        samples, q_params = model.encoder(noisy_data, tau, X_list_c, deterministic_s=True)
        z_total_test = q_params['z'][0].cpu().numpy()
        theta, _ = model.decoder.decode_only(samples['z'])
        log_p_x, samples_total_test, test_params_x = Evaluation.loglik_evaluation(normalized_data, types_dict, theta, normalization_params)

    samples_total_np = [s.detach().cpu().numpy() for s in samples_total_test]
    est_samples_transformed = Helpers.discrete_variables_transformation(np.concatenate(samples_total_np, axis=1), types_dict)

    counter_batch_size = 1
    data_concat = []
    data_concat_c = []
    counterfactuals = []
    latent_tilde = []
    latent = []

    search_samples = args.search_samples
    p = args.norm_latent_space

    scaler_test = StandardScaler().fit(test_data_counter)
    scaled_test = scaler_test.transform(test_data_counter)

    for i in tqdm(range(ncounterfactuals)):
        s = np.zeros((k, n_input))
        sz = np.zeros((k, args.dim_latent_z))
        ik = 0

        l = 0
        step = args.step_size
        test_data_c_replicated = np.repeat(test_data_c_counter[i, :].reshape(1, -1), search_samples, axis=0)
        replicated_scaled_test = np.repeat(scaled_test[i, :].reshape(1, -1), search_samples, axis=0)

        replicated_data_list = Helpers.replicate_data_list(Helpers.next_batch(test_data_counter, types_dict, counter_batch_size, index_batch=i), search_samples)
        replicated_data_list_c = Helpers.replicate_data_list(Helpers.next_batch(test_data_c_counter, types_dict_c, counter_batch_size, index_batch=i), search_samples)
        replicated_z = np.repeat(z_total_test[i].reshape(-1, args.dim_latent_z), search_samples, axis=0)

        h = l + step
        count = 0
        counter_step = 1
        max_step = 500

        while True:
            count += counter_step
            if count > max_step:
                sz = None
                s = None
                z = z_total_test[i].reshape(-1, args.dim_latent_z)
                break

            delta_z = np.random.randn(search_samples, replicated_z.shape[1])
            d = np.random.rand(search_samples) * (h - l) + l
            norm_p = np.linalg.norm(delta_z, ord=p, axis=1)
            d_norm = np.divide(d, norm_p).reshape(-1, 1)
            delta_z = np.multiply(delta_z, d_norm)
            z_tilde = replicated_z + delta_z

            z_tilde_tensor = torch.tensor(z_tilde, dtype=torch.float32, device=device)
            theta_perturbed, _ = decoder_test_time(model.decoder, z_tilde_tensor)
            theta_np = []
            for t in theta_perturbed:
                if isinstance(t, list):
                    theta_np.append([p.detach().cpu().numpy() for p in t])
                else:
                    theta_np.append(t.detach().cpu().numpy())
            normalization_params_np = [(m.detach().cpu().numpy(), v.detach().cpu().numpy()) for m, v in normalization_params]
            x_tilde, params_x_perturbed = Evaluation.loglik_evaluation_test(X_list,
                                                                            theta_np,
                                                                            normalization_params_np,
                                                                            types_dict)
            x_tilde = np.concatenate(x_tilde, axis=1)
            scaled_tilde = scaler_test.transform(x_tilde)
            d_scale = np.sum(np.abs(scaled_tilde - replicated_scaled_test), axis=1)

            x_tilde_full = np.c_[test_data_c_replicated, x_tilde]
            y_tilde = clf.predict(x_tilde_full)

            indices_adv = np.where(y_tilde == 0)[0]

            if len(indices_adv) == 0:
                l = h
                h = l + step
            elif all(s[k - 1, :] == 0):
                indx = indices_adv[np.argmin(d_scale[indices_adv])]
                s[ik, :] = x_tilde_full[indx, :]
                sz[ik, :] = z_tilde[indx, :]
                z = z_total_test[i].reshape(-1, args.dim_latent_z)
                ik += 1
                l = h
                h = l + step
            else:
                break

        data_concat.append(np.concatenate(Helpers.next_batch(test_data_counter, types_dict, counter_batch_size, index_batch=i), axis=1))
        data_concat_c.append(np.concatenate(Helpers.next_batch(test_data_c_counter, types_dict_c, counter_batch_size, index_batch=i), axis=1))
        counterfactuals.append(s)
        latent_tilde.append(sz)
        latent.append(z)

    cchvae_counterfactuals = np.array(counterfactuals)
    return cchvae_counterfactuals
