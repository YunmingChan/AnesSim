import os

import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import pandas as pd
import seaborn as sns

from tqdm import tqdm, trange

import torch
from AnesSim.anessim import AnesSim

### Evalution

def cal_error(pred, y, *, normalize=False, data_mean=None, data_std=None):
    if normalize:
        pred = pred * data_std + data_mean
        y = y * data_std + data_mean

    ae = np.abs(pred - y)
    se = ((pred - y)**2)
    pe = np.divide(np.abs(y - pred), y, out=np.zeros_like(y), where=np.abs(y)>1e-5)
    ape = np.abs(y - pred) / ((np.abs(y) + np.abs(pred)) / 2)
    return np.stack([ae, se, pe, ape])

@torch.no_grad()
def cal_crps(samples, y, qs=20, *, normalize=False, data_mean=None, data_std=None):
    n_sample = len(samples)
    if normalize:
        samples = samples * data_std + data_mean
        y = y * data_std + data_mean
    
    def quantile_loss(q, quantile, y):
        return (q - (y<quantile)) * (y-quantile)
    
    crps_m = 0
    samples_m = np.sort(samples, 0)
    for q in range(1, qs+1):
        crps_m += 2 * quantile_loss(q/qs, samples_m[round((n_sample-1)*q/qs)], y)
    crps_m = crps_m.sum() / qs / len(y)

    crps_sum = 0
    y_sum = y.sum()
    samples_sum = np.sort(samples.sum(-1))
    for q in range(1, qs+1):
        crps_sum += 2 * quantile_loss(q/qs, samples_sum[round((n_sample-1)*q/qs)], y_sum)
    
    return np.array([crps_m, crps_sum/qs])

@torch.no_grad()
def evaluate(model, dataset_path, device, start, *, normalize=False, data_mean=None, data_std=None, crps_n_samples=200, state_dim=5):
    model.eval()
    test_files = [os.path.join(dataset_path, filename) for filename in os.listdir(dataset_path)]
    total_err = []

    for test_data in tqdm(test_files):
        df = pd.read_excel(test_data)
        data = df.iloc[:, 1:].to_numpy(dtype=np.float32)

        if normalize:
            data = (data - data_mean) / data_std

        h = None
        
        crps = np.zeros((2, ))
        err = np.zeros((4, state_dim))
        for t in range(len(data)-1):
            if t < start:
                pstate = data[t].copy()
                state = torch.tensor(pstate).to(device).float().unsqueeze(0)
                sample, h = model.predict(state, h)
                s = sample.detach().cpu().numpy().squeeze()

            else:
                pstate = np.concatenate([s, data[t, state_dim:]])
                state = torch.tensor(pstate).to(device).float().unsqueeze(0)
                
                ### CRPS
                samples, _ = model.predict(state, h, crps_n_samples)
                samples = samples.squeeze().cpu().numpy()
                crps += cal_crps(samples, data[t+1, :state_dim], normalize=normalize, data_mean=data_mean[:state_dim], data_std=data_std[:state_dim])

                sample, h = model.predict(state, h)
                s = sample.detach().cpu().numpy().squeeze()
                err += cal_error(s, data[t+1, :state_dim], normalize=normalize, data_mean=data_mean[:state_dim], data_std=data_std[:state_dim])
            
        crps /= (len(data) - start)
        err = np.mean(err / (len(data) - start), 1)
        total_err.append([err[0], err[1], err[1]**0.5, err[2], err[3], crps[0], crps[1]])
    
    total_err = np.array(total_err).mean(0)
    print(f'MAE: {total_err[0]} | MSE: {total_err[1]} | RMSE: {total_err[2]} | MAPE: {total_err[3]} | sMAPE: {total_err[4]} | CRPS: {total_err[5]} | CRPS-Sum: {total_err[6]}')
    return total_err

### Simulation

def simple_reward(x, normalize=False, data_mean=None, data_std=None):
    if normalize:
        x = x.detach().cpu().numpy()*data_std + data_mean
    if 40 <= x[0] and x[0] <= 60:
        return 1
    return -5

@torch.no_grad()
def simulate(model, length, device, *, static_action=2, normalize=False, data_mean=None, data_std=None, plot=False):
    if normalize:
        state = np.random.normal(0, 1, 11)
        static_action = ((static_action - data_mean[5]) / data_std[5])
        state[5] = static_action
    else:
        state = np.random.normal(data_mean, data_std)
        state[5] = static_action
    
    state = torch.from_numpy(state).to(device).type(torch.float32)
    history = [state]
    ep_reward = simple_reward(state)
    
    h = None
    for i in range(length):
        state = state.reshape(1, 1, 11)
        next_state, h = model.predict(state, h)
        action = torch.tensor([static_action], device=device, dtype=torch.float)
        state = torch.cat([next_state.flatten(), action, state[..., 6:].flatten()])
        history.append(state)
        ep_reward += simple_reward(state, normalize, data_mean, data_std)
        
    history = torch.stack(history).squeeze().cpu().numpy()

    if normalize:
        history = history * data_std + data_mean
    
    if plot:
        print(history.shape)
        plt.plot(history[:, :5])
        plt.show()
    
    return history, ep_reward

def traj_distr(k, model, length, device, *, static_action=2, normalize=False, data_mean=None, data_std=None):
    trajs = []
    rewards = []
    for _ in trange(k):
        traj, reward = simulate(model, length, device, static_action=static_action, normalize=normalize, data_mean=data_mean, data_std=data_std)
        trajs.append(traj)
        rewards.append(reward)
    trajs = np.array(trajs)
    rewards = np.array(rewards)

    d = {'traj': trajs, 'reward': rewards}
    np.save('simulations.npy', d)

@torch.no_grad()
def recon(model, normalize, device, start):
    model.eval()
    state_dim = 5
    path = 'data/test'
    test_files = [os.path.join(path, filename) for filename in os.listdir(path)]
    n = np.random.randint(len(test_files))
    print(f'Test file: {n}, {test_files[n]}')
    df = pd.read_excel(test_files[n])
    data = df.iloc[:, 1:].to_numpy(dtype=np.float32)

    if normalize:
        data = (data - data_mean) / data_std

    sim = data[:, :6].copy()
    h = None
    for t in range(len(data)-1):
        if t < start:
            pstate = data[t].copy()
            state = torch.tensor(pstate).to(device).float().unsqueeze(0)
            sample, h = model.predict(state, h)
            s = sample.detach().cpu().numpy().squeeze()
            sim[t+1, :5] = s.copy()

        else:
            pstate = np.concatenate([s, data[t, state_dim:]])
            state = torch.tensor(pstate).to(device).float().unsqueeze(0)

            sample, h = model.predict(state, h)
            s = sample.detach().cpu().numpy().squeeze()
            sim[t+1, :5] = s.copy()
    
    if normalize:
        data = data * data_std + data_mean
        sim = sim * data_std[:6] + data_mean[:6]
        
    plt.rcParams["font.size"] = 14
    fig, ax = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(6, 6), dpi=120)

    sns.lineplot(sim[:, :5], ax=ax[0], errorbar=None)
    ax[0].legend(['BIS', 'HR', 'SBP', 'DBP', 'MBP'], bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=5)

    sns.lineplot(sim[:, 5], ax=ax[1])
    ax[1].set_xlabel('Time (min)')
    ax[1].set_ylabel('Propofol')
    
    fig.tight_layout()
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='path to saved model')
    parser.add_argument('--args_path', help='path to saved model arguments. Default: model_dir/args.txt')

    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--evaluate_start', default=8, type=int)
    
    parser.add_argument('--simulate', action='store_true')
    parser.add_argument('--simulate_length', default=500, help='episode length')
    parser.add_argument('--simulate_plot', action='store_true')
    parser.add_argument('--simulate_action', type=float, help='static input action on simulator')
    parser.add_argument('--simulate_num', default=1, type=int)

    parser.add_argument('--seed', default=None, type=int)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    if args.args_path is None:
        path = os.path.dirname(args.model_path)
        args.args_path = os.path.join(path, 'args.txt')
    with open(args.args_path, 'r') as f:
        args.__dict__.update(json.load(f))

    data_mean = np.array([46.45685828, 79.65172678, 120.60212733, 64.93746456, 84.81941889, 2.4204189, 0, 61.03742107, 162.46704818, 63.87425764, 24.24048627], dtype=np.float32)
    data_std = np.array([9.75071157, 15.96392099, 26.10452866, 19.66076933, 21.40473207, 0.63353726, 1, 12.00777728, 9.05078195, 12.00646209, 4.88932586], dtype=np.float32)

    model = AnesSim(
        args.input_size, 
        args.hidden_size, 
        args.num_layers, 
        args.output_size, 
        args.distr_type, 
        dropout=args.dropout, 
        inverse_dynamic=args.inverse_dynamic,
        device=args.device
    )
    model.load_state_dict(torch.load(args.model_path))
    model.eval().to(args.device)

    if args.evaluate:
        evaluate(model, args.testing_dataset_path, args.device, args.evaluate_start, normalize=args.normalize, data_mean=data_mean, data_std=data_std)
    elif args.simulate:
        if args.simulate_num > 1:
            traj_distr(args.simulate_num, model, args.simulate_length, args.device, static_action=args.simulate_action, normalize=args.normalize, data_mean=data_mean, data_std=data_std)
        else:
            simulate(model, args.simulate_length, args.device, static_action=args.simulate_action, normalize=args.normalize, data_mean=data_mean, data_std=data_std, plot=args.simulate_plot)
    else:
        recon(model, args.normalize, args.device, args.evaluate_start)
