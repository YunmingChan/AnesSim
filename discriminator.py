import os
import datetime

import json
import argparse
from argparse import Namespace

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from AnesSim.anessim import AnesSim
from dataset import AnesthesiaDataset

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        
        self.net = nn.Sequential(
            nn.Conv1d(input_size, 8, 5),
            nn.ReLU(),
            nn.Conv1d(8, 16, 5),
            nn.ReLU(),
            nn.Conv1d(16, 32, 5),
            nn.ReLU(),
            nn.Conv1d(32, 64, 5),
            nn.ReLU(),
            nn.Conv1d(64, 1, 5),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(80, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x.permute(1, 2, 0))

def train(dataloader, model, optimizer, simulator, device, condition_range=50, state_dim=5, loss_func=nn.BCELoss()):
    num = len(dataloader.dataset)
    model.train()
    train_loss = 0
    correct = 0

    for x in dataloader:
        x = x.to(device).permute(1, 0, 2)

        fake = simulator.rollout(x[:condition_range], x[condition_range:-1, :, state_dim])
        fake = torch.cat([x[:condition_range, :, :state_dim+1], torch.cat([fake, x[condition_range:, :, state_dim].unsqueeze(-1)], -1)], 0)

        real = x[:, :, :state_dim+1]

        pred = model(real)
        loss = loss_func(pred, torch.ones(pred.shape, device=device))
        correct += (pred > 0.5).sum().item()

        pred = model(fake)
        loss += loss_func(pred, torch.zeros(pred.shape, device=device))
        correct += (pred <= 0.5).sum().item()

        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_loss/len(dataloader)/2, correct/num/2

def test(dataloader, model, simulator, device, condition_range=50, state_dim=5, loss_func=nn.BCELoss()):
    num = len(dataloader.dataset)
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for x in dataloader:
            x = x.to(device).permute(1, 0, 2)
            fake = simulator.rollout(x[:condition_range], x[condition_range:-1, :, state_dim])
            fake = torch.cat([x[:condition_range, :, :state_dim+1], torch.cat([fake, x[condition_range:, :, state_dim].unsqueeze(-1)], -1)], 0)


            real = x[:, :, :state_dim+1]
            
            pred = model(real)
            loss = loss_func(pred, torch.ones(pred.shape).to(device))
            correct += (pred > 0.5).sum().item()

            pred = model(fake)
            loss += loss_func(pred, torch.zeros(pred.shape).to(device))
            test_loss += loss.item()
            correct += (pred <= 0.5).sum().item()
        
    return test_loss/len(dataloader)/2, correct/num/2

@torch.no_grad()
def confusion(dataloader, model, simulator, device, condition_range=50, state_dim=5, loss_func=nn.BCELoss()):
    correct = 0
    correctfake = 0
    correctreal = 0

    for x in dataloader:
        x = x.to(device).permute(1, 0, 2)
        fake = simulator.rollout(x[condition_range:], x[condition_range:-1, :, state_dim])
        fake = torch.cat([x[:condition_range, :, :state_dim+1], torch.cat([fake, x[condition_range:, :, state_dim].unsqueeze(-1)], -1)], 0)

        real = x[:, :, :state_dim+1]
        
        pred = model(real)
        loss = loss_func(pred, torch.ones(pred.shape).to(device))
        correct += (pred > 0.5).sum().item()
        correctreal += (pred > 0.5).sum().item()

        pred = model(fake)
        loss += loss_func(pred, torch.zeros(pred.shape).to(device))
        correct += (pred <= 0.5).sum().item()
        correctfake += (pred <= 0.5).sum().item()

    return correct / len(dataloader.dataset) / 2, correctreal / len(dataloader.dataset), correctfake / len(dataloader.dataset)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-4, type=float, help='discriminator learning rate')

    parser.add_argument('--training_dataset_path', default='data/train')
    parser.add_argument('--testing_dataset_path', default='data/test')
    parser.add_argument('--batch_size', default=256)
    parser.add_argument('--stride', default=10, type=int)
    parser.add_argument('--window_size', default=100, type=int)

    parser.add_argument('--input_size', default=11, type=int)
    parser.add_argument('--condition_range', default=8, type=int)

    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--simulator_path', required=True)
    parser.add_argument('--simulator_args_path', default=None)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    if args.simulator_args_path is None:
        path = os.path.dirname(args.simulator_path)
        args.simulator_args_path = os.path.join(path, 'args.txt')
    with open(args.simulator_args_path, 'r') as f:
        sim_args = Namespace()
        sim_args.__dict__.update(json.load(f))
    
    args.device = sim_args.device

    path = os.path.join('model', 'discriminator', f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(path, exist_ok=True)
    logfile = open(os.path.join(path, 'logfile.txt'), 'w+')
    with open(os.path.join(path, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    data_mean = np.array([46.45685828, 79.65172678, 120.60212733, 64.93746456, 84.81941889, 2.4204189, 0, 61.03742107, 162.46704818, 63.87425764, 24.24048627], dtype=np.float32)
    data_std = np.array([9.75071157, 15.96392099, 26.10452866, 19.66076933, 21.40473207, 0.63353726, 1, 12.00777728, 9.05078195, 12.00646209, 4.88932586], dtype=np.float32)

    train_dataset = AnesthesiaDataset(args.training_dataset_path, args.stride, args.window_size, normalize=sim_args.normalize, mean=data_mean, std=data_std)
    test_dataset = AnesthesiaDataset(args.testing_dataset_path, args.stride, args.window_size, normalize=sim_args.normalize, mean=data_mean, std=data_std)
        
    print(len(train_dataset))
    print(len(test_dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    simulator = AnesSim(sim_args.input_size, sim_args.hidden_size, sim_args.num_layers, sim_args.output_size, sim_args.distr_type, dropout=sim_args.dropout, inverse_dynamic=sim_args.inverse_dynamic)
    simulator.load_state_dict(torch.load(args.simulator_path))
    simulator.eval().to(sim_args.device)
    
    model = Discriminator(6).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    logs = []
    acc_log = []
    best_loss = 1e10
    for t in trange(args.epochs):
        train_loss, train_acc = train(train_dataloader, model, optimizer, simulator, args.device, args.condition_range)
        test_loss, test_acc = test(test_dataloader, model, simulator, args.device, args.condition_range)
        print(f"Epoch {t+1:>3d} - Avg train loss: {train_loss:>12.6f}, Avg test loss: {test_loss:>12.6f}, train acc: {train_acc}, test acc: {test_acc}", file=logfile, flush=True)
        logs.append([train_loss, test_loss])
        acc_log.append([train_acc, test_acc])

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), os.path.join(path, 'discriminator.pth'))
    print("Done!")

    np.save(os.path.join(path, 'log.npy'), np.array(logs))
    plt.plot(logs)
    plt.legend(["train_loss", "test_loss"])
    plt.savefig(os.path.join(path, 'train_stats.png'))

    plt.figure()
    np.save(os.path.join(path, 'acc.npy'), np.array(acc_log))
    plt.plot(acc_log)
    plt.legend(["train_acc", "test_acc"])
    plt.savefig(os.path.join(path, 'train_stats_acc.png'))

    acc, acc_real, acc_fake = confusion(test_dataloader, model, simulator, args.device, args.condition_range)
    print(f'Accuracy: {acc}, Accuracy (Real): {acc_real}, Accuracy (Fake): {acc_fake}')
    print(f'Accuracy: {acc}', file=logfile)
    print(f'     | Predict Real | Predict Fake', file=logfile)
    print(f'=================================', file=logfile)
    print(f'Real | {acc_real} | {1-acc_real}', file=logfile)
    print(f'Fake | {1-acc_fake} | {acc_fake}', file=logfile)