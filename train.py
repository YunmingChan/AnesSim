import os
import time
import datetime

import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import trange

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from AnesSim.anessim import AnesSim
from dataset import AnesthesiaDataset

def train(dataloader, model, optimizer):
    num = len(dataloader)
    model.train()
    train_loss = 0
    for x in dataloader:
        x = x.to(device).permute(1, 0, 2)
        _, _, loss = model(x[:-1], None, x[1:, :, :5])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        
        train_loss += loss.item()

    train_loss /= num
    return train_loss

@torch.no_grad()
def test(dataloader, model):
    num = len(dataloader)
    model.eval()
    test_loss = 0
    for x in dataloader:
        x = x.to(device).permute(1, 0, 2)
        _, _, loss = model(x[:-1], None, x[1:, :, :5])
        test_loss += loss.item()
        
    test_loss /= num
    return test_loss

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--wd', default=1e-6, type=float, help='weight decay')

    parser.add_argument('--inverse_dynamic', action='store_true')

    parser.add_argument('--training_dataset_path', default='data/train')
    parser.add_argument('--testing_dataset_path', default='data/test')
    parser.add_argument('--batch_size', default=256)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--stride', default=1, type=int)
    parser.add_argument('--window_size', default=100, type=int)

    parser.add_argument('--input_size', default=11, type=int)
    parser.add_argument('--output_size', default=5, type=int)
    parser.add_argument('--hidden_size', default=128, type=int)
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('--distr_type', default=None, choices=['no', 'deepar', 'tril', 'kernel'])
    parser.add_argument('--dropout', default=0.1, type=float)
    
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--save_model_frequency', default=25, type=int)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    args.device = device

    path = os.path.join('model', f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, 'checkpoints'), exist_ok=True)
    logfile = open(os.path.join(path, 'logfile.txt'), 'w+')
    with open(os.path.join(path, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)
    
    data_mean = np.array([46.45685828, 79.65172678, 120.60212733, 64.93746456, 84.81941889, 2.4204189, 0, 61.03742107, 162.46704818, 63.87425764, 24.24048627], dtype=np.float32)
    data_std = np.array([9.75071157, 15.96392099, 26.10452866, 19.66076933, 21.40473207, 0.63353726, 1, 12.00777728, 9.05078195, 12.00646209, 4.88932586], dtype=np.float32)

    train_dataset = AnesthesiaDataset(args.training_dataset_path, args.stride, args.window_size, normalize=args.normalize, mean=data_mean, std=data_std)
    test_dataset = AnesthesiaDataset(args.testing_dataset_path, args.stride, args.window_size, normalize=args.normalize, mean=data_mean, std=data_std)
        
    print(len(train_dataset))
    print(len(test_dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    model = AnesSim(
        args.input_size,
        args.hidden_size,
        args.num_layers,
        args.output_size,
        args.distr_type,
        dropout=args.dropout,
        inverse_dynamic=args.inverse_dynamic,
        device=device
    ).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    logs = []
    best_loss = 1e10
    ctime = time.time()
    for t in trange(args.epochs):
        train_loss = train(train_dataloader, model, optimizer)
        test_loss = test(test_dataloader, model)

        print(f"Epoch {t+1:>3d} ({time.time()-ctime:6.2f}) - Avg train loss: {train_loss:>12.6f}, Avg test loss: {test_loss:>12.6f}", file=logfile, flush=True)
        ctime = time.time()
        logs.append([train_loss, test_loss])

        if t % args.save_model_frequency == 0:
            torch.save(model.state_dict(), os.path.join(path, 'checkpoints', f'epoch{t}.pth'))
            np.save(os.path.join(path, 'log.npy'), np.array(logs))

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), os.path.join(path, 'best.pth'))

    torch.save(model.state_dict(), os.path.join(path, 'final.pth'))

    np.save(os.path.join(path, 'log.npy'), np.array(logs))
    plt.plot(logs)
    plt.legend(["train_loss", "test_loss"])
    plt.savefig(os.path.join(path, 'train_stats.png'))

