import torch
from torch import nn
from torch.nn import functional as F

class AnesSim(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, distr_type=None, *, dropout=0.1, inverse_dynamic=False, device='cpu'):
        super(AnesSim, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.distri_type = distr_type
        self.inverse_dynamic = inverse_dynamic
        self.device = device

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=False,
            dropout=dropout,
            bidirectional=False
        )
        
        if distr_type == 'no':
            self.cal_mu = nn.Linear(hidden_size, output_size)
        elif distr_type == 'tril':
            self.cal_mu = nn.Linear(hidden_size, output_size)
            self.cal_sigma = nn.Sequential(
                nn.Linear(hidden_size, output_size),
                nn.Softplus()
            )
            self.cal_tril = nn.Linear(hidden_size, output_size * (output_size-1) // 2)
        elif distr_type == 'kernel':
            self.cal_mu = nn.Linear(hidden_size, output_size)
            self.emb = nn.ModuleList([nn.Linear(hidden_size, output_size) for _ in range(output_size)])
        else:
            self.cal_mu = nn.Linear(hidden_size, output_size)
            self.cal_sigma = nn.Sequential(
                nn.Linear(hidden_size, output_size),
                nn.Softplus()
            )
        
        if inverse_dynamic:
            self.inverse = nn.Sequential(
                nn.Linear(hidden_size + hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )

    def forward(self, x, h, y):
        o, h = self.lstm(x, h)

        if self.distri_type == 'no':
            sample = self.cal_mu(o)
            loss = F.mse_loss(sample, y)
        else:
            distr = self.get_distribution(o)
            sample = distr.sample()
            loss = -torch.mean(distr.log_prob(y))
        
        if self.inverse_dynamic:
            ss = torch.cat([o[:-1], o[1:]], -1)
            pred_act = self.inverse(ss).squeeze(-1)
            real_act = x[:-1, ..., 5]
            inverse_loss = F.mse_loss(pred_act, real_act)
            loss += inverse_loss

        return sample, h, loss
    
    def reset(self):
        self.h = None
    
    def step(self, state):
        x = torch.tensor(state, device=self.device, dtype=torch.float).reshape(1, -1)
        sample, self.h = self.predict(x, self.h)
        return sample

    def predict(self, x, h, n_samples=0):
        o, h = self.lstm(x, h)
        if n_samples > 0:
            o = o.repeat(n_samples, *[1]*o.dim())
        
        if self.distri_type == 'no':
            sample = self.cal_mu(o)
        else:
            distr = self.get_distribution(o)
            sample = distr.sample()
        return sample, h
    
    def get_distribution(self, o):
        if self.distri_type == 'tril':
            mu = self.cal_mu(o)
            diag = self.cal_sigma(o)
            sigma = torch.diag_embed(diag)
            i = torch.tril_indices(self.output_size, self.output_size, -1)
            sigma[..., i[0], i[1]] = self.cal_tril(o)
            distr = torch.distributions.MultivariateNormal(mu, scale_tril=sigma)
        elif self.distri_type == 'kernel':
            mu = self.cal_mu(o)
            embs = []
            for i in range(len(self.emb)):
                embs.append(self.emb[i](o))
            
            sigma = []
            for i in range(len(self.emb)):
                r = []
                for j in range(len(self.emb)):
                    r.append(-torch.square(embs[i] - embs[j]).sum(-1))
                sigma.append(torch.stack(r, -1).exp())
            sigma = torch.stack(sigma, -1)
            sigma = sigma + torch.diag_embed(torch.ones_like(mu) * 1e-6)
            distr = torch.distributions.MultivariateNormal(mu, sigma)
        else:
            mu = self.cal_mu(o)
            sigma = self.cal_sigma(o)
            distr = torch.distributions.Normal(mu, sigma)
        return distr

    @torch.no_grad()
    def rollout(self, init_state, actions):
        o, h = self.predict(init_state, None)
        s = o[-1]

        static_s = init_state[0, ..., 6:]
        states = [s]
        actions = actions.unsqueeze(-1)
        for a in actions:
            s = torch.cat([s, a, static_s], -1).unsqueeze(0)
            s, h = self.predict(s, h)
            s = s[-1]
            states.append(s)

        return torch.stack(states)
