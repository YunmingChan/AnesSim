import os
import json
import torch
import matplotlib.pyplot as plt
import numpy as np

import gym
from gym import spaces
from PIL import Image
import seaborn as sns
from argparse import Namespace
from AnesSim.anessim import AnesSim

def simple_reward(x):
    bis = x[0]
    if 40 <= bis and bis <= 60:
        return 1
    return -5

class Renderer():
    def __init__(self):
        self.fig = None

    def show(self):
        self.fig.show()
    
    def reset(self):
        plt.close(self.fig)
        plt.rcParams["font.size"] = 14
        self.fig, self.ax = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(6, 6), dpi=120)
        self.imgs = []

    def render(self, step, history):
        self.history = history

        x = np.arange(step-100, step)
        k = min(step, 100)
        
        self.ax[0].clear()
        self.ax[0].plot(x[-k:], history[-k:, :5])
        self.ax[0].legend(['BIS', 'HR', 'SBP', 'DBP', 'MBP'], bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=5)

        self.ax[1].clear()
        self.ax[1].plot(x[-k:], history[-k:, 5])
        self.ax[1].set_xlabel('Time (min)')
        self.ax[1].set_ylabel('Propofol')
        # self.fig.tight_layout()

        if step < 100:
            self.ax[0].set_xbound(-5, 105)
            self.ax[1].set_xbound(-5, 105)
            
        self.fig.canvas.draw()
        self.fig.canvas.start_event_loop(0.01)

        img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        self.imgs.append(img)
    
    def close(self):
        plt.close(self.fig)
        if len(self.imgs) > 0:
            self.imgs = [Image.fromarray(img) for img in self.imgs]
            self.imgs[0].save('simulation.gif', save_all=True, append_images=self.imgs[1:], duration=100, loop=0)
            
            history = self.history
            fig, ax = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(6, 6), dpi=120)

            sns.lineplot(history[:, :5], ax=ax[0], errorbar=None)
            ax[0].legend(['BIS', 'HR', 'SBP', 'DBP', 'MBP'], bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=5)

            sns.lineplot(history[:, 5], ax=ax[1])
            ax[1].set_xlabel('Time (min)')
            ax[1].set_ylabel('Propofol')
            
            fig.tight_layout()
            fig.savefig('history.png')
            plt.close(fig)

class Simulator(gym.Env):
    def __init__(self, model_path, args_path, reward_func=None, render_mode='human'):
        with open(args_path, 'r') as f:
            sim_args = Namespace()
            sim_args.__dict__.update(json.load(f))

        self.sim_model = AnesSim(
            sim_args.input_size, 
            sim_args.hidden_size, 
            sim_args.num_layers, 
            sim_args.output_size, 
            sim_args.distr_type, 
            inverse_dynamic=sim_args.inverse_dynamic,
            device=sim_args.device,
        )
        self.sim_model.load_state_dict(torch.load(model_path))
        self.sim_model.eval().to(sim_args.device)
        self.sim_model.reset()

        self.render_mode = render_mode
        self.normalized = sim_args.normalize

        self.data_mean = np.array([46.45685828, 79.65172678, 120.60212733, 64.93746456, 84.81941889, 2.4204189, 0, 61.03742107, 162.46704818, 63.87425764, 24.24048627], dtype=np.float32)
        self.data_std = np.array([9.75071157, 15.96392099, 26.10452866, 19.66076933, 21.40473207, 0.63353726, 1, 12.00777728, 9.05078195, 12.00646209, 4.88932586], dtype=np.float32)

        """[bis, hr, sbp, dbp, mbp, gender, age, height, weight, BMI]"""
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(10,), dtype=np.float32)

        if sim_args.normalize:
            self.action_space = spaces.Box(-1, 1, (1,), dtype=np.float32)
        else:
            self.action_space = spaces.Box(1, 3, (1,), dtype=np.float32)

        self.renderer = Renderer()

        if reward_func is not None:
            self.reward_func = reward_func
        else:
            self.reward_func = simple_reward

    def _get_obs(self):
        return self.state[[0,1,2,3,4,6,7,8,9,10]]

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.n_step = 0
        self.history = None

        state = np.random.normal(
            [46.45685828,  79.65172678, 120.60212733,  64.93746456,  84.81941889, 2.4204189], 
            [9.75071157, 15.96392099, 26.10452866, 19.66076933, 21.40473207, 0.63353726]
        )
        gender = np.array([np.random.randint(2)])
        static = np.random.normal(
            [61.03742107, 162.46704818,  63.87425764],
            [12.00777728,  9.05078195, 12.00646209]
        )
        bmi = np.array([static[2] / (static[1] / 100) ** 2])
        
        sample = np.concatenate([state, gender, static, bmi])
        if self.normalized:
            sample = (sample - self.data_mean) / self.data_std
        self.state = sample

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.renderer.reset()

        return observation

    def step(self, action):
        self.n_step += 1
        if self.normalized:
            action = np.clip(action, -1, 1)
            action = action * 1.621 - 0.621
        else:
            action = np.clip(action, 1, 3)
        self.state[5] = action

        if self.history is None:
            self.history = self.state[:6].reshape(1, -1)
        else:
            self.history = np.append(self.history, self.state[:6].reshape(1, -1), axis=0)

        next_state = self.state.copy()
        next_state[:5] = self.sim_model.step(self.state).detach().cpu().numpy()
        self.state = next_state

        observation = self._get_obs()
        info = self._get_info()
        reward = self.reward_func(self.state * self.data_std + self.data_mean if self.normalized else self.state)

        return observation, reward, False, info

    def render(self, mode, **kwargs):
        if self.n_step <= 1:
            self.renderer.show()
        history = self.history.copy()
        if self.normalized:
            history = history * self.data_std[:6] + self.data_mean[:6]
        self.renderer.render(self.n_step, history)

    def close(self):
        self.renderer.close()