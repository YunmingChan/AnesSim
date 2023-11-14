# AnesSim

- Run your policy on AnesSim

```python
import gym
import AnesSim

env = gym.make('AnesSim-v0', model_path=model_path, args_path=configuration_path)

obs = env.reset()
while True:
    action = your_policy(obs)
    obs, reward, done, info = env.step(action)
    env.render()

    if done:
        break

env.close()

```