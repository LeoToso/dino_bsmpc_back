"""
Standalone evaluation for a saved TD-MPC checkpoint -- reports success rate
in addition to episode reward (the reference implementation's own evaluate(),
mirrored in train_tdmpc.py, only reports reward; TD-MPC's paper has no notion
of a binary success metric since it's pure reward-maximization).

"Success" is defined using PushT's own built-in task criterion: the env's
step() reward is coverage/success_threshold clipped to [0, 1], so reward
reaching 1.0 at any point in the episode means the block fully covered the
target -- same coverage-based objective the environment already reports.

Usage:
    python eval_tdmpc.py eval_checkpoint=/path/to/model_final.pth
    python eval_tdmpc.py eval_checkpoint=/path/to/model_3000.pth eval_episodes=20 visual_condition=C
"""
import warnings
warnings.filterwarnings("ignore")

import hydra
import numpy as np
import torch

from tdmpc.agent import TDMPC
from tdmpc.env_wrapper import make_env
from utils import seed as seed_all


@torch.no_grad()
def evaluate(env, agent, num_episodes, success_threshold=1.0):
    episode_rewards = []
    successes = []
    for i in range(num_episodes):
        obs, done, ep_reward, max_reward, t = env.reset(), False, 0.0, 0.0, 0
        while not done:
            # step=int(1e9) assumes the std/horizon schedules have fully annealed
            # (i.e. evaluating a converged checkpoint, not one mid-schedule).
            action = agent.plan(obs, eval_mode=True, step=int(1e9), t0=(t == 0))
            obs, reward, done, _ = env.step(action.cpu().numpy())
            ep_reward += reward
            max_reward = max(max_reward, reward)
            t += 1
        episode_rewards.append(ep_reward)
        successes.append(bool(max_reward >= success_threshold))
        print(f"  episode {i + 1}/{num_episodes}: reward={ep_reward:.2f}, success={successes[-1]}")

    return {
        "mean_episode_reward": float(np.mean(episode_rewards)),
        "success_rate": float(np.mean(successes)),
        "episode_rewards": episode_rewards,
        "successes": successes,
    }


@hydra.main(config_path="conf", config_name="tdmpc")
def main(cfg):
    assert torch.cuda.is_available(), "TD-MPC requires a CUDA-enabled device"
    assert cfg.eval_checkpoint, "Set eval_checkpoint=/path/to/model_*.pth"
    seed_all(cfg.seed)

    env = make_env(cfg)
    agent = TDMPC(cfg)
    agent.load(cfg.eval_checkpoint)

    print(f"Evaluating {cfg.eval_checkpoint} under visual_condition={cfg.visual_condition} "
          f"for {cfg.eval_episodes} episodes...")
    results = evaluate(env, agent, cfg.eval_episodes)

    print(f"\nSuccess rate: {results['success_rate']:.2%}")
    print(f"Mean episode reward: {results['mean_episode_reward']:.2f}")


if __name__ == "__main__":
    main()
