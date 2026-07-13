"""
TD-MPC baseline training entrypoint, adapted from nicklashansen/tdmpc's
src/train.py to this repo's Hydra conventions and the PushT environment.

Usage:
    python train_tdmpc.py
    python train_tdmpc.py train_steps=1000000 exp_name=my_run
"""
import warnings
warnings.filterwarnings("ignore")

import logging
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb

from tdmpc.agent import TDMPC
from tdmpc.env_wrapper import make_env
from tdmpc.helper import Episode, ReplayBuffer
from utils import cfg_to_dict, seed as seed_all

log = logging.getLogger(__name__)


def _resolve_ckpt_base_path(ckpt_base: str) -> Path:
    """Relative paths are resolved from the process cwd where the user launched
    this script (Hydra's run dir is outputs/..., not the repo root)."""
    p = Path(ckpt_base)
    if p.is_absolute():
        return p
    return Path(hydra.utils.get_original_cwd()) / p


@torch.no_grad()
def evaluate(env, agent, num_episodes, step):
    """Evaluate a trained agent (no exploration noise, deterministic policy trajectories)."""
    episode_rewards = []
    for _ in range(num_episodes):
        obs, done, ep_reward, t = env.reset(), False, 0.0, 0
        while not done:
            action = agent.plan(obs, eval_mode=True, step=step, t0=(t == 0))
            obs, reward, done, _ = env.step(action.cpu().numpy())
            ep_reward += reward
            t += 1
        episode_rewards.append(ep_reward)
    return float(np.nanmean(episode_rewards))


@hydra.main(config_path="conf", config_name="tdmpc")
def main(cfg):
    assert torch.cuda.is_available(), "TD-MPC requires a CUDA-enabled device (matches the reference implementation)"
    seed_all(cfg.seed)

    ckpt_dir = _resolve_ckpt_base_path(cfg.ckpt_base_path) / "outputs" / "tdmpc" / cfg.exp_name / str(cfg.seed)
    (ckpt_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    log.info(f"Checkpoints will be saved to {ckpt_dir / 'checkpoints'}")

    if cfg.wandb_logging:
        wandb_run = wandb.init(
            project="tdmpc_pusht", config=cfg_to_dict(cfg), name=f"{cfg.exp_name}_{cfg.seed}"
        )
    else:
        wandb_run = None

    env = make_env(cfg)
    agent = TDMPC(cfg)
    buffer = ReplayBuffer(cfg)

    episode_idx, start_time = 0, time.time()
    for step in range(0, cfg.train_steps + cfg.episode_length, cfg.episode_length):
        # Collect trajectory using TD-MPC planning (Algorithm 1)
        obs = env.reset()
        episode = Episode(cfg, obs)
        while not episode.done:
            action = agent.plan(obs, step=step, t0=episode.first)
            obs, reward, done, _ = env.step(action.cpu().numpy())
            episode += (obs, action, reward, done)
        assert len(episode) == cfg.episode_length, (
            f"episode ran for {len(episode)} steps, expected {cfg.episode_length} -- "
            "check that the 'pusht' gym registration's max_episode_steps (env/__init__.py) "
            "matches conf/tdmpc.yaml's episode_length"
        )
        buffer += episode

        # Update TOLD model (Algorithm 2)
        train_metrics = {}
        if step >= cfg.seed_steps:
            num_updates = cfg.seed_steps if step == cfg.seed_steps else cfg.episode_length
            for i in range(num_updates):
                train_metrics.update(agent.update(buffer, step + i))

        episode_idx += 1
        common_metrics = {
            "episode": episode_idx,
            "step": step,
            "env_step": int(step * cfg.action_repeat),
            "total_time": time.time() - start_time,
            "episode_reward": episode.cumulative_reward,
        }
        train_metrics.update(common_metrics)
        log.info(
            f"episode {episode_idx} | step {step}/{cfg.train_steps} | "
            f"reward {episode.cumulative_reward:.2f}"
        )
        if wandb_run is not None:
            wandb_run.log({f"train/{k}": v for k, v in train_metrics.items()})

        # Evaluate agent periodically
        if step % cfg.eval_freq == 0:
            eval_reward = evaluate(env, agent, cfg.eval_episodes, step)
            common_metrics["episode_reward"] = eval_reward
            log.info(f"  eval @ step {step}: reward {eval_reward:.2f}")
            if wandb_run is not None:
                wandb_run.log({f"eval/{k}": v for k, v in common_metrics.items()})
            if cfg.save_model:
                agent.save(str(ckpt_dir / "checkpoints" / f"model_{step}.pth"))

    if cfg.save_model:
        agent.save(str(ckpt_dir / "checkpoints" / "model_final.pth"))
    log.info("TD-MPC training completed successfully")


if __name__ == "__main__":
    main()
