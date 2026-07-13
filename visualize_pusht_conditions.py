"""Render PushT states/rollouts under each visual condition and save as PNG/GIF."""

import argparse

import imageio
import matplotlib.pyplot as plt
import numpy as np

from env.pusht.pusht_wrapper import PushTWrapper
from env.visual_conditions import VISUAL_COLUMNS

# [agent_x, agent_y, block_x, block_y, angle, agent_vx, agent_vy]
DEFAULT_INIT_STATE = np.array([180.0, 400.0, 256.0, 300.0, 0.0, 0.0, 0.0])


def render_state_grid(init_state, out_path="pusht_visual_conditions.png"):
    """Single fixed state rendered under every visual condition, side by side."""
    fig, axes = plt.subplots(1, len(VISUAL_COLUMNS), figsize=(4 * len(VISUAL_COLUMNS), 4))
    for ax, vc in zip(axes, VISUAL_COLUMNS):
        env = PushTWrapper(with_velocity=True, with_target=True, visual_condition=vc)
        obs, _ = env.prepare(seed=42, init_state=init_state)
        ax.imshow(obs["visual"])
        ax.set_title(vc, fontsize=12)
        ax.axis("off")
        env.close()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def render_rollout_gif(init_state, visual_condition, n_steps=30, out_path=None):
    """Short random-action rollout under a single condition, saved as a GIF."""
    out_path = out_path or f"pusht_rollout_{visual_condition}.gif"
    env = PushTWrapper(with_velocity=True, with_target=True, visual_condition=visual_condition)
    obs, _ = env.prepare(seed=42, init_state=init_state)
    frames = [obs["visual"]]

    rng = np.random.RandomState(0)
    for _ in range(n_steps):
        action = rng.uniform(-1, 1, size=2)
        obs, _, done, _ = env.step(action)
        frames.append(obs["visual"])
        if done:
            break
    env.close()

    imageio.mimsave(out_path, frames, fps=10)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--visual-condition", type=str, default="D", choices=list(VISUAL_COLUMNS),
        help="Condition to use for the rollout GIF (the static grid always covers all conditions)",
    )
    parser.add_argument("--rollout-steps", type=int, default=30)
    args = parser.parse_args()

    render_state_grid(DEFAULT_INIT_STATE)
    render_rollout_gif(DEFAULT_INIT_STATE, args.visual_condition, n_steps=args.rollout_steps)
