"""
Re-render a PushT dataset with a custom background color.

Usage:
    python generate_pusht_bg_dataset.py \
        --src_dir /path/to/pusht_noise/train \
        --dst_dir /path/to/pusht_bg/train \
        --bg_color 180 130 80

The script replays each episode's stored states through the PushT environment
with the specified background color, writes new MP4 videos, and copies the
non-visual files (states, actions, etc.) unchanged.
"""

import argparse
import pickle
import shutil
import numpy as np
import torch
import cv2
import os
from pathlib import Path

import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# Load pusht_env directly to avoid env/__init__.py triggering mujoco_py
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "pusht_env",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "env", "pusht", "pusht_env.py"),
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
PushTEnv = _mod.PushTEnv


BACKGROUND_CONFIGS = {
    "default":              (255, 255, 255),
    "slight_change":        (235, 230, 225),
    "color":                (180, 210, 240),
    "large_color":          (235, 235, 115),
    "large_color_gradient": (46,  13,  89),
}


def render_episode(env, ep_states, seq_len, out_path, render_size=224):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, 10, (render_size, render_size))
    for t in range(seq_len):
        state = ep_states[t].numpy()
        env.reset_to_state = state
        env.reset()
        frame = env.render(mode="rgb_array")  # (H, W, 3) RGB uint8
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def generate(src_dir: Path, dst_dir: Path, bg_color: tuple, n_rollout=None):
    (dst_dir / "obses").mkdir(parents=True, exist_ok=True)

    states = torch.load(src_dir / "states.pth").float()
    with open(src_dir / "seq_lengths.pkl", "rb") as f:
        seq_lengths = pickle.load(f)

    shapes_file = src_dir / "shapes.pkl"
    if shapes_file.exists():
        with open(shapes_file, "rb") as f:
            shapes = pickle.load(f)
    else:
        shapes = ["T"] * len(states)

    n = n_rollout if n_rollout else len(states)
    states = states[:n]
    seq_lengths = seq_lengths[:n]
    shapes = shapes[:n]

    env = PushTEnv(bg_color=bg_color, render_size=224)

    for i, (ep_states, seq_len, shape) in enumerate(zip(states, seq_lengths, shapes)):
        env.shape = shape
        out_path = dst_dir / "obses" / f"episode_{i:03d}.mp4"
        render_episode(env, ep_states, seq_len, out_path)
        if (i + 1) % 10 == 0 or i == n - 1:
            print(f"  rendered {i+1}/{n}")

    env.close()

    # Copy non-visual files unchanged
    for fname in ["states.pth", "rel_actions.pth", "abs_actions.pth",
                  "velocities.pth", "seq_lengths.pkl", "shapes.pkl"]:
        src_f = src_dir / fname
        if src_f.exists():
            shutil.copy(src_f, dst_dir / fname)
            print(f"  copied {fname}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", required=True,
                        help="Source dataset split dir, e.g. /data/pusht_noise/train")
    parser.add_argument("--dst_dir", required=True,
                        help="Destination dir for the new dataset split")
    parser.add_argument("--bg_name", default=None,
                        choices=list(BACKGROUND_CONFIGS.keys()),
                        help="Named background preset (overrides --bg_color)")
    parser.add_argument("--bg_color", nargs=3, type=int, default=[255, 255, 255],
                        metavar=("R", "G", "B"),
                        help="Background RGB color (default: 255 255 255 = white)")
    parser.add_argument("--n_rollout", type=int, default=None,
                        help="Limit number of episodes (default: all)")
    args = parser.parse_args()

    if args.bg_name:
        bg_color = BACKGROUND_CONFIGS[args.bg_name]
        print(f"Using preset '{args.bg_name}': bg_color={bg_color}")
    else:
        bg_color = tuple(args.bg_color)
        print(f"Using bg_color={bg_color}")

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)

    print(f"Source: {src_dir}")
    print(f"Dest:   {dst_dir}")
    generate(src_dir, dst_dir, bg_color, n_rollout=args.n_rollout)
    print("Done.")


if __name__ == "__main__":
    main()
