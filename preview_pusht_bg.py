"""
Preview a PushT background by rendering a few frames and saving a PNG grid.

Usage:
    python preview_pusht_bg.py --src_dir /data/pusht_noise/train --bg_name large_color
    python preview_pusht_bg.py --src_dir /data/pusht_noise/train --bg_color 180 130 80
"""

import argparse
import pickle
import os
import sys
import importlib.util as _ilu
import numpy as np
import torch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame  # noqa: E402 (must come after SDL env vars)

_spec = _ilu.spec_from_file_location(
    "pusht_env",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "env", "pusht", "pusht_env.py"),
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
PushTEnv = _mod.PushTEnv

from pathlib import Path

BACKGROUND_CONFIGS = {
    "default":              (255, 255, 255),
    "slight_change":        (235, 230, 225),
    "color":                (180, 210, 240),
    "large_color":          (235, 235, 115),
    "large_color_gradient": (46,  13,  89),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", required=True)
    parser.add_argument("--bg_name", default=None, choices=list(BACKGROUND_CONFIGS.keys()))
    parser.add_argument("--bg_color", nargs=3, type=int, default=[255, 255, 255], metavar=("R", "G", "B"))
    parser.add_argument("--n_frames", type=int, default=8, help="Number of frames to show")
    parser.add_argument("--out", type=str, default="pusht_bg_preview.png")
    args = parser.parse_args()

    bg_color = BACKGROUND_CONFIGS[args.bg_name] if args.bg_name else tuple(args.bg_color)
    print(f"bg_color={bg_color}")

    src_dir = Path(args.src_dir)
    states = torch.load(src_dir / "states.pth", weights_only=False).float()
    with open(src_dir / "seq_lengths.pkl", "rb") as f:
        seq_lengths = pickle.load(f)

    env = PushTEnv(bg_color=bg_color, render_size=224)

    frames = []
    ep_idx = 0
    while len(frames) < args.n_frames:
        T = seq_lengths[ep_idx]
        step = max(1, T // args.n_frames)
        for t in range(0, T, step):
            env.reset_to_state = states[ep_idx, t].numpy()
            env.reset()
            frame = env.render(mode="rgb_array")
            frames.append(frame)
            if len(frames) >= args.n_frames:
                break
        ep_idx += 1

    env.close()

    # Save as a horizontal strip PNG
    import cv2
    strip = np.concatenate(frames[:args.n_frames], axis=1)  # (H, W*N, 3)
    cv2.imwrite(args.out, cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    print(f"Saved preview to {args.out}  ({args.n_frames} frames, bg={bg_color})")


if __name__ == "__main__":
    main()
