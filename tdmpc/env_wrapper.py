"""
Adapts this repo's PushT gym env (dict obs, ~unit-scale continuous action
space) to the plain-array-obs, [-1,1]-action interface TD-MPC expects.

Mirrors nicklashansen/tdmpc's src/env.py::make_env chain (ActionDTypeWrapper +
action_scale.Wrapper + pixels.Wrapper + FrameStackWrapper + TimeStepToGymWrapper),
simplified since we wrap a single custom gym env instead of dm_control's.
"""
import numpy as np
import gym
import cv2
from collections import deque
from omegaconf import open_dict

import env as _env_pkg  # noqa: F401  registers the "pusht" gym id


class PushTTDMPCEnv:
    def __init__(self, img_size=84, frame_stack=3, visual_condition="NC", seed=None):
        self.img_size = img_size
        self.frame_stack = frame_stack
        self._env = gym.make(
            "pusht",
            with_velocity=True,
            with_target=True,
            visual_condition=visual_condition,
        )
        if seed is not None:
            self._env.seed(seed)
        self.action_dim = self._env.action_space.shape[0]
        # TD-MPC's policy/planner always operate in [-1, 1] (matching the reference
        # repo's action_scale.Wrapper(env, minimum=-1, maximum=1)). PushT's raw action
        # interface already expects roughly unit-scale actions -- datasets/pusht_dset.py
        # stores actions pre-divided by action_scale=100 -- so no rescaling is needed
        # beyond clipping to the [-1, 1] box.
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(3 * frame_stack, img_size, img_size), dtype=np.uint8
        )
        self._frames = deque([], maxlen=frame_stack)

    def _process_frame(self, obs):
        img = obs["visual"]
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        return img.transpose(2, 0, 1).copy()  # HWC uint8 -> CHW uint8

    def _stacked_obs(self):
        return np.concatenate(list(self._frames), axis=0)

    def reset(self):
        obs = self._env.reset()
        frame = self._process_frame(obs)
        for _ in range(self.frame_stack):
            self._frames.append(frame)
        return self._stacked_obs()

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        obs, reward, done, info = self._env.step(action)
        self._frames.append(self._process_frame(obs))
        return self._stacked_obs(), float(reward), bool(done), info

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)


def make_env(cfg):
    """
    Constructs the PushT env for TD-MPC training/eval and fills in the cfg
    fields TOLD needs (obs_shape, action_shape, action_dim), mirroring the
    "Convenience" section at the end of the reference src/env.py::make_env.
    """
    environment = PushTTDMPCEnv(
        img_size=cfg.img_size,
        frame_stack=cfg.frame_stack,
        visual_condition=cfg.get("visual_condition", "NC"),
        seed=cfg.get("seed"),
    )
    with open_dict(cfg):
        cfg.obs_shape = tuple(int(x) for x in environment.observation_space.shape)
        cfg.action_shape = tuple(int(x) for x in environment.action_space.shape)
        cfg.action_dim = environment.action_space.shape[0]
    return environment
