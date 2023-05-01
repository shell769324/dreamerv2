import datetime
import json
import pathlib

import imageio
import numpy as np
import gym


class Recorder:

  def __init__(
      self, env, directory):
    env = StatsRecorder(env, directory)
    self._env = env

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    return getattr(self._env, name)


class StatsRecorder:

  def __init__(self, env, directory):
    self._env = env
    self._directory = pathlib.Path(directory).expanduser()
    self._directory.mkdir(exist_ok=True, parents=True)
    self._file = (self._directory / 'stats.jsonl').open('a')
    self._length = None
    self._reward = None
    self._unlocked = None
    self._stats = None

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    return getattr(self._env, name)

  def reset(self):
    obs = self._env.reset()
    self._length = 0
    self._reward = 0
    self._unlocked = None
    self._stats = None
    return obs.transpose(2, 0, 1)

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    self._length += 1
    self._reward += info['reward']
    if done:
      self._stats = {'length': self._length, 'reward': round(self._reward, 1)}
      for key, value in info['achievements'].items():
        self._stats[f'achievement_{key}'] = value
      self._save()
    return obs.transpose(2, 0, 1), reward, done, info

  def _save(self):
    self._file.write(json.dumps(self._stats) + '\n')
    self._file.flush()

