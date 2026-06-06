"""Replay buffer and dataset utilities: Dataset base class, DatasetDict type
alias, and ReplayBuffer.
"""

from typing import Dict, Iterable, Optional, Tuple, Union
import collections
import copy
import pickle

import jax
import gym
import gym.spaces
import numpy as np
from gym.utils import seeding
from flax.core import frozen_dict

DataType = Union[np.ndarray, Dict[str, 'DataType']]
DatasetDict = Dict[str, DataType]


def concat_recursive(batches):
    new_batch = {}
    for k, v in batches[0].items():
        if isinstance(v, frozen_dict.FrozenDict):
            new_batch[k] = concat_recursive([batches[0][k], batches[1][k]])
        else:
            new_batch[k] = np.concatenate([b[k] for b in batches], 0)
    return new_batch


def _check_lengths(dataset_dict: DatasetDict,
                   dataset_len: Optional[int] = None) -> int:
    for v in dataset_dict.values():
        if isinstance(v, dict):
            dataset_len = dataset_len or _check_lengths(v, dataset_len)
        elif isinstance(v, np.ndarray):
            item_len = len(v)
            dataset_len = dataset_len or item_len
            assert dataset_len == item_len, 'Inconsistent item lengths in the dataset.'
        else:
            raise TypeError('Unsupported type.')
    return dataset_len


def _split(dataset_dict: DatasetDict,
           index: int) -> Tuple[DatasetDict, DatasetDict]:
    train_dataset_dict, test_dataset_dict = {}, {}
    for k, v in dataset_dict.items():
        if isinstance(v, dict):
            train_v, test_v = _split(v, index)
        elif isinstance(v, np.ndarray):
            train_v, test_v = v[:index], v[index:]
        else:
            raise TypeError('Unsupported type.')
        train_dataset_dict[k] = train_v
        test_dataset_dict[k] = test_v
    return train_dataset_dict, test_dataset_dict


def _sample(dataset_dict: Union[np.ndarray, DatasetDict],
            indx: np.ndarray) -> DatasetDict:
    if isinstance(dataset_dict, np.ndarray):
        return dataset_dict[indx]
    elif isinstance(dataset_dict, dict):
        batch = {}
        for k, v in dataset_dict.items():
            batch[k] = _sample(v, indx)
    else:
        raise TypeError("Unsupported type.")
    return batch


class Dataset(object):

    def __init__(self, dataset_dict: DatasetDict, seed: Optional[int] = None):
        self.dataset_dict = dataset_dict
        self.dataset_len = _check_lengths(dataset_dict)

        self._np_random = None
        if seed is not None:
            self.seed(seed)

    @property
    def np_random(self) -> np.random.RandomState:
        if self._np_random is None:
            self.seed()
        return self._np_random

    def seed(self, seed: Optional[int] = None) -> list:
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def __len__(self) -> int:
        return self.dataset_len

    def sample(self,
               batch_size: int,
               keys: Optional[Iterable[str]] = None,
               indx: Optional[np.ndarray] = None) -> frozen_dict.FrozenDict:
        if indx is None:
            if hasattr(self.np_random, 'integers'):
                indx = self.np_random.integers(len(self), size=batch_size)
            else:
                indx = self.np_random.randint(len(self), size=batch_size)

        batch = dict()

        if keys is None:
            keys = self.dataset_dict.keys()

        for k in keys:
            if isinstance(self.dataset_dict[k], dict):
                batch[k] = _sample(self.dataset_dict[k], indx)
            else:
                batch[k] = self.dataset_dict[k][indx]

        return frozen_dict.freeze(batch)

    def split(self, ratio: float) -> Tuple['Dataset', 'Dataset']:
        assert 0 < ratio and ratio < 1
        index = int(self.dataset_len * ratio)
        train_dataset_dict, test_dataset_dict = _split(self.dataset_dict,
                                                       index)
        return Dataset(train_dataset_dict), Dataset(test_dataset_dict)


def _init_replay_dict(obs_space: gym.Space,
                      capacity: int) -> Union[np.ndarray, DatasetDict]:
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Dict):
        data_dict = {}
        for k, v in obs_space.spaces.items():
            data_dict[k] = _init_replay_dict(v, capacity)
        return data_dict
    else:
        raise TypeError()


class ReplayBuffer(Dataset):

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, capacity: int, ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.capacity = capacity

        print("making replay buffer of capacity ", self.capacity)

        observations = _init_replay_dict(self.observation_space, self.capacity)
        next_observations = _init_replay_dict(self.observation_space, self.capacity)
        actions = np.empty((self.capacity, *self.action_space.shape), dtype=self.action_space.dtype)
        next_actions = np.empty((self.capacity, *self.action_space.shape), dtype=self.action_space.dtype)
        rewards = np.empty((self.capacity, ), dtype=np.float32)
        masks = np.empty((self.capacity, ), dtype=np.float32)
        discount = np.empty((self.capacity, ), dtype=np.float32)

        self.data = {
            'observations': observations,
            'next_observations': next_observations,
            'actions': actions,
            'next_actions': next_actions,
            'rewards': rewards,
            'masks': masks,
            'discount': discount,
        }

        self._weights = np.ones((self.capacity,), dtype=np.float32)
        self._has_nonuniform_weights = False

        self._np_random = None
        self.size = 0
        self._traj_counter = 0
        self._start = 0
        self.traj_bounds = dict()
        self.streaming_buffer_size = None  # this is for streaming the online data

    def __len__(self) -> int:
        return self.size

    def length(self) -> int:
        return self.size

    def increment_traj_counter(self):
        self.traj_bounds[self._traj_counter] = (self._start, self.size)  # [start, end)
        self._start = self.size
        self._traj_counter += 1

    def get_random_trajs(self, num_trajs: int):
        self.which_trajs = self.np_random.integers(0, self._traj_counter, num_trajs) if hasattr(self.np_random, 'integers') else self.np_random.randint(0, self._traj_counter, num_trajs)
        observations_list = []
        next_observations_list = []
        actions_list = []
        rewards_list = []
        terminals_list = []
        masks_list = []
        discount_list = []

        for i in self.which_trajs:
            start, end = self.traj_bounds[i]

            obs_dict_curr_traj = dict()
            for k in self.data['observations']:
                obs_dict_curr_traj[k] = self.data['observations'][k][start:end]
            observations_list.append(obs_dict_curr_traj)

            next_obs_dict_curr_traj = dict()
            for k in self.data['next_observations']:
                next_obs_dict_curr_traj[k] = self.data['next_observations'][k][start:end]
            next_observations_list.append(next_obs_dict_curr_traj)

            actions_list.append(self.data['actions'][start:end])
            rewards_list.append(self.data['rewards'][start:end])
            terminals_list.append(1 - self.data['masks'][start:end])
            masks_list.append(self.data['masks'][start:end])

        batch = {
            'observations': observations_list,
            'next_observations': next_observations_list,
            'actions': actions_list,
            'rewards': rewards_list,
            'terminals': terminals_list,
            'masks': masks_list,
        }
        return batch

    def insert(self, data_dict: DatasetDict, weight: float = 1.0):
        if self.size == self.capacity:
            # Double the capacity
            observations = _init_replay_dict(self.observation_space, self.capacity)
            next_observations = _init_replay_dict(self.observation_space, self.capacity)
            actions = np.empty((self.capacity, *self.action_space.shape), dtype=self.action_space.dtype)
            next_actions = np.empty((self.capacity, *self.action_space.shape), dtype=self.action_space.dtype)
            rewards = np.empty((self.capacity, ), dtype=np.float32)
            masks = np.empty((self.capacity, ), dtype=np.float32)
            discount = np.empty((self.capacity, ), dtype=np.float32)

            data_new = {
                'observations': observations,
                'next_observations': next_observations,
                'actions': actions,
                'next_actions': next_actions,
                'rewards': rewards,
                'masks': masks,
                'discount': discount,
            }

            for x in data_new:
                if isinstance(self.data[x], np.ndarray):
                    self.data[x] = np.concatenate((self.data[x], data_new[x]), axis=0)
                elif isinstance(self.data[x], dict):
                    for y in self.data[x]:
                        self.data[x][y] = np.concatenate((self.data[x][y], data_new[x][y]), axis=0)
                else:
                    raise TypeError()
            new_weights = np.ones((self.capacity,), dtype=np.float32)
            self._weights = np.concatenate((self._weights, new_weights), axis=0)
            self.capacity *= 2

        for x in data_dict:
            if x in self.data:
                if isinstance(data_dict[x], dict):
                    for y in data_dict[x]:
                        self.data[x][y][self.size] = data_dict[x][y]
                else:
                    self.data[x][self.size] = data_dict[x]
        self._weights[self.size] = weight
        if weight != 1.0:
            self._has_nonuniform_weights = True
        self.size += 1

    def compute_action_stats(self):
        actions = self.data['actions']
        return {'mean': actions.mean(axis=0), 'std': actions.std(axis=0)}

    def normalize_actions(self, action_stats):
        # do not normalize gripper dimension (last dimension)
        copy.deepcopy(action_stats)
        action_stats['mean'][-1] = 0
        action_stats['std'][-1] = 1
        self.data['actions'] = (self.data['actions'] - action_stats['mean']) / action_stats['std']
        self.data['next_actions'] = (self.data['next_actions'] - action_stats['mean']) / action_stats['std']

    def sample(self, batch_size: int, keys: Optional[Iterable[str]] = None, indx: Optional[np.ndarray] = None) -> frozen_dict.FrozenDict:
        n = self.streaming_buffer_size if self.streaming_buffer_size else self.size

        if self._has_nonuniform_weights:
            w = self._weights[:n]
            probs = w / w.sum()
            indices = self.np_random.choice(n, size=batch_size, replace=True, p=probs)
        else:
            indices = self.np_random.integers(0, n, batch_size) if hasattr(self.np_random, 'integers') else self.np_random.randint(0, n, batch_size)

        data_dict = {}
        for x in self.data:
            if isinstance(self.data[x], np.ndarray):
                data_dict[x] = self.data[x][indices]
            elif isinstance(self.data[x], dict):
                data_dict[x] = {}
                for y in self.data[x]:
                    data_dict[x][y] = self.data[x][y][indices]
            else:
                raise TypeError()

        return frozen_dict.freeze(data_dict)

    def get_iterator(self, batch_size: int, keys: Optional[Iterable[str]] = None, indx: Optional[np.ndarray] = None, queue_size: int = 2):
        # Prefetch to device (flax-style); queue_size=2 is fine for one GPU.
        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(batch_size, keys, indx)
                queue.append(jax.device_put(data))

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)

    def save(self, filename):
        save_dict = dict(
            data=self.data,
            size=self.size,
            _traj_counter=self._traj_counter,
            _start=self._start,
            traj_bounds=self.traj_bounds,
            _weights=self._weights,
        )
        with open(filename, 'wb') as f:
            pickle.dump(save_dict, f, protocol=4)

    def restore(self, filename):
        save_dict = np.load(filename, allow_pickle=True)[0]
        self.data = save_dict['data']
        self.size = save_dict['size']
        self._traj_counter = save_dict['_traj_counter']
        self._start = save_dict['_start']
        self.traj_bounds = save_dict['traj_bounds']
        if '_weights' in save_dict:
            self._weights = save_dict['_weights']
            self._has_nonuniform_weights = np.any(self._weights[:self.size] != 1.0)
        else:
            self._weights = np.ones((self.capacity,), dtype=np.float32)
            self._has_nonuniform_weights = False
