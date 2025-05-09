from typing import Any, Optional, Callable, Tuple
from abc import ABC, abstractmethod
from collections import namedtuple, deque
from easydict import EasyDict
import copy
import numpy as np
import torch

from ding.utils import lists_to_dicts
from ding.torch_utils import to_tensor, to_ndarray, tensor_to_list


class ISerialEvaluator(ABC):
    """
    Overview:
        Basic interface class for serial evaluator.
    Interfaces:
        reset, reset_policy, reset_env, close, should_eval, eval
    Property:
        env, policy
    """

    @classmethod
    def default_config(cls: type) -> EasyDict:
        """
        Overview:
            Get evaluator's default config. We merge evaluator's default config with other default configs\
                and user's config to get the final config.
        Return:
            cfg: (:obj:`EasyDict`): evaluator's default config
        """
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    @abstractmethod
    def reset_env(self, _env: Optional[Any] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset_policy(self, _policy: Optional[namedtuple] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset(self, _policy: Optional[namedtuple] = None, _env: Optional[Any] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def should_eval(self, train_iter: int) -> bool:
        raise NotImplementedError

    @abstractmethod
    def eval(
            self,
            save_ckpt_fn: Callable = None,
            train_iter: int = -1,
            envstep: int = -1,
            n_episode: Optional[int] = None
    ) -> Any:
        raise NotImplementedError


class VectorEvalMonitor(object):
    """
    Overview:
        In some cases,  different environment in evaluator may collect different length episode. For example, \
            suppose we want to collect 12 episodes in evaluator but only have 5 environments, if we didn’t do \
            any thing, it is likely that we will get more short episodes than long episodes. As a result, \
            our average reward will have a bias and may not be accurate. we use VectorEvalMonitor to solve the problem.
    Interfaces:
        __init__, is_finished, update_info, update_reward, get_episode_reward, get_latest_reward, get_current_episode,\
            get_episode_info
    """

    def __init__(self, env_num: int, n_episode: int) -> None:
        """
        Overview:
            Init method. According to the number of episodes and the number of environments, determine how many \
                episodes need to be opened for each environment, and initialize the reward, info and other \
                information
        Arguments:
            - env_num (:obj:`int`): the number of episodes need to be open
            - n_episode (:obj:`int`): the number of environments
        """
        assert n_episode >= env_num, "n_episode < env_num, please decrease the number of eval env"
        self._env_num = env_num
        self._n_episode = n_episode
        each_env_episode = [n_episode // env_num for _ in range(env_num)]
        for i in range(n_episode % env_num):
            each_env_episode[i] += 1
        self._reward = {env_id: deque(maxlen=maxlen) for env_id, maxlen in enumerate(each_env_episode)}
        self._info = {env_id: deque(maxlen=maxlen) for env_id, maxlen in enumerate(each_env_episode)}

    def is_finished(self) -> bool:
        """
        Overview:
            Determine whether the evaluator has completed the work.
        Return:
            - result: (:obj:`bool`): whether the evaluator has completed the work
        """
        return all([len(v) == v.maxlen for v in self._reward.values()])

    def update_info(self, env_id: int, info: Any) -> None:
        """
        Overview:
            Update the information of the environment indicated by env_id.
        Arguments:
            - env_id: (:obj:`int`): the id of the environment we need to update information
            - info: (:obj:`Any`): the information we need to update
        """
        info = tensor_to_list(info)
        self._info[env_id].append(info)

    def update_reward(self, env_id: int, reward: Any) -> None:
        """
        Overview:
            Update the reward indicated by env_id.
        Arguments:
            - env_id: (:obj:`int`): the id of the environment we need to update the reward
            - reward: (:obj:`Any`): the reward we need to update
        """
        if isinstance(reward, torch.Tensor):
            reward = reward.item()
        self._reward[env_id].append(reward)

    def get_episode_reward(self) -> list:
        """
        Overview:
            Get the total reward of one episode.
        """
        return sum([list(v) for v in self._reward.values()], [])  # sum(iterable, start)

    def get_latest_reward(self, env_id: int) -> int:
        """
        Overview:
            Get the latest reward of a certain environment.
        Arguments:
            - env_id: (:obj:`int`): the id of the environment we need to get reward.
        """
        return self._reward[env_id][-1]

    def get_current_episode(self) -> int:
        """
        Overview:
            Get the current episode. We can know which episode our evaluator is executing now.
        """
        return sum([len(v) for v in self._reward.values()])

    def get_episode_info(self) -> dict:
        """
        Overview:
            Get all episode information, such as total reward of one episode.
        """
        if len(self._info[0]) == 0:
            return None
        else:
            total_info = sum([list(v) for v in self._info.values()], [])
            total_info = lists_to_dicts(total_info)
            new_dict = {}
            for k in total_info.keys():
                if np.isscalar(total_info[k][0]):
                    new_dict[k + '_mean'] = np.mean(total_info[k])
            total_info.update(new_dict)
            return total_info
