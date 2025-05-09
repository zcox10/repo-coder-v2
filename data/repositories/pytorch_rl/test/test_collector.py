# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys

import numpy as np
import pytest
import torch
from _utils_internal import generate_seeds, PENDULUM_VERSIONED, PONG_VERSIONED
from mocking_classes import (
    ContinuousActionVecMockEnv,
    DiscreteActionConvMockEnv,
    DiscreteActionConvPolicy,
    DiscreteActionVecMockEnv,
    DiscreteActionVecPolicy,
    MockSerialEnv,
)
from tensordict.nn import TensorDictModule
from tensordict.tensordict import assert_allclose_td, TensorDict
from torch import nn
from torchrl._utils import seed_generator
from torchrl.collectors import aSyncDataCollector, SyncDataCollector
from torchrl.collectors.collectors import (
    MultiaSyncDataCollector,
    MultiSyncDataCollector,
    RandomPolicy,
)
from torchrl.collectors.utils import split_trajectories
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs import EnvCreator, ParallelEnv, SerialEnv
from torchrl.envs.libs.gym import _has_gym, GymEnv
from torchrl.envs.transforms import TransformedEnv, VecNorm
from torchrl.modules import Actor, LSTMNet, OrnsteinUhlenbeckProcessWrapper, SafeModule

# torch.set_default_dtype(torch.double)
_os_is_windows = sys.platform == "win32"
_python_is_3_10 = sys.version_info.major == 3 and sys.version_info.minor == 10
_python_is_3_7 = sys.version_info.major == 3 and sys.version_info.minor == 7


class WrappablePolicy(nn.Module):
    def __init__(self, out_features: int, multiple_outputs: bool = False):
        super().__init__()
        self.multiple_outputs = multiple_outputs
        self.linear = nn.LazyLinear(out_features)

    def forward(self, observation):
        output = self.linear(observation)
        if self.multiple_outputs:
            return output, output.sum(), output.min(), output.max()
        return self.linear(observation)


class TensorDictCompatiblePolicy(nn.Module):
    def __init__(self, out_features: int):
        super().__init__()
        self.in_keys = ["observation"]
        self.out_keys = ["action"]
        self.linear = nn.LazyLinear(out_features)

    def forward(self, tensordict):
        return TensorDict(
            {self.out_keys[0]: self.linear(tensordict.get(self.in_keys[0]))},
            [],
        )


class UnwrappablePolicy(nn.Module):
    def __init__(self, out_features: int):
        super().__init__()
        self.linear = nn.LazyLinear(out_features)

    def forward(self, observation, other_stuff):
        return self.linear(observation), other_stuff.sum()


class ParametricPolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.randn(1, requires_grad=True))

    def forward(self, obs):
        max_obs = (obs == obs.max(dim=-1, keepdim=True)[0]).cumsum(-1).argmax(-1)
        k = obs.shape[-1]
        max_obs = (max_obs + 1) % k
        action = torch.nn.functional.one_hot(max_obs, k)
        return action


class ParametricPolicy(Actor):
    def __init__(self):
        super().__init__(
            ParametricPolicyNet(),
            in_keys=["observation"],
        )


def make_make_env(env_name="conv"):
    def make_transformed_env(seed=None):
        if env_name == "conv":
            env = DiscreteActionConvMockEnv()
        elif env_name == "vec":
            env = DiscreteActionVecMockEnv()
        if seed is not None:
            env.set_seed(seed)
        return env

    return make_transformed_env


def dummypolicy_vec():
    policy = DiscreteActionVecPolicy()
    return policy


def dummypolicy_conv():
    policy = DiscreteActionConvPolicy()
    return policy


def make_policy(env):
    if env == "conv":
        return dummypolicy_conv()
    elif env == "vec":
        return dummypolicy_vec()
    else:
        raise NotImplementedError


def _is_consistent_device_type(
    device_type, policy_device_type, passing_device_type, tensordict_device_type
):
    if passing_device_type is None:
        if device_type is None:
            if policy_device_type is None:
                return tensordict_device_type == "cpu"

            return tensordict_device_type == policy_device_type

        return tensordict_device_type == device_type

    return tensordict_device_type == passing_device_type


@pytest.mark.skipif(
    _os_is_windows and _python_is_3_10,
    reason="Windows Access Violation in torch.multiprocessing / BrokenPipeError in multiprocessing.connection",
)
@pytest.mark.parametrize("num_env", [1, 3])
@pytest.mark.parametrize("device", ["cuda", "cpu", None])
@pytest.mark.parametrize("policy_device", ["cuda", "cpu", None])
@pytest.mark.parametrize("passing_device", ["cuda", "cpu", None])
def test_output_device_consistency(
    num_env, device, policy_device, passing_device, seed=40
):
    if (
        device == "cuda" or policy_device == "cuda" or passing_device == "cuda"
    ) and not torch.cuda.is_available():
        pytest.skip("cuda is not available")

    if _os_is_windows and _python_is_3_7:
        if device == "cuda" and policy_device == "cuda" and device is None:
            pytest.skip(
                "BrokenPipeError in multiprocessing.connection with Python 3.7 on Windows"
            )

    _device = "cuda:0" if device == "cuda" else device
    _policy_device = "cuda:0" if policy_device == "cuda" else policy_device
    _passing_device = "cuda:0" if passing_device == "cuda" else passing_device

    if num_env == 1:

        def env_fn(seed):
            env = make_make_env("vec")()
            env.set_seed(seed)
            return env

    else:

        def env_fn(seed):
            env = ParallelEnv(
                num_workers=num_env,
                create_env_fn=make_make_env("vec"),
                create_env_kwargs=[{"seed": i} for i in range(seed, seed + num_env)],
            )
            return env

    if _policy_device is None:
        policy = make_policy("vec")
    else:
        policy = ParametricPolicy().to(torch.device(_policy_device))

    collector = SyncDataCollector(
        create_env_fn=env_fn,
        create_env_kwargs={"seed": seed},
        policy=policy,
        frames_per_batch=20,
        max_frames_per_traj=2000,
        total_frames=20000,
        device=_device,
        passing_device=_passing_device,
        pin_memory=False,
    )
    for _, d in enumerate(collector):
        assert _is_consistent_device_type(
            device, policy_device, passing_device, d.device.type
        )
        break

    collector.shutdown()

    ccollector = aSyncDataCollector(
        create_env_fn=env_fn,
        create_env_kwargs={"seed": seed},
        policy=policy,
        frames_per_batch=20,
        max_frames_per_traj=2000,
        total_frames=20000,
        device=_device,
        passing_device=_passing_device,
        pin_memory=False,
    )

    for _, d in enumerate(ccollector):
        assert _is_consistent_device_type(
            device, policy_device, passing_device, d.device.type
        )
        break

    ccollector.shutdown()


@pytest.mark.parametrize("num_env", [1, 3])
@pytest.mark.parametrize("env_name", ["conv", "vec"])
def test_concurrent_collector_consistency(num_env, env_name, seed=40):
    if num_env == 1:

        def env_fn(seed):
            env = make_make_env(env_name)()
            env.set_seed(seed)
            return env

    else:

        def env_fn(seed):
            env = ParallelEnv(
                num_workers=num_env,
                create_env_fn=make_make_env(env_name),
                create_env_kwargs=[{"seed": i} for i in range(seed, seed + num_env)],
            )
            return env

    policy = make_policy(env_name)

    collector = SyncDataCollector(
        create_env_fn=env_fn,
        create_env_kwargs={"seed": seed},
        policy=policy,
        frames_per_batch=20,
        max_frames_per_traj=2000,
        total_frames=20000,
        device="cpu",
        pin_memory=False,
    )
    for i, d in enumerate(collector):
        if i == 0:
            b1 = d
        elif i == 1:
            b2 = d
        else:
            break
    with pytest.raises(AssertionError):
        assert_allclose_td(b1, b2)
    collector.shutdown()

    ccollector = aSyncDataCollector(
        create_env_fn=env_fn,
        create_env_kwargs={"seed": seed},
        policy=policy,
        frames_per_batch=20,
        max_frames_per_traj=2000,
        total_frames=20000,
        pin_memory=False,
    )
    for i, d in enumerate(ccollector):
        if i == 0:
            b1c = d
        elif i == 1:
            b2c = d
        else:
            break
    with pytest.raises(AssertionError):
        assert_allclose_td(b1c, b2c)

    assert_allclose_td(b1c, b1)
    assert_allclose_td(b2c, b2)

    ccollector.shutdown()


@pytest.mark.skipif(not _has_gym, reason="gym library is not installed")
def test_collector_env_reset():
    torch.manual_seed(0)

    def make_env():
        return GymEnv(PONG_VERSIONED, frame_skip=4)

    env = SerialEnv(2, make_env)
    # env = SerialEnv(3, lambda: GymEnv("CartPole-v1", frame_skip=4))
    env.set_seed(0)
    collector = SyncDataCollector(
        env, total_frames=10000, frames_per_batch=10000, split_trajs=False
    )
    for _data in collector:
        continue
    steps = _data["collector", "step_count"][..., 1:]
    done = _data["done"][..., :-1, :].squeeze(-1)
    # we don't want just one done
    assert done.sum() > 3
    # check that after a done, the next step count is always 1
    assert (steps[done] == 1).all()
    # check that if the env is not done, the next step count is > 1
    assert (steps[~done] > 1).all()
    # check that if step is 1, then the env was done before
    assert (steps == 1)[done].all()
    # check that split traj has a minimum total reward of -21 (for pong only)
    _data = split_trajectories(_data)
    assert _data["reward"].sum(-2).min() == -21


@pytest.mark.parametrize("num_env", [1, 3])
@pytest.mark.parametrize("env_name", ["vec"])
def test_collector_done_persist(num_env, env_name, seed=5):
    if num_env == 1:

        def env_fn(seed):
            env = MockSerialEnv(device="cpu")
            env.set_seed(seed)
            return env

    else:

        def env_fn(seed):
            def make_env(seed):
                env = MockSerialEnv(device="cpu")
                env.set_seed(seed)
                return env

            env = ParallelEnv(
                num_workers=num_env,
                create_env_fn=make_env,
                create_env_kwargs=[{"seed": i} for i in range(seed, seed + num_env)],
                allow_step_when_done=True,
            )
            env.set_seed(seed)
            return env

    policy = make_policy(env_name)

    collector = SyncDataCollector(
        create_env_fn=env_fn,
        create_env_kwargs={"seed": seed},
        policy=policy,
        frames_per_batch=200 * num_env,
        max_frames_per_traj=2000,
        total_frames=20000,
        device="cpu",
        pin_memory=False,
        reset_when_done=False,
    )
    for _, d in enumerate(collector):  # noqa
        break

    assert (d["done"].sum(-2) >= 1).all()
    assert torch.unique(d["collector", "traj_ids"], dim=-1).shape[-1] == 1

    del collector


@pytest.mark.parametrize("frames_per_batch", [200, 10])
@pytest.mark.parametrize("num_env", [1, 3])
@pytest.mark.parametrize("env_name", ["vec"])
def test_split_trajs(num_env, env_name, frames_per_batch, seed=5):
    if num_env == 1:

        def env_fn(seed):
            env = MockSerialEnv(device="cpu")
            env.set_seed(seed)
            return env

    else:

        def env_fn(seed):
            def make_env(seed):
                env = MockSerialEnv(device="cpu")
                env.set_seed(seed)
                return env

            env = SerialEnv(
                num_workers=num_env,
                create_env_fn=make_env,
                create_env_kwargs=[{"seed": i} for i in range(seed, seed + num_env)],
                allow_step_when_done=True,
            )
            env.set_seed(seed)
            return env

    policy = make_policy(env_name)

    collector = SyncDataCollector(
        create_env_fn=env_fn,
        create_env_kwargs={"seed": seed},
        policy=policy,
        frames_per_batch=frames_per_batch * num_env,
        max_frames_per_traj=2000,
        total_frames=20000,
        device="cpu",
        pin_memory=False,
        reset_when_done=True,
        split_trajs=True,
    )
    for _, d in enumerate(collector):  # noqa
        break

    assert d.ndimension() == 2
    assert d["collector", "mask"].shape == d.shape
    assert d["collector", "step_count"].shape == d.shape
    assert d["collector", "traj_ids"].shape == d.shape
    for traj in d.unbind(0):
        assert traj["collector", "traj_ids"].unique().numel() == 1
        assert (
            traj["collector", "step_count"][1:] - traj["collector", "step_count"][:-1]
            == 1
        ).all()

    del collector


# TODO: design a test that ensures that collectors are interrupted even if __del__ is not called
# @pytest.mark.parametrize("should_shutdown", [True, False])
# def test_shutdown_collector(should_shutdown, num_env=3, env_name="vec", seed=40):
#     def env_fn(seed):
#         env = ParallelEnv(
#             num_workers=num_env,
#             create_env_fn=make_make_env(env_name),
#             create_env_kwargs=[{"seed": i} for i in range(seed, seed + num_env)],
#         )
#         return env
#
#     policy = make_policy(env_name)
#
#     ccollector = aSyncDataCollector(
#         create_env_fn=env_fn,
#         create_env_kwargs={"seed": seed},
#         policy=policy,
#         frames_per_batch=20,
#         max_frames_per_traj=2000,
#         total_frames=20000,
#         pin_memory=False,
#     )
#     for i, d in enumerate(ccollector):
#         if i == 0:
#             b1c = d
#         elif i == 1:
#             b2c = d
#         else:
#             break
#     with pytest.raises(AssertionError):
#         assert_allclose_td(b1c, b2c)
#
#     if should_shutdown:
#         ccollector.shutdown()


@pytest.mark.parametrize("num_env", [1, 3])
@pytest.mark.parametrize("env_name", ["vec", "conv"])
def test_collector_batch_size(num_env, env_name, seed=100):
    if num_env == 3 and _os_is_windows:
        pytest.skip("Test timeout (> 10 min) on CI pipeline Windows machine with GPU")
    if num_env == 1:

        def env_fn():
            env = make_make_env(env_name)()
            return env

    else:

        def env_fn():
            env = ParallelEnv(
                num_workers=num_env, create_env_fn=make_make_env(env_name)
            )
            return env

    policy = make_policy(env_name)

    torch.manual_seed(0)
    np.random.seed(0)
    num_workers = 4
    frames_per_batch = 20
    ccollector = MultiaSyncDataCollector(
        create_env_fn=[env_fn for _ in range(num_workers)],
        policy=policy,
        frames_per_batch=frames_per_batch,
        max_frames_per_traj=1000,
        total_frames=frames_per_batch * 100,
        pin_memory=False,
    )
    ccollector.set_seed(seed)
    for i, b in enumerate(ccollector):
        assert b.numel() == -(-frames_per_batch // num_env) * num_env
        if i == 5:
            break
    ccollector.shutdown()

    ccollector = MultiSyncDataCollector(
        create_env_fn=[env_fn for _ in range(num_workers)],
        policy=policy,
        frames_per_batch=frames_per_batch,
        max_frames_per_traj=1000,
        total_frames=frames_per_batch * 100,
        pin_memory=False,
    )
    ccollector.set_seed(seed)
    for i, b in enumerate(ccollector):
        assert (
            b.numel()
            == -(-frames_per_batch // num_env // num_workers) * num_env * num_workers
        )
        if i == 5:
            break
    ccollector.shutdown()


@pytest.mark.parametrize("num_env", [1, 3])
@pytest.mark.parametrize("env_name", ["vec", "conv"])
def test_concurrent_collector_seed(num_env, env_name, seed=100):
    if num_env == 1:

        def env_fn():
            env = make_make_env(env_name)()
            return env

    else:

        def env_fn():
            env = ParallelEnv(
                num_workers=num_env, create_env_fn=make_make_env(env_name)
            )
            return env

    policy = make_policy(env_name)

    torch.manual_seed(0)
    np.random.seed(0)
    ccollector = aSyncDataCollector(
        create_env_fn=env_fn,
        create_env_kwargs={},
        policy=policy,
        frames_per_batch=20,
        max_frames_per_traj=20,
        total_frames=300,
        pin_memory=False,
    )
    ccollector.set_seed(seed)
    for i, data in enumerate(ccollector):
        if i == 0:
            b1 = data
            ccollector.set_seed(seed)
        elif i == 1:
            b2 = data
        elif i == 2:
            b3 = data
        else:
            break
    assert_allclose_td(b1, b2)
    with pytest.raises(AssertionError):
        assert_allclose_td(b1, b3)
    ccollector.shutdown()


@pytest.mark.parametrize("num_env", [1, 3])
@pytest.mark.parametrize("env_name", ["conv", "vec"])
def test_collector_consistency(num_env, env_name, seed=100):
    if num_env == 1:

        def env_fn(seed):
            env = make_make_env(env_name)()
            env.set_seed(seed)
            return env

    else:

        def env_fn(seed):
            env = ParallelEnv(
                num_workers=num_env,
                create_env_fn=make_make_env(env_name),
                create_env_kwargs=[{"seed": s} for s in generate_seeds(seed, num_env)],
            )
            return env

    policy = make_policy(env_name)

    torch.manual_seed(0)
    np.random.seed(0)

    # Get a single rollout with dummypolicy
    env = env_fn(seed)
    rollout1a = env.rollout(policy=policy, max_steps=20, auto_reset=True)
    env.set_seed(seed)
    rollout1b = env.rollout(policy=policy, max_steps=20, auto_reset=True)
    rollout2 = env.rollout(policy=policy, max_steps=20, auto_reset=True)
    assert_allclose_td(rollout1a, rollout1b)
    with pytest.raises(AssertionError):
        assert_allclose_td(rollout1a, rollout2)
    env.close()

    collector = SyncDataCollector(
        create_env_fn=env_fn,
        create_env_kwargs={"seed": seed},
        policy=policy,
        frames_per_batch=20 * num_env,
        max_frames_per_traj=20,
        total_frames=200,
        device="cpu",
        pin_memory=False,
    )
    collector_iter = iter(collector)
    b1 = next(collector_iter)
    b2 = next(collector_iter)
    with pytest.raises(AssertionError):
        assert_allclose_td(b1, b2)

    if num_env == 1:
        # rollouts collected through DataCollector are padded using pad_sequence, which introduces a first dimension
        rollout1a = rollout1a.unsqueeze(0)
    assert (
        rollout1a.batch_size == b1.batch_size
    ), f"got batch_size {rollout1a.batch_size} and {b1.batch_size}"

    assert_allclose_td(rollout1a, b1.select(*rollout1a.keys()))
    collector.shutdown()


@pytest.mark.parametrize("num_env", [1, 3])
@pytest.mark.parametrize("collector_class", [SyncDataCollector, aSyncDataCollector])
@pytest.mark.parametrize("env_name", ["conv", "vec"])
def test_traj_len_consistency(num_env, env_name, collector_class, seed=100):
    """Tests that various frames_per_batch lead to the same results."""

    if num_env == 1:

        def env_fn(seed):
            env = make_make_env(env_name)()
            env.set_seed(seed)
            return env

    else:

        def env_fn(seed):
            env = ParallelEnv(
                num_workers=num_env, create_env_fn=make_make_env(env_name)
            )
            env.set_seed(seed)
            return env

    max_frames_per_traj = 20

    policy = make_policy(env_name)

    def make_frames_per_batch(frames_per_batch):
        return -(-frames_per_batch // num_env) * num_env

    collector1 = collector_class(
        create_env_fn=env_fn,
        create_env_kwargs={"seed": seed},
        policy=policy,
        frames_per_batch=1 * num_env,
        max_frames_per_traj=2000,
        total_frames=2 * num_env * max_frames_per_traj,
        device="cpu",
        seed=seed,
        pin_memory=False,
    )
    count = 0
    data1 = []
    for d in collector1:
        data1.append(d)
        count += d.shape[1]
        if count > max_frames_per_traj:
            break

    data1 = torch.cat(data1, 1)
    data1 = data1[:, :max_frames_per_traj]

    collector1.shutdown()
    del collector1

    collector10 = collector_class(
        create_env_fn=env_fn,
        create_env_kwargs={"seed": seed},
        policy=policy,
        frames_per_batch=10 * num_env,
        max_frames_per_traj=20,
        total_frames=2 * num_env * max_frames_per_traj,
        device="cpu",
        seed=seed,
        pin_memory=False,
    )
    count = 0
    data10 = []
    for d in collector10:
        data10.append(d)
        count += d.shape[1]
        if count > max_frames_per_traj:
            break

    data10 = torch.cat(data10, 1)
    data10 = data10[:, :max_frames_per_traj]

    collector10.shutdown()
    del collector10

    collector20 = collector_class(
        create_env_fn=env_fn,
        create_env_kwargs={"seed": seed},
        policy=policy,
        frames_per_batch=20 * num_env,
        max_frames_per_traj=2000,
        total_frames=2 * num_env * max_frames_per_traj,
        device="cpu",
        seed=seed,
        pin_memory=False,
    )
    count = 0
    data20 = []
    for d in collector20:
        data20.append(d)
        count += d.shape[1]
        if count > max_frames_per_traj:
            break

    collector20.shutdown()
    del collector20
    data20 = torch.cat(data20, 1)
    data20 = data20[:, :max_frames_per_traj]

    assert_allclose_td(data1, data20)
    assert_allclose_td(data10, data20)


@pytest.mark.skipif(not _has_gym, reason="test designed with GymEnv")
@pytest.mark.parametrize("static_seed", [True, False])
def test_collector_vecnorm_envcreator(static_seed):
    """
    High level test of the following pipeline:
     (1) Design a function that creates an environment with VecNorm
     (2) Wrap that function in an EnvCreator to instantiate the shared tensordict
     (3) Create a ParallelEnv that dispatches this env across workers
     (4) Run several ParallelEnv synchronously
    The function tests that the tensordict gathered from the workers match at certain moments in time, and that they
    are modified after the collector is run for more steps.

    """
    from torchrl.envs.libs.gym import GymEnv

    num_envs = 4
    env_make = EnvCreator(lambda: TransformedEnv(GymEnv(PENDULUM_VERSIONED), VecNorm()))
    env_make = ParallelEnv(num_envs, env_make)

    policy = RandomPolicy(env_make.action_spec)
    num_data_collectors = 2
    c = MultiSyncDataCollector(
        [env_make] * num_data_collectors, policy=policy, total_frames=int(1e6)
    )

    init_seed = 0
    new_seed = c.set_seed(init_seed, static_seed=static_seed)
    if static_seed:
        assert new_seed == init_seed
    else:
        assert new_seed != init_seed

    seed = init_seed
    for _ in range(num_envs * num_data_collectors):
        seed = seed_generator(seed)
    if not static_seed:
        assert new_seed == seed
    else:
        assert new_seed != seed

    c_iter = iter(c)
    next(c_iter)
    next(c_iter)

    s = c.state_dict()

    td1 = s["worker0"]["env_state_dict"]["worker3"]["_extra_state"]["td"].clone()
    td2 = s["worker1"]["env_state_dict"]["worker0"]["_extra_state"]["td"].clone()
    assert (td1 == td2).all()

    next(c_iter)
    next(c_iter)

    s = c.state_dict()

    td3 = s["worker0"]["env_state_dict"]["worker3"]["_extra_state"]["td"].clone()
    td4 = s["worker1"]["env_state_dict"]["worker0"]["_extra_state"]["td"].clone()
    assert (td3 == td4).all()
    assert (td1 != td4).any()

    del c


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda device found")
def test_update_weights(use_async):
    def create_env():
        return ContinuousActionVecMockEnv()

    n_actions = ContinuousActionVecMockEnv().action_spec.shape[-1]
    policy = SafeModule(
        torch.nn.LazyLinear(n_actions), in_keys=["observation"], out_keys=["action"]
    )
    policy(create_env().reset())

    collector_class = (
        MultiSyncDataCollector if not use_async else MultiaSyncDataCollector
    )
    collector = collector_class(
        [create_env] * 3,
        policy=policy,
        devices=[torch.device("cuda:0")] * 3,
        passing_devices=[torch.device("cuda:0")] * 3,
    )
    # collect state_dict
    state_dict = collector.state_dict()
    policy_state_dict = policy.state_dict()
    for worker in range(3):
        for k in state_dict[f"worker{worker}"]["policy_state_dict"]:
            torch.testing.assert_close(
                state_dict[f"worker{worker}"]["policy_state_dict"][k],
                policy_state_dict[k].cpu(),
            )

    # change policy weights
    for p in policy.parameters():
        p.data += torch.randn_like(p)

    # collect state_dict
    state_dict = collector.state_dict()
    policy_state_dict = policy.state_dict()
    # check they don't match
    for worker in range(3):
        for k in state_dict[f"worker{worker}"]["policy_state_dict"]:
            with pytest.raises(AssertionError):
                torch.testing.assert_close(
                    state_dict[f"worker{worker}"]["policy_state_dict"][k],
                    policy_state_dict[k].cpu(),
                )

    # update weights
    collector.update_policy_weights_()

    # collect state_dict
    state_dict = collector.state_dict()
    policy_state_dict = policy.state_dict()
    for worker in range(3):
        for k in state_dict[f"worker{worker}"]["policy_state_dict"]:
            torch.testing.assert_close(
                state_dict[f"worker{worker}"]["policy_state_dict"][k],
                policy_state_dict[k].cpu(),
            )

    collector.shutdown()
    del collector


@pytest.mark.parametrize(
    "collector_class",
    [MultiSyncDataCollector, MultiaSyncDataCollector, SyncDataCollector],
)
@pytest.mark.parametrize("exclude", [True, False])
def test_excluded_keys(collector_class, exclude):
    if not exclude and collector_class is not SyncDataCollector:
        pytest.skip("defining _exclude_private_keys is not possible")

    def make_env():
        return ContinuousActionVecMockEnv()

    dummy_env = make_env()
    obs_spec = dummy_env.observation_spec["observation"]
    policy_module = nn.Linear(obs_spec.shape[-1], dummy_env.action_spec.shape[-1])
    policy = Actor(policy_module, spec=dummy_env.action_spec)
    policy_explore = OrnsteinUhlenbeckProcessWrapper(policy)

    collector_kwargs = {
        "create_env_fn": make_env,
        "policy": policy_explore,
        "frames_per_batch": 30,
    }
    if collector_class is not SyncDataCollector:
        collector_kwargs["create_env_fn"] = [
            collector_kwargs["create_env_fn"] for _ in range(3)
        ]

    collector = collector_class(**collector_kwargs)
    collector._exclude_private_keys = exclude
    for b in collector:
        keys = b.keys()
        if exclude:
            assert not any(key.startswith("_") for key in keys)
        else:
            assert any(key.startswith("_") for key in keys)
        break
    collector.shutdown()
    dummy_env.close()


@pytest.mark.skipif(not _has_gym, reason="test designed with GymEnv")
@pytest.mark.parametrize(
    "collector_class",
    [
        MultiaSyncDataCollector,
        MultiSyncDataCollector,
        SyncDataCollector,
    ],
)
@pytest.mark.parametrize("init_random_frames", [0, 50])
@pytest.mark.parametrize("explicit_spec", [True, False])
def test_collector_output_keys(collector_class, init_random_frames, explicit_spec):
    from torchrl.envs.libs.gym import GymEnv

    out_features = 1
    hidden_size = 12
    total_frames = 200
    frames_per_batch = 20
    num_envs = 3

    net = LSTMNet(
        out_features,
        {"input_size": hidden_size, "hidden_size": hidden_size},
        {"out_features": hidden_size},
    )

    policy_kwargs = {
        "module": net,
        "in_keys": ["observation", "hidden1", "hidden2"],
        "out_keys": [
            "action",
            "hidden1",
            "hidden2",
            ("next", "hidden1"),
            ("next", "hidden2"),
        ],
    }
    if explicit_spec:
        hidden_spec = UnboundedContinuousTensorSpec((1, hidden_size))
        policy_kwargs["spec"] = CompositeSpec(
            action=UnboundedContinuousTensorSpec(),
            hidden1=hidden_spec,
            hidden2=hidden_spec,
            next=CompositeSpec(hidden1=hidden_spec, hidden2=hidden_spec),
        )

    policy = SafeModule(**policy_kwargs)

    env_maker = lambda: GymEnv(PENDULUM_VERSIONED)

    policy(env_maker().reset())

    collector_kwargs = {
        "create_env_fn": env_maker,
        "policy": policy,
        "total_frames": total_frames,
        "frames_per_batch": frames_per_batch,
        "init_random_frames": init_random_frames,
    }

    if collector_class is not SyncDataCollector:
        collector_kwargs["create_env_fn"] = [
            collector_kwargs["create_env_fn"] for _ in range(num_envs)
        ]

    collector = collector_class(**collector_kwargs)

    keys = {
        "action",
        "done",
        "collector",
        "hidden1",
        "hidden2",
        ("collector", "mask"),
        ("next", "hidden1"),
        ("next", "hidden2"),
        ("next", "observation"),
        "next",
        "observation",
        "reward",
        ("collector", "step_count"),
        ("collector", "traj_ids"),
    }
    b = next(iter(collector))

    assert set(b.keys(True)) == keys
    collector.shutdown()
    del collector


@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("passing_device", ["cuda", "cpu"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda device found")
def test_collector_device_combinations(device, passing_device):
    if (
        _os_is_windows
        and _python_is_3_10
        and passing_device == "cuda"
        and device == "cuda"
    ):
        pytest.skip("Windows fatal exception: access violation in torch.storage")

    def env_fn(seed):
        env = make_make_env("conv")()
        env.set_seed(seed)
        return env

    policy = dummypolicy_conv()

    collector = SyncDataCollector(
        create_env_fn=env_fn,
        create_env_kwargs={"seed": 0},
        policy=policy,
        frames_per_batch=20,
        max_frames_per_traj=2000,
        total_frames=20000,
        device=device,
        passing_device=passing_device,
        pin_memory=False,
    )
    batch = next(collector.iterator())
    assert batch.device == torch.device(passing_device)
    collector.shutdown()

    collector = MultiSyncDataCollector(
        create_env_fn=[
            env_fn,
        ],
        create_env_kwargs=[
            {"seed": 0},
        ],
        policy=policy,
        frames_per_batch=20,
        max_frames_per_traj=2000,
        total_frames=20000,
        devices=[
            device,
        ],
        passing_devices=[
            passing_device,
        ],
        pin_memory=False,
    )
    batch = next(collector.iterator())
    assert batch.device == torch.device(passing_device)
    collector.shutdown()

    collector = MultiaSyncDataCollector(
        create_env_fn=[
            env_fn,
        ],
        create_env_kwargs=[
            {"seed": 0},
        ],
        policy=policy,
        frames_per_batch=20,
        max_frames_per_traj=2000,
        total_frames=20000,
        devices=[
            device,
        ],
        passing_devices=[
            passing_device,
        ],
        pin_memory=False,
    )
    batch = next(collector.iterator())
    assert batch.device == torch.device(passing_device)
    collector.shutdown()


@pytest.mark.skipif(not _has_gym, reason="test designed with GymEnv")
@pytest.mark.parametrize(
    "collector_class",
    [
        SyncDataCollector,
        MultiaSyncDataCollector,
        MultiSyncDataCollector,
    ],
)
class TestAutoWrap:
    num_envs = 3

    @pytest.fixture
    def env_maker(self):
        from torchrl.envs.libs.gym import GymEnv

        return lambda: GymEnv(PENDULUM_VERSIONED)

    def _create_collector_kwargs(self, env_maker, collector_class, policy):
        collector_kwargs = {"create_env_fn": env_maker, "policy": policy}

        if collector_class is not SyncDataCollector:
            collector_kwargs["create_env_fn"] = [
                collector_kwargs["create_env_fn"] for _ in range(self.num_envs)
            ]

        return collector_kwargs

    @pytest.mark.parametrize("multiple_outputs", [False, True])
    def test_auto_wrap_modules(self, collector_class, multiple_outputs, env_maker):
        policy = WrappablePolicy(
            out_features=env_maker().action_spec.shape[-1],
            multiple_outputs=multiple_outputs,
        )
        collector = collector_class(
            **self._create_collector_kwargs(env_maker, collector_class, policy)
        )

        out_keys = ["action"]
        if multiple_outputs:
            out_keys.extend(f"output{i}" for i in range(1, 4))

        if collector_class is not SyncDataCollector:
            assert all(
                isinstance(p, TensorDictModule) for p in collector._policy_dict.values()
            )
            assert all(p.out_keys == out_keys for p in collector._policy_dict.values())
            assert all(p.module is policy for p in collector._policy_dict.values())
        else:
            assert isinstance(collector.policy, TensorDictModule)
            assert collector.policy.out_keys == out_keys
            assert collector.policy.module is policy

    def test_no_wrap_compatible_module(self, collector_class, env_maker):
        policy = TensorDictCompatiblePolicy(
            out_features=env_maker().action_spec.shape[-1]
        )

        collector = collector_class(
            **self._create_collector_kwargs(env_maker, collector_class, policy)
        )

        if collector_class is not SyncDataCollector:
            assert all(
                isinstance(p, TensorDictCompatiblePolicy)
                for p in collector._policy_dict.values()
            )
            assert all(
                p.out_keys == ["action"] for p in collector._policy_dict.values()
            )
            assert all(p is policy for p in collector._policy_dict.values())
        else:
            assert isinstance(collector.policy, TensorDictCompatiblePolicy)
            assert collector.policy.out_keys == ["action"]
            assert collector.policy is policy

    def test_auto_wrap_error(self, collector_class, env_maker):
        policy = UnwrappablePolicy(out_features=env_maker().action_spec.shape[-1])

        with pytest.raises(
            TypeError,
            match=(
                "Arguments to policy.forward are incompatible with entries in "
                "env.observation_spec."
            ),
        ):
            collector_class(
                **self._create_collector_kwargs(env_maker, collector_class, policy)
            )


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
