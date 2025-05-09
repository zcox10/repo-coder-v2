import unittest
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F

from betty.engine import Engine
from betty.configs import Config, EngineConfig
from betty.problems import ImplicitProblem


class ChildNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.w = torch.nn.Parameter(torch.zeros(20))

    def forward(self, inputs):
        outs = inputs @ self.w
        return outs, self.w


class ParentNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.w = torch.nn.Parameter(torch.ones(20))

    def forward(self):
        return self.w


class Outer(ImplicitProblem):
    def training_step(self, batch):
        inputs, targets = batch
        outs = self.inner(inputs)[0]
        loss = F.binary_cross_entropy_with_logits(outs, targets)
        return loss

    def param_callback(self):
        for p in self.trainable_parameters():
            p.data.clamp_(min=1e-8)


class Inner(ImplicitProblem):
    def training_step(self, batch):
        inputs, targets = batch
        outs, params = self.module(inputs)
        loss = (
            F.binary_cross_entropy_with_logits(outs, targets)
            + 0.5
            * (
                params.unsqueeze(0) @ torch.diag(self.outer()) @ params.unsqueeze(1)
            ).sum()
        )
        return loss

    def on_inner_loop_start(self):
        self.module.w.data.zero_()


class EngineTest(unittest.TestCase):
    def setUp(self):
        # data preparation
        w_gt = np.random.randn(20)
        x = np.random.randn(100, 20)
        y = x @ w_gt + 0.1 * np.random.randn(100)
        y = (y > 0).astype(float)

        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.5)
        x_train, y_train = (
            torch.from_numpy(x_train).float(),
            torch.from_numpy(y_train).float(),
        )
        x_val, y_val = (
            torch.from_numpy(x_val).float(),
            torch.from_numpy(y_val).float(),
        )

        # data_loader
        self.train_loader = [(x_train, y_train)]
        self.valid_loader = [(x_val, y_val)]

        # module
        self.train_module = ChildNet()
        self.valid_module = ParentNet()

        # optimizer
        self.train_optimizer = torch.optim.SGD(self.train_module.parameters(), lr=0.1)
        self.valid_optimizer = torch.optim.SGD(
            self.valid_module.parameters(), lr=0.1, momentum=0.9
        )

        self.train_config = Config(unroll_steps=10)
        self.valid_config = Config()
        self.engine_config = EngineConfig(train_iters=20)

        # problem
        self.outer = Outer(
            name="outer",
            module=self.valid_module,
            optimizer=self.valid_optimizer,
            train_data_loader=self.valid_loader,
            config=self.valid_config,
        )
        self.inner = Inner(
            name="inner",
            module=self.train_module,
            optimizer=self.train_optimizer,
            train_data_loader=self.train_loader,
            config=self.train_config,
        )

        problems = [self.outer, self.inner]
        u2l = {self.outer: [self.inner]}
        l2u = {self.inner: [self.outer]}
        dependencies = {"l2u": l2u, "u2l": u2l}

        self.engine = Engine(
            config=self.engine_config, problems=problems, dependencies=dependencies
        )

    def test_find_paths(self):
        found_paths = self.outer.paths
        self.assertTrue(len(found_paths) == 1)
        self.assertTrue(len(found_paths[0]) == 3)
        self.assertTrue(found_paths[0][0] == self.outer)
        self.assertTrue(found_paths[0][1] == self.inner)
        self.assertTrue(found_paths[0][2] == self.outer)

    def test_check_leaf(self):
        self.assertTrue(self.inner.leaf)
        self.assertFalse(self.outer.leaf)

    def test_set_problem_attr(self):
        self.assertTrue(hasattr(self.engine, "inner"))
        self.assertTrue(hasattr(self.engine, "outer"))

    def test_parse_dependency(self):
        self.assertTrue(self.outer in self.inner.parents)
        self.assertTrue(self.inner in self.outer.children)
        self.assertTrue(hasattr(self.inner, "outer"))
        self.assertTrue(hasattr(self.outer, "inner"))

    def test_train_step(self):
        for i in range(10):
            self.assertTrue(self.inner.count == i)
            self.assertTrue(self.outer.count == 0)
            self.engine.train_step()
        self.assertTrue(self.inner.count == 10)
        self.assertTrue(self.outer.count == 1)


if __name__ == "__main__":
    unittest.main()
