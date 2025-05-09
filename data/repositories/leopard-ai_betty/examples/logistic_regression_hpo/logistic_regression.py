import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F

from betty.engine import Engine
from betty.configs import Config, EngineConfig
from betty.problems import ImplicitProblem, HigherIterativeProblem

# hyperparameters
DATA_NUM = 1000
DATA_DIM = 20

# data preparation
w_gt = np.random.randn(DATA_DIM)
x = np.random.randn(DATA_NUM, DATA_DIM)
y = x @ w_gt + 0.1 * np.random.randn(DATA_NUM)
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

data = {"train": (x_train, y_train), "valid": (x_val, y_val)}
torch.save(data, "data.t7")


def make_data_loader(xs, ys):
    datasets = [(xs, ys)]

    return datasets


class ChildNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.w = torch.nn.Parameter(torch.zeros(DATA_DIM))

    def forward(self, inputs):
        outs = inputs @ self.w
        return outs, self.w


class ParentNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.w = torch.nn.Parameter(torch.ones(DATA_DIM))

    def forward(self):
        return self.w


class Parent(ImplicitProblem):
    def training_step(self, batch):
        inputs, targets = batch
        outs = self.inner(inputs)[0]
        loss = F.binary_cross_entropy_with_logits(outs, targets)
        return loss

    def configure_train_data_loader(self):
        return make_data_loader(x_val, y_val)

    def configure_module(self):
        return ParentNet()

    def configure_optimizer(self):
        return torch.optim.SGD(self.module.parameters(), lr=1, momentum=0.9)

    def param_callback(self):
        for p in self.trainable_parameters():
            p.data.clamp_(min=1e-8)


class Child(HigherIterativeProblem):
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

    def configure_train_data_loader(self):
        return make_data_loader(x_train, y_train)

    def configure_module(self):
        return ChildNet()

    def configure_optimizer(self):
        return torch.optim.SGD(self.module.parameters(), lr=0.1)

    def on_inner_loop_start(self):
        self.module.w.data.zero_()


engine_config = EngineConfig(train_iters=10000, logger_type="none")
parent_config = Config(log_step=1, first_order=False, retain_graph=True)
child_config = Config(unroll_steps=100)

parent = Parent(name="outer", config=parent_config)
child = Child(name="inner", config=child_config)

problems = [parent, child]
u2l = {parent: [child]}
l2u = {child: [parent]}
dependencies = {"l2u": l2u, "u2l": u2l}

engine = Engine(config=engine_config, problems=problems, dependencies=dependencies)
engine.run()
