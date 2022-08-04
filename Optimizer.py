import numpy as np
from Layer import *


class Optimizer:
    def __init__(self, parameters, learning_rate=0.001, weight_decay=0.0, decay_type='l2'):
        assert decay_type in ['l1', 'l2'], "only support decay_type 'l1' and 'l2', but got {}.".format(decay_type)
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.decay_type = decay_type

    def step(self):
        raise NotImplementedError

    def clear_grad(self):
        for p in self.parameters:
            p.clear_grad()

    def get_decay(self, g):
        if self.decay_type == 'l1':
            return self.weight_decay
        elif self.decay_type == 'l2':
            return self.weight_decay * g


class SGD(Optimizer):
    def __init__(self, momentum=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum
        self.velocity = []
        for p in self.parameters:
            self.velocity.append(np.zeros_like(p.grad))

    def step(self):
        for p, v in zip(self.parameters, self.velocity):
            decay = self.get_decay(p.grad)
            v = self.momentum * v + p.grad + decay  # 动量计算
            p.data = p.data - self.learning_rate * v


class MSE(Layer):
    def __init__(self, name='mse', reduction='mean', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        assert reduction in ['mean', 'none',
                             'sum'], "reduction only support 'mean', 'none' and 'sum', but got {}.".format(reduction)
        self.reduction = reduction
        self.pred = None
        self.target = None

    def forward(self, y, target):
        assert y.shape == target.shape, "The shape of y and target is not same, y shape = {} but target shape = {}".format(
            y.shape, target.shape)
        self.pred = y
        self.target = target
        loss = 0.5 * np.square(y - target)
        if self.reduction is 'mean':
            return loss.mean()
        elif self.reduction is 'none':
            return loss
        else:
            return loss.sum()

    def backward(self):
        gradient = self.pred - self.target
        return gradient

