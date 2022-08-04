import numpy as np
from Tensor import Tensor, Normal, Constant


class Layer:
    def __init__(self, name='layer', *args, **kwargs):
        self.name = name

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def parameters(self):
        return []

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __str__(self):
        return self.name


class Linear(Layer):
    def __init__(
            self,
            in_features,
            out_features,
            name='linear',
            weight_attr=Normal(),
            bias_attr=Constant(),
            *args,
            **kwargs
    ):
        super().__init__(name=name, *args, **kwargs)
        self.weights = Tensor((in_features, out_features))
        self.weights.data = weight_attr(self.weights.data.shape)
        self.bias = Tensor((1, out_features))
        self.bias.data = bias_attr(self.bias.data.shape)
        self.input = None

    def forward(self, x):
        self.input = x
        output = np.dot(x, self.weights.data) + self.bias.data
        return output

    def backward(self, gradient):
        self.weights.grad += np.dot(self.input.T, gradient)  # dy / dw
        self.bias.grad += np.sum(gradient, axis=0, keepdims=True)  # dy / db
        input_grad = np.dot(gradient, self.weights.data.T)  # dy / dx
        return input_grad

    def parameters(self):
        return [self.weights, self.bias]

    def __str__(self):
        string = "linear layer, weight shape: {}, bias shape: {}".format(self.weights.data.shape, self.bias.data.shape)
        return string


class ReLU(Layer):
    def __init__(self, name='relu', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.activated = None

    def forward(self, x):
        x[x < 0] = 0
        self.activated = x
        return self.activated

    def backward(self, gradient):
        return gradient * (self.activated > 0)

