import numpy as np
import Tensor
from Layer import *


class Sequential:
    def __init__(self, *args, **kwargs):
        self.graphs = []
        self._parameters = []
        for arg_layer in args:
            if isinstance(arg_layer, Layer):
                self.graphs.append(arg_layer)
                self._parameters += arg_layer.parameters()

    def add(self, layer):
        assert isinstance(layer, Layer), "The type of added layer must be Layer, but got {}.".format(type(layer))
        self.graphs.append(layer)
        self._parameters += layer.parameters()

    def forward(self, x):
        for graph in self.graphs:
            x = graph(x)
        return x

    def backward(self, grad):
        # grad backward in inverse order of graph
        for graph in self.graphs[::-1]:
            grad = graph.backward(grad)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __str__(self):
        string = 'Sequential:\n'
        for graph in self.graphs:
            string += graph.__str__() + '\n'
        return string

    def parameters(self):
        return self._parameters
