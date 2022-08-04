import numpy as np
import Tensor
import Layer

class Sequential:
    def __init__(self,*args,**kwargs):
        self.graphs = []
        self.parameters = []
        for arg_layer in args:
            if isinstance(arg_layer,Layer):
                self.graphs.append(arg_layer)
                self.parameters += arg_layer.parameters()

    def add(self,layer):
        self.graphs.append(layer)
        self.parameters += layer.parameters()

    def forward(self,x):
        for graph in self.graphs:
            x = graph(x)
        return x

    def backward(self,grad):
        for graph in self.graphs:
            grad = graph.backward(grad)

    def __call__(self,*args,**kwargs):
        return self.forward(*args,**kwargs)

    def parameters(self):
        return self.parameters
