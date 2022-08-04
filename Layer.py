import numpy as np
import Tensor
class Layer:
    def __init__(self,name='layer',*args,**kwargs):
        self.name = name

    def forward(self,*args,**kwargs):
        raise NotImplementedError

    def backward(self,*args,**kwargs):
        raise NotImplementedError

    def parameters(self):
        return []

    def __call__(self,*args,**kwargs):
        return self.forward(*args,**kwargs)

    def __str__(self):
        return self.name

class Linear(Layer):
    def __init__(self,in_features,out_features,name='linear',weight_attr=Normal(),bias_attr=Constant(),*args,**kwargs):
        super().__init__(name=name,*args,**kwargs)
        self.weight = Tensor((in_features,out_features))
        self.weight.data = weight_attr(self.weight.data.shape)
        self.bias = Tensor((1,out_features))
        self.bias.data = bias_attr(self.bias.data.shape)
        self.input = None

    def forward(self,x):
        self.input = x
        return np.dot(x,self.weight.data) + self.bias.data

    def backward(self,gradient):
        self.weight.grad += np.dot(self.input.T,gradient)
        self.bias.grad += np.sum(gradient,axis=0,keepdims=True)
        input_grad = np.dot(gradient,self.weight.data.T)
        return input_grad

    def parameters(self):
        return [self.weight,self.bias]

class ReLU(Layer):
    def __init__(self,name='relu',*args,**kwargs):
        super().__init__(name=name,*args,**kwargs)
        self.activated = None`

    def forward(self,x):
        x[x<0] = 0
        self.activated = x
        return self.activated

    def backward(self,gradient):
        return gradient * (self.activated < 0)

