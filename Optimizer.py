import numpy as np


class Optimizer:
    def __init__(self,parameters,learning_rate=0.001,weigth_decay=0.0,decay_type='l2'):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.weigth_decay = weigth_decay
        self.decay_type = decay_type

    def step(self):
        raise NotImplementedError

    def clear_grad(self):
        for par in self.parameters:
            par.clear_grad()

    def get_decay(self,gra):
        if self.decay_type == 'l1':
            return self.weigth_decay
        elif self.decay_type == 'l2':
            return self.weigth_decay * gra

class SGD(Optimizer):
    def __init__(self,momentum=0.9,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.momentum = momentum
        self.velocity = []
        for par in self.parameters:
            self.velocity.append(np.zeros_like(par.grad))

    def step(self):
        for p,v in zip(self.parameters,self.velocity):
            decay = self.get_decay(p.grad)
            v = self.momentum * v + p.grad + decay
            p.data = p.data - self.learning_rate * v

class MSE(Layer):
    def __init__(self,name='mse',reduction='mean',*args,**kwargs):
        super().__init__(name=name,*args,**kwargs)
        self.reduction = reduction
        self.pred = None
        self.target = None

    def forward(self,y,target):
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
