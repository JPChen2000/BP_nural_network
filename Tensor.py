import numpy as np

class Tensor:
    def __init__(self,shape):
        self.data = np.zeros(shape=shape,dtype=np.float32)
        self.grad = np.zeros(shape=shape,dtype=np.float32)
    
    def clear_grad(self):
        self.grad = np.zeros_like(self.grad)

    def __str__(self):
        return "Tensor shape: {}, data: {}".format(self.data.shape, self.data)

class Initializer:
    def __init__(self,shape=None,name='initializer'):
        self.shape = shape
        self.name = name

    def __call__(self,*args,**kwargs):
        raise NotImplementedError

    def __str__(self):
        return self.name

class Normal(Initializer):
    def __init__(self,mean=0.,std=0.01,name='normal initializer',*args,**kwargs):
        super().__init__(name=name,*args,**kwargs)
        self.mean = mean
        self.std = std
    
    def __call__(self,shape=None,*args,**kwargs):
        if shape:
            self.shape = shape
        assert shape is not None, "the shape of initializer must not be None."
        return np.random.normal(self.mean,self.std,size=self.shape)

class Constant(Initializer):
    def __init__(self,value=0.,name='normal initializer',*args,**kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.value = value

    def __call__(self,shape=None,*args,**kwargs):
       if shape:
           self.shape = shape
       assert shape is not None, "the shape of initializer must not be None."
       return self.value + np.zeros(shape=self.shape)


