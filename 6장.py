import numpy as np
import matplotlib.pyplot as plt

class SGD:
    def __init__(self, lr):
        self.lr = lr #학습률 저장
    
    def update(self, network, grads): # 네트워크는 딕셔너리 변수, W와 기울기를 저장함.
        for key in network.keys():
            network[key] -= self.lr * grads[key] 

class Momentum:
    def __init__(self, lr = 0.01, momentum = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, network, grads):
        if self.v is None:
            self.v = {}
            for key, val in network.items():
                self.v[key] = np.zeros_like(val)
                
        for key in network.keys():
            self.v[key] = self.momentum*self.v[key]
            
class Adagrad:
    def __init__(self, lr = 0.01):
        self.lr = lr
        self.h = None
        
    def update(self, network, grads):
        if self.h is None:
            self.h = {}
            for key, val in network.items():
                self.h[key] = np.zeros_like(val)
                
        for key in network.keys():
            self.h[key] += grads[key] * grads[key]
            network[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)