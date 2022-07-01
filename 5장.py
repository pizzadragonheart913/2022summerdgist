
from typing import OrderedDict
import numpy as np
import os, sys
import matplotlib.pyplot as plt
sys.path.append(os.pardir)
from mnist import load_mnist
def Relu(x):
    return np.maximum(0,x)
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))  

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)
            
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def softmax(x): # 입력받은 값을 출력으로 0~1사이의 값으로 모두 정규화함
    if x.ndim == 2: # 인덱싱이 두번 가능하다면
        x = x.T # T는 전치하는 메소드
        x = x - np.max(x, axis=0) # 최댓값을 뺴주는 연산
        y = np.exp(x) / np.sum(np.exp(x), axis=0) # 일반적인 e/sigma(e)
        return y.T #y의 전치행렬 반환

    x = x - np.max(x) # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x)) # 일반적인 e/sigma(e)

def sigmoid_grad( x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

class Affine:
    def __init__(self,W,b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.dx = None
        
    def forward(self,x):
        self.x = x # 엑스
        ret = np.dot(x,self.W) + self.b # (X *(dot product) W)+ b
        return ret

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW  = np.dot(self.x.T, dout)
        self.db = np.sum(dout,axis=0) # 편향의 역전파는 0번째 축의 합으로 구함.
        
        return dx
        # dx = dx.reshpae(*self.original)
        
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        
        return dx
    
    
class ReLu:
    def __init__(self):
        self.mask = None
        
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx
    
class DL:
    def __init__(self, input_size, hidden_size, output_size,weight_init_std=0.01):#가중치 설정하기
        self.network = {} #네트워크 초기화
        self.network['W1'] = weight_init_std * np.random.randn(input_size,hidden_size) #가우시안으로 분포를 만들고 계산하기
        self.network['W2'] = weight_init_std * np.random.randn(hidden_size,50) #입력층이 784개이고 1번 은닉층이 100개
        self.network['W3'] = weight_init_std * np.random.randn(50,10)
        self.network['b1'] = np.zeros(hidden_size) 
        self.network['b2'] = np.zeros(50)
        self.network['b3'] = np.zeros(10)
        
        
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.network['W1'], self.network['b1'])
        self.layers['ReLU1'] = ReLu()
        self.layers['Affine2'] = Affine(self.network['W2'], self.network['b2'])
        self.layers['ReLU2'] = ReLu()
        self.layers['Affine3'] = Affine(self.network['W3'], self.network['b3'])
        
        self.lastLayer = SoftmaxWithLoss()
        
        
        
        

    def predict(self,x):
        W1, W2, W3= self.network['W1'], self.network['W2'], self.network['W3']
        b1, b2, b3= self.network['b1'], self.network['b2'], self.network['b3']
        # 어파인 변환과 소프트맥스위드로스를 구현했기 때문에 새로 작성
        # 어파인변환으로  np.dot(x, W1) + b1를 대체
        # 렐루로 시그모이드 대체
        # 소프트맥스 위드 로스로 소프트맥스 대체
        # a1 = np.dot(x, W1) + b1
        # z1 = sigmoid(a1)
        # a2 = np.dot(z1, W2) + b2
        # z2 = sigmoid(a2)
        #a3 = np.dot(z2, W3) + b3
        #y = softmax(a3)
        #y = softmax(z2)
        
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
        
    def loss(self,x,t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy   

    def numeric_gradient(self,f, x):
        h = 1e-4  # 0.0001
        grad = np.zeros_like(x)

        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            tmp_val = x[idx]
            x[idx] = float(tmp_val) + h
            fxh1 = f(x)  # f(x+h)

            x[idx] = tmp_val - h
            fxh2 = f(x)  # f(x-h)
            grad[idx] = (fxh1 - fxh2) / (2*h)

            x[idx] = tmp_val  # 값 복원
            it.iternext()

        return grad

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = self.numeric_gradient(loss_W, self.network['W1'])
        grads['b1'] = self.numeric_gradient(loss_W, self.network['b1'])
        grads['W2'] = self.numeric_gradient(loss_W, self.network['W2'])
        grads['b2'] = self.numeric_gradient(loss_W, self.network['b2'])
        grads['W3'] = self.numeric_gradient(loss_W, self.network['W3'])
        grads['b3'] = self.numeric_gradient(loss_W, self.network['b3'])

        return grads
    
    def gradient(self, x, t):
        self.loss(x,t)
        
        dout= 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        
        for layer in layers:
            dout = layer.backward(dout)
        
        grads = {}
        grads['W3'] = self.layers['Affine3'].dW
        grads['b3'] = self.layers['Affine3'].db
        
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db

        return grads

if __name__ == '__main__':
    (x_train, t_train), (x_test,t_test) = load_mnist(normalize=True,one_hot_label=True)

    net = DL(input_size = 784, hidden_size = 100, output_size = 10)
    iters_num = 10000 #미니배치 반복횟수
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.11

    train_loss_list = [] 
    train_acc_list = []
    test_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1) 
    #1에폭 = 미니배치로 모든 데이터를 돌았을때

    for i in range(iters_num):
        # 미니배치 획득
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        #기울기 계산
        #grad = net.numerical_gradient(x_batch, t_batch)
        grad = net.gradient(x_batch, t_batch)
        #매개변수 갱신
        for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
            net.network[key] -= learning_rate * grad[key]
            
        # 학습 경과 기록
        loss = net.loss(x_batch, t_batch)
        train_loss_list.append(loss)
        
        if i % iter_per_epoch == 0:
            train_acc = net.accuracy(x_train, t_train)
            test_acc = net.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("train acc, test acc | " + str(train_acc) + "," + str(test_acc))
            
            
    markers = {'train' : 'o', 'test' : 's'}
    x = np.arange(len(train_acc_list))
    plt.plot(x, train_acc_list, label='train acc')
    plt.plot(x, test_acc_list, label= 'testacc', linestyle='--')
    plt.xlabel('epo')
    plt.ylabel('acc')
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()