import numpy as np
import os, sys
import matplotlib.pyplot as plt
sys.path.append(os.pardir)
from mnist import load_mnist

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

class DL:
    def __init__(self, input_size, hidden_size, output_size,weight_init_std=0.01):#가중치 설정하기
        self.network = {} #네트워크 초기화
        self.network['W1'] = weight_init_std * np.random.randn(input_size,hidden_size) #가우시안으로 분포를 만들고 계산하기
        self.network['W2'] = weight_init_std * np.random.randn(hidden_size,output_size) #입력층이 784개이고 1번 은닉층이 100개
        #self.network['W3'] = weight_init_std * np.random.randn(50,10)
        self.network['b1'] = np.zeros(hidden_size) 
        self.network['b2'] = np.zeros(output_size)
        #self.network['b3'] = np.zeros(10)

    def predict(self,x):
        W1,W2= self.network['W1'], self.network['W2']#, self.network['W3']
        b1, b2= self.network['b1'], self.network['b2']#, self.network['b3']
        
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid(a2)
        #a3 = np.dot(z2, W3) + b3
        #y = softmax(a3)
        y = softmax(z2)
        return y
        
    def loss(self,x,t):
        y = self.predict(x)
        
        return cross_entropy_error(y,t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
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

        return grads
    
    def gradient(self, x, t):
        W1, W2 = self.network['W1'], self.network['W2']#, self.network['W3']
        b1, b2 = self.network['b1'], self.network['b2']#, self.network['b3']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num  
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads

if __name__ == '__main__':
    (x_train, t_train), (x_test,t_test) = load_mnist(normalize=True,one_hot_label=True)

    net = DL(input_size = 784, hidden_size = 50, output_size = 10)
    iters_num = 10000 #미니배치 반복횟수
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1

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
        for key in ('W1', 'b1', 'W2', 'b2'):
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