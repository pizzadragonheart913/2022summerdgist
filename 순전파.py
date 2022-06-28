import numpy as np
import os, sys
sys.path.append(os.pardir)
from mnist import load_mnist


def relu(x):
    return np.maximum(0,x)

def getdata():
    (x_train, t_train), (x_test,t_test) = load_mnist(normalize=True,flatten=True,one_hot_label=False)# normal : 0~1로 불러오기,
    # flatten : 28*28 이미지를 1*784로 바꿔줌 one hot label : 0~9까지 정답인 것만 1로 만듦.
    return x_test, t_test # 테스트 데이터 셋만 활용한다는 뜻

def init_network():#가중치 설정하기
    network = {} #네트워크 초기화
    network['W1'] = np.random.randn(784,100) #가우시안으로 분포를 만들고 계산하기
    network['W2'] = np.random.randn(100,50) #입력층이 784개이고 1번 은닉층이 100개
    network['W3'] = np.random.randn(50,10) #2번 은닉층이 50개이고 출력층이 10개임.
    network['b1'] = np.zeros(100) 
    network['b2'] = np.zeros(50)
    network['b3'] = np.zeros(10)    
    
    return network
def predict(network,x):
    W1,W2,W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = relu(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = relu(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    
    return y
    
def softmax(a): # 입력받은 값을 출력으로 0~1사이의 값으로 모두 정규화함
    exp_a=np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y


x, t = getdata()
network = init_network()
accuracy_cnt = 0

for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt+=1
        
print("accuracy:",str(float(accuracy_cnt)/len(x)))