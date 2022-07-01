from re import L
import numpy as np

class Mullayer:
    def __init__(self): # 변수 유지용 초기화 함수
        self.x = None # 순전파로 계산한 x와 y를 저장함.
        self.y = None
    def forward(self,x,y):
        self.x = x
        self.y = y
        out = x*y
        
        return out
    
    def backward(self, dout):
        dx = dout*self.y # 곱셈노드에서는 x를 구하기 위해 y값을 곱해준다.
        dy = dout * self.x
        
        return dx,dy

class Addlayer():
    def __init__(self):
        pass # 왜 패스를 하는건지?????????
    
    def forward(self, x, y):
        out = x + y
        return out
    
    def backward(self, dout):
        dx = dout * 1
        dy = dout *1 
        return dx, dy 
        
        
def orangeandapple():
    apple = 100
    tax = 1.1
    apple_ea = 2
    orange = 150
    orange_ea = 3
    dprice = 1
    apple_ea_mul = Mullayer()
    orange_ea_mul = Mullayer()
    add_apple_orange = Addlayer()
    tax_mul = Mullayer()
    
    apple_price = apple_ea_mul.forward(apple, apple_ea)
    orange_price = orange_ea_mul.forward(orange, orange_ea)
    total_price = add_apple_orange.forward(apple_price, orange_price)
    final_price = tax_mul.forward(total_price, tax)
    print("순전파",final_price)
    dtotal,dtax = tax_mul.backward(dprice)
    dorange_price, dapple_price = add_apple_orange.backward(dtotal)
    dorange, dorange_ea = orange_ea_mul.backward(dorange_price)
    dapple, dapple_ea = apple_ea_mul.backward(dapple_price)
    
    print("tax",dtax,"dtotal",dtotal,"dorange_price",dorange_price,"dapple_price",dapple_price, "dorange",dorange,"dorange_ea", dorange_ea,"dapple",dapple,"dapple_ea",dapple_ea)
    
    
    
    
orangeandapple()

def apple():
    apple = 100
    tax = 1.1
    ea = 2
    dprice = 1
    applemullayer = Mullayer()
    taxmullayer = Mullayer()

    first = applemullayer.forward(apple,ea)
    result = taxmullayer.forward(first, tax)
    print("순전파",result)
    
    dtotal, dtax = taxmullayer.backward(dprice)
    dea, dprice = applemullayer.backward(dtotal)
    
    print("total, tax, ea, price",dtotal, dtax, dea, dprice)
    
# apple()



# total = 
# 배치용 아핀이랑 그냥 아핀 비교해서 추가할거 추가하기
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