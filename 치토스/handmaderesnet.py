import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import numpy as np
import torch
import torchvision
import torchvision.datasets as dsets
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim

class basicblock(nn.Module):
    def __init__(self, in_planes, planes, stride = 1):
        super(basicblock,self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride = 1, padding=1, bias=1)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut =nn.Sequential()
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self,block, num_classes = 10):
        super(ResNet, self).__init__()
        self.in_planes= 64
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size= 3, stride = 1, padding =1, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block,64, 2, stride = 1) #num_block[2,2,2,2]
        self.layer2 = self._make_layer(block,128, 2, stride = 2) # 스트라이드를 두번째부터 줘서 너비와 높이가 줄어들 수 있게
        self.layer3 = self._make_layer(block,256, 2, stride = 2)
        self.layer4 = self._make_layer(block,512,2, stride = 2)
        self.linear = nn.Linear(512, num_classes) # 완전연결 레이어
        
    def _make_layer(self, block, planes,num_blocks, stride):
        layers = []
        if stride ==1: 
            layers.append(block(self.in_planes, planes, 1))
            self.in_planes = planes
            layers.append(block(self.in_planes, planes, 1))
            self.in_planes = planes
        elif stride == 2:
            layers.append(block(self.in_planes, planes, 2))
            self.in_planes = planes
            layers.append(block(self.in_planes, planes, 1))
            self.in_planes = planes
        print(layers)
        return nn.Sequential(*layers)
    
    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4) #풀링
        out = out.view(out.size(0), -1)
        out = self.linear(out) # 완전연결 레이어
        return out
    
def ResNet18():
    return ResNet(basicblock)

transform_train = transforms.Compose([
    transforms.ToTensor(),
])

transfrom_test = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = dsets.MNIST(root='MNIST_data/',
                          train = True,
                          transform=transforms.ToTensor(),
                          download = True)
test_dataset = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download= True)
dataset_size = len(train_dataset)
validation_size = int(dataset_size * 0.17)
train_size = int(dataset_size*0.83)
validation_dataset, train_dataset = torch.utils.data.random_split(train_dataset,[validation_size, train_size])

print(f"trainsize: {len(train_dataset)}")
print(f"trainsize: {len(test_dataset)}")
print(f"trainsize: {len(validation_dataset)}")

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size= 128,
                                          shuffle= True,
                                          num_workers=4)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size= 128,
                                          shuffle= True,
                                          num_workers=4)
        
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                          batch_size= 128,
                                          shuffle= True,
                                          num_workers=4)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

net = ResNet18()
net = net.to(device) ####################
net = torch.nn.DataParallel(net)############################
cudnn.benchmark = True# ####

learning_rate = 0.01
file_name = 'resnet18_mnist.pt'

criterion = nn.CrossEntropyLoss() # 로스함수는 크로스엔트로피로스를 사용하겠습니다.
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0002)
train_loss = []
def train(epoch):
    print('\n [ Train epoch: %d]' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader): #트레인로더를 이용해 batch_idx에 넣은 후에 
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        benign_outputs = net(inputs) # 트레인 로더에서 얻어온 인풋을 네트워크에 넣은 후에
        loss = criterion(benign_outputs, targets) # 교차 엔트로피 로스에 따라서
        loss.backward() # 역전파 진행
        
        optimizer.step() # 전체 모델 파라미터를 업데이트
        train_loss += loss.item() # 학습되는 도중 로스값을 출력
        _, predicted = benign_outputs.max(1)
        
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 100 == 0:
            print("\n current batch: ", str(batch_idx))
            print('Current bengin train accuracy: ', str(predicted.eq(targets).sum().item() / targets.size(0)))
            
            print('current benign train Loss :', loss.item())
            
    print('\nTotal benign train accuarcy:', 100. * correct / total)
    print('Total benign train loss:', train_loss)
    train_loss.append(train_loss)
    
    
valid_loss = []
def valid(epoch):
    with torch.no_grad():
        net.eval() # 평가모드로 바꿈
        loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(validation_loader): #테스트로더를 돌려가며 사용
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            outputs = net(inputs) # 단순히 모델에 입력을 넣기만 해줌
            loss += criterion(outputs, targets).item()

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

    print('\nTest accuarcy:', 100. * correct / total)
    print('Test average loss:', loss / total)
    valid_loss.append(loss)
    
test_loss = []
def test(epoch):
    print('\n[ Test epoch: %d ]' % epoch)
    net.eval() # 평가모드로 바꿈
    loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader): #테스트로더를 돌려가며 사용
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)

        outputs = net(inputs) # 단순히 모델에 입력을 넣기만 해줌
        loss += criterion(outputs, targets).item()

        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()

    print('\nTest accuarcy:', 100. * correct / total)
    print('Test average loss:', loss / total)

    state = {
        'net': net.state_dict()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/' + file_name)
    print('Model Saved!')
    test_loss.append(loss)
    
def adjust_learning_rate(optimizer, epoch): # 학습률 다섯번 에포크 돌고 낮춰서 조절
    lr = learning_rate
    if epoch >= 5:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
        
if __name__ == "__main__":
    for epoch in range(0, 10):
        adjust_learning_rate(optimizer, epoch)
        train(epoch)        
        valid(epoch)
        
    for epoch in range(0, 10):
        test(epoch)
        print("trainacc:",*train_loss)
        print("validacc:",*valid_loss)
        print("testacc:", *test_loss)

import matplotlib.pyplot as plt

x = np.arange(0,10,1)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='best', ncol=2)
plt.plot(x,train_loss, 'r',x,valid_loss,'b',x,test_loss,'g')

plt.show()