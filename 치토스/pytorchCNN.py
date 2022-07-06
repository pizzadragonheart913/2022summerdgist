import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)# 시드지정
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

learning_rate = 0.001
training_epochs = 15
batch_size = 100

mnist_train = dsets.MNIST(root='MNIST_data/',
                          train = True,
                          transform=transforms.ToTensor(),
                          download = True)
mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download= True)

data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size= batch_size,
                                          shuffle= True,
                                          drop_last = True)

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Sequential( # 입력은 mnist 데이터이기 때문에 [1,28,28]로 들어옴 거기에 제로패딩 1 추가.[1,30,30]이 됨 얘들을 크기가 3인 필터로 1씩 스트라이드 하여 콘볼루션 연산
            torch.nn.Conv2d(1,32, kernel_size= 3, stride = 1, padding = 1), # 각각 32개의 필터들에 대해 연산 하기 때문에 [32,28,28]의 텐서가 됨.
            torch.nn.ReLU(), 
            torch.nn.MaxPool2d(kernel_size=2,stride=2) # 풀링연산 2X2를 두칸씩 해서 [32, 28, 28]의 텐서를 [32, 14, 14]로 바꿈
        )
        self.layer2 = torch.nn.Sequential( #두 번째 콘볼루션 레이어는 채널이 32이므로 입력을 32개를 받고 출력을 64개로 할거임. 얘도 마찬가지로 제로패딩 1칸 넣어줘서 [32,15, 15]으로 만들고 
            torch.nn.Conv2d(32,64, kernel_size=3, stride=1, padding=1), # 크기가 3인 필터로 스트라이드 1씩 돌려서 [64, 14, 14]을 만듦.
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,stride=2)) # 얘도 풀링연산 2X2를 스트라이드 2로 하여 [64, 7, 7]로 만듦.
        self.fc = torch.nn.Linear(7 * 7 * 64, 10, bias = True) #input을 7, 7 , 64로 받겠다는 것이고 output을 10으로 내보내겠다는 뜻.
        torch.nn.init.kaiming_uniform_(self.fc.weight) # 초깃값을 카이밍 히 초기화를 통해 가중치를 초기화시킴. 나는 렐루를 쓰기 때문에 하비에르보다 카이밍히가 낫다고 판단함.
        
    def forward(self, x):
        out = self.layer1(x) #순전파를 진행함. 입력을 레이어 1에 삽입 하면 콘볼루션, 렐루, 풀링을 거쳐 뱉음
        out = self.layer2(out) # 1layer를 거친 출력을 다시 2layer에 삽입.
        out = out.view(out.size(0), -1) # 출력으로는 [7,7,64] 가 나오는데 이를 fc에 대입 해준다. 
        out = self.fc(out)
        return out
    
model = CNN().to(device)

criterion = torch.nn.CrossEntropyLoss().to(device) # 비용함수를 정하는 과정 
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate) # 옵티마이저(경사하강법 사용할 때 아담을 쓰겠다는 뜻)

total_batch = len(data_loader) # 총 배치의 수는 600이 출력될 것. 왜냐하면 60000개의 학습데이터를 100의 배치사이즈로 나누면 600개의 배치가 나오니깐
print("학습 시작")

for epoch in range(training_epochs):
    avg_cost = 0
    
    for X,Y in data_loader: # 미니배치 단위로 데이터 get!
        X = X.to(device)
        Y = Y.to(device)
        
        optimizer.zero_grad()
        
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()
        
        avg_cost += cost / total_batch
        
    print('[epoch : {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))
    
print('learning finish')
####### 여기까지가 학습하는 과정

with torch.no_grad():
    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)
    
    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('accuracy', accuracy.item())

torch.save(model, 'model.pt')