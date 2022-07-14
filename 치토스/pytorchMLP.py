# 3 레이어 다중 퍼셉트론 구현
# 박현규 2022summer dgist

# 의존성 정리
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import random

# 디바이스를 엔비디아 있으면 엔비디아 지피유 쓰고 아니면 씨피유로
device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(777) # 랜덤에 시드를 주어 예측가능하게하기
torch.manual_seed(777)

# 지피유라면 시드를 쿠다 시드로 설정해줘야
if device =="cuda":
    torch.cuda.manual_seed_all(777)
    
# 하이퍼 파라미터를 설정하기
learning_rate = 0.001 #학습률을 0.001
training_epochs = 15 # 에포크는 15
batch_size = 100 # 배치사이즈
drop_prob = 0.3 # 드롭아웃 확률 지정

# 트레인 데이터
mnist_train = dsets.MNIST(root='MNIST_data/',train=True, 
                          transform= transforms.ToTensor(),download=True) #트레인데이터라 트레인에 트루
# 테스트 데이터
mnist_test = dsets.MNIST(root = 'MNIST_data/',train=False,
                         transform= transforms.ToTensor(),download=True)#테스트데이터라 트레인에 폴스
#####################
# 추후 트레인 데이터를 나누어서 벨리데이션도 만들 수 있으면 만들기
#####################
data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size= batch_size,
                                          shuffle = True,
                                          drop_last = True) # drop_last는 트루로 주게 되면 마지막 배치가 배치사이즈보다 작을때 마지막은 배치를 사용하지 않음


# 블럭은 28*28 을 flatten 시키고 히든레이어의 노드수가 100, 50, 10인 히든레이어를ㄹ 거치며 학습함
linear1 = torch.nn.Linear(784, 100, bias = True) # bias 는 편향을 학습할지 말지 결정하는 옵션
linear2 = torch.nn.Linear(100, 50, bias = True)
linear3 = torch.nn.Linear(50, 10, bias = True)

relu = torch.nn.ReLU() # 활성화 함수로 렐루를 사용
dropout = torch.nn.Dropout(p = drop_prob) # 드롭아웃: 무작위로 히든레이어의 노드를 삭제해서 오버피팅을 막는 기능인데 아주 고맙게도 토치에 있음

##################### 가중치를 초기화  하는 블럭 카이밍 히 사용 (활성화함수가 렐루이기 때문)
torch.nn.init.kaiming_uniform_(linear1.weight) 
torch.nn.init.kaiming_uniform_(linear2.weight)
torch.nn.init.kaiming_uniform_(linear3.weight)
##########################################

# nn.sequential은 괄호 안의 내용을 순서대로 실행하는 것.
model = torch.nn.Sequential(linear1, relu, dropout,
                            linear2, relu, dropout,
                            linear3)
criterion = torch.nn.CrossEntropyLoss().to(device) 
# 파이토치느 소프트맥스 크로스엔트로피 를 합쳐서 제공하기 때문에 사용하면 된다.
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
#gradient descent할 때 아담을 사용함.
print(len(data_loader))
total_batch = len(data_loader) # 
model.train()
for epoch in range(training_epochs):
    avg_cost = 0
    
    for X,Y in data_loader:
        X = X.view(-1, 28 * 28).to(device)
        Y = Y.to(device) #to(device) 는 cpu로 보내버린다는 뜻
        
        optimizer.zero_grad() #매번 백프로파게이션 할때 값을 초기화 해줘야 함.
        hypothesis = model(X)
        cost = criterion(hypothesis, Y) # 실제 답과 추론치의 비교를 크로스엔트로피 + 소프트맥스로 비교
        cost.backward() # 코스트를 미분하여 기울기를 계산
        optimizer.step() #W 와 b를 업데이트
        
        avg_cost += cost / total_batch
        
    print('Epoch : ' '%04d' % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))
    # 에포크를 출력하는 단계이면서, 소수점 9째자리 까지 avg_cost를 출력하는 문
print('Learning Finished')

with torch.no_grad(): # 그라디언트 연산 옵션을 끌 때 사용하는 컨텍스트 매니져
    model.eval() # 모델을 평가하는 함수 드롭아웃을 안함
    X_test = mnist_test.test_data.view(-1, 28*28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)
    
    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())
    #임의의 자료를 가져와서 대입하는 과정
    # r = random.randint(0, len(mnist_test) - 1)
    # X_single_data = mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float().to(device)
    # Y_single_data = mnist_test.test_labels[r: r + 1].to(device)
    from PIL import Image
    import numpy as np
    a= Image.open('test1.png')
    a = a.convert("L")
    array = np.array(a)
    array= array.reshape(-1,784)
    X_test_data = torch.Tensor(array)
    # print('Label : ', Y_single_data.item())
    single_prediction = model(X_test_data)
    print('Prediction: ', torch.argmax(single_prediction, 1).item())
    