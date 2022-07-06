import numpy as np
import cv2
import matplotlib.pyplot as plt    
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision.datasets as dsets
import torch.nn.init
from torchvision import transforms

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
        out = out.view(-1)
        out = self.fc(out)
        return out
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = torch.load('model.pt')
model.eval()

with torch.no_grad():
    image = Image.open('testnum/0.png').convert('L') ################### 여기에서 숫자만 바꾸기
    tf_toTensor = transforms.ToTensor() 
    img_RGB_tensor_from_PIL = tf_toTensor(image)
    print(img_RGB_tensor_from_PIL.size())
    prediction = model(img_RGB_tensor_from_PIL)
    ans = prediction.max(dim=0)
    print(ans)
    
    print(prediction)

