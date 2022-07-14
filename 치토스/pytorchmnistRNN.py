import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import os
import matplotlib.pyplot as plt
import numpy as np
torch.__version__

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(777)
num_gpu = 1
if torch.cuda.device_count() > 1: # gpu 개수 파악
    num_gpu = torch.cuda.device_count()

print("let's use", num_gpu, "GPU!")
print('our device', device)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first = True)
        self.gru = nn.GRU(input_size,self.hidden_size,self.num_layers, batch_first = True)
        self.fc = nn.Linear(self.hidden_size, num_classes)
        
    def forward(self, x, rnn):
        if rnn == 'lstm':
            rnn_layer = self.lstm
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            out, _ = self.lstm(x,(h0,c0))
        else:
            rnn_layer = self.gru
            h = torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(device)
            out, _ = self.gru(x,h)
            
        out = self.fc(out[:, -1, :])
        return out
    
    
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 10
learning_rate = 0.01

model = RNN(input_size, hidden_size, num_layers,num_classes).to(device)

for p in model.parameters():
    print(p.size())
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model_hp = count_parameters(model)
print('model"s hyper parameters', model_hp)

train_loader = torch.utils.data.DataLoader(datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()),batch_size=batch_size, shuffle=True)
print(len(train_loader)) # 600
test_loader = torch.utils.data.DataLoader(datasets.MNIST('data', train=False, transform=transforms.ToTensor()),batch_size=1000)
print(len(test_loader)) # 10

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

model.train()
total_loss = 0
total_acc = 0
train_loss = []
train_accuracy = []
i =0

for epoch in range(num_epochs):
    for data, target in train_loader:
        data = data.reshape(-1, sequence_length, input_size).to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        output= model(data,"lstm")
        loss = criterion(output, target)
        
        loss.backward()
        
        total_loss += loss
        train_loss.append(total_loss/i)
        optimizer.step()
        
        prediction = output.data.max(1)[1]
        accuracy = prediction.eq(target.data).sum()/batch_size* 100
        
        total_acc += accuracy
        
        train_accuracy.append(total_acc/i)
        
        if i%10 == 0:
            print('epoch:{}\t train step: {}\t loss: {:,3f}\tAccuracy: {:,3f}'.format(epoch+1,i,loss,accuracy))
        i += 1
        
    print('epoch: {} finished'.format(epoch+1))
    
    
plt.figure()
plt.plot(np.arange(len(train_loss)), train_loss)
plt.show()

plt.figure()
plt.plot(np.arrange(len(train_accuracy)), train_accuracy)
plt.show()

with torch.no_grad():
    model.eval()
    correct = 0
   
    for data, target in test_loader:
        
        data = data.reshape(-1, sequence_length, input_size).to(device)
        target = target.to(device)        
        output = model(data, 'lstm')

        prediction = output.data.max(1)[1]
        correct += prediction.eq(target.data).sum()

print('\nTest set: Accuracy: {:.2f}%'.format(100. * correct / len(test_loader.dataset)))
# Test set: Accuracy: 97.63%