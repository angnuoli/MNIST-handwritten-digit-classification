# -*- coding: utf-8 -*-

import torch
from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import net

batch_size = 128
learning_rate = 1e-2
num_epoches = 20

data_tf = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])]
)

train_dataset = datasets.MNIST(root='./data', train=True, transform=data_tf, download=False)
test_dataset = datasets.MNIST(root='/data', train=False, transform=data_tf, download=False)

train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = net.CNN(1, 10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


for e in range(num_epoches):
    print('epoch {}\n*******'.format(e))
    acc = 0
    for i, data in enumerate(train_loader, 1):
        img, label = data        
        img = Variable(img)
        label = Variable(label)
        out = model(img)
        loss = criterion(out, label)        
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        acc += num_correct.data[0]
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 300 == 0:
            print('[{}/{}] Loss: {:.5f}, Acc: {:.6f}'.format(i, len(train_loader), loss, acc.float() / (i * label.size(0))))
              

model.eval()
eval_loss = 0
eval_acc = 0
for data in test_loader:
    img, label = data
    img = Variable(img, volatile=True)
    label = Variable(label, volatile=True)
    out = model(img)
    loss = criterion(out, label)
    eval_loss += loss.data[0] * label.size(0)
    _, pred = torch.max(out, 1)
    num_correct = (pred == label).sum()
    eval_acc += num_correct.data[0]

print('Test Loss: {:.6f}, Acc: {:.6f}'.format(
    eval_loss / len(test_dataset),
    eval_acc.float() / len(test_dataset)
))