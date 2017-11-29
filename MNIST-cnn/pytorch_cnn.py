# -*- coding = utf-8 -*-

import os
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import torch.nn.functional as F

# hyper parameters
EPOCH = 10
BATCH_SIZE = 50
LR = 0.001              # learning rate
DOWNLOAD_MNIST = False

# load data
if not os.path.exists('../data/mnist') or not os.listdir('../data/mnist'):
    DOWNLOAD_MNIST = True

train_data = datasets.MNIST(
    root='../data/mnist',
    train=True,
    transform=transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)

test_data = datasets.MNIST(
    root='../data/mnist',
    train=False,
    transform=transforms.ToTensor()
)

train_loader = data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)

test_loader = data.DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4
)

# model definition


class CNNnet(nn.Module):
    def __init__(self):
        super(CNNnet, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=16,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
                # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (1, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)  # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)

        return output

model = CNNnet()
print(model)

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    params=model.parameters(),
    lr=LR
)

# model training


def train(epoch):
    model.train()
    for batch_idx, (features, target) in enumerate(train_loader):
        features, target = Variable(features), Variable(target)

        optimizer.zero_grad()
        output = model(features)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(features),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.data[0]
                )
            )

# model evaluation


def evaluate():
    model.eval()
    test_loss = 0
    correct = 0

    for features, target in test_loader:
        features, target = Variable(features, volatile=True), Variable(target)

        output = model(features)
        test_loss += loss_func(output, target).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)

    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)
        )
    )

if __name__ == '__main__':
    for epoch in range(0, EPOCH):
        train(epoch)
        evaluate()
