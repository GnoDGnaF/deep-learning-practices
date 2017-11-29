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
BATCH_SIZE = 64
LR = 0.001              # learning rate
DOWNLOAD_MNIST = False


TIME_STEP = 28
INPUT_SIZE = 28
HIDDEN_SIZE = 64
NUM_LAYERS = 1

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


class RNNnet(nn.Module):
    def __init__(self):
        super(RNNnet, self).__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True
        )
        self.gru = nn.GRU(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True
        )
        self.out = nn.Linear(HIDDEN_SIZE, 10)

    def forward(self, x):

        # r_out, (h_n, h_c) = self.lstm(x, None)    # use LSTM
        r_out, h_n = self.gru(x, None)              # use GRU
        out = self.out(r_out[:, -1, :])

        return out

model = RNNnet()
print(model)

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    params=model.parameters(),
    lr=LR
)

# model training


def train(epoch):
    model.train()
    for batch_idx, (features, target) in enumerate(train_loader):
        features, target = Variable(features.view(-1, 28, 28)), Variable(target)

        output = model(features)
        loss = loss_func(output, target)
        optimizer.zero_grad()
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
        features, target = Variable(features.view(-1, 28, 28), volatile=True), Variable(target)
        output = model(features)
        test_loss += loss_func(output, target).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
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
