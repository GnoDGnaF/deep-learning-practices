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

N_INPUT = 784
N_HIDDEN = 120
N_OUT = 10

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

# print(train_data.train_data.size())                 # (60000, 28, 28)
# print(train_data.train_labels.size())               # (60000)
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[0])
# plt.show()

# model definition


class SoftmaxNet(nn.Module):

    def __init__(self, n_input, n_hidden, n_out):
        super(SoftmaxNet, self).__init__()
        self.n_input = n_input
        self.hidden = nn.Linear(n_input, n_hidden)
        self.out = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        x = x.view(-1, self.n_input)   # Flatten the data (n, 1, 28, 28)-> (n, 784)
        x = F.relu(self.hidden(x))
        x = self.out(x)

        return x

model = SoftmaxNet(n_input=N_INPUT, n_hidden=N_HIDDEN, n_out=N_OUT)

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

        if batch_idx % 10 == 0:
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
