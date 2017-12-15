# -*- coding = utf-8 -*-

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable

# hyper parameters
CONTEXT_SIZE = 2
EMBEDDING_DIM = 50
LEARNING_RATE = 1e-3
EPOCH = 100

# load data
sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

tri_gram = [
    ((sentence[i], sentence[i+1]), sentence[i+2])
    for i in range(len(sentence) - 2)
]
vocb = set(sentence)
word_to_idx = {word: i for i, word in enumerate(vocb)}
idx_to_word = {word_to_idx[word]: word for word in word_to_idx}

# model definition


class NGramModel(nn.Module):
    def __init__(self, vocb_size, context_size, n_dim):
        super(NGramModel, self).__init__()
        self.n_word = vocb_size
        self.embedding = nn.Embedding(self.n_word, n_dim)
        self.linear1 = nn.Linear(context_size * n_dim, 128)
        self.linear2 = nn.Linear(128, self.n_word)

    def forward(self, x):
        embedding = self.embedding(x)
        embedding = embedding.view(1, -1)
        hidden = self.linear1(embedding)
        hidden = F.relu(hidden)
        out = self.linear2(hidden)

        return out

model = NGramModel(len(vocb), CONTEXT_SIZE, EMBEDDING_DIM)
print(model)

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    params=model.parameters(),
    lr=LEARNING_RATE
)

# model training


def train(epoch):
    model.train()
    running_loss = 0
    for data in tri_gram:
        word, label = data
        word = Variable(torch.LongTensor([word_to_idx[i] for i in word]))
        label = Variable(torch.LongTensor([word_to_idx[label]]))
        # forward
        out = model(word)
        loss = loss_func(out, label)
        running_loss += loss.data[0]

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch: {0}, Loss: {1:.6f}'.format(epoch, running_loss / len(tri_gram)))

# model evaluation


def evaluate():
    model.eval()
    test_loss = 0
    correct = 0

    for data in tri_gram:
        word, label = data
        word = Variable(torch.LongTensor([word_to_idx[i] for i in word]))
        label = Variable(torch.LongTensor([word_to_idx[label]]))

        # forward
        out = model(word)
        loss = loss_func(out, label)
        test_loss += loss.data[0]

        pred = out.data.max(1, keepdim=True)[1]
        correct += pred.eq(label.data.view_as(pred)).sum()

    print('Accuracy: {0:.3f}%, Loss: {1:.6f}'.format(100. * correct / len(tri_gram), test_loss / len(word_to_idx)))


# model inference


def inference(instance):
    model.eval()

    word = Variable(torch.LongTensor([word_to_idx[i] for i in instance]))

    # forward
    out = model(word)
    index = out.data.max(1, keepdim=True)[1]

    return idx_to_word[int(index)]


if __name__ == '__main__':
    for i in range(0, EPOCH):
        train(i)
        evaluate()

    word, label = tri_gram[5]
    result = inference(word)
    print()
    print('real word is `{}`, predict word is `{}`'.format(label, result))
