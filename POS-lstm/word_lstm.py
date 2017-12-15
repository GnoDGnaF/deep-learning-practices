# -*- coding = utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn

# hyper parameters
EMBEDDING_DIM = 50
LEARNING_RATE = 1e-3
EPOCH = 100
HIDDEN_DIM = 6
NUM_LAYERS = 1

# load data

training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]

word_to_idx = {}
tag_to_idx = {}
idx_to_tags = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)
    for label in tags:
        if label not in tag_to_idx:
            tag_to_idx[label] = len(tag_to_idx)
            idx_to_tags[len(idx_to_tags)] = label

# print(word_to_idx)
# print(tag_to_idx)


def prepare_sequence(seq, to_idx):
    idxs = [to_idx[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

# model definition


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=NUM_LAYERS,
        )
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, h = self.lstm(embeds.view(len(sentence), 1, -1), None)
        out = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return out


model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_idx), len(tag_to_idx))
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    params=model.parameters(),
    lr=LEARNING_RATE
)

# model training


def train(epoch):
    model.train()
    training_loss = 0

    for sentence, tags in training_data:
        sentence_in = prepare_sequence(sentence, word_to_idx)
        targets = prepare_sequence(tags, tag_to_idx)

        # forward
        out = model(sentence_in)
        loss = loss_func(out, targets)
        training_loss += loss.data[0]

        # backward
        model.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch: {0}, Loss: {1:.6f}'.format(epoch, training_loss / len(training_data)))

# model evaluation


def evaluate():
    model.eval()
    test_loss = 0

    for sentence, tags in training_data:
        sentence_in = prepare_sequence(sentence, word_to_idx)
        targets = prepare_sequence(tags, tag_to_idx)

        # forward
        out = model(sentence_in)
        loss = loss_func(out, targets)
        test_loss += loss.data[0]

    print('Loss: {0:.6f}'.format(test_loss / len(training_data)))

# model inference


def inference(instance):
    model.eval()

    sentence_in = prepare_sequence(instance, word_to_idx)

    out = model(sentence_in)
    tags = [idx_to_tags[int(i)] for i in out.data.max(1, keepdim=True)[1]]
    return ','.join(tags)

if __name__ == '__main__':
    for i in range(0, EPOCH):
        train(i)
        evaluate()

    # print(idx_to_tags)
    result = inference(training_data[0][0])
    label = ','.join(training_data[0][1])
    print('real POS is `{0}`, predict POS is `{1}`'.format(label, result))
