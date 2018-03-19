import torch
import torch.nn as nn
from torch.autograd import Variable


class Net(nn.Module):

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128):
        super(Net, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding = nn.Linear(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.out_linear = nn.Linear(hidden_dim, 2)

    def step(self, input, hidden=None):
        output, hidden = self.lstm(input, hidden)

    def forward(self, input):
        batch_size = input.size()[0]
        h0 = Variable(torch.randn(batch_size, 1, self.hidden_dim))
        c0 = Variable(torch.randn(batch_size, 1, self.hidden_dim))
        if (torch.cuda.is_available()):
            h0 = h0.cuda()
            c0 = c0.cuda()

        output = self.embedding(input)
        output, hidden = self.lstm(output, (h0, c0))
        output = output[:, -1, :]
        output = self.out_linear(output)
        return output
