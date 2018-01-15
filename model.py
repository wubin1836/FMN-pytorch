import torch
import torch.nn as nn
from torch.autograd import Variable
from config import MAX_LENGTH,SOS_token,EOS_token
use_cuda = torch.cuda.is_available()
import torch.nn.functional as F

class TermEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(TermEncoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

class ContextEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(ContextEncoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding_c = nn.Embedding(input_size, hidden_size)
        self.embedding_a = nn.Embedding(input_size, hidden_size)

        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden_c, hidden_a):
        embedded_c = self.embedding_c(input).view(1, 1, -1)
        output_c = embedded_c
        for i in range(self.n_layers):
            output_c, hidden_c = self.gru(output_c, hidden_c)

        embedded_a = self.embedding_a(input).view(1, 1, -1)
        output_a = embedded_a
        for i in range(self.n_layers):
            output_a, hidden_a = self.gru(output_a, hidden_a)

        return output_c, output_a, hidden_c, hidden_a

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

class QueryEncoder(nn.Module):
    def __init__(self, input_size, hidden_size , n_layers=1):
        super(QueryEncoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.gru = nn.GRU(input_size, hidden_size)

    def forward(self, input, hidden):
        output = input
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

class QueryDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1):
        super(QueryDecoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result