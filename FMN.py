import torch
import torch.nn as nn
from torch.autograd import Variable
from config import MAX_LENGTH,SOS_token,EOS_token
use_cuda = torch.cuda.is_available()
import torch.nn.functional as F
from model import *

class FMN(nn.Module):
    def __init__(self, dict_num, term_hidden_size, query_hidden_size, document_hidden_size, n_layers=1, pos_enc=1):
        super(FMN, self).__init__()
        self.n_layers = n_layers
        self.dict_num = dict_num
        self.term_hidden_size = term_hidden_size
        self.query_hidden_size = query_hidden_size
        self.document_hidden_size = document_hidden_size
        self.softmax = torch.nn.Softmax()

        self.position_embedding = nn.Embedding(15, 4) #set fixed number
        self.position_dense = nn.Linear(term_hidden_size+4, term_hidden_size)
        self.position_encoding = pos_enc

        self.term_encoder =  TermEncoder(dict_num,term_hidden_size)
        self.query_encoder = QueryEncoder(term_hidden_size,query_hidden_size)
        self.document_encoder = ContextEncoder(dict_num,term_hidden_size)
        self.query_decoder = QueryDecoder(query_hidden_size,dict_num)

    def forward(self, input_variable, document_variable):
        query_num = len(input_variable)
        q_encoder_outputs = Variable(torch.zeros(query_num, self.query_hidden_size))
        q_encoder_outputs = q_encoder_outputs.cuda() if use_cuda else q_encoder_outputs

        for qi in range(len(input_variable)):
            q_input_variable = input_variable[qi]
            q_input_variable = q_input_variable.cuda() if use_cuda else q_input_variable

            input_length = q_input_variable.size()[0]
            # term encoding
            encoder_hidden = self.term_encoder.initHidden()
            for ei in range(input_length):
                encoder_output, encoder_hidden = self.term_encoder.forward(
                    q_input_variable[ei], encoder_hidden)

            # pos memory network
            p_index = 0
            n_index = 0
            pos_c = pos_a = neg_c = neg_a = Variable()
            if use_cuda:
                pos_c = pos_c.cuda()
                pos_a = pos_c.cuda()
                neg_c = neg_c.cuda()
                neg_a = neg_a.cuda()

            for p in document_variable[0]:
                hidden_a = self.document_encoder.initHidden()
                hidden_c = self.document_encoder.initHidden()
                d_length = p.size()[0]
                for pi in range(d_length):
                    output_c, output_a, hidden_c, hidden_a = self.document_encoder.forward(
                        p[pi], hidden_c, hidden_a
                    )
                if self.position_encoding == 1:
                    hidden_c = self.position_dense(torch.cat((hidden_c, self.position_embedding(p_index)), 0))
                if p_index == 0:
                    pos_c = hidden_c
                    pos_a = hidden_a
                else:
                    pos_c = torch.cat((pos_c, hidden_c), 1)
                    pos_a = torch.cat((pos_a, hidden_a), 1)
                p_index += 1

            # neg memory network
            for n in document_variable[1]:
                hidden_a = self.document_encoder.initHidden()
                hidden_c = self.document_encoder.initHidden()
                d_length = n.size()[0]
                for ni in range(d_length):
                    output_c, output_a, hidden_c, hidden_a = self.document_encoder.forward(
                        n[ni], hidden_c, hidden_a
                    )
                if self.position_encoding == 1:
                    hidden_c = self.position_dense(torch.cat((hidden_c, self.position_embedding(n_index)), 0))
                if n_index == 0:
                    neg_c = hidden_c
                    neg_a = hidden_a
                else:
                    neg_c = torch.cat((neg_c, hidden_c), 1)
                    neg_a = torch.cat((neg_a, hidden_a), 1)
                n_index += 1

            pos_c = pos_c.view(-1, 256)
            pos_a = pos_a.view(-1, 256)
            neg_c = neg_c.view(-1, 256)
            neg_a = neg_a.view(-1, 256)

            q_pos_neg = encoder_hidden
            q_pos_neg = q_pos_neg.cuda() if use_cuda else q_pos_neg

            # pos memory
            if p_index > 0:
                a = torch.mm(pos_a, torch.transpose(encoder_hidden.view(1, 256), 0, 1)).view(1, -1)
                a = self.softmax(a).view(-1, 1)
                pos_memory = torch.mul(pos_c, a)
                for i in range(pos_memory.size()[0]):
                    q_pos_neg = q_pos_neg + pos_memory[i].view(1, 1, 256)

            # neg memory
            if n_index > 0:
                a = torch.mm(neg_a, torch.transpose(encoder_hidden.view(1, 256), 0, 1)).view(1, -1)
                a = self.softmax(a).view(-1, 1)
                neg_memory = torch.mul(neg_c, a)
                for i in range(neg_memory.size()[0]):
                    q_pos_neg = q_pos_neg + neg_memory[i].view(1, 1, 256)

            q_encoder_hidden = self.query_encoder.initHidden()
            q_encoder_output, q_encoder_hidden = self.query_encoder.forward(
                q_pos_neg, q_encoder_hidden
            )
            q_encoder_outputs[qi] = q_encoder_output[0][0]

        return q_encoder_hidden