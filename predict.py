import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from config import MAX_LENGTH,MAX_SESSION,TERM_HIDDEN_SIZE,QUERY_HIDDEN_SIZE,SOS_token,EOS_token,MAX_DIC

import cPickle as pickle
S = pickle.load(open("./data/S.pkl","r")) # source query
T = pickle.load(open("./data/T.pkl","r")) # target query
D = pickle.load(open("./data/D.pkl","r")) # docuemnts

SESSION_NUM = len(S)
Click_thread = 1 #from 0 to max_click_index+1

softmax = torch.nn.Softmax()

sources = []
targets = []
documents = []

use_cuda = torch.cuda.is_available()

from FMN import FMN

def variableFromSentence(sentence): #change index to variable
    result = Variable(torch.LongTensor(sentence).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result

for i in range(SESSION_NUM):
    #source variable
    sources.append([variableFromSentence(s) for s in S[i]])

    #target variable
    targets.append(variableFromSentence(T[i]))
    #document variable
    for item in D:
        document_tmp = []
        query_num = len(item)

        for q_item in item:
            document_pos_tmp = []
            document_neg_tmp = []
            max_click = 0
            for index in range(len(q_item),0,-1):
                if q_item[index-1][1] == 1:
                    max_click = index - 1
            for index in range(0,min(len(q_item),max_click+1)):
                if q_item[index][1] == 1: #click document
                    click_v = variableFromSentence(q_item[index][0])
                    document_pos_tmp.append(click_v)
                else:#un-click document
                    if index <= max_click:
                        un_click_v = variableFromSentence(q_item[index][0])
                        document_neg_tmp.append(un_click_v)
            document_tmp.append(document_pos_tmp)
            document_tmp.append(document_neg_tmp)
        documents.append(document_tmp)

def score(input_variable, target_variable, document_variable, fmn, q_decoder, criterion, max_length=MAX_LENGTH):


    loss = 0
    q_encoder_hidden = fmn(input_variable, document_variable)

    # query decoding
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = q_encoder_hidden

    target_length = target_variable.size()[0]
    for di in range(target_length):
        decoder_output, decoder_hidden = q_decoder(
            decoder_input, decoder_hidden,)
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        loss += criterion(decoder_output, target_variable[di])
        if ni == EOS_token:
            break

    return loss.data[0] / target_length


from tqdm import tqdm
def score_test(fmn, q_decoder):
    print_losses = []
    print_loss_total = 0

    criterion = nn.NLLLoss()

    for iter in tqdm(range(1, len(S) + 1)):
        input_variable = sources[iter - 1]
        target_variable = targets[iter - 1]
        document_variable = documents[iter - 1]

        loss = score(input_variable, target_variable, document_variable, fmn, q_decoder, criterion)
        print loss

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from model import QueryDecoder
fmn = FMN(MAX_DIC, TERM_HIDDEN_SIZE, QUERY_HIDDEN_SIZE, TERM_HIDDEN_SIZE, n_layers=1)
q_decoder = QueryDecoder(QUERY_HIDDEN_SIZE, MAX_DIC,1)

if use_cuda:
    fmn = fmn.cuda()
    q_decoder = q_decoder.cuda()

fmn.load_state_dict(torch.load("./model/300.enc"))
q_decoder.load_state_dict(torch.load("./model/300.dec"))

score_test(fmn, q_decoder)
