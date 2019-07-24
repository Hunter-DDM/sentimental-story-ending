import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import models
import math
import numpy as np
from torch.autograd import Function

class ContextEncoder(nn.Module):

    def __init__(self, config, embedding=None):
        super(ContextEncoder, self).__init__()

        self.embedding = embedding if embedding is not None else nn.Embedding(config.input_vocab_size, config.emb_size)
        self.hidden_size = config.hidden_size
        self.config = config
        self.sigmoid = nn.Sigmoid()

        if config.attention == 'simple':
            self.attention = models.simple_attention(config.hidden_size)

        if config.cell == 'gru':
            self.rnn = nn.GRU(input_size=config.emb_size, hidden_size=config.hidden_size,
                              num_layers=config.enc_num_layers, dropout=config.dropout,
                              bidirectional=config.bidirectional)
        else:
            self.rnn = nn.LSTM(input_size=config.emb_size, hidden_size=config.hidden_size,
                               num_layers=config.enc_num_layers, dropout=config.dropout,
                               bidirectional=config.bidirectional)

    def forward(self, inputs, lengths):
        inputs = inputs.t()
        embs = pack(self.embedding(inputs), lengths)
        outputs, state = self.rnn(embs)
        outputs = unpack(outputs)[0]
        if self.config.bidirectional:
            outputs = outputs[:,:,:self.config.hidden_size] + outputs[:,:,self.config.hidden_size:]

        # self attention
        outputs, weights = self.attention(outputs.transpose(0, 1))

        if self.config.cell == 'gru':
            state = state
        else:
            state = (state[0], state[1])

        return outputs, state


class EmotionRegressor(nn.Module):

    def __init__(self, config):
        super(EmotionRegressor, self).__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.emotion_types = config.emotion_types
        self.linear_out = nn.Sequential(nn.Linear(self.hidden_size, 32), nn.ReLU(), nn.Dropout(p=config.dropout), nn.Linear(32, 1))

    def forward(self, contexts):
        outputs = self.linear_out(contexts)
        return outputs

        
class DomainClassifier(nn.Module):

    def __init__(self, config):
        super(DomainClassifier, self).__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.domain_types = config.domain_types
        self.linear_out = nn.Sequential(nn.Linear(self.hidden_size, 32), nn.ReLU(), nn.Dropout(p=config.dropout), nn.Linear(32, self.domain_types))

    def forward(self, contexts):
        contexts = grad_reverse(contexts, 0.3)
        outputs = self.linear_out(contexts)
        return outputs
        

def grad_reverse(x, lambd):
    return GradReverse(lambd)(x)
        
        
class GradReverse(Function):

    def __init__(self, lambd):
        super(GradReverse, self).__init__()
        self.lambd = lambd
        
    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return grad_output * (-self.lambd)
