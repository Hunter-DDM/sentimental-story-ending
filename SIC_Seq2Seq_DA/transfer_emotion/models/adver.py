import torch
import torch.nn as nn
import utils
import models
import random

'''
class AdverNet(nn.Module):
    def __init__(self, config):
        super(AdverNet, self).__init__()
        self.encoder = models.ContextEncoder(config)
        self.emotion_classfier = models.EmotionClassifer(config)
        self.domain_classfier = models.DomainClassifer(config)
        self.use_cuda = config.use_cuda
        self.config = config
        self.criterion = nn.CrossEntropyLoss(ignore_index=utils.PAD, reduction='none')
        if config.use_cuda:
            self.criterion.cuda()

    def compute_loss(self, scores, targets):
        scores = scores.view(-1, scores.size(-1))
        loss = self.criterion(scores, targets.contiguous().view(-1))
        return loss

    def forward(self, input, input_len, emotion, belong, teacher_ratio=1.0):
        input = input.t()
        contexts, state = self.encoder(input, input_len.tolist())
        # outputs_domain = self.domain_classfier(contexts)
        outputs_emotion = self.emotion_classfier(contexts)

        loss = self.compute_loss(outputs_emotion, emotion)
        return loss, outputs_emotion
'''