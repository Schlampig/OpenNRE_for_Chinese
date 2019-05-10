# -*- coding: utf-8 -*-

import torch.nn as nn
from networks.embedding import *
from networks.encoder import *
from networks.selector import *
from networks.classifier import *


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.embedding = Embedding(config)
        self.encoder = None
        self.selector = None
        self.classifier = Classifier(config)

    def forward(self):
        embedding = self.embedding()
        sen_embedding = self.encoder(embedding)
        logits = self.selector(sen_embedding)
        return self.classifier(logits)

    def test(self):
        embedding = self.embedding()
        sen_embedding = self.encoder(embedding)
        return self.selector.test(sen_embedding)


class CNN_ATT(Model):
    def __init__(self, config):
        super(CNN_ATT, self).__init__(config)
        self.encoder = CNN(config)
        self.selector = Attention(config, config.hidden_size)


class CNN_AVE(Model):
    def __init__(self, config):
        super(CNN_AVE, self).__init__(config)
        self.encoder = CNN(config)
        self.selector = Average(config, config.hidden_size)


class CNN_ONE(Model):
    def __init__(self, config):
        super(CNN_ONE, self).__init__(config)
        self.encoder = CNN(config)
        self.selector = One(config, config.hidden_size)


class PCNN_ATT(Model):
    def __init__(self, config):
        super(PCNN_ATT, self).__init__(config)
        self.encoder = PCNN(config)
        self.selector = Attention(config, config.hidden_size * 3)


class PCNN_AVE(Model):
    def __init__(self, config):
        super(PCNN_AVE, self).__init__(config)
        self.encoder = PCNN(config)
        self.selector = Average(config, config.hidden_size * 3)


class PCNN_ONE(Model):
    def __init__(self, config):
        super(PCNN_ONE, self).__init__(config)
        self.encoder = PCNN(config)
        self.selector = One(config, config.hidden_size * 3)
