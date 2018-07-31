import os
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import \
    Layer, \
    InputLayer, Conv1d, MaxPool1d, \
    Conv1dLayer, \
    RNNLayer, DropoutLayer, DenseLayer, \
    LambdaLayer, ReshapeLayer, ConcatLayer, \
    Conv2d, MaxPool2d, FlattenLayer, \
    DeConv2d, BatchNormLayer, EmbeddingInputlayer, \
    Seq2Seq, retrieve_seq_length_op2, DynamicRNNLayer, \
    retrieve_seq_length_op

import numpy as np
import logging

import config

class Base_Model():

    def __init__(
            self,
            model_name,
            start_learning_rate,
            decay_rate,
            decay_steps
    ):
        self.model_name = model_name
        self.start_learning_rate = start_learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

        self.__create_placeholders__()
        self.__create_model__()
        self.__create_loss__()
        self.__create_training_op__()

    def __create_placeholders__(self):
        # the placeholder for inputs
        pass

    def __create_model__(self):
        # train_net
        # test_net
        pass

    def __get_network__(self, model_name, reuse=False, is_train=True):
        # the architecture of networks
        pass

    def __create_loss__(self):
        # loss function
        # train_loss
        # test_loss if necessary
        pass

    def __create_training_op__(self):
        # learning rate operators
        # optim operators
        pass
