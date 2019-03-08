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
import model_base

class Model4Seen(model_base.Base_Model):

    def __init__(
            self,
            model_name,
            start_learning_rate,
            decay_rate,
            decay_steps,
            number_of_seen_classes,
            word_embedding_dim=config.word_embedding_dim,
            max_length=config.max_length
    ):
        self.number_of_seen_classes = number_of_seen_classes
        self.word_embedding_dim = word_embedding_dim
        self.max_length = max_length

        super(Model4Seen, self).__init__(model_name, start_learning_rate, decay_rate, decay_steps)


    def __create_placeholders__(self):
        self.encode_seqs = tf.placeholder(dtype=tf.float32, shape=[config.batch_size, self.max_length, self.word_embedding_dim], name="encode_seqs")
        self.category_target_index = tf.placeholder(dtype=tf.int32, shape=[config.batch_size, ], name="category_target_index")

    def __create_model__(self):
        self.train_net = self.__get_network__(self.model_name, self.encode_seqs, reuse=False, is_train=True)
        self.test_net = self.__get_network__(self.model_name, self.encode_seqs, reuse=True, is_train=False)

    def __get_network__(self, model_name, encode_seqs, reuse=False, is_train=True):
        with tf.variable_scope(model_name, reuse=reuse):
            tl.layers.set_name_reuse(reuse)

            net_in = InputLayer(
                inputs=encode_seqs,
                name="in_word_embed"
            )

            '''
            net_in = ReshapeLayer(
                net_in,
                (-1, self.max_length, self.word_embedding_dim, 1),
                name="reshape"
            )
            '''

            filter_length = [2, 4, 8]
            n_filter = 600

            net_cnn_list = list()

            for fsz in filter_length:

                net_cnn = Conv1d(
                    net_in,
                    n_filter=n_filter,
                    filter_size=fsz,
                    stride=1,
                    act=tf.nn.relu,
                    name="cnn%d" % fsz
                )
                net_cnn.outputs = tf.reduce_max(net_cnn.outputs, axis=1, name="global_maxpool%d" % fsz)
                net_cnn_list.append(net_cnn)

            net_cnn = ConcatLayer(net_cnn_list, concat_dim=-1)

            '''
            net_cnn = Conv1d(net_in, 400, 8, act=tf.nn.relu, name="cnn_1")
            net_cnn = MaxPool1d(net_cnn, 2, 2, padding="valid", name="maxpool_1")

            net_cnn = Conv1d(net_cnn, 600, 4, act=tf.nn.relu, name="cnn_2")
            net_cnn = MaxPool1d(net_cnn, 2, 2, padding="valid", name="maxpool_2")

            net_cnn = Conv1d(net_cnn, 600, 2, act=tf.nn.relu, name="cnn_3")
            net_cnn = MaxPool1d(net_cnn, 2, 2, padding="valid", name="maxpool_3")

            net_cnn = FlattenLayer(net_cnn, name="flatten")
            '''
            '''
            net_cnn = Conv2d(net_in, 64, (8, 8), act=tf.nn.relu, name="cnn_1")
            net_cnn = MaxPool2d(net_cnn, (2, 2), padding="valid", name="maxpool_1")

            net_cnn = Conv2d(net_cnn, 32, (4, 4), act=tf.nn.relu, name="cnn_2")
            net_cnn = MaxPool2d(net_cnn, (2, 4), padding="valid", name="maxpool_2")

            net_cnn = Conv2d(net_cnn, 8, (2, 2), act=tf.nn.relu, name="cnn_3")
            net_cnn = MaxPool2d(net_cnn, (2, 2), padding="valid", name="maxpool_3")

            net_cnn = FlattenLayer(net_cnn, name="flatten")
            '''

            net_cnn = DropoutLayer(net_cnn, keep=0.5, is_fix=True, is_train=is_train, name='drop1')

            net_fc = DenseLayer(
                net_cnn,
                n_units=400,
                act=tf.nn.relu,
                name="fc_1"
            )

            net_fc = DropoutLayer(net_fc, keep=0.5, is_fix=True, is_train=is_train, name='drop2')

            net_fc = DenseLayer(
                net_fc,
                n_units=100,
                act=tf.nn.relu,
                name="fc_2"
            )

            net_fc = DropoutLayer(net_fc, keep=0.5, is_fix=True, is_train=is_train, name='drop3')

            net_fc = DenseLayer(
                net_fc,
                n_units=self.number_of_seen_classes,
                act=tf.nn.relu,
                name="fc_3"
            )

        return net_fc

    def __create_loss__(self):
        self.train_loss = tl.cost.cross_entropy(
            output=self.train_net.outputs,
            target=self.category_target_index,
            name="train_loss"
        )
        self.test_loss = tl.cost.cross_entropy(
            output=self.test_net.outputs,
            target=self.category_target_index,
            name="test_loss"
        )

    def __create_training_op__(self):
        self.global_step = tf.placeholder(
            dtype=tf.int32,
            shape=[],
            name="global_step"
        )
        self.learning_rate = tf.train.exponential_decay(
            learning_rate=self.start_learning_rate,
            global_step=self.global_step,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            staircase=True,
            name="learning_rate"
        )
        self.optim = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5) \
            .minimize(self.train_loss, var_list=self.train_net.all_params)

if __name__ == "__main__":
    model = Model4Seen(
        model_name="model4seen_test",
        start_learning_rate=0.001,
        decay_rate=0.8,
        decay_steps=1000,
        number_of_seen_classes=11
    )
