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

class Model4Reject():

    def __init__(
            self,
            model_name,
            start_learning_rate,
            decay_rate,
            decay_steps,
            main_class,
            seen_classes,
            unseen_classes,
            word_embedding_dim=config.word_embedding_dim,
            max_length=config.max_length
    ):
        self.model_name = model_name
        self.start_learning_rate = start_learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.word_embedding_dim = word_embedding_dim
        self.max_length = max_length
        self.main_class = main_class
        self.seen_classes = seen_classes
        self.unseen_classes = unseen_classes
        self.threshold = 0.5

        self.__create_placeholders__()
        self.__create_model__()
        self.__create_loss__()
        self.__create_training_op__()

    def __create_placeholders__(self):
        # the placeholder for inputs
        self.encode_seqs = tf.placeholder(dtype=tf.float32, shape=[None, self.max_length, self.word_embedding_dim], name="encode_seqs")
        self.label_logits = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="label_logits")
        
        # self.encode_seqs_anc = tf.placeholder(dtype=tf.float32, shape=[None, self.max_length, self.word_embedding_dim], name="encode_seqs_anc")
        # self.encode_seqs_pos = tf.placeholder(dtype=tf.float32, shape=[None, self.max_length, self.word_embedding_dim], name="encode_seqs_pos")
        # self.encode_seqs_neg = tf.placeholder(dtype=tf.float32, shape=[None, self.max_length, self.word_embedding_dim], name="encode_seqs_neg")
        # pass

    def __create_model__(self):
        # train_net
        # self.train_net = self.__get_network__(self.model_name, self.encode_seqs, reuse=False, is_train=True)
        self.train_net, _ = self.__get_network__(self.model_name, self.encode_seqs, reuse=False, is_train=True)
        # _, self.cnn_anchor = self.__get_network__(self.model_name, self.encode_seqs_anc, reuse=True, is_train=True)
        # _, self.cnn_pos = self.__get_network__(self.model_name, self.encode_seqs_pos, reuse=True, is_train=True)
        # _, self.cnn_neg = self.__get_network__(self.model_name, self.encode_seqs_neg, reuse=True, is_train=True)
        
        # test_net
        # self.test_net = self.__get_network__(self.model_name, self.encode_seqs, reuse=True, is_train=False)
        self.test_net, _ = self.__get_network__(self.model_name, self.encode_seqs, reuse=True, is_train=False)
        
        # pass

    def __get_network__(self, model_name, encode_seqs, reuse=False, is_train=True):
        # the architecture of networks
        with tf.variable_scope(model_name, reuse=reuse):
            # tl.layers.set_name_reuse(reuse)
            net_in = InputLayer(
                inputs=encode_seqs,
                name="in_word_embed"
            )

            filter_length = [3, 4, 5]
            n_filter = 200
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
            net_fc = DenseLayer(
                net_cnn,
                n_units=300,
                act=tf.nn.relu,
                name="fc_1"
            )

            net_fc = DenseLayer(
                net_fc,
                n_units=1,
                act=tf.nn.sigmoid,
                name="fc_2"
            )
        return net_fc, net_cnn
        # return net_fc
        # pass

    def __create_loss__(self):
        # loss function
        # train_loss
        train_predicted_logits = self.train_net.outputs
        self.cross_entropy_loss = tl.cost.binary_cross_entropy(
            output=train_predicted_logits,
            target=self.label_logits,
        )

        # anchor_output = self.cnn_anchor.outputs
        # positive_output = self.cnn_pos.outputs
        # negative_output = self.cnn_neg.outputs

        # d_pos = tf.reduce_sum(tf.square(anchor_output - positive_output), 1)
        # d_neg = tf.reduce_sum(tf.square(anchor_output - negative_output), 1)

        # triplet_loss = tf.maximum(0., 0.1 + d_pos - d_neg)
        # self.triplet_loss = tf.reduce_mean(triplet_loss)

        self.train_loss = self.cross_entropy_loss
        # self.train_loss = self.cross_entropy_loss + self.triplet_loss


        # test_loss if necessary
        test_predicted_logits = self.test_net.outputs
        self.test_loss = tl.cost.binary_cross_entropy(
            output=test_predicted_logits,
            target=self.label_logits,
        )
        # pass

    def __create_training_op__(self):
        # learning rate operators
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

        # optim operators
        self.optim = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5) \
            .minimize(self.train_loss, var_list=self.train_net.all_params)
        # self.optim = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5) \
        #     .minimize(self.train_loss, var_list=self.train_net.all_params + self.cnn_anchor.all_params + self.cnn_pos.all_params + self.cnn_neg.all_params)
        # pass

if __name__ == "__main__":
    model = Reject_Model(
        model_name="cnn_binary_classification",
        start_learning_rate=0.001,
        decay_rate=0.8,
        decay_steps=1000
    )
    pass
