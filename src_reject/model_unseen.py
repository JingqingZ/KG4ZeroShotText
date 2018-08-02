
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

class Model4Unseen(model_base.Base_Model):

    def __init__(
            self,
            model_name,
            start_learning_rate,
            decay_rate,
            decay_steps,
            word_embedding_dim=config.word_embedding_dim,
            kg_embedding_dim=config.kg_embedding_dim,
            max_length=config.max_length
    ):
        self.word_embedding_dim = word_embedding_dim
        self.kg_embedding_dim = kg_embedding_dim
        self.max_length = max_length

        super(Model4Unseen, self).__init__(model_name, start_learning_rate, decay_rate, decay_steps)

    def __create_placeholders__(self):
        self.encode_seqs = tf.placeholder(dtype=tf.float32, shape=[config.batch_size, self.max_length, self.word_embedding_dim], name="encode_seqs")
        self.class_label_seqs = tf.placeholder(dtype=tf.float32, shape=[config.batch_size, self.max_length, self.word_embedding_dim], name="class_label_seqs")
        self.kg_vector = tf.placeholder(dtype=tf.float32, shape=[config.batch_size, self.max_length, self.kg_embedding_dim], name="kg_score")
        self.category_logits = tf.placeholder(dtype=tf.float32, shape=[config.batch_size, 1], name="category_logits")

    def __create_model__(self):
        self.train_net = self.__get_network__(self.model_name, self.encode_seqs, self.class_label_seqs, self.kg_vector, reuse=False, is_train=True)
        self.test_net = self.__get_network__(self.model_name, self.encode_seqs, self.class_label_seqs, self.kg_vector, reuse=True, is_train=False)

    def __get_network__(self, model_name, encode_seqs, class_label_seqs, kg_vector, reuse=False, is_train=True):
        with tf.variable_scope(model_name, reuse=reuse):
            tl.layers.set_name_reuse(reuse)

            net_word_embed = InputLayer(
                inputs=encode_seqs,
                name="in_word_embed"
            )

            net_class_label_embed = InputLayer(
                inputs=class_label_seqs,
                name="in_class_label_embed"
            )

            net_kg = InputLayer(
                inputs=kg_vector,
                name='in_kg'
            )

            net_kg = ReshapeLayer(
                net_kg,
                shape=(-1, self.kg_embedding_dim),
                name="reshape_kg_1"
            )

            net_kg = ReshapeLayer(
                net_kg,
                shape=(-1, self.max_length, self.kg_embedding_dim),
                name="reshape_kg_2"
            )

            net_in = ConcatLayer(
                 [net_word_embed, net_kg, net_class_label_embed],
                 concat_dim=-1,
                 name='concat_kg_word'
            )

            # TODO: place to change inputs
            # net_in = ConcatLayer(
            #     [net_word_embed, net_kg],
            #     concat_dim=-1,
            #     name='concat_kg_word'
            # )

            # net_in = ConcatLayer(
            #     [net_word_embed, net_class_label_embed],
            #     concat_dim=-1,
            #     name='concat_kg_word'
            # )
            # net_in = ConcatLayer(
            #     [net_kg],
            #     concat_dim=-1,
            #     name='concat_kg_word'
            # )

            filter_length = [2, 4, 8]
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

            net_cnn = DropoutLayer(net_cnn, keep=0.8, is_fix=True, is_train=is_train, name='drop1')

            net_fc = DenseLayer(
                net_cnn,
                n_units=300,
                act=tf.nn.relu,
                name="fc_1"
            )

            net_fc = DropoutLayer(net_fc, keep=0.8, is_fix=True, is_train=is_train, name='drop2')

            net_fc = DenseLayer(
                net_fc,
                n_units=100,
                act=tf.nn.relu,
                name="fc_2"
            )

            net_fc = DenseLayer(
                net_fc,
                n_units=1,
                act=tf.nn.sigmoid,
                name="fc_3"
            )
        return net_fc

    def __create_loss__(self):
        self.train_loss = tl.cost.binary_cross_entropy(
            output=self.train_net.outputs,
            target=self.category_logits,
            name="train_loss"
        )

        self.test_loss = tl.cost.binary_cross_entropy(
            output=self.test_net.outputs,
            target=self.category_logits,
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
    model = Model4Unseen(
        model_name="model4unseen_test",
        start_learning_rate=0.001,
        decay_rate=0.8,
        decay_steps=1000
    )


