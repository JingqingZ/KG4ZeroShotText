
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
            max_length=config.max_length,
            vocab_size=30000,
    ):
        self.word_embedding_dim = word_embedding_dim
        self.kg_embedding_dim = kg_embedding_dim
        self.max_length = max_length
        self.vocab_size = vocab_size

        super(Model4Unseen, self).__init__(model_name, start_learning_rate, decay_rate, decay_steps)

    def __create_placeholders__(self):
        self.target_seqs = tf.placeholder(dtype=tf.int32, shape=[config.batch_size, self.max_length], name="target_seqs")
        self.target_mask = tf.placeholder(dtype=tf.int32, shape=[config.batch_size, self.max_length], name="target_mask")
        self.decode_seqs = tf.placeholder(dtype=tf.float32, shape=[config.batch_size, self.max_length, self.word_embedding_dim], name="decode_seqs")

        self.encode_seqs = tf.placeholder(dtype=tf.float32, shape=[config.batch_size, self.max_length, self.word_embedding_dim], name="encode_seqs")
        self.class_label_seqs = tf.placeholder(dtype=tf.float32, shape=[config.batch_size, self.max_length, self.word_embedding_dim], name="class_label_seqs")
        self.kg_vector = tf.placeholder(dtype=tf.float32, shape=[config.batch_size, self.max_length, self.kg_embedding_dim], name="kg_score")
        self.category_logits = tf.placeholder(dtype=tf.float32, shape=[config.batch_size, 1], name="category_logits")

        # self.class_label_single = tf.placeholder(dtype=tf.float32, shape=[config.batch_size, self.word_embedding_dim], name="class_label_single")
        # self.positive_encode_seqs = tf.placeholder(dtype=tf.float32, shape=[config.batch_size, self.max_length, self.word_embedding_dim], name="encode_seqs_positive")
        # self.positive_class_label_seqs = tf.placeholder(dtype=tf.float32, shape=[config.batch_size, self.max_length, self.word_embedding_dim], name="class_label_seqs_positive")
        # self.positive_kg_vector = tf.placeholder(dtype=tf.float32, shape=[config.batch_size, self.max_length, self.kg_embedding_dim], name="kg_score_positive")

        # self.negative_encode_seqs = tf.placeholder(dtype=tf.float32, shape=[config.batch_size, self.max_length, self.word_embedding_dim], name="encode_seqs_negative")
        # self.negative_class_label_seqs = tf.placeholder(dtype=tf.float32, shape=[config.batch_size, self.max_length, self.word_embedding_dim], name="class_label_seqs_negative")
        # self.negative_kg_vector = tf.placeholder(dtype=tf.float32, shape=[config.batch_size, self.max_length, self.kg_embedding_dim], name="kg_score_negative")

    def __create_model__(self):
        if config.model == "autoencoder":
            self.train_net, self.train_seq2seq = self.__get_network_autoencoder__(self.model_name, self.encode_seqs, self.decode_seqs, reuse=False, is_train=True)
            self.test_net, self.test_seq2seq = self.__get_network_autoencoder__(self.model_name, self.encode_seqs, self.decode_seqs, reuse=True, is_train=False)
            self.train_text_state = self.train_seq2seq.final_state_encode
            self.test_text_state = self.test_seq2seq.final_state_encode
            self.train_y = tf.nn.softmax(self.train_net.outputs)
            self.test_y = tf.nn.softmax(self.test_net.outputs)
        elif config.model == "cnnfc":
            self.train_net = self.__get_network_cnnfc__(self.model_name, self.encode_seqs, self.class_label_seqs, reuse=False, is_train=True)
            self.test_net = self.__get_network_cnnfc__(self.model_name, self.encode_seqs, self.class_label_seqs, reuse=True, is_train=False)
        elif config.model == "rnnfc":
            self.train_net = self.__get_network_rnnfc__(self.model_name, self.encode_seqs, self.class_label_seqs, reuse=False, is_train=True)
            self.test_net = self.__get_network_rnnfc__(self.model_name, self.encode_seqs, self.class_label_seqs, reuse=True, is_train=False)
        else:
            self.train_net, self.train_net_cnn = self.__get_network__(self.model_name, self.encode_seqs, self.class_label_seqs, self.kg_vector, reuse=False, is_train=True)
            self.test_net, self.test_net_cnn = self.__get_network__(self.model_name, self.encode_seqs, self.class_label_seqs, self.kg_vector, reuse=True, is_train=False)

        # self.positive_train_net, self.positive_train_net_cnn = self.__get_network__(self.model_name, self.encode_seqs, self.class_label_seqs, self.kg_vector, reuse=False, is_train=True)
        # self.positive_test_net, self.positive_test_net_cnn = self.__get_network__(self.model_name, self.encode_seqs, self.class_label_seqs, self.kg_vector, reuse=True, is_train=False)


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

            if config.model == "vwvcvkg":
                # dbpedia and 20news
                net_in = ConcatLayer(
                    [net_word_embed, net_class_label_embed, net_kg],
                    concat_dim=-1,
                    name='concat_vw_vwc_vc'
                )
            elif config.model == "vwvc":
                net_in = ConcatLayer(
                    [net_word_embed, net_class_label_embed],
                    concat_dim=-1,
                    name='concat_vw_vc'
                )
            elif config.model == "vwvkg":
                net_in = ConcatLayer(
                    [net_word_embed, net_kg],
                    concat_dim=-1,
                    name='concat_vw_vwc'
                )
            elif config.model == "vcvkg":
                net_in = ConcatLayer(
                    [net_class_label_embed, net_kg],
                    concat_dim=-1,
                    name='concat_vc_vwc'
                )
            elif config.model == "kgonly":
                net_in = ConcatLayer(
                    [net_kg],
                    concat_dim=-1,
                    name='concat_vwc'
                )
            else:
                raise Exception("config.model value error")

            filter_length = [2, 4, 8]
            # dbpedia
            n_filter = 600
            # n_filter = 200

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

            '''
            if config.model == "vwvcvkg":
                net_class_label_embed.outputs = tf.slice(
                    net_class_label_embed.outputs,
                    [0, 0, 0],
                    [config.batch_size, 1, self.word_embedding_dim],
                    name="slice_word"
                )
                net_class_label_embed.outputs = tf.squeeze(
                    net_class_label_embed.outputs,
                    name="squeeze_word"
                )
                net_cnn = ConcatLayer(net_cnn_list + [net_class_label_embed], concat_dim=-1)
            else:
                net_cnn = ConcatLayer(net_cnn_list, concat_dim=-1)
            '''
            net_cnn = ConcatLayer(net_cnn_list, concat_dim=-1)

            net_fc = DropoutLayer(net_cnn, keep=0.5, is_fix=True, is_train=is_train, name='drop1')

            net_fc = DenseLayer(
                net_fc,
                n_units=400,
                act=tf.nn.relu,
                name="fc_1"
            )

            net_fc = DropoutLayer(net_fc, keep=0.5, is_fix=True, is_train=is_train, name='drop2')

            # dbpedia
            net_fc = DenseLayer(
                net_fc,
                n_units=100,
                act=tf.nn.relu,
                name="fc_2"
            )
            net_fc = DropoutLayer(net_fc, keep=0.5, is_fix=True, is_train=is_train, name='drop3')

            net_fc = DenseLayer(
                net_fc,
                n_units=1,
                act=tf.nn.sigmoid,
                name="fc_3"
            )
        return net_fc, net_cnn

    def __get_network_cnnfc__(self, model_name, encode_seqs, class_label_seqs, reuse=False, is_train=True):
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

            net_class_label_embed.outputs = tf.slice(
                net_class_label_embed.outputs,
                [0, 0, 0],
                [config.batch_size, 1, self.word_embedding_dim],
                name="slice_word"
            )

            net_class_label_embed.outputs = tf.squeeze(
                net_class_label_embed.outputs,
                name="squeeze_word"
            )

            net_in = ConcatLayer(
                [net_word_embed],
                concat_dim=-1,
                name='concat_vw'
            )

            filter_length = [2, 4, 8]
            # dbpedia
            n_filter = 600
            # n_filter = 200

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

            net_cnn = ConcatLayer(net_cnn_list + [net_class_label_embed], concat_dim=-1)

            net_fc = DropoutLayer(net_cnn, keep=0.5, is_fix=True, is_train=is_train, name='drop1')

            net_fc = DenseLayer(
                net_fc,
                n_units=400,
                act=tf.nn.relu,
                name="fc_1"
            )

            net_fc = DropoutLayer(net_fc, keep=0.5, is_fix=True, is_train=is_train, name='drop2')

            # dbpedia
            net_fc = DenseLayer(
                net_fc,
                n_units=100,
                act=tf.nn.relu,
                name="fc_2"
            )
            net_fc = DropoutLayer(net_fc, keep=0.5, is_fix=True, is_train=is_train, name='drop3')

            net_fc = DenseLayer(
                net_fc,
                n_units=1,
                act=tf.nn.sigmoid,
                name="fc_3"
            )
        return net_fc

    def __get_network_rnnfc__(self, model_name, encode_seqs, class_label_seqs, reuse=False, is_train=True):
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

            net_class_label_embed.outputs = tf.slice(
                net_class_label_embed.outputs,
                [0, 0, 0],
                [config.batch_size, 1, self.word_embedding_dim],
                name="slice_word"
            )

            net_class_label_embed.outputs = tf.squeeze(
                net_class_label_embed.outputs,
                name="squeeze_word"
            )

            net_in = ConcatLayer(
                [net_word_embed],
                concat_dim=-1,
                name='concat_vw'
            )

            net_rnn = RNNLayer(
                net_in,
                cell_fn=tf.contrib.rnn.BasicLSTMCell,
                n_hidden = 512,
                n_steps = self.max_length,
                return_last = True,
                name = 'lstm'
            )

            net_fc = ConcatLayer([net_rnn, net_class_label_embed], concat_dim=-1)

            net_fc = DropoutLayer(net_fc, keep=0.5, is_fix=True, is_train=is_train, name='drop1')

            net_fc = DenseLayer(
                net_fc,
                n_units=400,
                act=tf.nn.relu,
                name="fc_1"
            )

            net_fc = DropoutLayer(net_fc, keep=0.5, is_fix=True, is_train=is_train, name='drop2')

            # dbpedia
            net_fc = DenseLayer(
                net_fc,
                n_units=100,
                act=tf.nn.relu,
                name="fc_2"
            )
            net_fc = DropoutLayer(net_fc, keep=0.5, is_fix=True, is_train=is_train, name='drop3')

            net_fc = DenseLayer(
                net_fc,
                n_units=1,
                act=tf.nn.sigmoid,
                name="fc_3"
            )
        return net_fc

    def __get_network_autoencoder__(self, model_name, encode_seqs, decode_seqs, reuse=False, is_train=True):
        with tf.variable_scope(model_name, reuse=reuse):
            tl.layers.set_name_reuse(reuse)

            net_encode = InputLayer(
                inputs=encode_seqs,
                name="in_word_embed_encode"
            )
            net_decode = InputLayer(
                inputs=decode_seqs,
                name="in_word_embed_decode"
            )

            net_seq2seq = Seq2Seq(
                net_encode, net_decode,
                cell_fn = tf.contrib.rnn.BasicLSTMCell,
                n_hidden = 512,
                initializer = tf.random_uniform_initializer(-0.1, 0.1),
                encode_sequence_length = retrieve_seq_length_op(encode_seqs),
                decode_sequence_length = retrieve_seq_length_op(decode_seqs),
                initial_state_encode = None,
                n_layer = 1,
                return_seq_2d = True,
                name = 'seq2seq'
            )
            net_out = DenseLayer(net_seq2seq, n_units=self.vocab_size, act=tf.identity, name='output')

        return net_out, net_seq2seq


    def __create_loss__(self):
        if config.model == "autoencoder":

            self.train_loss = tl.cost.cross_entropy_seq_with_mask(
                logits=self.train_net.outputs,
                target_seqs=self.target_seqs,
                input_mask=self.target_mask,
                name='train_loss'
            )
            self.test_loss = tl.cost.cross_entropy_seq_with_mask(
                logits=self.test_net.outputs,
                target_seqs=self.target_seqs,
                input_mask=self.target_mask,
                name='test_loss'
            )
        else:
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


