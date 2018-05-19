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

class Model_KG4Text():

    def __init__(
            self,
            model_name,
            start_learning_rate,
            decay_rate,
            decay_steps,
            word_embedding_dim=config.word_embedding_dim,
            hidden_dim=config.hidden_dim,
            kg_embedding_dim=config.kg_embedding_dim,
            vocab_size=config.vocab_size,
            max_length=config.max_length
    ):
        self.model_name = model_name
        self.start_learning_rate = start_learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.word_embedding_dim = word_embedding_dim
        self.kg_embedding_dim = kg_embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_length = max_length

        self.__create_placeholders__()
        self.__create_model__()
        self.__create_loss__()
        self.__create_training_op__()

    def __create_placeholders__(self):
        # self.encode_seqs = tf.placeholder(dtype=tf.int32, shape=[config.batch_size, self.max_length], name="encode_seqs")
        self.encode_seqs = tf.placeholder(dtype=tf.float32, shape=[config.batch_size, self.max_length, self.word_embedding_dim], name="encode_seqs")
        self.class_label_seqs = tf.placeholder(dtype=tf.float32, shape=[config.batch_size, self.max_length, self.word_embedding_dim], name="class_label_seqs")
        self.kg_vector = tf.placeholder(dtype=tf.float32, shape=[config.batch_size, self.max_length, self.kg_embedding_dim], name="kg_score")
        self.category_logits = tf.placeholder(dtype=tf.float32, shape=[config.batch_size, 1], name="category_logits")

        self.class_id_seqs = tf.placeholder(dtype=tf.int32, shape=[config.batch_size, 1], name="class_id_seqs")

        # self.encode_seqs_infer = tf.placeholder(dtype=tf.int32, shape=[1, self.max_length], name="encode_seqs_inference")
        self.encode_seqs_infer = tf.placeholder(dtype=tf.float32, shape=[1, self.max_length, self.word_embedding_dim], name="encode_seqs_inference")
        self.class_label_seqs_infer = tf.placeholder(dtype=tf.float32, shape=[1, self.max_length, self.word_embedding_dim], name="class_label_inference")
        self.kg_vector_infer = tf.placeholder(dtype=tf.float32, shape=[1, self.max_length, self.kg_embedding_dim], name="kg_score_inference")
        self.category_logits_infer = tf.placeholder(dtype=tf.float32, shape=[1, 1], name="category_logits_inference")

        self.class_id_seqs_infer = tf.placeholder(dtype=tf.int32, shape=[1, 1], name="class_id_seqs")

    def __create_model__(self):
        # self.train_net, self.train_align = self.__get_network__(self.model_name, self.encode_seqs, self.class_label_seqs, self.kg_vector, reuse=False, is_train=True)
        # self.test_net, self.test_align = self.__get_network__(self.model_name, self.encode_seqs, self.class_label_seqs, self.kg_vector, reuse=True, is_train=False)
        # self.infer_net, self.infer_align = self.__get_network__(self.model_name, self.encode_seqs_infer, self.class_label_seqs_infer, self.kg_vector_infer, reuse=True, is_train=False)
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
                shape=(-1, config.kg_embedding_dim),
                name="reshape_kg_1"
            )

            net_kg = DenseLayer(
                net_kg,
                n_units=200,
                act=tf.nn.relu,
                name='dense_kg'
            )

            net_kg = ReshapeLayer(
                net_kg,
                shape=(-1, self.max_length, 200),
                name="reshape_kg_2"
            )

            # TODO: kg_vector for training
            # net_in = ConcatLayer(
            #      [net_word_embed, net_class_label_embed, net_kg],
            #     concat_dim=-1,
            #     name='concat_kg_word'
            # )

            net_in = ConcatLayer(
                [net_word_embed, net_class_label_embed],
                concat_dim=-1,
                name='concat_kg_word'
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
        return net_fc

    '''
    def __get_network__(self, model_name, encode_seqs, class_label_seqs, kg_vector, reuse=False, is_train=True):
        with tf.variable_scope(model_name, reuse=reuse):

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
                 shape=(-1, config.kg_embedding_dim),
                 name="reshape_kg_1"
             )

             net_kg = DenseLayer(
                 net_kg,
                 n_units=200,
                 act=tf.nn.relu,
                 name='dense_kg'
             )

             net_kg = ReshapeLayer(
                 net_kg,
                 shape=(-1, self.max_length, 200),
                 name="reshape_kg_2"
             )

             net_in = ConcatLayer(
                 [net_word_embed, net_class_label_embed, net_kg],
                 concat_dim=-1,
                 name='concat_kg_word'
             )

             net_encoder = RNNLayer(
                 net_in,
                 cell_fn = tf.contrib.rnn.BasicLSTMCell,
                 n_hidden = self.hidden_dim,
                 n_steps = self.max_length,
                     return_last = False,
                     return_seq_2d = False,
                     name = 'net_encoder'
                 )
     
                 net_align = LambdaLayer(
                     net_encoder,
                     fn=self.__attention_align_1__,
                     name="attention_align"
                 )
     
                 # net_align = LambdaLayer(
                 #     net_word_embed,
                 #     fn=self.__attention_align_2__,
                 #     fn_args={"classlabel": net_class_label_embed.outputs, "kgvector": net_kg.outputs},
                 #     name="attention_align"
                 # )
     
                 net_attention = LambdaLayer(
                     net_encoder,
                     fn=self.__attention_h_star__,
                     fn_args={'align': net_align.outputs},
                     name="attention_h_star"
                 )
     
                 # net_encoder.outputs = net_encoder.outputs[:, -1, :]
     
                 net_out = DenseLayer(
                     net_attention,
                     # net_encoder,
                     n_units=1,
                     act=tf.sigmoid,
                     name='dense'
                 )
     
             return net_out, net_align
     '''

    def __attention_align_1__(self, inputs):
        # print(inputs.get_shape())

        W_a = tf.get_variable(
            name="attention_w_a",
            shape=(inputs.get_shape()[-1], inputs.get_shape()[-1]),
            initializer=tf.random_uniform_initializer(-0.1, 0.1),
            trainable=True
        )

        hs = tf.unstack(inputs[:, :-1, :], axis=0)
        ct = tf.unstack(inputs[:, -1:, :], axis=0)

        align = list()

        for i in range(inputs.get_shape()[0]):
            align.append(tf.matmul(
                a=tf.matmul(
                    a=hs[i],
                    b=W_a
               ),
                b=tf.transpose(ct[i])
            ))
            # align.append(tf.matmul(
            #     a=hs[i],
            #     b=ct[i],
            #     transpose_b=True
            # ))

        align = tf.stack(align, axis=0)
        align = tf.squeeze(align, axis=[-1])

        align = tf.nn.softmax(align)
        self.attention_softmax = align
        align = tf.expand_dims(align, axis=-1)

        return align

    def __attention_align_2__(self, word, classlabel, kgvector):

        norm_word = tf.nn.l2_normalize(word, dim=-1)
        norm_class = tf.nn.l2_normalize(classlabel, dim=-1)

        align_word = tf.reduce_mean(tf.multiply(norm_word, norm_class), axis=-1)
        align_word = tf.expand_dims(align_word, axis=-1)

        W_kg = tf.get_variable(
            name="attention_w_kg",
            shape=(kgvector.get_shape()[-1], 1),
            initializer=tf.random_uniform_initializer(-0.1, 0.1),
            trainable=True
        )
        align_kg = tf.matmul(
            a=tf.reshape(kgvector, shape=[-1, 200]),
            b=W_kg
        )
        align_kg = tf.reshape(align_kg, shape=[-1, self.max_length, 1])

        align_score = align_kg + align_word

        align_score = tf.slice(align_score, begin=[0, 0, 0], size=[-1, self.max_length - 1, 1])
        return align_score

    def __attention_h_star__(self, inputs, align):

        h_star = tf.matmul(
            a=inputs[:, :-1, :],
            b=align,
            transpose_a=True
        )

        h_star = tf.squeeze(h_star, axis=[-1])

        # ct = tf.squeeze(ct, axis=[1])
        # h_star = tf.concat([ct, h_star], axis=-1)

        return h_star

    def __create_loss__(self):
        self.train_loss = tl.cost.binary_cross_entropy(
            output=self.train_net.outputs,
            target=self.category_logits,
        )

        # self.train_align_loss =

        self.test_loss = tl.cost.binary_cross_entropy(
            output=self.test_net.outputs,
            target=self.category_logits,
        )
        # self.infer_loss = tl.cost.binary_cross_entropy(
        #     output=self.infer_net.outputs,
        #     target=self.category_logits_infer,
        # )

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
    model = Model_KG4Text(
        model_name="text_encoding",
        start_learning_rate=0.001,
        decay_rate=0.8,
        decay_steps=1000
    )
    pass
