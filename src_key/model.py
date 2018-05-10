import os
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import \
    InputLayer, Conv1d, MaxPool1d, \
    RNNLayer, DropoutLayer, DenseLayer, \
    LambdaLayer, ReshapeLayer, ConcatLayer, \
    Conv2d, MaxPool2d, FlattenLayer, \
    DeConv2d, BatchNormLayer, EmbeddingInputlayer, \
    Seq2Seq, retrieve_seq_length_op2

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
            vocab_size=config.vocab_size
    ):
        self.model_name = model_name
        self.start_learning_rate = start_learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.word_embedding_dim = word_embedding_dim
        self.kg_embedding_dim = kg_embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.__create_placeholders__()
        self.__create_model__()
        self.__create_loss__()
        self.__create_training_op__()


    def __create_placeholders__(self):
        self.encode_seqs = tf.placeholder(dtype=tf.int64, shape=[config.batch_size, None], name="encode_seqs")
        self.kg_score = tf.placeholder(dtype=tf.int64, shape=[config.batch_size, None, self.kg_embedding_dim], name="kg_score")

        self.encode_seqs_inference = tf.placeholder(dtype=tf.int64, shape=[1, None], name="encode_seqs_inference")


    def __create_model__(self):
        self.__get_network__(self.model_name, self.encode_seqs, self.kg_score, reuse=False, is_train=True)


    def __get_network__(self, model_name, encode_seqs, kg_score, reuse=False, is_train=True):
        with tf.variable_scope(model_name, reuse=reuse):

            net_encode = EmbeddingInputlayer(
                inputs=encode_seqs,
                vocabulary_size = self.vocab_size,
                embedding_size = self.word_embedding_dim,
                name='seq_embedding')

            

        return


    def __create_loss__(self):
        pass

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
        # self.optim = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5) \
        #     .minimize(self.train_loss, var_list=self.train_net.all_params)


if __name__ == "__main__":
    model = Model_KG4Text(
        model_name="text_encoding",
        start_learning_rate=0.001,
        decay_rate=0.8,
        decay_steps=1000
    )
    pass
