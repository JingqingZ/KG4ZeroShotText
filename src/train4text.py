import os
import random
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorlayer as tl
from datetime import datetime, timedelta

import utils
import config
import model4text
import dataloader4text

results_path = "../results/"

class Controller():

    def __init__(self, model):
        self.model = model
        self.saver = tf.train.Saver(max_to_keep=200)
        self.sess = tf.Session()
        tl.layers.initialize_global_variables(self.sess)
        self.__init_path__()
        self.__init_mkdir__()

    def __init_path__(self):
        self.model_save_dir = "%s/%s/models/" % (results_path, self.model.model_name)
        self.log_save_dir = "%s/%s/logs/" % (results_path, self.model.model_name)
        self.figs_save_dir = "%s/%s/figs/" % (results_path, self.model.model_name)

    def __init_mkdir__(self):
        dirlist = [
            self.model_save_dir,
            self.log_save_dir,
            self.figs_save_dir
        ]
        utils.make_dirlist(dirlist)

    def save_model(self, path, global_step=None):
        save_path = self.saver.save(self.sess, path, global_step=global_step)
        print("[S] Model saved in ckpt %s" % save_path)
        return save_path

    def restore_model(self, path, global_step=None):
        model_path = "%s-%s" % (path, global_step)
        self.saver.restore(self.sess, model_path)
        print("[R] Model restored from ckpt %s" % model_path)
        return True

    def save_model_npz_dict(self, path, global_step=None):
        name = "%s-%s.npz" % (path, global_step)
        save_list_names = [tensor.name for tensor in self.model.train_net.all_params]
        save_list_var = self.sess.run(self.model.train_net.all_params)
        save_var_dict = {save_list_names[idx]: val for idx, val in enumerate(save_list_var)}
        np.savez(name, **save_var_dict)
        save_list_var = None
        save_var_dict = None
        del save_list_var
        del save_var_dict
        print("[S] Model saved in npz_dict %s" % name)

    def load_assign_model_npz_dict(self, path, global_step=None):
        name = "%s-%s.npz" % (path, global_step)
        params = np.load(name)
        if len(params.keys()) != len(set(params.keys())):
            raise Exception("Duplication in model npz_dict %s" % name)
        ops = list()
        for key in params.keys():
            # tensor = tf.get_default_graph().get_tensor_by_name(key)
            varlist = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=key)
            if len(varlist) > 1:
                raise Exception("Multiple candidate variables to be assigned for name %s" % key)
            elif len(varlist) == 0:
                print("Warning: Tensor named %s not found in network." % key)
            else:
                ops.append(varlist[0].assign(params[key]))
                print("[R] Tensor restored: %s" % key)

        self.sess.run(ops)
        print("[R] Model restored from npz_dict %s" % name)


class Controller4Text(Controller):

    def __init__(
            self,
            model,
            base_epoch
    ):
        Controller.__init__(self, model=model)
        self.base_epoch = base_epoch

    def __run__(self, epoch, text_seqs, vocab, mode):

        train_order = list(range(len(text_seqs)))
        random.shuffle(train_order)

        start_time = time.time()
        step_time = time.time()

        all_loss = np.zeros(1)

        train_steps = len(train_order) // config.batch_size

        for cstep in range(train_steps):

            text_seqs_mini = [text_seqs[idx] for idx in train_order[cstep * config.batch_size : (cstep + 1) * config.batch_size]]

            encode_seqs_mini = dataloader4text.prepro_encode(text_seqs_mini.copy(), vocab)
            decode_seqs_mini = dataloader4text.prepro_decode(text_seqs_mini.copy(), vocab)
            target_seqs_mini = dataloader4text.prepro_target(text_seqs_mini.copy(), vocab)
            target_mask_mini = tl.prepro.sequences_get_mask(target_seqs_mini, vocab.pad_id)

            assert decode_seqs_mini.shape[0] == encode_seqs_mini.shape[0]
            assert decode_seqs_mini.shape[1] == encode_seqs_mini.shape[1] + 1
            assert decode_seqs_mini.shape == target_seqs_mini.shape
            assert decode_seqs_mini.shape == target_mask_mini.shape

            global_step = cstep + epoch * train_steps

            if mode == "train":
                results = self.sess.run([
                    self.model.train_loss,
                    self.model.learning_rate,
                    self.model.optim
                ], feed_dict={
                    self.model.encode_seqs: encode_seqs_mini,
                    self.model.decode_seqs: decode_seqs_mini,
                    self.model.target_seqs: target_seqs_mini,
                    self.model.target_mask: target_mask_mini,
                    self.model.global_step: global_step,
                })

            elif mode == "test":
                results = self.sess.run([
                    self.model.test_loss,
                    tf.constant(-1),
                    tf.constant(-1)
                ], feed_dict={
                    self.model.encode_seqs: encode_seqs_mini,
                    self.model.decode_seqs: decode_seqs_mini,
                    self.model.target_seqs: target_seqs_mini,
                    self.model.target_mask: target_mask_mini,
                })

            del text_seqs_mini, encode_seqs_mini, decode_seqs_mini, target_seqs_mini, target_mask_mini

            all_loss += results[:1]

            if cstep % 20 == 0 and cstep > 0:
                print(
                    "[T%s] Epoch: [%3d][%4d/%4d] time: %.4f, lr: %.8f, loss: %s" %
                    (mode[1:], epoch, cstep, train_steps, time.time() - step_time, results[-2], all_loss / (cstep + 1))
                )
                step_time = time.time()

        print(
            "[T%s Sum] Epoch: [%3d] time: %.4f, lr: %.8f, loss: %s" %
            (mode[1:], epoch, time.time() - start_time, results[-2], all_loss / train_steps)
        )

        return all_loss / train_steps


    def __inference__(self, epoch, text_seqs, vocab, verify=False):

        start_time = time.time()
        step_time = time.time()

        text_state_list = list()

        for cstep in range(len(text_seqs)):

            text_seqs_mini = text_seqs[cstep : cstep + 1]
            encode_seqs_mini = dataloader4text.prepro_encode(text_seqs_mini.copy(), vocab)
            decode_seqs_mini = dataloader4text.prepro_decode(text_seqs_mini.copy(), vocab)

            state = self.sess.run(
                self.model.inference_seq2seq.final_state_encode,
                feed_dict={
                    self.model.encode_seqs_inference: encode_seqs_mini,
                })

            text_state_list.append(state)

            # for verification
            if verify:

                o, state = self.sess.run([
                    self.model.inference_y,
                    self.model.inference_seq2seq.final_state_decode
                ], feed_dict={
                    self.model.inference_seq2seq.initial_state_decode: state,
                    self.model.decode_seqs_inference: [[vocab.start_id]],
                })

                word_id = tl.nlp.sample_top(o[0], top_k=3)
                word = vocab.id_to_word(word_id)

                sentence = [word]
                sentence_id = [word_id]

                for _ in range(1, len(text_seqs_mini[0]) - 1):
                    o, state = self.sess.run([
                        self.model.inference_y,
                        self.model.inference_seq2seq.final_state_decode
                    ], feed_dict={
                        self.model.inference_seq2seq.initial_state_decode: state,
                        self.model.decode_seqs_inference: [[word_id]]
                    })

                    word_id = tl.nlp.sample_top(o[0], top_k=3)
                    word = vocab.id_to_word(word_id)

                    if word_id == vocab.end_id:
                        break

                    sentence = sentence + [word]
                    sentence_id = sentence_id + [word_id]

                full_out = self.sess.run(
                    self.model.inference_y
                , feed_dict={
                    self.model.encode_seqs_inference: encode_seqs_mini,
                    self.model.decode_seqs_inference: decode_seqs_mini
                })

                decode_recons_sentence = list()
                for o in full_out:
                    word_id = tl.nlp.sample_top(o, top_k=3)
                    word = vocab.id_to_word(word_id)
                    decode_recons_sentence.append(word)

                origin_sentence = [vocab.id_to_word(text_id) for text_id in text_seqs_mini[0]]
                encode_sentence = [vocab.id_to_word(text_id) for text_id in encode_seqs_mini[0]]

                print(" origin > ", ' '.join(origin_sentence))
                print(" encode > ", ' '.join(encode_sentence))
                print(" decode > ", ' '.join(decode_recons_sentence))
                print(" recons > ", ' '.join(sentence))
                print("--------------------")

        return np.array(text_state_list)


    def controller(self, text_seqs, vocab, inference=False, train_epoch=config.train_epoch):

        if inference:
            assert self.base_epoch > 0

        last_save_epoch = self.base_epoch
        global_epoch = self.base_epoch + 1

        if last_save_epoch >= 0:
            self.restore_model(
                path=self.model_save_dir,
                global_step=last_save_epoch
            )

        if inference:

            state = self.__inference__(global_epoch, text_seqs[:3], vocab, verify=False)
            print(state.shape)
            state = self.__inference__(global_epoch, text_seqs[-3:], vocab, verify=False)
            print(state.shape)

        else:

            for epoch in range(train_epoch + 1):

                self.__run__(global_epoch, text_seqs[:-len(text_seqs) // 10], vocab, mode="train")
                self.__run__(global_epoch, text_seqs[-len(text_seqs) // 10:], vocab, mode="test")

                self.__inference__(global_epoch, text_seqs[:3], vocab)
                self.__inference__(global_epoch, text_seqs[-3:], vocab)

                if global_epoch > self.base_epoch and global_epoch % 1 == 0:
                    self.save_model(
                        path=self.model_save_dir,
                        global_step=global_epoch
                    )
                    last_save_epoch = global_epoch

                global_epoch += 1


if __name__ == "__main__":

    vocab = dataloader4text.build_vocabulary_from_full_corpus(
        config.wiki_full_data_path, config.wiki_vocab_path, column="text", force_process=False
    )

    text_seqs = dataloader4text.load_data_from_text_given_vocab(
        config.wiki_train_data_path, vocab, config.wiki_train_processed_path,
        column="text", force_process=False
    )

    with tf.Graph().as_default() as graph:
        tl.layers.clear_layers_name()
        mdl = model4text.Model4Text(
            model_name="text_encoding_wiki",
            start_learning_rate=0.0001,
            decay_rate=0.8,
            decay_steps=4e3,
            # vocab_size=5e3
        )
        ctl = Controller4Text(model=mdl, base_epoch=-1)
        # ctl.controller(text_seqs, vocab, train_epoch=20)
        ctl.controller(text_seqs, vocab, inference=True)
        ctl.sess.close()

    '''

    vocab = dataloader4text.build_vocabulary_from_full_corpus(
        config.arxiv_full_data_path, config.arxiv_vocab_path, column="abstract", force_process=False
    )

    text_seqs = dataloader4text.load_data_from_text_given_vocab(
        config.arxiv_train_data_path, vocab, config.arxiv_train_processed_path,
        column="abstract", force_process=False
    )

    # text_seqs = text_seqs[:len(text_seqs) // 10]

    with tf.Graph().as_default() as graph:
        tl.layers.clear_layers_name()
        mdl = model4text.Model4Text(
            model_name="text_encoding_arxiv",
            start_learning_rate=0.0001,
            decay_rate=0.8,
            decay_steps=4e3,
            # vocab_size=config.vocab_size
        )
        ctl = Controller4Text(model=mdl, base_epoch=20)
        # ctl.controller(text_seqs, vocab, train_epoch=20)
        ctl.controller(text_seqs, vocab, inference=True)
        ctl.sess.close()
    '''
    pass








