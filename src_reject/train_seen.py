import os
import sys
import random
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import math
import sklearn
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorlayer as tl
from datetime import datetime, timedelta
from random import randint
import progressbar
import nltk

import log
import utils
import error
import config
import model_seen
import train_base
import dataloader

results_path = "../results/"

class Controller4Seen(train_base.Base_Controller):

    def __init__(
            self,
            model,
            vocab,
            class_dict,
            word_embed_mat,
            random_unseen_class=False,
            random_percentage=0.25,
            random_unseen_class_list=None,
            base_epoch=-1,
    ):
        super(Controller4Seen, self).__init__(model)

        logging = log.Log(sys.stdout, self.log_save_dir + "log-%s" % utils.now2string())
        sys.stdout = logging

        self.base_epoch = base_epoch
        self.vocab = vocab
        self.class_dict = class_dict
        self.word_embed_mat = word_embed_mat

        self.full_class_list = sorted(list(self.class_dict.keys()))
        self.full_class_map2index = dict()
        for idx, class_id in enumerate(self.full_class_list):
            assert class_id not in self.full_class_map2index
            self.full_class_map2index[class_id] = idx
        print("All class map to index %s" % ([(k, v) for k, v in self.full_class_map2index.items()]))

        if random_unseen_class:
            num_class = len(class_dict.keys())
            num_unseen_class = int(num_class * random_percentage)
            class_id_list = list(class_dict.keys())
            random.shuffle(class_id_list)
            self.unseen_class = class_id_list[:num_unseen_class]
            print("Random selected unseen class %s" % (self.unseen_class))
        else:
            assert random_unseen_class_list is not None
            self.unseen_class = random_unseen_class_list
            print("Assigned unseen class %s" % (self.unseen_class))

        self.seen_class = list()
        self.seen_class_map2index = dict()
        indx = 0
        for key in self.class_dict.keys():
            if key not in self.unseen_class:
                assert key not in self.seen_class_map2index
                self.seen_class.append(key)
                self.seen_class_map2index[key] = indx
                indx += 1
        print("Seen class %s" % (self.seen_class))
        print("Seen class map to index %s" % ([(k, v) for k, v in self.seen_class_map2index.items()]))

        assert len(self.seen_class) == self.model.number_of_seen_classes

    def check_seen(self, class_id):
        return not class_id in self.unseen_class

    def get_text_of_seen_class(self, text_seqs, class_list):
        print("Getting text of seen classes")
        assert len(text_seqs) == len(class_list)
        seen_text_seqs = list()
        seen_class_list = list()

        with progressbar.ProgressBar(max_value=len(text_seqs)) as bar:
            for idx, text in enumerate(text_seqs):
                if self.check_seen(class_list[idx]):
                    seen_text_seqs.append(text)
                    seen_class_list.append(class_list[idx])
                bar.update(idx)
        assert len(seen_text_seqs) == len(seen_class_list)
        print("Text seqs of seen classes: %d" % len(seen_text_seqs))
        return seen_text_seqs, seen_class_list

    def prepro_encode(self, textlist):
        newtextlist = list()
        for idx, text in enumerate(textlist):
            if len(text[1:-1]) > self.model.max_length:
                startid = 1 + randint(0, len(text[1:-1]) - self.model.max_length)
            else:
                startid = 1
            newtextlist.append(text[startid:-1] + [self.vocab.pad_id])
        newtextlist = tl.prepro.pad_sequences(newtextlist, maxlen=self.model.max_length, dtype='int64', padding='post', truncating='post', value=self.vocab.pad_id)
        for idx, text in enumerate(newtextlist):
            newtextlist[idx] = text[:-1] + [self.vocab.pad_id]

        text_array = np.zeros((len(newtextlist), self.model.max_length, self.model.word_embedding_dim))
        for idx, text in enumerate(newtextlist):
            for widx, word_id in enumerate(text):
                text_array[idx][widx] = self.word_embed_mat[word_id]

        return np.array(newtextlist), text_array

    def get_pred_class_topk(self, pred_mat, k=1):
        assert k > 0
        pred_k = list()
        for i in range(pred_mat.shape[0]):
            confidence = pred_mat[i]
            topk = np.array([self.full_class_list[_] for _ in np.argsort(confidence)[-k:]])
            for class_id in topk:
                assert class_id in self.class_dict
            pred_k.append(topk)
        pred_k = np.array(pred_k)
        pred_k = np.reshape(pred_k, newshape=(pred_mat.shape[0], k))
        assert pred_k.shape == (pred_mat.shape[0], k)
        return pred_k

    def get_one_hot_results(self, class_list):
        one_hot_res = np.zeros(shape=(class_list.shape[0], len(self.class_dict)))
        for idx, item in enumerate(class_list):
            for class_id in item:
                assert class_id in self.class_dict
                one_hot_res[idx, class_id - 1] = 1
        return one_hot_res

    def __train__(self, epoch, text_seqs, class_list, max_train_steps=None):

        assert len(text_seqs) == len(class_list)

        train_order = list(range(len(text_seqs)))

        random.shuffle(train_order)

        start_time = time.time()
        step_time = time.time()

        all_loss = np.zeros(1)

        train_steps = len(train_order) // config.batch_size

        if max_train_steps is not None and max_train_steps < train_steps:
            train_steps = max_train_steps

        for cstep in range(train_steps):
            global_step = cstep + epoch * train_steps

            class_idx_mini = [self.seen_class_map2index[class_list[idx]] for idx in train_order[cstep * config.batch_size : (cstep + 1) * config.batch_size]]
            text_seqs_mini = [text_seqs[idx] for idx in train_order[cstep * config.batch_size : (cstep + 1) * config.batch_size]]

            encode_seqs_id_mini, encode_seqs_mat_mini = self.prepro_encode(text_seqs_mini)

            results = self.sess.run([
                self.model.train_loss,
                self.model.train_net.outputs,
                # self.model.train_align.outputs,
                self.model.learning_rate,
                self.model.optim
            ], feed_dict={
                self.model.encode_seqs: encode_seqs_mat_mini,
                self.model.category_target_index: np.array(class_idx_mini),
                self.model.global_step: global_step,
            })

            all_loss += results[:1]

            if cstep % config.cstep_print == 0 and cstep > 0:
                print(
                    "[Train] Epoch: [%3d][%4d/%4d] time: %.4f, lr: %.8f, loss: %s" %
                    (epoch, cstep, train_steps, time.time() - step_time, results[-2], all_loss / (cstep + 1) )
                )
                step_time = time.time()

        print(
            "[Train Sum] Epoch: [%3d] time: %.4f, lr: %.8f, loss: %s" %
            (epoch, time.time() - start_time, results[-2], all_loss / train_steps )
        )

        return all_loss / train_steps

    def __test__(self, epoch, text_seqs, class_list):

        assert len(text_seqs) == len(class_list)

        start_time = time.time()
        step_time = time.time()

        test_steps = len(text_seqs) // config.batch_size

        topk_list = list()
        pred_class_list = list()

        all_loss = np.zeros(1)

        for cstep in range(test_steps):

            text_seqs_mini = text_seqs[cstep * config.batch_size : (cstep + 1) * config.batch_size]
            class_idx_mini = [self.seen_class_map2index[_] for _ in class_list[cstep * config.batch_size : (cstep + 1) * config.batch_size]]

            encode_seqs_id_mini, encode_seqs_mat_mini = self.prepro_encode(text_seqs_mini)

            pred_mat = np.zeros([config.batch_size, len(self.class_dict)])

            test_loss, out  = self.sess.run([
                self.model.test_loss,
                self.model.test_net.outputs,
            ], feed_dict={
                self.model.encode_seqs: encode_seqs_mat_mini,
                self.model.category_target_index: class_idx_mini
            })

            all_loss[0] += test_loss

            pred = np.array([_ / np.sum(_) for _ in np.exp(out)])

            for i in range(len(self.seen_class)):
                pred_mat[:, self.full_class_map2index[self.seen_class[i]]] = pred[:, i]

            topk = self.get_pred_class_topk(pred_mat, k=1)
            topk_list.append(topk)
            pred_class_list.append(pred_mat)

            if cstep % config.cstep_print == 0 and cstep > 0:
                tmp_topk = np.concatenate(topk_list, axis=0)
                tmp_topk = self.get_one_hot_results(np.array(tmp_topk[: (cstep + 1) * config.batch_size]))
                tmp_gt = self.get_one_hot_results(np.reshape(np.array(class_list[ : (cstep + 1) * config.batch_size]), newshape=(-1, 1)))
                tmp_stats = utils.get_statistics(tmp_topk, tmp_gt, single_label_pred=True)

                print(
                    "[Test] Epoch: [%3d][%4d/%4d] time: %.4f, loss: %s \n %s" %
                    (epoch, cstep, test_steps, time.time() - step_time, all_loss / (cstep + 1), utils.dict_to_string_4_print(tmp_stats))
                )
                step_time = time.time()


        prediction_topk = np.concatenate(topk_list, axis=0)
        prediction_topk = self.get_one_hot_results(np.array(prediction_topk[: test_steps * config.batch_size]))
        ground_truth = self.get_one_hot_results(np.reshape(np.array(class_list[: test_steps * config.batch_size]), newshape=(-1, 1)))

        stats = utils.get_statistics(prediction_topk, ground_truth, single_label_pred=True)

        print(
            "[Test Sum] Epoch: [%3d] time: %.4f, loss: %s \n %s" %
            (epoch, time.time() - start_time, all_loss / test_steps , utils.dict_to_string_4_print(stats))
        )

        return stats, prediction_topk, ground_truth, np.array([0]), np.array([0])

    def controller(self, train_text_seqs, train_class_list, test_text_seqs, test_class_list, train_epoch=config.train_epoch, save_test_per_epoch=1):

        last_save_epoch = self.base_epoch
        global_epoch = self.base_epoch + 1

        if last_save_epoch >= 0:
            self.restore_model(
                path=self.model_save_dir,
                global_step=last_save_epoch
            )

        train_text_seqs, train_class_list = self.get_text_of_seen_class(train_text_seqs, train_class_list)

        seen_test_text_seqs, seen_test_class_list = self.get_text_of_seen_class(test_text_seqs, test_class_list)

        # double check if text of unseen classes are removed
        for class_id in train_class_list:
            assert self.check_seen(class_id)
        # seen
        for class_id in seen_test_class_list:
            assert self.check_seen(class_id)

        for epoch in range(train_epoch + 1):

            self.__train__(
                global_epoch,
                train_text_seqs,
                train_class_list,
                # max_train_steps=3000
            )

            if global_epoch > self.base_epoch and global_epoch % save_test_per_epoch == 0:
                self.save_model(
                    path=self.model_save_dir,
                    global_step=global_epoch
                )
                last_save_epoch = global_epoch

            if global_epoch % save_test_per_epoch == 0:

                # TODO: remove to get full test
                print("[Test] Testing seen classes")
                state_seen, pred_seen, gt_seen, align_seen, kg_vector_seen = self.__test__(
                    global_epoch,
                    [_ for idx, _ in enumerate(seen_test_text_seqs) if idx % 1 == 0],
                    [_ for idx, _ in enumerate(seen_test_class_list) if idx % 1 == 0],
                )

                np.savez(
                    self.log_save_dir + "test_%d" % global_epoch,
                    seen_class=self.seen_class,
                    unseen_class=self.unseen_class,
                    pred_seen=pred_seen,
                    gt_seen=gt_seen,
                    align_seen=align_seen,
                    kg_vector_seen=kg_vector_seen,
                )
                # error.rejection_single_label(self.log_save_dir + "test_%d.npz" % global_epoch)
                # error.classify_single_label(self.log_save_dir + "test_%d.npz" % global_epoch)
            global_epoch += 1

    def controller4test(self, test_text_seqs, test_class_list, unseen_class_list, base_epoch=None):

        self.unseen_class = unseen_class_list

        last_save_epoch = self.base_epoch if base_epoch is None else base_epoch
        global_epoch = self.base_epoch if base_epoch is None else base_epoch

        if last_save_epoch >= 0:
            self.restore_model(
                path=self.model_save_dir,
                global_step=last_save_epoch
            )

        seen_test_text_seqs, seen_test_class_list = self.get_text_of_seen_class(test_text_seqs, test_class_list)

        # seen
        for class_id in seen_test_class_list:
            assert self.check_seen(class_id)

        # TODO: remove to get full test
        print("[Test] Testing seen classes")
        state_seen, pred_seen, gt_seen, align_seen, kg_vector_seen = self.__test__(
            global_epoch,
            [_ for idx, _ in enumerate(seen_test_text_seqs) if idx % 1 == 0],
            [_ for idx, _ in enumerate(seen_test_class_list) if idx % 1 == 0],
        )

        np.savez(
            self.log_save_dir + "test_%d" % global_epoch,
            seen_class=self.seen_class,
            unseen_class=self.unseen_class,
            pred_seen=pred_seen,
            gt_seen=gt_seen,
            align_seen=align_seen,
            kg_vector_seen=kg_vector_seen,
        )

if __name__ == "__main__":

    # DBpedia
    vocab = dataloader.build_vocabulary_from_full_corpus(
        # config.zhang15_dbpedia_full_data_path, config.zhang15_dbpedia_vocab_path, column="selected", force_process=False,
        config.zhang15_dbpedia_full_data_path, config.zhang15_dbpedia_vocab_path, column="text", force_process=False,
        min_word_count=55
    )

    glove_mat = dataloader.load_glove_word_vector(
        config.word_embed_file_path, config.zhang15_dbpedia_word_embed_matrix_path, vocab, force_process=False
    )
    assert np.sum(glove_mat[vocab.start_id]) == 0
    assert np.sum(glove_mat[vocab.end_id]) == 0
    assert np.sum(glove_mat[vocab.unk_id]) == 0
    assert np.sum(glove_mat[vocab.pad_id]) == 0

    class_dict = dataloader.load_class_dict(
        class_file=config.zhang15_dbpedia_class_label_path,
        class_code_column="ClassCode",
        class_name_column="ConceptNet"
    )

    print("Check NaN in csv ...")
    check_nan_train = dataloader.check_df(config.zhang15_dbpedia_train_path)
    check_nan_test = dataloader.check_df(config.zhang15_dbpedia_test_path)
    print("Train NaN %s, Test NaN %s" % (check_nan_train, check_nan_test))

    assert not check_nan_train
    assert not check_nan_test

    train_class_list = dataloader.load_data_class(
        filename=config.zhang15_dbpedia_train_path,
        column="class",
    )

    train_text_seqs = dataloader.load_data_from_text_given_vocab(
        config.zhang15_dbpedia_train_path, vocab, config.zhang15_dbpedia_train_processed_path,
        column="text", force_process=False
        # column="selected", force_process=False
        # column="selected_tfidf", force_process=False
        # column="selected_tfidf", force_process=False
    )

    test_class_list = dataloader.load_data_class(
        filename=config.zhang15_dbpedia_test_path,
        column="class",
    )

    test_text_seqs = dataloader.load_data_from_text_given_vocab(
        config.zhang15_dbpedia_test_path, vocab, config.zhang15_dbpedia_test_processed_path,
        column="text", force_process=False
        # column="selected", force_process=False
        # column="selected_tfidf", force_process=False
        # column="selected_tfidf", force_process=False
    )

    # print([vocab.id_to_word(wordid) for wordid in train_text_seqs[0][:10]])
    # lenlist = [len(text) for text in test_text_seqs]
    # print(np.mean(lenlist))  # full 49.52 selected_tfidf 32.03
    # print(np.percentile(lenlist, 95))  # full 85 selected_tfidf 58
    # print(np.percentile(lenlist, 90))  # full 80 selected_tfidf 53
    # print(np.percentile(lenlist, 80))  # full 73 selected_tfidf 47
    # exit()

    for i in range(10):

        unseen_percentage = 0.25
        max_length = 50

        with tf.Graph().as_default() as graph:
            tl.layers.clear_layers_name()

            mdl = model_seen.Model4Seen(
                model_name="seen_full_zhang15_dbpedia_kg3_cluster_3group_random%d_unseen%.2f_max%d_cnn" \
                           % (i + 1, unseen_percentage, max_length),
                start_learning_rate=0.001,
                decay_rate=0.5,
                decay_steps=20e3,
                max_length=max_length,
                number_of_seen_classes=11
            )
            # TODO: if unseen_classes are already selected, set randon_unseen_class=False and provide a list of unseen_classes
            ctl = Controller4Seen(
                model=mdl,
                vocab=vocab,
                class_dict=class_dict,
                word_embed_mat=glove_mat,
                random_unseen_class=True,
                random_unseen_class_list=None,
                base_epoch=-1,
            )
            ctl.controller(train_text_seqs, train_class_list, test_text_seqs, test_class_list, train_epoch=2)
            # ctl.controller4test(test_text_seqs, test_class_list, unseen_class_list, base_epoch=5)
            ctl.controller4test(test_text_seqs, test_class_list, unseen_class_list=ctl.unseen_class, base_epoch=2)

            ctl.sess.close()
    pass





