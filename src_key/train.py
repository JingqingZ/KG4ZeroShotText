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
import model
import dataloader

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


class Controller_KG4Text(Controller):

    def __init__(
            self,
            model,
            vocab,
            class_dict,
            kg_vector_dict,
            word_embed_mat,
            num_fold=4,
            current_fold=1,
            random_unseen=True,
            random_percentage=0.25,
            base_epoch=-1,
            lemma=False
    ):

        Controller.__init__(self, model=model)

        logging = log.Log(sys.stdout, self.log_save_dir + "log-%s" % utils.now2string())
        sys.stdout = logging

        self.base_epoch = base_epoch
        self.vocab = vocab
        self.class_dict = class_dict
        self.kg_vector_dict = kg_vector_dict
        self.word_embed_mat = word_embed_mat
        self.lemma = lemma

        if self.lemma:
            from nltk.stem import WordNetLemmatizer
            self.lemmatizer = WordNetLemmatizer()

        self.num_fold = num_fold
        self.current_fold = current_fold

        num_class = len(class_dict.keys())
        if random_unseen:
            num_unseen_class = int(num_class * random_percentage)
            class_id_list = list(class_dict.keys())
            random.shuffle(class_id_list)
            self.unseen_class = class_id_list[:num_unseen_class]
            print("Random selected unseen class %s" % (self.unseen_class))

        else:
            num_of_class_in_each_fold = list()
            remain = num_class

            unseen_class_tuple =[0, 0]

            assert current_fold > 0
            assert current_fold <= num_fold
            for i in range(num_fold):
                this_fold = math.ceil(remain / (num_fold - i))
                remain -= this_fold
                num_of_class_in_each_fold.append(this_fold)

                if i == self.current_fold - 1:
                    unseen_class_tuple[1] = num_class - remain
                    unseen_class_tuple[0] = num_class - remain - this_fold + 1

            self.unseen_class = [range(unseen_class_tuple[0], unseen_class_tuple[1] + 1)]
            print("%d/%d fold class %s" % (current_fold, num_fold, self.unseen_class))

    def get_seen_class(self):
        seen_class = list()
        for key in self.class_dict.keys():
            if key not in self.unseen_class:
                seen_class.append(key)
        return seen_class

    def check_seen(self, class_id):
        return not class_id in self.unseen_class

    def get_kg_vector_given_class(self, encode_text_seqs, class_id_list):

        # TODO: remove to add kg_vector for training
        # return np.zeros((config.batch_size, self.model.max_length, config.kg_embedding_dim))

        assert encode_text_seqs.shape[0] == class_id_list.shape[0]

        kg_vector_list = list()

        for idx, class_id in enumerate(class_id_list):

            kg_vector = np.zeros([self.model.max_length, config.kg_embedding_dim])

            for widx, word_id in enumerate(encode_text_seqs[idx]):
                word = self.vocab.id_to_word(word_id)
                if self.lemma:
                    new_word = nltk.pos_tag([word])  # a list of words à a list of words with part of speech
                    new_word = [self.lemmatizer.lemmatize(t[0], config.pos_dict[t[1]]) for t in new_word if t[1] in config.pos_dict]
                    if len(new_word) > 0:
                        word = new_word[0]
                kg_vector[widx, :] = dataloader.get_kg_vector(self.kg_vector_dict, self.class_dict[class_id], word)

            kg_vector_list.append(kg_vector)

        kg_vector_list = np.array(kg_vector_list)

        assert kg_vector_list.shape == (config.batch_size, self.model.max_length, config.kg_embedding_dim)

        return kg_vector_list

    def get_random_class(self, true_class_id_list, category_logits):
        class_id_list = list()

        for idx, logit in enumerate(category_logits):
            true_class_id = true_class_id_list[idx]
            assert self.check_seen(true_class_id)
            if logit == 1:
                class_id_list.append(true_class_id)
            elif logit == 0:
                while True:
                    random_class_id = random.choice(list(self.class_dict))
                    # random choice class must be seen class
                    if random_class_id != true_class_id and self.check_seen(random_class_id):
                        break
                class_id_list.append(random_class_id)
            else:
                raise ValueError("category logit has to be 0 or 1.")
        return np.array(class_id_list)

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

    def get_text_of_unseen_class(self, text_seqs, class_list):
        print("Getting text of unseen classes")
        assert len(text_seqs) == len(class_list)
        unseen_text_seqs = list()
        unseen_class_list = list()

        with progressbar.ProgressBar(max_value=len(text_seqs)) as bar:
            for idx, text in enumerate(text_seqs):
                if not self.check_seen(class_list[idx]):
                    unseen_text_seqs.append(text)
                    unseen_class_list.append(class_list[idx])
                bar.update(idx)
        assert len(unseen_text_seqs) == len(unseen_class_list)
        print("Text seqs of unseen classes: %d" % len(unseen_text_seqs))
        return unseen_text_seqs, unseen_class_list

    # TODO: design for single-label case
    def get_one_hot_results(self, class_list):
        one_hot_res = np.zeros(shape=(class_list.shape[0], len(self.class_dict)))
        for idx, item in enumerate(class_list):
            for class_id in item:
                assert class_id in self.class_dict
                one_hot_res[idx, class_id - 1] = 1
        return one_hot_res

    # TODO: this is designed for single-label classification
    def get_pred_class_topk(self, pred_mat, k=1):
        assert k > 0
        pred_k = list()
        for i in range(pred_mat.shape[0]):
            confidence = pred_mat[i]
            topk = np.argsort(confidence)[-k:] + 1
            for class_id in topk:
                assert class_id in self.class_dict
            pred_k.append(topk)
        pred_k = np.array(pred_k)
        pred_k = np.reshape(pred_k, newshape=(pred_mat.shape[0], k))
        assert pred_k.shape == (pred_mat.shape[0], k)
        return pred_k

    def get_class_label_embed(self, class_id_list):
        class_embed = np.zeros((len(class_id_list), self.model.max_length, self.model.word_embedding_dim))

        for idx, class_id in enumerate(class_id_list):
            class_embed[idx, :, :] = np.repeat([self.word_embed_mat[self.vocab.word_to_id(self.class_dict[class_id])]], self.model.max_length, axis=0)

        return class_embed

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
            newtextlist[idx] = text[:-1] + [vocab.pad_id]

        text_array = np.zeros((len(newtextlist), self.model.max_length, self.model.word_embedding_dim))
        for idx, text in enumerate(newtextlist):
            for widx, word_id in enumerate(text):
                text_array[idx][widx] = self.word_embed_mat[word_id]

        return np.array(newtextlist), text_array

    def get_random_text(self, num):
        random_text = list()
        for i in range(num):
            text = [randint(0, self.vocab.unk_id) for _ in range(self.model.max_length)]
            random_text.append(text)
        return random_text

    def __train__(self, epoch, text_seqs, class_list, max_train_steps=None):

        assert len(text_seqs) == len(class_list)

        train_order = list(range(len(text_seqs)))

        random.shuffle(train_order)

        # TODO: uncomment to train on all data
        # train_order = train_order[: 5000 * config.batch_size]

        start_time = time.time()
        step_time = time.time()

        all_loss = np.zeros(1)

        train_steps = len(train_order) // config.batch_size

        if max_train_steps is not None and max_train_steps < train_steps:
            train_steps = max_train_steps

        seen_class_list = self.get_seen_class()
        print("Seen classes: %s" % seen_class_list)

        for cstep in range(train_steps):
            global_step = cstep + epoch * train_steps

            # category_logits = [1 if randint(0, config.negative_sample) == 0 else 0 for _ in range(config.batch_size)]
            category_logits = [1 if randint(0, config.negative_sample + epoch * 2) == 0 else 0 for _ in range(config.batch_size)]
            # category_logits = [1 if randint(0, config.negative_sample + epoch * 3) == 0 else 0 for _ in range(config.batch_size)]

            true_class_id_mini = [class_list[idx] for idx in train_order[cstep * config.batch_size : (cstep + 1) * config.batch_size]]
            text_seqs_mini = [text_seqs[idx] for idx in train_order[cstep * config.batch_size : (cstep + 1) * config.batch_size]]

            # random text
            true_class_id_mini = true_class_id_mini[:-3]  + [-1, -1, -1]
            text_seqs_mini = text_seqs_mini[:-3] + self.get_random_text(3)

            encode_seqs_id_mini, encode_seqs_mat_mini = self.prepro_encode(text_seqs_mini)

            # for class_id in seen_class_list:

            category_logits = category_logits[:-3] + [0, 0, 0]
            class_id_mini = self.get_random_class(true_class_id_mini, category_logits)
            # class_id_mini = np.array([class_id] * config.batch_size)
            # category_logits = [int(class_id_mini[i] == true_class_id_mini[i]) for i in range(config.batch_size)]

            class_label_embed_mini = self.get_class_label_embed(class_id_mini)
            kg_vector_seqs_mini = self.get_kg_vector_given_class(encode_seqs_id_mini, class_id_mini)

            # np.set_printoptions(threshold=np.nan)
            # print("encode", encode_seqs_mat_mini[:2, :20, :10])
            # print("class", class_label_embed_mini[:2, :10, :10])
            # print("kg", np.sum(kg_vector_seqs_mini[:2, :20, :10], axis=-1))
            # print("encode", encode_seqs_id_mini[:2, :10])
            # print("encode", [[self.vocab.id_to_word(word_id) for word_id in text] for text in encode_seqs_id_mini[:2, :20]])
            # print("class", [self.class_dict[class_id] for class_id in class_id_mini[:2]])
            # print("class", class_id_mini[:2])
            # print("true_class", true_class_id_mini[:2])
            # print("cate", category_logits[:2])

            results = self.sess.run([
                self.model.train_loss,
                self.model.train_net.outputs,
                # self.model.train_align.outputs,
                self.model.learning_rate,
                self.model.optim
            ], feed_dict={
                self.model.encode_seqs: encode_seqs_mat_mini,
                self.model.class_label_seqs: class_label_embed_mini,
                self.model.kg_vector: kg_vector_seqs_mini,
                self.model.category_logits: np.expand_dims(np.array(category_logits), -1),
                self.model.global_step: global_step,
            })
            # print("model", time.time() - step_time)

            # print("out", results[1][:2])
            # print("align", results[2][:2, :10])
            # print("trainloss", results[0])
            # exit()

            all_loss += results[:1]

            if cstep % config.cstep_print == 0 and cstep > 0:
                print(
                    "[Train] Epoch: [%3d][%4d/%4d] time: %.4f, lr: %.8f, loss: %s" %
                    (epoch, cstep, train_steps, time.time() - step_time, results[-2], all_loss / (cstep + 1) )
                )
                step_time = time.time()
            # print(time.time() - step_time)
            # exit()

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
        # align_list = list()

        # kg_vector_list = list()

        for cstep in range(test_steps):

            text_seqs_mini = text_seqs[cstep * config.batch_size : (cstep + 1) * config.batch_size]
            true_class_id_mini = class_list[cstep * config.batch_size : (cstep + 1) * config.batch_size]

            encode_seqs_id_mini, encode_seqs_mat_mini = self.prepro_encode(text_seqs_mini)

            pred_mat = np.zeros([config.batch_size, len(self.class_dict)])
            # align_mat = np.zeros([config.batch_size, len(self.class_dict), self.model.max_length - 1])
            # kg_vector_mat = np.zeros([config.batch_size, len(self.class_dict), config.max_length, config.kg_embedding_dim])

            for class_id in self.class_dict:

                class_id_mini = np.array([class_id] * config.batch_size)

                class_label_embed_mini = self.get_class_label_embed(class_id_mini)

                category_logits = np.zeros([config.batch_size, 1])
                for b in range(config.batch_size):
                    category_logits[b, 0] = int(class_id == true_class_id_mini[b])

                kg_vector_seqs_mini = self.get_kg_vector_given_class(encode_seqs_id_mini, class_id_mini)

                test_loss, pred  = self.sess.run([
                    self.model.test_loss,
                    self.model.test_net.outputs,
                    # self.model.test_align.outputs,
                ], feed_dict={
                    self.model.encode_seqs: encode_seqs_mat_mini,
                    self.model.class_label_seqs: class_label_embed_mini,
                    self.model.kg_vector: kg_vector_seqs_mini,
                    self.model.category_logits: category_logits,
                })

                # print(test_loss)

                pred_mat[:, class_id - 1] = pred[:, 0]
                # align_mat[:, class_id - 1, :] = align[:, :, 0]
                # kg_vector_mat[:, class_id - 1, :, :] = kg_vector_seqs_mini

            topk = self.get_pred_class_topk(pred_mat, k=1)
            topk_list.append(topk)
            pred_class_list.append(pred_mat)
            # kg_vector_list.append(kg_vector_mat)

            # align_list.append(align_mat)

            # print(align_mat)
            # print(pred_mat)
            # print(pred_class_list)
            # print([np.argmax(pred) + 1 for pred in pred_mat])
            # print(true_class_id_mini)
            # exit()

            if cstep % config.cstep_print == 0:
                tmp_pred = np.concatenate(pred_class_list, axis=0)
                threshold = 0.5
                tmp_pred[tmp_pred > threshold] = 1
                tmp_pred[tmp_pred <= threshold] = 0
                tmp_gt = self.get_one_hot_results(np.reshape(np.array(class_list[ : (cstep + 1) * config.batch_size]), newshape=(-1, 1)))
                tmp_stats = utils.get_statistics(tmp_pred, tmp_gt, single_label_pred=False)

                print(
                    "[Test] Epoch: [%3d][%4d/%4d] time: %.4f, \n %s" %
                    (epoch, cstep, test_steps, time.time() - step_time, utils.dict_to_string_4_print(tmp_stats))
                )
                step_time = time.time()

        prediction = np.concatenate(pred_class_list, axis=0)

        tmp_pred = np.concatenate(pred_class_list, axis=0)
        threshold = 0.5
        tmp_pred[tmp_pred > threshold] = 1
        tmp_pred[tmp_pred <= threshold] = 0

        # topk_pred = self.get_one_hot_results(np.concatenate(topk_list, axis=0))
        ground_truth = self.get_one_hot_results(np.reshape(np.array(class_list[: test_steps * config.batch_size]), newshape=(-1, 1)))

        # align = np.concatenate(align_list, axis=0)
        # kg_vector_full = np.concatenate(kg_vector_list, axis=0)

        stats = utils.get_statistics(tmp_pred, ground_truth, single_label_pred=False)

        print(
            "[Test Sum] Epoch: [%3d] time: %.4f, \n %s" %
            (epoch, time.time() - start_time, utils.dict_to_string_4_print(stats))
        )

        # return stats, prediction, ground_truth, align, kg_vector_full
        return stats, prediction, ground_truth, np.array([0]), np.array([0])

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
        unseen_test_text_seqs, unseen_test_class_list = self.get_text_of_unseen_class(test_text_seqs, test_class_list)

        # double check if text of unseen classes are removed
        for class_id in train_class_list:
            assert self.check_seen(class_id)
        # seen
        for class_id in seen_test_class_list:
            assert self.check_seen(class_id)
        # unseen
        for class_id in unseen_test_class_list:
            assert not self.check_seen(class_id)

        for epoch in range(train_epoch + 1):

            # TODO: apply constrain on training set size
            for _ in range(max(2 - 10 * global_epoch, 2)):
                print("epoch %d %d" % (global_epoch, _))
                self.__train__(
                    global_epoch,
                    train_text_seqs,
                    train_class_list,
                    max_train_steps=3000
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
                    [_ for idx, _ in enumerate(seen_test_text_seqs) if idx % 10 == 0],
                    [_ for idx, _ in enumerate(seen_test_class_list) if idx % 10 == 0],
                )

                # TODO: remove to get full test
                print("[Test] Testing unseen classes")
                state_unseen, pred_unseen, gt_unseen, align_unseen, kg_vector_unseen = self.__test__(
                    global_epoch,
                    [_ for idx, _ in enumerate(unseen_test_text_seqs) if idx % 2 == 0],
                    [_ for idx, _ in enumerate(unseen_test_class_list) if idx % 2 == 0],
                )

                np.savez(
                    self.log_save_dir + "test_%d" % global_epoch,
                    pred_seen=pred_seen,
                    pred_unseen=pred_unseen,
                    gt_seen=gt_seen,
                    gt_unseen=gt_unseen,
                    align_seen=align_seen,
                    align_unseen=align_unseen,
                    kg_vector_seen=kg_vector_seen,
                    kg_vector_unseen=kg_vector_unseen,
                    )
                # error.rejection_single_label(self.log_save_dir + "test_%d.npz" % global_epoch)
                # error.classify_single_label(self.log_save_dir + "test_%d.npz" % global_epoch)
                error.classify_single_label2(self.log_save_dir + "test_%d.npz" % global_epoch)
                class_distance_matrix = np.loadtxt(config.zhang15_dbpedia_dir + 'class_distance.txt')
                error.classify_adjust_single_label(self.log_save_dir + "test_%d.npz" % global_epoch, class_distance_matrix)

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
        unseen_test_text_seqs, unseen_test_class_list = self.get_text_of_unseen_class(test_text_seqs, test_class_list)

        # seen
        for class_id in seen_test_class_list:
            assert self.check_seen(class_id)
        # unseen
        for class_id in unseen_test_class_list:
            assert not self.check_seen(class_id)

        # TODO: remove to get full test
        print("[Test] Testing seen classes")
        state_seen, pred_seen, gt_seen, align_seen, kg_vector_seen = self.__test__(
            global_epoch,
            [_ for idx, _ in enumerate(seen_test_text_seqs) if idx % 1 == 0],
            [_ for idx, _ in enumerate(seen_test_class_list) if idx % 1 == 0],
        )

        # TODO: remove to get full test
        print("[Test] Testing unseen classes")
        state_unseen, pred_unseen, gt_unseen, align_unseen, kg_vector_unseen = self.__test__(
            global_epoch,
            [_ for idx, _ in enumerate(unseen_test_text_seqs) if idx % 1 == 0],
            [_ for idx, _ in enumerate(unseen_test_class_list) if idx % 1 == 0],
        )

        np.savez(
            self.log_save_dir + "test_%d" % global_epoch,
            pred_seen=pred_seen,
            pred_unseen=pred_unseen,
            gt_seen=gt_seen,
            gt_unseen=gt_unseen,
            align_seen=align_seen,
            align_unseen=align_unseen,
            kg_vector_seen=kg_vector_seen,
            kg_vector_unseen=kg_vector_unseen,
        )

        # error.rejection_single_label(self.log_save_dir + "test_%d.npz" % global_epoch)
        # error.classify_single_label(self.log_save_dir + "test_%d.npz" % global_epoch)
        error.classify_single_label2(self.log_save_dir + "test_%d.npz" % global_epoch)
        class_distance_matrix = np.loadtxt(config.zhang15_dbpedia_dir + 'class_distance.txt')
        error.classify_adjust_single_label(self.log_save_dir + "test_%d.npz" % global_epoch, class_distance_matrix)

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

    # check class label in vocab and glove
    for class_id in class_dict:
        class_label = class_dict[class_id]
        class_label_word_id = vocab.word_to_id(class_label)
        assert class_label_word_id != vocab.unk_id
        assert np.sum(glove_mat[class_label_word_id]) != 0

    kg_vector_dict = dataloader.load_kg_vector(
        config.zhang15_dbpedia_kg_vector_dir,
        config.zhang15_dbpedia_kg_vector_prefix,
        class_dict
    )

    print("Check NaN in csv ...")
    check_nan_train = dataloader.check_df(config.zhang15_dbpedia_train_path)
    check_nan_test = dataloader.check_df(config.zhang15_dbpedia_test_path)
    print("Train NaN %s, Test NaN %s" % (check_nan_train, check_nan_test))

    train_class_list = dataloader.load_data_class(
        filename=config.zhang15_dbpedia_train_path,
        column="class",
    )

    train_text_seqs = dataloader.load_data_from_text_given_vocab(
        config.zhang15_dbpedia_train_path, vocab, config.zhang15_dbpedia_train_processed_path,
        # column="text", force_process=check_nan_train
        # column="selected", force_process=check_nan_train
        column="selected_tfidf", force_process=check_nan_train
    )

    test_class_list = dataloader.load_data_class(
        filename=config.zhang15_dbpedia_test_path,
        column="class",
    )

    test_text_seqs = dataloader.load_data_from_text_given_vocab(
        config.zhang15_dbpedia_test_path, vocab, config.zhang15_dbpedia_test_processed_path,
        # column="text", force_process=check_nan_test
        # column="selected", force_process=check_nan_test
        column="selected_tfidf", force_process=check_nan_test
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

            mdl = model.Model_KG4Text(
                # model_name="selected_zhang15_dbpedia_kg3_random%d_unseen%.2f_max%d_cnn_negative%dincrease2_randomtext" \
                # model_name="selected_zhang15_dbpedia_kg3_random%d_unseen%.2f_max%d_cnn_negative%dincrease3_randomtext" \
                # model_name="selected_zhang15_dbpedia_kg3_cluster_3group_random%d_unseen%.2f_max%d_cnn_negative%dincrease2_randomtext" \
                # model_name="selected_zhang15_dbpedia_kg3_cluster_allgroup_random%d_unseen%.2f_max%d_cnn_negative%dincrease2_randomtext" \
                # model_name="full_zhang15_dbpedia_kg3_cluster_allgroup_only_random%d_unseen%.2f_max%d_cnn_negative%dincrease2_randomtext" \
                # model_name="full_zhang15_dbpedia_kg3_cluster_3group_only_random%d_unseen%.2f_max%d_cnn_negative%dincrease2_randomtext" \
                # model_name="selected_tfidf_zhang15_dbpedia_kg3_cluster_3group_random%d_unseen%.2f_max%d_cnn_negative%dincrease2_randomtext" \
                model_name="selected_tfidf_zhang15_dbpedia_kg3_cluster_3group_only_random%d_unseen%.2f_max%d_cnn_negative%dincrease2_randomtext" \
                               % (i + 1, unseen_percentage, max_length, config.negative_sample),
                start_learning_rate=0.0001,
                decay_rate=0.8,
                decay_steps=5e3,
                vocab_size=23000,
                max_length=max_length
            )
            ctl = Controller_KG4Text(
                model=mdl,
                vocab=vocab,
                class_dict=class_dict,
                kg_vector_dict=kg_vector_dict,
                word_embed_mat=glove_mat,
                # lemma=False,
                lemma=True,
                random_unseen=True,
                random_percentage=unseen_percentage,
                base_epoch=-1,
            )
            ctl.controller(train_text_seqs, train_class_list, test_text_seqs, test_class_list, train_epoch=5)
            # unseen_class_list = [4, 2, 11]
            # ctl.controller4test(test_text_seqs, test_class_list, unseen_class_list, base_epoch=5)
            ctl.controller4test(test_text_seqs, test_class_list, unseen_class_list=ctl.unseen_class, base_epoch=5)

            ctl.sess.close()

    # chen14 dataset
    '''
    vocab = dataloader.build_vocabulary_from_full_corpus(
        config.chen14_elec_full_data_path, config.chen14_elec_vocab_path, column="text", force_process=False,
        min_word_count=10
    )

    glove_mat = dataloader.load_glove_word_vector(
        config.word_embed_file_path, config.chen14_elec_word_embed_matrix_path, vocab, force_process=False
    )
    assert np.sum(glove_mat[vocab.start_id]) == 0
    assert np.sum(glove_mat[vocab.end_id]) == 0
    assert np.sum(glove_mat[vocab.unk_id]) == 0
    assert np.sum(glove_mat[vocab.pad_id]) == 0

    class_dict = dataloader.load_class_dict(
        class_file=config.chen14_elec_class_label_path,
        class_code_column="ClassCode",
        class_name_column="ConceptNet"
    )

    # check class label in vocab and glove
    for class_id in class_dict:
        class_label = class_dict[class_id]
        class_label_word_id = vocab.word_to_id(class_label)
        # assert class_label_word_id != vocab.unk_id
        # assert np.sum(glove_mat[class_label_word_id]) != 0
        # if class_label_word_id == vocab.unk_id:
        #     print("UNK", class_label)
        # if np.sum(glove_mat[class_label_word_id]) == 0:
        #     print("GLO", class_label)

    kg_vector_dict = dataloader.load_kg_vector(
        config.chen14_elec_kg_vector_dir,
        config.chen14_elec_kg_vector_prefix,
        class_dict
    )

    train_class_list = dataloader.load_data_class(
        filename=config.chen14_elec_train_path,
        column="class",
    )

    train_text_seqs = dataloader.load_data_from_text_given_vocab(
        config.chen14_elec_train_path, vocab, config.chen14_elec_train_processed_path,
        column="text", force_process=False
    )

    test_class_list = dataloader.load_data_class(
        filename=config.chen14_elec_test_path,
        column="class",
    )

    test_text_seqs = dataloader.load_data_from_text_given_vocab(
        config.chen14_elec_test_path, vocab, config.chen14_elec_test_processed_path,
        column="text", force_process=False
    )

    # lenlist = [len(text) for text in test_text_seqs]
    # print(np.mean(lenlist))
    # print(np.percentile(lenlist, 95))
    # print(np.percentile(lenlist, 90))
    # print(np.percentile(lenlist, 80))
    # exit()

    for i in range(10):

        unseen_percentage = 0.25
        max_length = 200

        with tf.Graph().as_default() as graph:
            tl.layers.clear_layers_name()

            mdl = model.Model_KG4Text(
                model_name="selected_chen14_elec_kg3_random%d_unseen%.2f_max%d_cnn_negative%dincrease2_randomtext" \
                           % (i + 1, unseen_percentage, max_length, config.negative_sample),
                start_learning_rate=0.001,
                decay_rate=0.6,
                decay_steps=2000,
                vocab_size=15000,
                max_length=max_length,
                hidden_dim=512,
            )
            ctl = Controller_KG4Text(
                model=mdl,
                vocab=vocab,
                class_dict=class_dict,
                kg_vector_dict=kg_vector_dict,
                word_embed_mat=glove_mat,
                # lemma=False,
                lemma=True,
                random_unseen=True,
                random_percentage=unseen_percentage,
                base_epoch=-1,
            )
            ctl.controller(train_text_seqs, train_class_list, test_text_seqs, test_class_list, train_epoch=20)
            # unseen_class_list = [35, 29, 31, 19, 15, 25, 11, 50, 33, 46, 3, 36]
            # unseen_class_list = [10, 17, 14, 11, 8]
            # ctl.controller4test(test_text_seqs, test_class_list, unseen_class_list, base_epoch=20)
            ctl.controller4test(test_text_seqs, test_class_list, unseen_class_list=ctl.unseen_class, base_epoch=20)

            ctl.sess.close()
    '''

    # 20 news
    '''
    vocab = dataloader.build_vocabulary_from_full_corpus(
        config.news20_full_data_path, config.news20_vocab_path, column="selected", force_process=False,
        min_word_count=10
    )

    glove_mat = dataloader.load_glove_word_vector(
        config.word_embed_file_path, config.news20_word_embed_matrix_path, vocab, force_process=False
    )
    assert np.sum(glove_mat[vocab.start_id]) == 0
    assert np.sum(glove_mat[vocab.end_id]) == 0
    assert np.sum(glove_mat[vocab.unk_id]) == 0
    assert np.sum(glove_mat[vocab.pad_id]) == 0

    class_dict = dataloader.load_class_dict(
        class_file=config.news20_class_label_path,
        class_code_column="ClassCode",
        class_name_column="ConceptNet"
    )

    # score_matrix = np.zeros((len(class_dict), len(class_dict)))
    # class_list = [class_dict[class_id] for class_id in class_dict]
    # glove_matrix = np.array([glove_mat[vocab.word_to_id(class_dict[class_id])] for class_id in class_dict])
    # score_matrix = sklearn.metrics.pairwise.cosine_similarity(glove_matrix, glove_matrix)

    # plt.imshow(score_matrix, cmap='hot', interpolation='nearest')
    # import seaborn as sns
    # sns.heatmap(score_matrix)
    # plt.yticks(range(len(class_list)), class_list)
    # plt.xticks(range(len(class_list)), class_list)
    # plt.yticks(rotation=0)
    # plt.xticks(rotation=90)
    # plt.tight_layout()
    # plt.savefig("correlation.png")
    # plt.clf()

    # softmax_score_matrix = np.exp(score_matrix)
    # for i in range(softmax_score_matrix.shape[0]):
    #     softmax_score_matrix[i] /= np.sum(softmax_score_matrix[i])

    # plt.imshow(softmax_score_matrix, cmap='hot', interpolation='nearest')
    # sns.heatmap(softmax_score_matrix)
    # plt.yticks(range(len(class_list)), class_list)
    # plt.xticks(range(len(class_list)), class_list)
    # plt.yticks(rotation=0)
    # plt.xticks(rotation=90)
    # plt.tight_layout()
    # plt.savefig("correlation_softmax.png")
    # plt.clf()
    # exit()


    # check class label in vocab and glove
    for class_id in class_dict:
        class_label = class_dict[class_id]
        class_label_word_id = vocab.word_to_id(class_label)
        assert class_label_word_id != vocab.unk_id
        assert np.sum(glove_mat[class_label_word_id]) != 0

    kg_vector_dict = dataloader.load_kg_vector(
        config.news20_kg_vector_dir,
        config.news20_kg_vector_prefix,
        class_dict
    )

    train_class_list = dataloader.load_data_class(
        filename=config.news20_train_path,
        column="class",
    )

    train_text_seqs = dataloader.load_data_from_text_given_vocab(
        config.news20_train_path, vocab, config.news20_train_processed_path,
        column="selected", force_process=False
    )

    test_class_list = dataloader.load_data_class(
        filename=config.news20_test_path,
        column="class",
    )

    test_text_seqs = dataloader.load_data_from_text_given_vocab(
        config.news20_test_path, vocab, config.news20_test_processed_path,
        column="selected", force_process=False
    )

    # for idx in range(1000, 1010):
    #     print(test_class_list[idx], class_dict[test_class_list[idx]])
    #     print(test_text_seqs[idx])
    #     print([vocab.id_to_word(word_id) for word_id in test_text_seqs[idx]])
    #     print([1 if np.sum(glove_mat[word_id]) else 0 for word_id in test_text_seqs[idx]])
    # lenlist = [len(text) for text in test_text_seqs]
    # print(np.mean(lenlist))
    # print(np.percentile(lenlist, 95))
    # print(np.percentile(lenlist, 90))
    # print(np.percentile(lenlist, 80))

    for i in range(10):

        unseen_percentage = 0.25
        max_length = 100

        with tf.Graph().as_default() as graph:
            tl.layers.clear_layers_name()

            mdl = model.Model_KG4Text(
                model_name="selected_news20_kg3_random%d_unseen%.2f_max%d_cnn_negative%dincrease2_randomtext" \
                           % (i + 1, unseen_percentage, max_length, config.negative_sample),
                start_learning_rate=0.0001,
                decay_rate=0.6,
                decay_steps=300,
                vocab_size=20000,
                max_length=max_length,
                hidden_dim=512,
            )
            ctl = Controller_KG4Text(
                model=mdl,
                vocab=vocab,
                class_dict=class_dict,
                kg_vector_dict=kg_vector_dict,
                word_embed_mat=glove_mat,
                # lemma=False,
                lemma=True,
                random_unseen=True,
                random_percentage=unseen_percentage,
                base_epoch=-1,
            )
            ctl.controller(train_text_seqs, train_class_list, test_text_seqs, test_class_list, train_epoch=10)
            # unseen_class_list = [35, 29, 31, 19, 15, 25, 11, 50, 33, 46, 3, 36]
            # unseen_class_list = [10, 17, 14, 11, 8]
            # ctl.controller4test(test_text_seqs, test_class_list, unseen_class_list, base_epoch=20)
            ctl.controller4test(test_text_seqs, test_class_list, unseen_class_list=ctl.unseen_class, base_epoch=10)

            ctl.sess.close()
    '''

    pass






