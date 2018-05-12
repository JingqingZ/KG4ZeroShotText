import os
import random
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorlayer as tl
from datetime import datetime, timedelta
from random import randint
import progressbar

import utils
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
            num_fold=4,
            current_fold=1,
            base_epoch=-1
    ):
        Controller.__init__(self, model=model)
        self.base_epoch = base_epoch
        self.vocab = vocab
        self.class_dict = class_dict
        self.kg_vector_dict = kg_vector_dict
        self.num_fold = num_fold
        self.current_fold = current_fold

        num_class = len(class_dict.keys())
        num_of_class_in_each_fold = list()
        remain = num_class


        self.unseen_class =[0, 0]

        for i in range(num_fold):
            this_fold = math.ceil(remain / (num_fold - i))
            remain -= this_fold
            num_of_class_in_each_fold.append(this_fold)

            if i == self.current_fold - 1:
                self.unseen_class[1] = num_class - remain
                self.unseen_class[0] = num_class - remain - this_fold + 1

    def check_seen(self, class_id):
        return class_id < self.unseen_class[0] or class_id > self.unseen_class[1]

    def get_kg_vector_given_class(self, encode_text_seqs, class_id_list):
        assert encode_text_seqs.shape[0] == class_id_list.shape[0]

        kg_vector_list = list()

        for idx, class_id in enumerate(class_id_list):

            kg_vector = np.zeros([config.max_length, config.kg_embedding_dim])

            for widx, word_id in enumerate(encode_text_seqs[idx]):
                kg_vector[widx, :] = dataloader.get_kg_vector(self.kg_vector_dict, self.class_dict[class_id], self.vocab.id_to_word(word_id))

            kg_vector_list.append(kg_vector)

        kg_vector_list = np.array(kg_vector_list)

        assert kg_vector_list.shape == (config.batch_size, config.max_length, config.kg_embedding_dim)

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

    def __train__(self, epoch, text_seqs, class_list):

        assert len(text_seqs) == len(class_list)

        train_order = list(range(len(text_seqs)))

        random.shuffle(train_order)

        train_order = train_order[: 5000 * config.batch_size]

        start_time = time.time()
        step_time = time.time()

        all_loss = np.zeros(1)

        train_steps = len(train_order) // config.batch_size

        for cstep in range(train_steps):

            text_seqs_mini = [text_seqs[idx] for idx in train_order[cstep * config.batch_size : (cstep + 1) * config.batch_size]]
            true_class_id_mini = [class_list[idx] for idx in train_order[cstep * config.batch_size : (cstep + 1) * config.batch_size]]
            encode_seqs_mini = dataloader.prepro_encode(text_seqs_mini.copy(), vocab)

            for logit in range(2):

                # category_logits = np.array([randint(0, 1) for b in range(config.batch_size)])
                category_logits = np.array([logit] * config.batch_size)

                class_id_mini = self.get_random_class(true_class_id_mini, category_logits)
                kg_vector_seqs_mini = self.get_kg_vector_given_class(encode_seqs_mini, class_id_mini)

                global_step = cstep + epoch * train_steps

                results = self.sess.run([
                    self.model.train_loss,
                    self.model.learning_rate,
                    self.model.optim
                ], feed_dict={
                    self.model.encode_seqs: encode_seqs_mini,
                    self.model.kg_vector: kg_vector_seqs_mini,
                    self.model.category_logits: np.expand_dims(category_logits, -1),
                    self.model.global_step: global_step,
                })

                all_loss += results[:1]

            if cstep % 100 == 0 and cstep > 0:
                print(
                    "[Train] Epoch: [%3d][%4d/%4d] time: %.4f, lr: %.8f, loss: %s" %
                    (epoch, cstep, train_steps, time.time() - step_time, results[-2], all_loss / (cstep + 1) / 2)
                )
                step_time = time.time()

        print(
            "[Train Sum] Epoch: [%3d] time: %.4f, lr: %.8f, loss: %s" %
            (epoch, time.time() - start_time, results[-2], all_loss / train_steps / 2)
        )

        return all_loss / train_steps

    def __test__(self, epoch, text_seqs, class_list):

        assert len(text_seqs) == len(class_list)

        start_time = time.time()
        step_time = time.time()

        test_steps = len(text_seqs) // config.batch_size

        pred_class_list = list()
        align_list = list()

        for cstep in range(test_steps):

            text_seqs_mini = text_seqs[cstep * config.batch_size : (cstep + 1) * config.batch_size]
            true_class_id_mini = class_list[cstep * config.batch_size : (cstep + 1) * config.batch_size]

            encode_seqs_mini = dataloader.prepro_encode(text_seqs_mini.copy(), vocab)

            pred_mat = np.zeros([config.batch_size, len(self.class_dict)])
            align_mat = np.zeros([config.batch_size, len(self.class_dict), config.max_length - 1])

            for class_id in self.class_dict:

                class_id_mini = np.array([class_id] * config.batch_size)
                category_logits = np.zeros([config.batch_size, 1])
                for b in range(config.batch_size):
                    category_logits[b, 0] = int(class_id == true_class_id_mini[b])


                kg_vector_seqs_mini = self.get_kg_vector_given_class(encode_seqs_mini, class_id_mini)

                test_loss, pred, align = self.sess.run([
                    self.model.test_loss,
                    self.model.test_net.outputs,
                    self.model.test_align.outputs,
                ], feed_dict={
                    self.model.encode_seqs: encode_seqs_mini,
                    self.model.kg_vector: kg_vector_seqs_mini,
                    self.model.category_logits: category_logits,
                })

                pred_mat[:, class_id - 1] = pred[:, 0]
                align_mat[:, class_id - 1, :] = align[:, :, 0]

            topk = self.get_pred_class_topk(pred_mat, k=1)
            pred_class_list.append(topk)

            align_list.append(align_mat)

            # print(pred_mat)
            # print(pred_class_list)
            # print(class_list[: (cstep + 1) * config.batch_size])
            # exit()

            if cstep % 100 == 0:
                tmp_pred = self.get_one_hot_results(np.concatenate(pred_class_list, axis=0))
                tmp_gt = self.get_one_hot_results(np.reshape(np.array(class_list[ : (cstep + 1) * config.batch_size]), newshape=(-1, 1)))
                tmp_stats = utils.get_statistics(tmp_pred, tmp_gt, single_label_pred=True)

                print(
                    "[Test] Epoch: [%3d][%4d/%4d] time: %.4f, \n %s" %
                    (epoch, cstep, test_steps, time.time() - step_time, utils.dict_to_string_4_print(tmp_stats))
                )
                step_time = time.time()

        prediction = self.get_one_hot_results(np.concatenate(pred_class_list, axis=0))
        ground_truth = self.get_one_hot_results(np.reshape(np.array(class_list[: test_steps * config.batch_size]), newshape=(-1, 1)))
        align = np.concatenate(align_list, axis=0)

        stats = utils.get_statistics(prediction, ground_truth, single_label_pred=True)

        print(
            "[Test Sum] Epoch: [%3d] time: %.4f, \n %s" %
            (epoch, time.time() - start_time, utils.dict_to_string_4_print(stats))
        )
        return stats, prediction, ground_truth, align



    def controller(self, train_text_seqs, train_class_list, test_text_seqs, test_class_list, train_epoch=config.train_epoch):

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

            self.__train__(
                global_epoch,
                train_text_seqs,
                train_class_list
            )

            print("[Test] Testing seen classes")
            state_seen, pred_seen, gt_seen, align_seen = self.__test__(
                global_epoch,
                seen_test_text_seqs,
                seen_test_class_list
            )

            print("[Test] Testing unseen classes")
            state_unseen, pred_unseen, gt_unseen, align_unseen = self.__test__(
                global_epoch,
                unseen_test_text_seqs,
                unseen_test_class_list
            )

            np.savez(
                self.log_save_dir + "test_%d" % global_epoch,
                pred_seen=pred_seen,
                pred_unseen=pred_unseen,
                gt_seen=gt_seen,
                gt_unseen=gt_unseen,
                align_seen=align_seen,
                align_unseen=align_unseen
            )

            if global_epoch > self.base_epoch and global_epoch % 1 == 0:
                self.save_model(
                    path=self.model_save_dir,
                    global_step=global_epoch
                )
                last_save_epoch = global_epoch

            global_epoch += 1

    def controller4test(self, test_text_seqs, test_class_list, base_epoch=None):

        last_save_epoch = self.base_epoch if base_epoch is None else base_epoch
        global_epoch = self.base_epoch + 1 if base_epoch is None else base_epoch + 1

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

        # TODO: uncomment this
        # print("[Test] Testing seen classes")
        # state_seen, pred_seen, gt_seen, align_seen = self.__test__(
        #     global_epoch,
        #     seen_test_text_seqs,
        #     seen_test_class_list
        # )

        print("[Test] Testing unseen classes")
        state_unseen, pred_unseen, gt_unseen, align_unseen = self.__test__(
            global_epoch,
            unseen_test_text_seqs,
            unseen_test_class_list
        )

        np.savez(
            self.log_save_dir + "test_%d" % global_epoch,
            pred_seen=pred_seen,
            pred_unseen=pred_unseen,
            gt_seen=gt_seen,
            gt_unseen=gt_unseen,
            align_seen=align_seen,
            align_unseen=align_unseen
        )


if __name__ == "__main__":

    vocab = dataloader.build_vocabulary_from_full_corpus(
        config.zhang15_dbpedia_full_data_path, config.zhang15_dbpedia_vocab_path, column="text", force_process=False
    )

    kg_vector_dict = dataloader.load_kg_vector(config.kg_vector_data_path)

    class_dict = dataloader.load_class_dict(
        class_file=config.zhang15_dbpedia_class_label_path,
        class_code_column="ClassCode",
        class_name_column="ConceptNet"
    )

    train_class_list = dataloader.load_data_class(
        filename=config.zhang15_dbpedia_train_path,
        column="class",
    )

    train_text_seqs = dataloader.load_data_from_text_given_vocab(
        config.zhang15_dbpedia_train_path, vocab, config.zhang15_dbpedia_train_processed_path,
        column="text", force_process=False
    )

    # train_kg_vector = dataloader.load_kg_vector_given_text_seqs(
    #     train_text_seqs, vocab, class_dict, kg_vector_dict,
    #     processed_file=config.zhang15_dbpedia_kg_vector_train_processed_path,
    #     force_process=False
    # )

    test_class_list = dataloader.load_data_class(
        filename=config.zhang15_dbpedia_test_path,
        column="class",
    )

    test_text_seqs = dataloader.load_data_from_text_given_vocab(
        config.zhang15_dbpedia_test_path, vocab, config.zhang15_dbpedia_test_processed_path,
        column="text", force_process=False
    )

    # test_kg_vector = dataloader.load_kg_vector_given_text_seqs(
    #     test_text_seqs, vocab, class_dict, kg_vector_dict,
    #     processed_file=config.zhang15_dbpedia_kg_vector_test_processed_path,
    #     force_process=False
    # )

    with tf.Graph().as_default() as graph:
        tl.layers.clear_layers_name()

        current_fold = 4
        num_fold = 4

        mdl = model.Model_KG4Text(
            model_name="key_zhang15_dbpedia_%dof%d" % (current_fold, num_fold),
            start_learning_rate=0.0004,
            decay_rate=0.8,
            decay_steps=5e3,
            vocab_size=15000
        )
        ctl = Controller_KG4Text(
            model=mdl,
            vocab=vocab,
            class_dict=class_dict,
            kg_vector_dict=kg_vector_dict,
            num_fold=num_fold,
            current_fold=current_fold,
            base_epoch=-1
        )
        ctl.controller(train_text_seqs, train_class_list, test_text_seqs, test_class_list, train_epoch=5)
        # ctl.controller4test(test_text_seqs, test_class_list, base_epoch=2)

        ctl.sess.close()
    pass








