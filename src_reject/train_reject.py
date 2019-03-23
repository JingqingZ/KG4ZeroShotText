import os
import sys
import random
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import math
import sklearn
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorlayer as tl
from datetime import datetime, timedelta
from random import randint
from collections import Counter
import progressbar
import nltk

import log
import utils
import error
import config
import model_reject
import train_base
import dataloader

# results_path = "../results/Model4Reject" + "/" + datetime.now().strftime("%Y%m%d%H%M%S")
results_path = "../results/"


class Controller4Reject(train_base.Base_Controller):

    def __init__(
            self,
            model,
            vocab,
            class_dict,
            word_embed_mat,
            base_epoch=-1,
    ):
        super(Controller4Reject, self).__init__(model)

        logging = log.Log(sys.stdout, self.log_save_dir + "log-%s" % utils.now2string())
        sys.stdout = logging

        self.base_epoch = base_epoch
        self.vocab = vocab
        self.class_dict = class_dict
        self.word_embed_mat = word_embed_mat
        self.main_class = model.main_class
        self.seen_classes = model.seen_classes
        self.unseen_classes = model.unseen_classes

    def check_seen(self, class_id):
        return not class_id in self.unseen_classes

    def get_text_of_seen_class(self, text_seqs, class_list):
        print("Getting text of seen classes")
        assert len(text_seqs) == len(class_list), "Unequal numbers of texts and classes: %d, %d" % (len(text_seqs), len(class_list))
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

    def get_binary_training_data(self, text_seqs, class_list):
        print("Creating training dataset for class %d" % (self.main_class))
        assert len(text_seqs) == len(class_list)
        seen_text_seqs = list()
        seen_class_list = list()
        negative_text_seqs = list()

        # Positive examples
        with progressbar.ProgressBar(max_value=len(text_seqs)) as bar:
            for idx, text in enumerate(text_seqs):
                if class_list[idx] == self.main_class:
                    seen_text_seqs.append(text)
                    seen_class_list.append(1.0)
                else:
                    negative_text_seqs.append(text)
                bar.update(idx)
        assert len(seen_text_seqs) == len(seen_class_list)
        
        # Negative examples
        random.shuffle(negative_text_seqs)
        # seen_text_seqs.extend(negative_text_seqs[:2*len(seen_text_seqs)])
        # seen_class_list.extend([0.0] * (2*len(seen_class_list)))

        seen_text_seqs.extend(negative_text_seqs)
        seen_class_list.extend([0.0] * len(negative_text_seqs))
        assert len(seen_text_seqs) == len(seen_class_list)

        print("Text seqs of main class + negative classes: %d" % len(seen_text_seqs))
        return seen_text_seqs, seen_class_list

    def get_binary_test_data(self, text_seqs, class_list):
        print("Creating test dataset for class %d" % (self.main_class))
        ans_class_list = list()
        with progressbar.ProgressBar(max_value=len(text_seqs)) as bar:
            for idx, text in enumerate(text_seqs):
                if class_list[idx] == self.main_class:
                    ans_class_list.append(1.0)
                else:
                    ans_class_list.append(0.0)
                bar.update(idx)
        assert all([i == 0.0 or i == 1.0 for i in ans_class_list])
        return text_seqs, ans_class_list

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

    def get_adjusted_threshold(trained_pos_logits):
        all_logits = []
        for l in trained_pos_logits:
            assert l >= 0 and l <= 1
            all_logits.append(l)
            all_logits.append(1 + (1-l))
        print('Candidate threshold =', 1 - 3*np.std(np.array(all_logits)))
        return max(0.5, 1 - 3*np.std(np.array(all_logits)))


    def __train__(self, epoch, text_seqs, class_list, max_train_steps=None):
        assert len(text_seqs) == len(class_list)

        train_order = list(range(len(text_seqs)))
        random.shuffle(train_order)

        start_time = time.time()
        step_time = time.time()

        all_loss = np.zeros(1)
        train_steps = len(train_order) // config.batch_size
        if train_steps * config.batch_size < len(text_seqs):
            train_steps += 1

        if max_train_steps is not None and max_train_steps < train_steps:
            train_steps = max_train_steps

        logits_of_positive_examples = np.array([]).reshape(-1,1)

        for cstep in range(train_steps):
            global_step = cstep + epoch * train_steps

            class_idx_mini = [class_list[idx] for idx in train_order[cstep * config.batch_size : min((cstep + 1) * config.batch_size, len(text_seqs))]]
            text_seqs_mini = [text_seqs[idx] for idx in train_order[cstep * config.batch_size : min((cstep + 1) * config.batch_size, len(text_seqs))]]

            encode_seqs_id_mini, encode_seqs_mat_mini = self.prepro_encode(text_seqs_mini)

            results = self.sess.run([
                self.model.train_loss,
                self.model.train_net.outputs,
                # self.model.train_align.outputs,
                self.model.learning_rate,
                self.model.optim
            ], feed_dict={
                self.model.encode_seqs: encode_seqs_mat_mini,
                self.model.label_logits: np.array(class_idx_mini).reshape(-1,1),
                self.model.global_step: global_step,
            })

            all_loss += results[:1]

            positive_examples_indices = tuple([idx for idx, x in enumerate(class_idx_mini) if x == 1.0])
            logits_of_positive_examples_this_step = results[1][positive_examples_indices,:]
            logits_of_positive_examples = np.concatenate((logits_of_positive_examples, logits_of_positive_examples_this_step), axis = 0)


            if cstep % config.cstep_print == 0 and cstep > 0:
                print(
                    "[Train] Epoch: [%3d][%4d/%4d] time: %.4f, lr: %.8f, loss: %s" %
                    (epoch, cstep, train_steps, time.time() - step_time, results[-2], all_loss / (cstep + 1) )
                )
                step_time = time.time()

        assert logits_of_positive_examples.shape[0] == sum(class_list), "Did not collect all logits of positive examples %d/%d" % (logits_of_positive_examples.shape[0], sum(class_list))
        logits_of_positive_examples = np.ravel(logits_of_positive_examples)
        self.model.threshold = Controller4Reject.get_adjusted_threshold(logits_of_positive_examples)

        print(
            "[Train Sum] Epoch: [%3d] time: %.4f, lr: %.8f, loss: %s, threshold: %.4f" %
            (epoch, time.time() - start_time, results[-2], all_loss / train_steps, self.model.threshold)
        )

        return all_loss / train_steps

    def __test__(self, epoch, text_seqs, class_list):

        assert len(text_seqs) == len(class_list)

        start_time = time.time()
        step_time = time.time()

        test_steps = len(text_seqs) // config.batch_size
        if test_steps * config.batch_size < len(text_seqs):
            test_steps += 1

        # topk_list = list()
        pred_all = np.array([])

        all_loss = np.zeros(1)

        for cstep in range(test_steps):

            text_seqs_mini = text_seqs[cstep * config.batch_size : min((cstep + 1) * config.batch_size, len(text_seqs))]
            class_idx_mini = class_list[cstep * config.batch_size : min((cstep + 1) * config.batch_size, len(text_seqs))]

            encode_seqs_id_mini, encode_seqs_mat_mini = self.prepro_encode(text_seqs_mini)

            # pred_mat = np.zeros([config.batch_size, len(self.class_dict)])

            test_loss, out  = self.sess.run([
                self.model.test_loss,
                self.model.test_net.outputs,
            ], feed_dict={
                self.model.encode_seqs: encode_seqs_mat_mini,
                self.model.label_logits: np.array(class_idx_mini).reshape(-1,1)
            })

            all_loss[0] += test_loss

            pred = np.array([1.0 if x[0] >= self.model.threshold else 0.0 for x in out])


            # pred = np.array([_ / np.sum(_) for _ in np.exp(out)])

            # for i in range(len(self.seen_class)):
            #     pred_mat[:, self.full_class_map2index[self.seen_class[i]]] = pred[:, i]

            # topk = self.get_pred_class_topk(pred_mat, k=1)
            # topk_list.append(topk)
            pred_all = np.concatenate((pred_all, pred), axis = 0)

            if cstep % config.cstep_print == 0 and cstep > 0:
                # tmp_topk = np.concatenate(topk_list, axis=0)
                # tmp_topk = self.get_one_hot_results(np.array(tmp_topk[: (cstep + 1) * config.batch_size]))
                # tmp_gt = self.get_one_hot_results(np.reshape(np.array(class_list[ : (cstep + 1) * config.batch_size]), newshape=(-1, 1)))
                # tmp_stats = utils.get_statistics(tmp_topk, tmp_gt, single_label_pred=True)
                tmp_stats = utils.get_precision_recall_f1(pred_all, np.array(class_list[:len(pred_all)]), with_confusion_matrix = True)

                print(
                    "[Test] Epoch: [%3d][%4d/%4d] time: %.4f, loss: %s, threshold: %.4f \n %s" %
                    (epoch, cstep, test_steps, time.time() - step_time, all_loss / (cstep + 1), self.model.threshold, utils.dict_to_string_4_print(tmp_stats))
                )
                step_time = time.time()


        # prediction_topk = np.concatenate(topk_list, axis=0)
        # prediction_topk = self.get_one_hot_results(np.array(prediction_topk[: test_steps * config.batch_size]))
        # ground_truth = self.get_one_hot_results(np.reshape(np.array(class_list[: test_steps * config.batch_size]), newshape=(-1, 1)))

        stats = utils.get_precision_recall_f1(pred_all, np.array(class_list), with_confusion_matrix = True)

        print(
            "[Test Sum] Epoch: [%3d] time: %.4f, loss: %s, threshold: %.4f \n %s" %
            (epoch, time.time() - start_time, all_loss / test_steps, self.model.threshold , utils.dict_to_string_4_print(stats))
        )

        return stats, pred_all, class_list

    def controller(self, train_text_seqs, train_class_list, test_text_seqs, test_class_list, train_epoch=config.train_epoch, save_test_per_epoch=1):

        last_save_epoch = self.base_epoch
        global_epoch = self.base_epoch + 1

        if last_save_epoch >= 0:
            self.restore_model(
                path=self.model_save_dir,
                global_step=last_save_epoch
            )


        seen_train_text_seqs, seen_train_class_list = self.get_text_of_seen_class(train_text_seqs, train_class_list)
        binary_train_text_seqs, binary_train_class_list = self.get_binary_training_data(seen_train_text_seqs, seen_train_class_list)
        binary_test_text_seqs, binary_test_class_list = self.get_binary_test_data(test_text_seqs, test_class_list)
        print(len(binary_train_text_seqs), len(binary_train_class_list), len(binary_test_text_seqs), len(binary_test_class_list))
        print(Counter(seen_train_class_list), Counter(binary_train_class_list), Counter(binary_test_class_list))

        for epoch in range(train_epoch + 1):
            print("[Train] Training class", self.main_class, "Epoch", epoch)
            self.__train__(
                global_epoch,
                binary_train_text_seqs,
                binary_train_class_list,
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
                print("[Test] Testing class", self.main_class)
                stat, pred, gt = self.__test__(
                    global_epoch,
                    binary_test_text_seqs, 
                    binary_test_class_list
                )

                stat['threshold'] = self.model.threshold

                np.savez(
                    self.log_save_dir + "test_%d" % global_epoch,
                    seen_classes=self.seen_classes,
                    unseen_classes=self.unseen_classes,
                    stat=stat,
                    pred=pred,
                    gt=gt,
                )
                # error.rejection_single_label(self.log_save_dir + "test_%d.npz" % global_epoch)
                # error.classify_single_label(self.log_save_dir + "test_%d.npz" % global_epoch)
            global_epoch += 1


    def controller4test(self, test_text_seqs, test_class_list, base_epoch=None):
        last_save_epoch = self.base_epoch if base_epoch is None else base_epoch
        global_epoch = self.base_epoch if base_epoch is None else base_epoch

        if last_save_epoch >= 0:
            self.restore_model(
                path=self.model_save_dir,
                global_step=last_save_epoch
            )

        binary_test_text_seqs, binary_test_class_list = self.get_binary_test_data(test_text_seqs, test_class_list)
        print(Counter(binary_test_class_list))

        # TODO: remove to get full test
        print("[Final Test] Testing class", self.main_class)
        stat, pred, gt = self.__test__(
            global_epoch,
            binary_test_text_seqs, 
            binary_test_class_list
        )

        stat['threshold'] = self.model.threshold

        # accepted_index = [idx for idx, p in enumerate(pred) if p == 1.0]
        texts_accepted_from_class = Counter([c for idx, c in enumerate(test_class_list) if pred[idx] == 1.0])
        stat['texts_accepted_from_class'] = texts_accepted_from_class
        print(stat['texts_accepted_from_class'])

        np.savez(
            self.log_save_dir + "test_%d" % global_epoch,
            seen_classes=self.seen_classes,
            unseen_classes=self.unseen_classes,
            stat=stat,
            pred=pred,
            gt=gt,
        )

        return stat, pred, gt


def get_seen_unseen(class_names, seen_ratio):
    num_seen = seen_ratio * len(class_names)
    if num_seen - int(num_seen) >= 0.5:
        num_seen = int(num_seen) + 1
    else:
        num_seen = int(num_seen)
        
    all_classes = list(class_names)
    random.shuffle(all_classes)
    seen_classes = all_classes[:num_seen]
    unseen_classes = all_classes[num_seen:]
    seen_classes.sort()
    unseen_classes.sort()
    return seen_classes, unseen_classes

def read_random_set(filename):
    ans = []
    with open(filename, "r") as f:
        for line in f:
            seen = line[:line.index("|")].split(',')
            unseen = line[line.index("|")+1:].split(',')
            ans.append({
                'seen': [int(a) for a in seen],
                'unseen': [int(a) for a in unseen]
                })
            ans[-1]['seen'].sort()
            ans[-1]['unseen'].sort()
        print("Finish load seen - unseen classes %d sets" % (len(ans)))
    return ans

def print_summary_of_one_iteration(summary):
    print('----------------------------------------------------')
    print('Iteration:', summary['iteration'])
    print('Seen classes:', summary['seen_classes'])
    print('Unseen classes:', summary['unseen_classes'])
    print('Classifier statistics:')
    for idx, stat in enumerate(summary['stat_list']):
        print('- Class', summary['seen_classes'][idx], utils.dict_to_string_4_print(stat))
    print('Average classifier statistics:')
    print(utils.dict_to_string_4_print(summary['avg_classifier_stat']))
    print('Rejection power (TP = correctly accept):')
    print(utils.dict_to_string_4_print(summary['accepted_stats']))
    cal_result(utils.dict_to_string_4_print(summary['accepted_stats']))

def cal_result(s):
    s = s.strip().split(', ')
    stat = dict()
    for x in s:
        m, v = x.split(': ')
        stat[m] = float(v)
    print('Positive acc:', stat['TP'] / (stat['TP'] + stat['FN']))
    print('Negative acc:', stat['TN'] / (stat['TN'] + stat['FP']))
    print('Overall  acc:', (stat['TP'] + stat['TN'])/ (stat['TP'] + stat['FN'] + stat['TN'] + stat['FP']))
    print('%.3f/%.3f/%.3f' % (stat['TP'] / (stat['TP'] + stat['FN']), stat['TN'] / (stat['TN'] + stat['FP']), (stat['TP'] + stat['TN'])/ (stat['TP'] + stat['FN'] + stat['TN'] + stat['FP'])))


def print_summary_of_all_iterations(iteration_statistics):
    avg_rejection_performance = dict()
    for key in iteration_statistics[0]['accepted_stats']:
        avg_rejection_performance[key] = sum([summary['accepted_stats'][key] for summary in iteration_statistics]) / len(iteration_statistics)
    print('====================================================')
    print('The results of all iterations')
    for summary in iteration_statistics:
        print_summary_of_one_iteration(summary)
    print('====================================================')
    print('Rejection power: averaging over', len(iteration_statistics),'iterations (TP = correctly accept)')
    print(utils.dict_to_string_4_print(avg_rejection_performance))



if __name__ == "__main__":

    if config.dataset == "dbpedia":
        # DBpedia
        vocab = dataloader.build_vocabulary_from_full_corpus(
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
        )

        test_class_list = dataloader.load_data_class(
            filename=config.zhang15_dbpedia_test_path,
            column="class",
        )

        test_text_seqs = dataloader.load_data_from_text_given_vocab(
            config.zhang15_dbpedia_test_path, vocab, config.zhang15_dbpedia_test_processed_path,
            column="text", force_process=False
        )


        random_set = read_random_set(config.zhang15_dbpedia_random_set_path)
        max_length = 50
    
    elif config.dataset == "20news":
        # 20 news 
        vocab = dataloader.build_vocabulary_from_full_corpus(
            config.news20_full_data_path, config.news20_vocab_path, column="text", force_process=False,
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

        train_class_list = dataloader.load_data_class(
            filename=config.news20_train_path,
            column="class",
        )

        train_text_seqs = dataloader.load_data_from_text_given_vocab(
            config.news20_train_path, vocab, config.news20_train_processed_path,
            column="text", force_process=False
        )

        test_class_list = dataloader.load_data_class(
            filename=config.news20_test_path,
            column="class",
        )

        test_text_seqs = dataloader.load_data_from_text_given_vocab(
            config.news20_test_path, vocab, config.news20_test_processed_path,
            column="text", force_process=False
        )

        random_set = read_random_set(config.news20_random_set_path)
        max_length = 100
        # --------------------------------------------------------------------------------------------------------------
    else:
        raise Exception("config.dataset %s not found" % config.dataset)

    print(len(train_class_list), len(train_text_seqs), len(test_class_list), len(test_text_seqs))
    iteration_statistics = []
    num_epoch = config.args.nepoch
    unseen_percentage = config.unseen_rate
    pass_to_phase2 = []

    for i in range(config.args.rgidx-1, 10):
        seen_classes, unseen_classes = random_set[i]['seen'], random_set[i]['unseen']
        print(seen_classes, unseen_classes)
        stat_list = []
        pred_list = []

        with tf.Graph().as_default() as graph:
            # tl.layers.clear_layers_name()
            for a_class in seen_classes:           
                mdl = model_reject.Model4Reject(
                    model_name="reject_full_" + config.dataset + "_random%d_class%d_unseen%.2f_negativeAll_max%d_cnn" \
                               % (i + 1, a_class, unseen_percentage, max_length),
                    start_learning_rate=0.001,
                    decay_rate=0.5,
                    decay_steps=20e3,
                    main_class=a_class,
                    seen_classes=seen_classes,
                    unseen_classes=unseen_classes,
                    max_length=max_length,
                )
                
                ctl = Controller4Reject(
                    model=mdl,
                    vocab=vocab,
                    class_dict=class_dict,
                    word_embed_mat=glove_mat,
                    base_epoch=-1,
                )

                ctl.controller(train_text_seqs, train_class_list, test_text_seqs, test_class_list, train_epoch=num_epoch)
                stat, pred, _ = ctl.controller4test(test_text_seqs, test_class_list, base_epoch=num_epoch)
                stat_list.append(stat)
                pred_list.append(pred)
                ctl.sess.close()

        prediction = []
        accepted = []
        gt_accepted = []
        for idx, gt in enumerate(test_class_list):
            accepted_classes = [seen_classes[cid] for cid, x in enumerate(pred_list) if x[idx] == 1.0]
            prediction.append(accepted_classes)
            if len(accepted_classes) > 0:
                accepted.append(1)
            else:
                accepted.append(0)
            if gt in seen_classes:
                gt_accepted.append(1)
            else:
                gt_accepted.append(0)
        accepted_stats = utils.get_precision_recall_f1(np.array(accepted), np.array(gt_accepted), with_confusion_matrix = True)
        pass_to_phase2.append(accepted)

        avg_classifier_stat = dict()
        for key in stat_list[0]:
            if key != 'texts_accepted_from_class':
                avg_classifier_stat[key] = sum([stat[key] for stat in stat_list]) / len(stat_list)

        summary_dict = {'iteration':i,
                        'seen_classes':seen_classes,
                        'unseen_classes':unseen_classes,
                        'test_class_list':test_class_list,
                        'stat_list':stat_list,
                        'avg_classifier_stat':avg_classifier_stat,
                        'pred_list':pred_list,
                        'prediction':prediction,
                        'accepted_stats':accepted_stats}
        print_summary_of_one_iteration(summary_dict)
        iteration_statistics.append(summary_dict)
    print_summary_of_all_iterations(iteration_statistics)
    
    np.savez(
        results_path + "reject_full_" + config.dataset + "_unseen%.2f_negativeAll_max%d_cnn_iteration_statistics" % (unseen_percentage, max_length),
        iteration_statistics=iteration_statistics
    )

    pickle.dump(pass_to_phase2, results_path + "%s_unseen%.2f_augmented0.pickle" % (dataset_name, unseen_percentage))

    pass



