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

    def __train__(self, epoch, encode_seqs, decode_seqs, target_seqs, target_mask):

        assert len(encode_seqs) == len(decode_seqs)
        assert len(encode_seqs) == len(target_seqs)
        assert len(encode_seqs) == len(target_mask)

        train_order = list(range(len(encode_seqs)))
        random.shuffle(train_order)

        start_time = time.time()
        step_time = time.time()

        all_loss = np.zeros(1)

        train_steps = len(train_order) // config.batch_size

        for cstep in range(train_steps):

            encode_seqs_mini = [encode_seqs[idx] for idx in train_order[cstep * config.batch_size : (cstep + 1) * config.batch_size]]
            decode_seqs_mini = [decode_seqs[idx] for idx in train_order[cstep * config.batch_size : (cstep + 1) * config.batch_size]]
            target_seqs_mini = [target_seqs[idx] for idx in train_order[cstep * config.batch_size : (cstep + 1) * config.batch_size]]
            target_mask_mini = [target_mask[idx] for idx in train_order[cstep * config.batch_size : (cstep + 1) * config.batch_size]]

            global_step = cstep + epoch * train_steps

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

            all_loss += results[:-2]

            if cstep % 100 == 0 and cstep > 0:
                print(
                    "[Train] Epoch: [%3d][%4d/%4d] time: %.4f, lr: %.8f, loss: %s" %
                    (epoch, cstep, train_steps, time.time() - step_time, results[-1], all_loss / (cstep + 1))
                )
                step_time = time.time()

        print(
            "[Train Sum] Epoch: [%3d] time: %.4f, lr: %.8f, loss: %s" %
            (epoch, time.time() - start_time, results[-1], all_loss / train_steps)
        )

        return all_loss / train_steps

    def controller_train(self, train_epoch=config.train_epoch):
        encode_seqs, decode_seqs, target_seqs, target_mask = dataloader4text.load_data(dataloader4text.data_filename)

        last_save_epoch = self.base_epoch
        global_epoch = self.base_epoch + 1

        if last_save_epoch >= 0:
            self.restore_model(
                path=self.model_save_dir,
                global_step=last_save_epoch
            )

        for epoch in range(train_epoch + 1):

            self.__train__(global_epoch, encode_seqs, decode_seqs, target_seqs, target_mask)

            if global_epoch > self.base_epoch and global_epoch % 10 == 0:
                self.save_model(
                    path=self.model_save_dir,
                    global_step=global_epoch
                )
                last_save_epoch = global_epoch

            global_epoch += 1


if __name__ == "__main__":
    with tf.Graph().as_default() as graph:
        tl.layers.clear_layers_name()
        mdl = model4text.Model4Text(
            model_name="text_encoding",
            start_learning_rate=0.001,
            decay_rate=0.8,
            decay_steps=1000
        )
        ctl = Controller4Text(model=mdl, base_epoch=-1)
        ctl.controller_train(train_epoch=50)
        ctl.sess.close()
    pass








