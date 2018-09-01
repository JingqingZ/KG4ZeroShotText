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
import model_base
import dataloader

results_path = "../results/"

class Base_Controller():

    def __init__(self, model, gpu_config=None):
        self.model = model
        self.saver = tf.train.Saver(max_to_keep=200)
        if gpu_config is None:
            self.sess = tf.Session()
        else:
            self.sess = tf.Session(config=gpu_config)
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

if __name__ == "__main__":
    pass





