import pickle, json, requests, csv, copy, os, re, sys, math, random, copy
import numpy as np
import pprint as pp
import urllib.request, urllib.parse
from sklearn.metrics.pairwise import cosine_similarity
from json import JSONDecodeError
from text_to_uri import *

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import time
import tensorflow as tf
import tensorlayer as tl

def regression_tf_idf():
    # -------------------------------------- prepare data -------------------------------------------
    def prepare_data():
        class_info, class_id_to_word, class_id_to_name = get_class_info("../data/dbpedia/classLabelsDBpedia.csv")
        count_of_class = pickle.load(open("../wordEmbeddings/dbpedia_count_of_class.pickle", "rb"))
        count_word_of_class = pickle.load(open("../wordEmbeddings/dbpedia_count_word_of_class.pickle", "rb"))
        tf_idf_of_class = pickle.load(open("../wordEmbeddings/dbpedia_tf_idf_of_class.pickle", "rb"))
        tf_of_class = pickle.load(open("../wordEmbeddings/dbpedia_tf_of_class.pickle", "rb"))
        idf_of_words = pickle.load(open("../wordEmbeddings/dbpedia_idf_of_words.pickle", "rb"))
        seen_classes, unseen_classes = get_seen_unseen(list(count_of_class.keys()), seen_ratio = 0.75)
        print(seen_classes, unseen_classes)
        train_tf_idf, train_tf, train_idf = create_training_data(count_of_class, count_word_of_class, unseen_classes)
        # glove_vec = loadGloveModel("../data/glove/glove.6B.200d.txt")
        # pickle.dump(glove_vec, open("../data/glove/glove.6B.200d.pickle", "wb"))
        glove_vec = pickle.load(open("../data/glove/glove.6B.200d.pickle", "rb"))
        # print(list(glove_vec.keys())[:10])
        # print(glove_vec['officeholder'])
        X_train = []
        y_train = []
        for a_class in train_tf_idf:
            class_word = class_id_to_word[a_class]
            assert class_word in glove_vec, "Class word %s is not in Glove" % (class_word)
            for word in train_tf_idf[a_class]:
                if word in glove_vec:
                    X_train.append(np.concatenate((glove_vec[class_word], glove_vec[word]), axis = 0))
                    y_train.append(train_tf_idf[a_class][word])
        X_train = np.array(X_train)
        y_train = np.array(y_train).reshape(-1, 1)
        assert X_train.shape[0] == y_train.shape[0]
        print("Finish creating X_train")

        all_index = list(range(X_train.shape[0]))
        random.shuffle(all_index)
        validate_index = all_index[:int(X_train.shape[0]*0.1)]
        train_index = all_index[int(X_train.shape[0]*0.1):]
        # validate_index = random.sample(list(range(X_train.shape[0])), int(X_train.shape[0]*0.1))
        # train_index = [i for i in range(X_train.shape[0]) if i not in validate_index]
        validate_index = tuple(validate_index)
        train_index = tuple(train_index)
        print("Finish setting train and validate index")
        X_train_a = X_train[train_index, :]
        y_train_a = y_train[train_index, :]
        X_val = X_train[validate_index, :]
        y_val = y_train[validate_index, :]
        print("Finish creating train and val dataset")
        # print(X_train.shape, y_train.shape)
        # print(X_train[0], y_train[0])
        # print(X_train[1], y_train[1])

        X_test = []
        y_test = []
        for a_class in tf_idf_of_class:
            if a_class not in unseen_classes:
                continue
            class_word = class_id_to_word[a_class]
            assert class_word in glove_vec, "Class word %s is not in Glove" % (class_word)
            for word in tf_idf_of_class[a_class]:
                if word in glove_vec:
                    X_test.append(np.concatenate((glove_vec[class_word], glove_vec[word]), axis = 0))
                    y_test.append(tf_idf_of_class[a_class][word])
        X_test = np.array(X_test)
        y_test = np.array(y_test).reshape(-1, 1)
        assert X_test.shape[0] == y_test.shape[0]
        print("Finish creating test dataset")

        X_top_test = []
        y_top_test = []
        word_top_test = []
        topk = 100
        for a_class in tf_idf_of_class:
            if a_class not in unseen_classes:
                continue
            class_word = class_id_to_word[a_class]
            assert class_word in glove_vec, "Class word %s is not in Glove" % (class_word)
            ranking = tf_idf_of_class[a_class].items()
            ranking = [x for x in ranking if x[0] in glove_vec]
            # ranking = [(word, tf_idf_of_class[a_class][word]) for word in tf_idf_of_class[a_class] if word in glove_vec]
            ranking.sort(key=lambda x: x[1], reverse=True)
            ranking = ranking[:topk]
            for r in ranking:
                X_top_test.append(np.concatenate((glove_vec[class_word], glove_vec[r[0]]), axis = 0))
                y_top_test.append(r[1])
                word_top_test.append((class_word, r[0]))
        X_top_test = np.array(X_top_test)
        y_top_test = np.array(y_top_test).reshape(-1, 1)
        print("Finish creating top test dataset")

        print("Shape", X_train_a.shape, y_train_a.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape, X_top_test.shape, y_top_test)
        return X_train_a, y_train_a, X_val, y_val, X_test, y_test, X_top_test, y_top_test, word_top_test, class_info, class_id_to_word, class_id_to_name

    def loadGloveModel(gloveFile):
        print("Loading Glove Model")
        f = open(gloveFile,'rb')
        model = {}
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word.decode('utf-8')] = embedding
        print("Done.",len(model)," words loaded!")
        return model

    def get_class_info(filename):
        class_id_to_word = dict()
        class_id_to_name = dict()
        with open(filename, encoding = 'utf-8') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            ans = [row for row in reader]
            print("No. of classes =", len(ans))
            print("Header =", ans[0].keys())
            for row in ans:
                class_id_to_word[row['ClassCode']] = row['ClassWord']
                class_id_to_name[row['ClassCode']] = row['ClassLabel']
            return ans, class_id_to_word, class_id_to_name

    def entropy(a):
        return np.sum([-x*math.log2(x) for x in a if x!=0])

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
        return seen_classes, unseen_classes

    def create_training_data(count_of_class, count_word_of_class, unseen_classes):
        new_count_of_class = copy.deepcopy(count_of_class)
        new_count_word_of_class = copy.deepcopy(count_word_of_class)
        for key in unseen_classes:
            new_count_of_class.pop(key, None)
            new_count_word_of_class.pop(key, None)
        return calculate_tfidf(new_count_of_class, new_count_word_of_class)
    
    def calculate_tfidf(count_of_class, count_word_of_class):
        tf_of_class = dict()
        all_words = set()
        idf_of_words = dict()
        for key in count_word_of_class:
            tf_of_class[key] = copy.deepcopy(count_word_of_class[key])
            for k in tf_of_class[key]:
                tf_of_class[key][k] = tf_of_class[key][k] / count_of_class[key]
            all_words = all_words.union(set(tf_of_class[key].keys()))
            print("Finish calculating tf of class", key)
        for w in all_words:
            profile = []
            for key in tf_of_class:
                if w in tf_of_class[key]:
                    profile.append(tf_of_class[key][w])
                else:
                    profile.append(0)
            profile = np.array(profile)/np.sum(np.array(profile))
            idf_of_words[w] = 1 - (entropy(profile) / math.log2(len(count_of_class)))
        print("Finish calculating idf")
        tf_idf_of_class = dict()
        for key in tf_of_class:
            tf_idf_of_class[key] = copy.deepcopy(tf_of_class[key])
            for k in tf_idf_of_class[key]:
                tf_idf_of_class[key][k] = tf_idf_of_class[key][k] * idf_of_words[k] 
            print("Finish calculating tf-idf of class", key)
        return tf_idf_of_class, tf_of_class, idf_of_words

    # -------------------------------------- define the network -----------------------------------------------
    def mlp(x, is_train=True, reuse=False):
        with tf.variable_scope("MLP", reuse=reuse):
            network = tl.layers.InputLayer(x, name='input')
            network = tl.layers.DropoutLayer(network, keep=0.8, is_fix=True, is_train=is_train, name='drop1')
            network = tl.layers.DenseLayer(network, n_units=600, act=tf.nn.relu, name='relu1')
            network = tl.layers.DropoutLayer(network, keep=0.8, is_fix=True, is_train=is_train, name='drop2')
            network = tl.layers.DenseLayer(network, n_units=600, act=tf.nn.relu, name='relu2')
            network = tl.layers.DropoutLayer(network, keep=0.8, is_fix=True, is_train=is_train, name='drop3')
            network = tl.layers.DenseLayer(network, n_units=600, act=tf.nn.relu, name='relu3')
            network = tl.layers.DropoutLayer(network, keep=0.8, is_fix=True, is_train=is_train, name='drop4')
            network = tl.layers.DenseLayer(network, n_units=1, act=None, name='output')
        return network

    X_train, y_train, X_val, y_val, X_test, y_test, X_top_test, y_top_test, word_top_test, class_info, class_id_to_word, class_id_to_name = prepare_data()
    # sys.exit()

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    sess = tf.InteractiveSession()

    
    
    # X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 784))
    # define placeholder
    x = tf.placeholder(tf.float32, shape=[None, 400], name='x')
    y_ = tf.placeholder(tf.float32, shape=[None, 1], name='y_')


    


    # define inferences
    net_train = mlp(x, is_train=True, reuse=False)
    net_test = mlp(x, is_train=False, reuse=True)

    # cost for training
    y = net_train.outputs
    # cost = tl.cost.cross_entropy(y, y_, name='xentropy')
    cost = tl.cost.mean_squared_error(y, y_, name='mse_train')

    # cost and accuracy for evalution
    y2 = net_test.outputs
    # cost_test = tl.cost.cross_entropy(y2, y_, name='xentropy2')
    cost_test = tl.cost.mean_squared_error(y2, y_, name='mse_test')
    # correct_prediction = tf.equal(tf.argmax(y2, 1), y_)
    # acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # define the optimizer
    train_params = tl.layers.get_variables_with_name('MLP', train_only=True, printable=False)
    train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost, var_list=train_params)

    # initialize all variables in the session
    tl.layers.initialize_global_variables(sess)

    n_epoch = 25
    batch_size = 500
    print_freq = 5

    for epoch in range(n_epoch):
        start_time = time.time()
        for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
            sess.run(train_op, feed_dict={x: X_train_a, y_: y_train_a})

        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
            train_loss, train_acc, n_batch = 0, 0, 0
            for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
                # err, ac = sess.run([cost_test, acc], feed_dict={x: X_train_a, y_: y_train_a})
                err = sess.run(cost_test, feed_dict={x: X_train_a, y_: y_train_a})
                train_loss += err
                # train_acc += ac
                n_batch += 1
            print("   train loss: %f" % (train_loss / n_batch))
            # print("   train acc: %f" % (train_acc / n_batch))
            val_loss, val_acc, n_batch = 0, 0, 0
            for X_val_a, y_val_a in tl.iterate.minibatches(X_val, y_val, batch_size, shuffle=True):
                # err, ac = sess.run([cost_test, acc], feed_dict={x: X_val_a, y_: y_val_a})
                err = sess.run(cost_test, feed_dict={x: X_val_a, y_: y_val_a})
                val_loss += err
                # val_acc += ac
                n_batch += 1
            print("   val loss: %f" % (val_loss / n_batch))
            # print("   val acc: %f" % (val_acc / n_batch))

    print('Evaluation')
    test_loss, test_acc, n_batch = 0, 0, 0
    for X_test_a, y_test_a in tl.iterate.minibatches(X_test, y_test, batch_size, shuffle=True):
        # err, ac = sess.run([cost_test, acc], feed_dict={x: X_test_a, y_: y_test_a})
        err = sess.run(cost_test, feed_dict={x: X_test_a, y_: y_test_a})
        test_loss += err
        # test_acc += ac
        n_batch += 1
    print("   test loss: %f" % (test_loss / n_batch))
    # print("   test acc: %f" % (test_acc / n_batch))

    predicted_top_test, err = sess.run([y2, cost_test], feed_dict={x: X_top_test, y_: y_top_test})
    for i in range(len(word_top_test)):
        print(word_top_test[i], y_top_test[i][0], predicted_top_test[i][0])
    print("   top test loss: %f" % (err))

    argsort = np.argsort(predicted_top_test, axis=None)[::-1]
    for i in argsort:
        print(word_top_test[i])
regression_tf_idf()