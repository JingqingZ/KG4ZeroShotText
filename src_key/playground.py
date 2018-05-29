import os
import pickle
import pandas as pd
import numpy as np
import random

import config
import utils
import dataloader
import nltk


def doing_sth_on_chen14():

    # get class_dict
    # collect data from each class folder and randomly split to train.csv and test.csv
    # build a complete dictionary

    class_uri = [
        '/c/en/app',  #
        '/c/en/appliance',  #
        '/c/en/art',  #
        '/c/en/automotive',
        '/c/en/baby',
        '/c/en/bag',
        '/c/en/beauty',
        '/c/en/bike',
        '/c/en/book',  #
        '/c/en/cable',
        '/c/en/care',
        '/c/en/clothing',
        '/c/en/conditioner',
        '/c/en/diaper',
        '/c/en/dining',
        '/c/en/dumbbell',
        '/c/en/flashlight',
        '/c/en/food',
        '/c/en/glove',  #
        '/c/en/golf',
        '/c/en/home',  #
        '/c/en/supplies',  #
        '/c/en/jewelry',
        '/c/en/kindle',  #
        '/c/en/kitchen',
        '/c/en/knife',
        '/c/en/luggage',
        '/c/en/magazine',  #
        '/c/en/mat',
        '/c/en/mattress',
        '/c/en/movies',
        '/c/en/music',
        '/c/en/instrument',  #
        '/c/en/office',  #
        '/c/en/garden',  #
        '/c/en/pet',  #
        '/c/en/pillow',
        '/c/en/sandal',
        '/c/en/scooter',
        '/c/en/shoes',
        '/c/en/software',
        '/c/en/sports',
        '/c/en/table',  #
        '/c/en/tent',
        '/c/en/tire',
        '/c/en/toy',  #
        '/c/en/video_game',  #
        '/c/en/vitamin',  #
        '/c/en/clock',  #
        '/c/en/filter'
    ]  #

    home_dir = "../data/chen14/Non-Electronics/"
    class_dir_list = os.listdir(home_dir)
    class_dir_list = sorted(class_dir_list)

    def create_class_dict():
        for idx, class_dir in enumerate(class_dir_list):
            current_dir = "%s%s/" % (home_dir, class_dir)
            conceptnet_name = ""
            for concept_class in class_uri:
                concept_class_label = concept_class.split("/")[-1]
                if "_" in concept_class_label:
                    concept_class_label = concept_class_label.split("_")[-1]
                if concept_class_label in class_dir.lower():
                    conceptnet_name = concept_class
            # TODO: tricky
            if conceptnet_name == "":
                conceptnet_name = "/c/en/supplies"

            with open(current_dir + "%s.docs" % class_dir) as f:
                for i, l in enumerate(f):
                    pass
                num_line = i + 1
                try:
                    assert class_uri.index(conceptnet_name) == idx
                    print("%d,%s,%s,%d" % (class_uri.index(conceptnet_name) + 1, class_dir, conceptnet_name.split("/")[-1], num_line))
                except:
                    print(class_dir)

    # create_class_dict()

    def collect_full_and_split():
        class_dict_concept = dataloader.load_class_dict(
            class_file=config.chen14_class_label_path,
            class_code_column="ConceptNet",
            class_name_column="ClassCode",
        )

        class_dict_label = dataloader.load_class_dict(
            class_file=config.chen14_class_label_path,
            class_code_column="ClassLabel",
            class_name_column="ConceptNet",
        )

        class_dict_count = dataloader.load_class_dict(
            class_file=config.chen14_class_label_path,
            class_code_column="ClassLabel",
            class_name_column="Count",
        )

        train_df = pd.DataFrame(columns=["class", "text"])
        test_df = pd.DataFrame(columns=["class", "text"])
        df = pd.DataFrame(columns=["class", "text"])
        index_df = 0
        for idx, class_dir in enumerate(class_dir_list):
            filename = "%s%s/%s" % (home_dir, class_dir, class_dir)
            print(idx, filename)
            vocab_dict = dict()
            with open(filename + ".vocab", 'r') as f:
                for line in f:
                    word_id = line.split(":")[0]
                    word = line.split(":")[1].replace("\n", "")
                    assert word_id not in vocab_dict
                    vocab_dict[word_id] = word
            order = list(range(class_dict_count[class_dir]))
            random.shuffle(order)
            test_order = order[:int(class_dict_count[class_dir] * 0.3)]
            with open(filename + ".docs", 'r') as f:
                for lidx, line in enumerate(f):
                    content = line.replace("\n", "")
                    content = content.split(" ")
                    content = ' '.join([vocab_dict[c] for c in content])

                    if lidx in test_order:
                        test_df.loc[index_df] = [class_dict_concept[class_dict_label[class_dir]], content]
                    else:
                        train_df.loc[index_df] = [class_dict_concept[class_dict_label[class_dir]], content]
                    df.loc[index_df] = [class_dict_concept[class_dict_label[class_dir]], content]
                    index_df += 1
            print(test_df)
            print(train_df)
            print(df)
            exit()
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        df = df.reset_index(drop=True)
        train_df.to_csv("../data/chen14/clean/train.csv")
        test_df.to_csv("../data/chen14/clean/test.csv")
        df.to_csv("../data/chen14/clean/full.csv")

    collect_full_and_split()

    def collect_full_vocab():
        for idx, class_dir in enumerate(class_dir_list):
            filename = "%s%s/%s" % (home_dir, class_dir, class_dir)
            print(idx, filename)
            vocab_dict = dict()
            with open(filename + ".vocab", 'r') as f:
                for line in f:
                    word_id = line.split(":")[0]
                    word = line.split(":")[1].replace("\n", "")
                    assert word not in vocab_dict
                    vocab_dict[word] = word
        pass

    # collect_full_vocab()

def doing_sth_on_20_news():

    # get class_dict
    # collect data from each class folder and randomly split to train.csv and test.csv
    # build a complete dictionary

    class_uri = [
        '/c/en/atheism',
        '/c/en/graphics',
        '/c/en/operating',
        '/c/en/ibm',
        '/c/en/mac',
        '/c/en/windows',
        '/c/en/sale',
        '/c/en/auto',
        '/c/en/motorcycle',
        '/c/en/baseball',
        '/c/en/hockey',
        '/c/en/crypt',
        '/c/en/electronics',
        '/c/en/medical',
        '/c/en/space',
        '/c/en/christian',
        '/c/en/gun',
        '/c/en/middle-east',
        '/c/en/politics',
        '/c/en/religion'
    ]  #

    class_dir_list = [
        'alt.atheism',
        'comp.graphics',
        'comp.os.ms-windows.misc',
        'comp.sys.ibm.pc.hardware',
        'comp.sys.mac.hardware',
        'comp.windows.x',
        'misc.forsale',
        'rec.autos',
        'rec.motorcycles',
        'rec.sport.baseball',
        'rec.sport.hockey',
        'sci.crypt',
        'sci.electronics',
        'sci.med',
        'sci.space',
        'soc.religion.christian',
        'talk.politics.guns',
        'talk.politics.mideast',
        'talk.politics.misc',
        'talk.religion.misc'
    ]

    home_dir = "../data/20-newsgroups/processed/"

    # for i in range(20):
    #     dir = home_dir + class_dir_list[i]
    #     numfile = len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])
    #     print("%d,%s,%s,%d" % (i + 1, class_dir_list[i], class_uri[i], numfile))
    # exit()

    def collect_full_and_split():

        train_df = pd.DataFrame(columns=["class", "text"])
        test_df = pd.DataFrame(columns=["class", "text"])
        df = pd.DataFrame(columns=["class", "text"])
        index_df = 0
        for idx, class_dir in enumerate(class_dir_list):
            dirname = "%s%s/" % (home_dir, class_dir)
            print(idx + 1, dirname)

            docfiles = [name for name in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, name))]

            number_of_files = len(docfiles)
            order = list(range(number_of_files))
            random.shuffle(order)
            test_order = order[:int(number_of_files * 0.3)]

            for didx, doc in enumerate(docfiles):
                with open(dirname + doc, 'r') as f:
                    for lidx, line in enumerate(f):
                        content = line.replace("\n", "")
                        content = content.split(" ")
                        content = ' '.join([c.lower() for c in content])

                        if didx in test_order:
                            test_df.loc[index_df] = [idx + 1, content]
                        else:
                            train_df.loc[index_df] = [idx + 1, content]
                        df.loc[index_df] = [idx + 1, content]
                        index_df += 1

        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        df = df.reset_index(drop=True)
        train_df.to_csv("../data/20-newsgroups/clean/train.csv")
        test_df.to_csv("../data/20-newsgroups/clean/test.csv")
        df.to_csv("../data/20-newsgroups/clean/full.csv")

    collect_full_and_split()

def get_a_and_n(text):
    text_with_tag = nltk.pos_tag(text.split())  # a list of words à a list of words with part of speech
    selected = [text_tag[0] for text_tag in text_with_tag \
                   if text_tag[1] in config.pos_dict and \
                   (config.pos_dict[text_tag[1]] == 'a' or config.pos_dict[text_tag[1]] == 'n')]
    selected = ' '.join(selected)
    return selected

def preprocessing(filename, column="text", ncolumn="selected"):
    print(filename)
    df = pd.read_csv(filename)
    df[ncolumn] = df[column].apply(get_a_and_n)
    df.to_csv(filename)

def combine_zhang15_dbpedia_train_test():
    # df_train = pd.read_csv(config.zhang15_dbpedia_train_path, names=["class", "title", "text"])
    # df_test = pd.read_csv(config.zhang15_dbpedia_test_path, names=["class", "title", "text"])
    df_train = pd.read_csv(config.zhang15_dbpedia_train_path, index_col=0)
    df_test = pd.read_csv(config.zhang15_dbpedia_test_path, index_col=0)
    df_train = df_train[["class", "title", "text", "selected"]]
    df_test = df_test[["class", "title", "text", "selected"]]
    # df_train.to_csv(config.zhang15_dbpedia_train_path)
    # df_test.to_csv(config.zhang15_dbpedia_test_path)
    df_full = pd.concat([df_train, df_test], ignore_index=True)
    df_full.to_csv(config.zhang15_dbpedia_full_data_path)

def combine_20news_train_test():
    df_train = pd.read_csv(config.news20_train_path, index_col=0)
    df_test = pd.read_csv(config.news20_test_path, index_col=0)
    df_train = df_train[["class", "text", "selected"]]
    df_test = df_test[["class", "text", "selected"]]
    # df_train.to_csv(config.news20_train_path)
    # df_test.to_csv(config.news20_test_path)
    df_full = pd.concat([df_train, df_test], ignore_index=True)
    df_full.to_csv(config.news20_full_data_path)

if __name__ == "__main__":
    # kg_vector_1 = pickle.load(open("../wordEmbeddings/KG_VECTORS_1.pickle", "rb"))
    # print(kg_vector_1.keys())
    # kg_vector_2 = pickle.load(open("../wordEmbeddings/KG_VECTORS_2.pickle", "rb"))
    # print(kg_vector_2["/c/en/building"]['/c/en/syriac'].shape)

    # data = np.load("../results/key_zhang15_dbpedia_4of4/logs/test_5.npz")
    # data = np.load("../results/key_zhang15_dbpedia_4of4/logs/test_1_kg.npz")
    # print(data["pred_unseen"].shape)
    # print(data["gt_unseen"].shape)
    # print(data["align_unseen"].shape)
    # print(data["kg_vector_unseen"].shape)
    # print(data["pred_seen"].shape)
    # print(data["gt_seen"].shape)
    # print(data["align_seen"].shape)
    # print(data["kg_vector_seen"].shape)

    '''
    gt_sum = np.sum(data["gt_unseen"][:5000], axis=0)
    print(gt_sum)
    kg_sum = np.sum(data["kg_vector_unseen"][:5000], axis=(0, 2, 3))
    print(kg_sum)

    gt_sum = np.sum(data["gt_unseen"][5000:10000], axis=0)
    print(gt_sum)
    kg_sum = np.sum(data["kg_vector_unseen"][5000:10000], axis=(0, 2, 3))
    print(kg_sum)

    gt_sum = np.sum(data["gt_unseen"][10000:], axis=0)
    print(gt_sum)
    kg_sum = np.sum(data["kg_vector_unseen"][10000:], axis=(0, 2, 3))
    print(kg_sum)
    '''

    # print(data["pred_unseen"])

    # doing_sth_on_chen14()
    # doing_sth_on_20_news()

    # preprocessing(config.zhang15_dbpedia_test_path)
    # preprocessing(config.zhang15_dbpedia_train_path)
    # preprocessing(config.zhang15_dbpedia_full_data_path)

    # preprocessing(config.news20_test_path)
    # preprocessing(config.news20_train_path)
    # preprocessing(config.news20_full_data_path)

    # combine_zhang15_dbpedia_train_test()
    combine_20news_train_test()




