import os
import pickle
import pandas as pd
import numpy as np
import random
import progressbar

import config
import utils
import dataloader
import nltk
import json


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

def doing_sth_on_chen14_elec():

    # get class_dict
    # collect data from each class folder and randomly split to train.csv and test.csv
    # build a complete dictionary

    class_uri = os.listdir("../data/chen14/clean_elec/KG_VECTOR_3_Lem/")
    class_uri = ["/c/en/" + uri.split("_elec_")[-1].split(".")[0] for uri in class_uri]
    # class_uri = sorted(class_uri)
    class_uri_dict = dict()

    clean_dir = "../data/chen14/clean_elec/"

    home_dir = "../data/chen14/Electronics/"
    class_dir_list = os.listdir(home_dir)
    class_dir_list = sorted(class_dir_list)

    def create_class_dict():
        with open(clean_dir + "classLabelsChen14Elec.csv", "w") as f_c:
            f_c.write("ClassCode,ClassLabel,ConceptNet,Count\n")
            for idx, class_dir in enumerate(class_dir_list):
                current_dir = "%s%s/" % (home_dir, class_dir)
                conceptnet_name = ""
                for concept_class in class_uri:
                    if concept_class in class_uri_dict:
                        continue
                    concept_class_label = concept_class.split("/")[-1]
                    if "_" in concept_class_label:
                        concept_class_label = concept_class_label.split("_")[0]
                    if concept_class_label in class_dir.lower():
                        conceptnet_name = concept_class
                        class_uri_dict[conceptnet_name] = 1
                        break
                # TODO: tricky
                # if conceptnet_name == "":
                #     conceptnet_name = "/c/en/supplies"

                with open(current_dir + "%s.docs" % class_dir) as f:
                    for i, l in enumerate(f):
                        pass
                    num_line = i + 1
                    try:
                        # assert class_uri.index(conceptnet_name) == idx
                        print("%d,%s,%s,%d" % (idx + 1, class_dir, conceptnet_name.split("/")[-1], num_line))
                        f_c.write("%d,%s,%s,%d\n" % (idx + 1, class_dir, conceptnet_name.split("/")[-1], num_line))
                    except:
                        print(class_dir)

    # create_class_dict()
    # exit()

    def collect_full_and_split():
        class_dict_concept = dataloader.load_class_dict(
            class_file=config.chen14_elec_class_label_path,
            class_code_column="ConceptNet",
            class_name_column="ClassCode",
        )

        class_dict_label = dataloader.load_class_dict(
            class_file=config.chen14_elec_class_label_path,
            class_code_column="ClassLabel",
            class_name_column="ConceptNet",
        )

        class_dict_count = dataloader.load_class_dict(
            class_file=config.chen14_elec_class_label_path,
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
            with open(filename + ".vocab", 'r', encoding="utf8", errors="ignore") as f:
                for line in f:
                    word_id = line.split(":")[0]
                    word = line.split(":")[1].replace("\n", "")
                    assert word_id not in vocab_dict
                    vocab_dict[word_id] = word
            order = list(range(class_dict_count[class_dir]))
            random.shuffle(order)
            test_order = order[:int(class_dict_count[class_dir] * 0.3)]
            with open(filename + ".docs", 'r', encoding="utf8", errors="ignore") as f:
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
            # print(test_df)
            # print(train_df)
            # print(df)
            # exit()
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        df = df.reset_index(drop=True)
        train_df.to_csv("../data/chen14/clean_elec/train.csv")
        test_df.to_csv("../data/chen14/clean_elec/test.csv")
        df.to_csv("../data/chen14/clean_elec/full.csv")

    collect_full_and_split()

    def collect_full_vocab():
        for idx, class_dir in enumerate(class_dir_list):
            filename = "%s%s/%s" % (home_dir, class_dir, class_dir)
            vocab_dict = dict()
            with open(filename + ".vocab", 'r', encoding="utf8", errors="ignore") as f:
                for nidx, line in enumerate(f):
                    word_id = line.split(":")[0]
                    word = line.split(":")[1].replace("\n", "")
                    assert word not in vocab_dict
                    vocab_dict[word] = word
                print(idx, filename, nidx)
        pass

    # collect_full_vocab()

def doing_sth_on_chen14_elec_2():
    clean_dir = "../data/chen14/clean_elec/"
    datafile = "../data/chen14/clean_elec/50EleReviews.json"

    with open(datafile, 'r') as f:
        data = json.load(f)

        def create_class_dict():
            class_uri = os.listdir("../data/chen14/clean_elec/KG_VECTOR_3_Lem/")
            class_uri = ["/c/en/" + uri.split("_elec_")[-1].split(".")[0] for uri in class_uri]
            class_uri_dict = dict()

            with open(clean_dir + "classLabelsChen14Elec.csv", "w") as f_c:
                f_c.write("ClassCode,ClassLabel,ConceptNet,Count\n")

                for idx, class_dir in enumerate(data["target_names"]):
                    conceptnet_name = ""
                    for concept_class in class_uri:
                        if concept_class in class_uri_dict:
                            continue
                        concept_class_label = concept_class.split("/")[-1]
                        if "_" in concept_class_label:
                            concept_class_label = concept_class_label.split("_")[0]
                        if concept_class_label in class_dir.lower():
                            conceptnet_name = concept_class
                            class_uri_dict[conceptnet_name] = 1
                            break

                    try:
                        num_line = len([_ for _ in data["y"] if _ == idx])
                        print("%d,%s,%s,%d" % (idx + 1, class_dir, conceptnet_name.split("/")[-1], num_line))
                        f_c.write("%d,%s,%s,%d\n" % (idx + 1, class_dir, conceptnet_name.split("/")[-1], num_line))
                    except:
                        print(class_dir)

        # create_class_dict()

        def collect_full_and_split():
            class_dict_concept = dataloader.load_class_dict(
                class_file=config.chen14_elec_class_label_path,
                class_code_column="ConceptNet",
                class_name_column="ClassCode",
            )

            class_dict_label = dataloader.load_class_dict(
                class_file=config.chen14_elec_class_label_path,
                class_code_column="ClassLabel",
                class_name_column="ConceptNet",
            )

            class_dict_count = dataloader.load_class_dict(
                class_file=config.chen14_elec_class_label_path,
                class_code_column="ClassLabel",
                class_name_column="Count",
            )

            train_df = pd.DataFrame(columns=["class", "text"])
            test_df = pd.DataFrame(columns=["class", "text"])
            df = pd.DataFrame(columns=["class", "text"])
            index_df = 0
            for idx, class_dir in enumerate(data["target_names"]):
                print(idx, class_dir)

                order = list(range(class_dict_count[class_dir]))
                random.shuffle(order)
                test_order = order[:int(class_dict_count[class_dir] * 0.3)]

                class_samples = [x for xid, x in enumerate(data["X"]) if data["y"][xid] == idx]

                for lidx, content in enumerate(class_samples):

                    if lidx in test_order:
                        test_df.loc[index_df] = [class_dict_concept[class_dict_label[class_dir]], content]
                    else:
                        train_df.loc[index_df] = [class_dict_concept[class_dict_label[class_dir]], content]
                    df.loc[index_df] = [class_dict_concept[class_dict_label[class_dir]], content]
                    index_df += 1

            train_df = train_df.reset_index(drop=True)
            test_df = test_df.reset_index(drop=True)
            df = df.reset_index(drop=True)
            train_df.to_csv("../data/chen14/clean_elec/train.csv")
            test_df.to_csv("../data/chen14/clean_elec/test.csv")
            df.to_csv("../data/chen14/clean_elec/full.csv")

        collect_full_and_split()

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

def check_utf8(filename):
    try:
        # infile = open(filename, 'r')
        with open(filename, encoding="utf8") as f:
            for line in f:
                pass
    except:
        print(filename)
        with open(filename, encoding="utf8", errors="ignore") as f:
            for idx, line in enumerate(f):
                print(line)
                if idx > 3:
                    break
        '''
        infile = open(filename, 'rb')
        contents = infile.read()
        infile.close()
        outfile = open(filename, "w", encoding="utf-8")
        outfile.write(contents)
        outfile.close()
        '''

def analyse_definition():
    from nltk.corpus import wordnet as wn

    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    all_words = set(_ for _ in wn.words())
    all_defs = set()

    with progressbar.ProgressBar(max_value=len(list(all_words))) as bar:
        for idx, word in enumerate(all_words):
            synsets = wn.synsets(word)
            definitions = ' '.join([_.definition() for _ in synsets])
            definitions = definitions.replace(";", "")
            defwords = [defw for defw in definitions.split(' ') if defw in all_words]
            defwords += [lemmatizer.lemmatize(defw) for defw in definitions.split(' ') if defw not in all_words and lemmatizer.lemmatize(defw) in all_words]
            all_defs.update(defwords)
            bar.update(idx)

    print("Full voc:", len(list(all_words)))
    print("Full def:", len(list(all_defs)))


def analysis_num_in_vocab():
    with open(config.zhang15_dbpedia_vocab_path) as f:
        num = 0
        for line in f:
            content = line.split(" ")
            try:
                c = float(content[0])
                num += 1
                print(c)
            except:
                continue
        print(num)


def visualise_wordvector():

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

    train_class_list = dataloader.load_data_class(
        filename=config.zhang15_dbpedia_train_path,
        column="class",
    )

    train_text_seqs = dataloader.load_data_from_text_given_vocab(
        config.zhang15_dbpedia_train_path, vocab, config.zhang15_dbpedia_train_processed_path,
        column="text", force_process=False
        # column="selected", force_process=False
    )

    import tsne, pylab

    x = list()
    labels = list()
    with progressbar.ProgressBar(max_value=len(train_class_list)) as bar:
        for idx, sentence in enumerate(train_text_seqs):
            if idx % 1000 != 0:
                continue
            for wordid in sentence:
                x.append(glove_mat[wordid])
                labels.append(train_class_list[idx])
            bar.update(idx)
    x = np.array(x)
    labels = np.array(labels)
    print(x.shape)

    from sklearn.manifold import TSNE

    y = TSNE(n_components=2).fit_transform(x)

    pylab.scatter(y[:, 0], y[:, 1], 4, labels)
    pylab.show()

def tf_idf_document(vocab, glove_mat, text_seqs, df_filename, df_out_filename):

    df = pd.read_csv(
        df_filename,
        index_col=0
    )
    df.drop(columns=[col for col in df.columns if "Unnamed:" in str(col)], inplace=True)

    df["selected_tfidf"] = ""

    assert len(text_seqs) == df.shape[0]

    print("IDF")
    appearance_of_word = np.zeros(vocab.unk_id + 1)
    for idx, document in enumerate(text_seqs):
        word_set = set()
        for wordid in document:
            if wordid not in word_set:
                appearance_of_word[wordid] += 1
            word_set.add(wordid)

    total_number_documents = len(text_seqs)
    idf_word = np.log(total_number_documents / (1 + appearance_of_word))

    print("DF")
    with progressbar.ProgressBar(max_value=len(text_seqs)) as bar:
        # counting numbers
        for idx, document in enumerate(text_seqs):
            number_of_word = dict()
            for wordid in document:
                # if the word is not found in glove then ignore the word
                if np.sum(glove_mat[wordid]) == 0:
                    continue
                if wordid not in number_of_word:
                    number_of_word[wordid] = 0
                number_of_word[wordid] += 1
            # tf and tfidf
            tf_word = {k: v / len(document) for k, v in number_of_word.items()}
            tfidf_word = {k: v * idf_word[k] for k, v in tf_word.items()}

            # find top k words in this doument
            k = max(len(document) // 3, 10)
            k = min(k, len(document))

            # k = min(100, len(document) // 2)

            import operator
            sorted_tfidf = sorted(tfidf_word.items(), key=operator.itemgetter(1), reverse=True)
            topk = {t[0]: t[1] for t in sorted_tfidf[:k]}

            # save to df and then csv
            df.set_value(idx, "selected_tfidf", " ".join([vocab.id_to_word(wordid) for wordid in document if wordid in topk]))
            # print(df.iloc[idx]["text"])
            # print(df.iloc[idx]["selected_tfidf"])
            bar.update(idx)

    df.to_csv(df_out_filename)



def tf_idf_category():
    vocab = dataloader.build_vocabulary_from_full_corpus(
        config.zhang15_dbpedia_full_data_path, config.zhang15_dbpedia_vocab_path, column="text", force_process=False,
        min_word_count=55
    )

    class_dict = dataloader.load_class_dict(
        class_file=config.zhang15_dbpedia_class_label_path,
        class_code_column="ClassCode",
        class_name_column="ConceptNet"
    )

    glove_mat = dataloader.load_glove_word_vector(
        config.word_embed_file_path, config.zhang15_dbpedia_word_embed_matrix_path, vocab, force_process=False
    )

    assert np.sum(glove_mat[vocab.start_id]) == 0
    assert np.sum(glove_mat[vocab.end_id]) == 0
    assert np.sum(glove_mat[vocab.unk_id]) == 0
    assert np.sum(glove_mat[vocab.pad_id]) == 0

    train_class_list = dataloader.load_data_class(
        filename=config.zhang15_dbpedia_train_path,
        column="class",
    )

    train_text_seqs = dataloader.load_data_from_text_given_vocab(
        config.zhang15_dbpedia_train_path, vocab, config.zhang15_dbpedia_train_processed_path,
        column="text", force_process=False
    )

    print("Combing documents in the same category together ...")
    all_content_for_each_class_dict = dict()
    for idx, document in enumerate(train_text_seqs):
        class_id = train_class_list[idx]
        if class_id not in all_content_for_each_class_dict:
            all_content_for_each_class_dict[class_id] = list()
        all_content_for_each_class_dict[class_id] += document

    import math
    total_number_of_category = len(all_content_for_each_class_dict.keys())
    occur_of_word_in_category_list = np.zeros(vocab.unk_id + 1)
    number_of_word_in_each_category_dict_list = dict()

    print("Counting number of appearance ...")
    for class_id in all_content_for_each_class_dict:
        full_text = all_content_for_each_class_dict[class_id]

        assert class_id not in number_of_word_in_each_category_dict_list
        number_of_word_in_each_category_dict_list[class_id] = np.zeros(vocab.unk_id + 1)

        word_set = set()

        for word_id in full_text:
            number_of_word_in_each_category_dict_list[class_id][word_id] += 1
            if word_id in word_set:
                continue
            word_set.add(word_id)
            occur_of_word_in_category_list[word_id] += 1

    print("IDF")
    idf_list = np.array([math.log(total_number_of_category / (1 + _)) for _ in occur_of_word_in_category_list])

    print("TF")
    tf_dict_list = dict()
    for class_id in number_of_word_in_each_category_dict_list:
        assert class_id not in tf_dict_list
        most_freq = np.max(number_of_word_in_each_category_dict_list[class_id])
        # tf_dict_list[class_id] = np.array([ 0.5 + 0.5 * _ / most_freq for _ in number_of_word_in_each_category_dict_list[class_id]])
        # tf_dict_list[class_id] = np.array([_ / most_freq for _ in number_of_word_in_each_category_dict_list[class_id]])
        tf_dict_list[class_id] = np.array([math.log(1 + _) for _ in number_of_word_in_each_category_dict_list[class_id]])

    print("TFIDF")
    tfidf_dict_list = dict()
    for class_id in number_of_word_in_each_category_dict_list:
        assert class_id not in tfidf_dict_list
        tfidf_dict_list[class_id] = tf_dict_list[class_id] * idf_list

        # manually set some special words to 0
        for word_id in range(vocab.unk_id + 1):
            if np.sum(glove_mat[word_id]) == 0:
                tfidf_dict_list[class_id][word_id] = 0

    print("samples of top indicative words ...")
    for class_id in tfidf_dict_list:
        tfidf_scores = tfidf_dict_list[class_id]
        k = 10
        topk = tfidf_scores.argsort()[-k:][::-1]
        print(class_dict[class_id], [vocab.id_to_word(idx) for idx in topk])
        print(class_dict[class_id], [tfidf_scores[idx] for idx in topk])


    with open(config.zhang15_dbpedia_dir + "TFIDF_class.pkl", "wb") as f:
        pickle.dump(tfidf_dict_list, f)

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
    # combine_20news_train_test()
    # doing_sth_on_chen14_elec()
    # doing_sth_on_chen14_elec_2()

    '''
    home_dir = "../data/chen14/Electronics/"
    class_dir_list = os.listdir(home_dir)
    class_dir_list = sorted(class_dir_list)
    for idx, class_dir in enumerate(class_dir_list):
        filename = "%s%s/%s" % (home_dir, class_dir, class_dir)
        # print(filename)
        check_utf8(filename + ".vocab")
        check_utf8(filename + ".docs")
    '''

    # analyse_definition()
    # analysis_num_in_vocab()
    # visualise_wordvector()

    # tf_idf_document()
    # tf_idf_category()

    df = pd.read_csv(config.zhang15_dbpedia_train_path, index_col=0)
    for i in range(df.shape[0]):
        if type(df.iloc[i]["selected_tfidf"]) != str:
            print(i)
            print(df.iloc[i]["text"])
            print(df.iloc[i]["selected_tfidf"])
            exit()
    pass




