
import os
import re
import pickle
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorlayer as tl
import progressbar

import config


START_ID = '<START_ID>'
END_ID = '<END_ID>'
PAD_ID = '<PAD_ID>'
UNK_ID = '<UNK_ID>'

def preprocess(textlist):
    print("Preprocessing ...")
    with progressbar.ProgressBar(max_value=len(textlist)) as bar:
        for idx, text in enumerate(textlist):
            # textlist[idx].replace(",", " ")
            # textlist[idx].replace(".", " ")
            textlist[idx] = re.sub(r'[\W_]+', ' ', textlist[idx])
            textlist[idx] = tl.nlp.process_sentence(textlist[idx], start_word=START_ID, end_word=END_ID)
            # textlist[idx] = textlist[idx].split() # no empty string in the list
            bar.update(idx + 1)

    return textlist

def create_vocab_given_text(textlist, vocab_path, min_word_count=config.prepro_min_word_count):
    # create dictionary
    tl.nlp.create_vocab(textlist, word_counts_output_file=vocab_path, min_word_count=min_word_count)
    vocab = tl.nlp.Vocabulary(vocab_path, start_word=START_ID, end_word=END_ID, unk_word=UNK_ID)
    return vocab

def sentence_word_to_id(textlist, vocab):
    for idx, text in enumerate(textlist):
        textlist[idx] = [vocab.word_to_id(word) for word in text]
    return textlist

def prepro_encode_kg_vector(kg_vector_list):
    for idx, kg_vector in enumerate(kg_vector_list):
        new_kg_vector = np.zeros([config.max_length, config.kg_embedding_dim])
        new_kg_vector[:kg_vector.shape[0] - 2, :] = kg_vector[1:-1]
        kg_vector_list[idx] = new_kg_vector
    return np.array(kg_vector_list)

def prepro_encode(textlist, vocab):
    for idx, text in enumerate(textlist):
        textlist[idx] = text[1:-1] + [vocab.pad_id]
    textlist = tl.prepro.pad_sequences(textlist, maxlen=config.prepro_max_sentence_length, dtype='int64', padding='post', truncating='post', value=vocab.pad_id)
    for idx, text in enumerate(textlist):
        textlist[idx] = text[:-1] + [vocab.pad_id]
    return np.array(textlist)

def get_text_list(df, column):
    if type(column) == str:
        full_text_list = df[column].tolist()
    elif type(column) == list:
        df["text"] = df[column].apply(
            lambda x: ' '.join([item if type(item) == str else ' ' for item in x]), axis=1)
        full_text_list = df["text"].tolist()
    else:
        raise Exception("column should be either a string or a list of string")
    return full_text_list


def load_data(filename, vocab_file, processed_file, column, min_word_count=config.prepro_min_word_count, force_process=False):
    print("Loading data ...")

    if not force_process and os.path.exists(processed_file) and os.path.exists(vocab_file):
        print("Processed data found in local files. Loading ...")
        if processed_file.endswith(".pkl"):
            with open(processed_file, 'rb') as f:
                full_text_list = pickle.load(f)
        else:
            with open(processed_file, "r") as f:
                full_text_list = eval(f.read())
        vocab = tl.nlp.Vocabulary(vocab_file, start_word=START_ID, end_word=END_ID, unk_word=UNK_ID)
    else:
        df = pd.read_csv(filename, index_col=0)

        full_text_list = get_text_list(df, column)
        full_text_list = preprocess(full_text_list)
        vocab = create_vocab_given_text(full_text_list, vocab_path=vocab_file, min_word_count=min_word_count)
        full_text_list = sentence_word_to_id(full_text_list, vocab)

        if processed_file.endswith(".pkl"):
            with open(processed_file, "wb") as f:
                pickle.dump(full_text_list, f)
        else:
            with open(processed_file, "w") as f:
                f.write(str(full_text_list))
        print("Processed data saved to %s" % processed_file)

    print("Data loaded: num of seqs %s" % len(full_text_list))
    return full_text_list, vocab


def build_vocabulary_from_full_corpus(filename, vocab_file, column, min_word_count=config.prepro_min_word_count, force_process=False):
    if not force_process and os.path.exists(vocab_file):
        print("Load vocab from local file")
        vocab = tl.nlp.Vocabulary(vocab_file, start_word=START_ID, end_word=END_ID, unk_word=UNK_ID)
    else:
        print("Creating vocab ...")
        df = pd.read_csv(filename, index_col=0)

        full_text_list = get_text_list(df, column)
        full_text_list = preprocess(full_text_list)
        vocab = create_vocab_given_text(full_text_list, vocab_path=vocab_file, min_word_count=min_word_count)
        print("Vocab created and saved in %s" % vocab_file)
    return vocab

def load_data_class(filename, column):
    df = pd.read_csv(filename, index_col=0)
    data_class_list = df[column].tolist()
    return data_class_list

def load_class_dict(class_file, class_code_column, class_name_column):
    class_df = pd.read_csv(class_file)
    class_dict = dict(zip(class_df[class_code_column], class_df[class_name_column]))

    return class_dict

def load_data_from_text_given_vocab(filename, vocab, processed_file, column, force_process=False):
    print("Loading data given vocab ...")

    if not force_process and os.path.exists(processed_file):
        print("Processed data found in local files. Loading ...")
        if processed_file.endswith(".pkl"):
            with open(processed_file, 'rb') as f:
                full_text_list = pickle.load(f)
        else:
            with open(processed_file, "r") as f:
                full_text_list = eval(f.read())

    else:

        df = pd.read_csv(filename, index_col=0)

        full_text_list = get_text_list(df, column)
        full_text_list = preprocess(full_text_list)
        full_text_list = sentence_word_to_id(full_text_list, vocab)

        if processed_file.endswith(".pkl"):
            with open(processed_file, "wb") as f:
                pickle.dump(full_text_list, f)
        else:
            with open(processed_file, "w") as f:
                f.write(str(full_text_list))
        print("Processed data saved to %s" % processed_file)

    print("Data loaded: num of seqs %s" % len(full_text_list))
    return full_text_list

def get_kg_vector(kg_vector_dict, class_label, word):

    prefix = '/c/en/'

    if not class_label.startswith(prefix):
        class_label = prefix + class_label.lower()

    assert class_label in kg_vector_dict

    if not word.startswith(prefix):
        word = prefix + word.lower()

    if word in kg_vector_dict[class_label]:
        return kg_vector_dict[class_label][word]
    else:
        return np.zeros(config.kg_embedding_dim)

def get_kg_vector_sentence(encode_text_seqs_mini, true_class_mini, class_dict, category_logits, kg_vector_dict, vocab):
    kg_vector_seqs_mini = list()

    for idx, logit in enumerate(category_logits):

        if logit == 1:
            class_id = true_class_mini[idx]
        else:
            while True:
                false_class_id = random.choice(list(class_dict))
                if false_class_id != true_class_mini[idx]:
                    break
            class_id = false_class_id

        kg_vector = np.zeros([len(encode_text_seqs_mini[idx]), config.kg_embedding_dim])
        for widx, word_id in enumerate(encode_text_seqs_mini[idx]):
            kg_vector[widx, :] = get_kg_vector(kg_vector_dict, class_dict[class_id], vocab.id_to_word(word_id))
        kg_vector_seqs_mini.append(kg_vector)

    return kg_vector_seqs_mini

def load_kg_vector(filename):
    with open(filename, 'rb') as f:
        kg_vector_dict = pickle.load(f)
    return kg_vector_dict

def load_kg_vector_given_text_seqs(text_seqs, vocab, class_dict, kg_vector_dict, processed_file, force_process=False):

    print("Loading KG Vector ...")
    if not force_process and os.path.exists(processed_file):
        print("Processed data found in local files. Loading ...")
        with open(processed_file, 'rb') as f:
            kg_vector_seqs = pickle.load(f)
    else:
        kg_vector_seqs = list()

        with progressbar.ProgressBar(max_value=len(text_seqs)) as bar:
            for idx, text in enumerate(text_seqs):
                kg_vector_text_dict = dict()
                for class_id in class_dict:
                    kg_vector = np.zeros([len(text), config.kg_embedding_dim])
                    for widx, word_id in enumerate(text):
                        kg_vector[widx, :] = get_kg_vector(kg_vector_dict, class_dict[class_id], vocab.id_to_word(word_id))
                    kg_vector_text_dict[class_id] = kg_vector
                kg_vector_seqs.append(kg_vector_text_dict)
                bar.update(idx)
        with open(processed_file, "wb") as f:
            pickle.dump(kg_vector_seqs, f)
    return kg_vector_seqs

if __name__ == "__main__":
    # text_seqs, vocab = load_data(config.wiki_train_data_path, config.wiki_vocab_path, config.wiki_train_processed_path, column="text", force_process=True)
    # text_seqs, vocab = load_data(config.arxiv_train_data_path, config.arxiv_vocab_path, config.arxiv_train_processed_path, column="abstract", force_process=True)

    # vocab = build_vocabulary_from_full_corpus(config.wiki_full_data_path, config.wiki_vocab_path, column="text", force_process=True)
    # vocab = build_vocabulary_from_full_corpus(config.arxiv_full_data_path, config.arxiv_vocab_path, column="abstract", force_process=True)
    # vocab = build_vocabulary_from_full_corpus(config.zhang15_dbpedia_full_data_path, config.zhang15_dbpedia_vocab_path, column="text", force_process=False)
    # vocab = build_vocabulary_from_full_corpus(config.zhang15_yahoo_full_data_path, config.zhang15_yahoo_vocab_path, column=["question_title", "question_content", "best_answer"], force_process=False)

    # text_seqs = load_data_from_text_given_vocab(
    #     config.zhang15_dbpedia_test_path, vocab, config.zhang15_dbpedia_test_processed_path,
    #     column="text", force_process=False
    # )

    # text_seqs = load_data_from_text_given_vocab(
    #     config.zhang15_dbpedia_train_path, vocab, config.zhang15_dbpedia_train_processed_path,
    #     column="text", force_process=False
    # )

    # text_seqs = load_data_from_text_given_vocab(
    #     config.zhang15_yahoo_test_path, vocab, config.zhang15_yahoo_test_processed_path,
    #     column=["question_title", "question_content", "best_answer"], force_process=False
    # )

    # print(len(text_seqs))
    # print(text_seqs[0])

    # load_kg_vector(config.kg_vector_data_path)
    data_class_list = load_data_class(
        filename=config.zhang15_dbpedia_train_path,
        column="class",
    )
    print(data_class_list)

    class_dict = load_class_dict(
        class_file=config.zhang15_dbpedia_class_label_path,
        class_code_column="ClassCode",
        class_name_column="ConceptNet"
    )
    print(class_dict)

    pass

