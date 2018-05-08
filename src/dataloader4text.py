
import os
import re
import pickle
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

def prepro_encode(textlist, vocab):
    for idx, text in enumerate(textlist):
        textlist[idx] = text[1:-1] + [vocab.pad_id]
    textlist = tl.prepro.pad_sequences(textlist, maxlen=config.prepro_max_sentence_length - 1, dtype='int64', padding='post', truncating='post', value=vocab.pad_id)
    for idx, text in enumerate(textlist):
        textlist[idx] = text[:-1] + [vocab.pad_id]
    return np.array(textlist)

def prepro_decode(textlist, vocab):
    for idx, text in enumerate(textlist):
        textlist[idx] = text[:-1] + [vocab.pad_id]
    textlist = tl.prepro.pad_sequences(textlist, maxlen=config.prepro_max_sentence_length, dtype='int64', padding='post', truncating='post', value=vocab.pad_id)
    for idx, text in enumerate(textlist):
        textlist[idx] = text[:-1] + [vocab.pad_id]
    return np.array(textlist)

def prepro_target(textlist, vocab):
    for idx, text in enumerate(textlist):
        textlist[idx] = text[1:] + [vocab.pad_id]
    textlist = tl.prepro.pad_sequences(textlist, maxlen=config.prepro_max_sentence_length, dtype='int64', padding='post', truncating='post', value=vocab.pad_id)
    for idx, text in enumerate(textlist):
        if text[-1] != vocab.pad_id:
            textlist[idx] = text[:-2] + [vocab.end_id, vocab.pad_id]
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


if __name__ == "__main__":
    # text_seqs, vocab = load_data(config.wiki_train_data_path, config.wiki_vocab_path, config.wiki_train_processed_path, column="text", force_process=True)
    # text_seqs, vocab = load_data(config.arxiv_train_data_path, config.arxiv_vocab_path, config.arxiv_train_processed_path, column="abstract", force_process=True)

    # vocab = build_vocabulary_from_full_corpus(config.wiki_full_data_path, config.wiki_vocab_path, column="text", force_process=True)
    # vocab = build_vocabulary_from_full_corpus(config.arxiv_full_data_path, config.arxiv_vocab_path, column="abstract", force_process=True)
    # vocab = build_vocabulary_from_full_corpus(config.zhang15_dbpedia_full_data_path, config.zhang15_dbpedia_vocab_path, column="text", force_process=False)
    vocab = build_vocabulary_from_full_corpus(config.zhang15_yahoo_full_data_path, config.zhang15_yahoo_vocab_path, column=["question_title", "question_content", "best_answer"], force_process=False)

    # text_seqs = load_data_from_text_given_vocab(
    #     config.zhang15_dbpedia_test_path, vocab, config.zhang15_dbpedia_test_processed_path,
    #     column="text", force_process=False
    # )

    # text_seqs = load_data_from_text_given_vocab(
    #     config.zhang15_dbpedia_train_path, vocab, config.zhang15_dbpedia_train_processed_path,
    #     column="text", force_process=False
    # )

    text_seqs = load_data_from_text_given_vocab(
        config.zhang15_yahoo_test_path, vocab, config.zhang15_yahoo_test_processed_path,
        column=["question_title", "question_content", "best_answer"], force_process=False
    )

    print(len(text_seqs))
    print(text_seqs[0])

    pass

