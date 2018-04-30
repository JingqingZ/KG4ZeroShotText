
import re
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorlayer as tl

import config

data_filename = "../data/wiki/simple_wiki_2.csv"

START_ID = '<START_ID>'
END_ID = '<END_ID>'
PAD_ID = '<PAD_ID>'
UNK_ID = '<UNK_ID>'

def preprocess(textlist, vocabulary_path="../data/wiki/vocab.txt"):
    for idx, text in enumerate(textlist):
        # textlist[idx].replace(",", " ")
        # textlist[idx].replace(".", " ")
        # textlist[idx] = re.sub(r'[\W_]+', ' ', textlist[idx])
        textlist[idx] = tl.nlp.process_sentence(text, start_word=START_ID, end_word=END_ID)
        # textlist[idx] = textlist[idx].split() # no empty string in the list

    # create dictionary
    tl.nlp.create_vocab(textlist, word_counts_output_file=vocabulary_path, min_word_count=1)
    vocab = tl.nlp.Vocabulary(vocabulary_path, start_word=START_ID, end_word=END_ID, unk_word=UNK_ID)

    for idx, text in enumerate(textlist):
        textlist[idx] = [vocab.word_to_id(word) for word in text]

    return textlist, vocab

def prepro_encode(textlist, vocab):
    for idx, text in enumerate(textlist):
        textlist[idx] = text[1:-1] + [vocab.pad_id]
    textlist = tl.prepro.pad_sequences(textlist, maxlen=None, dtype='int64', padding='post', truncating='pre', value=vocab.pad_id)
    return np.array(textlist)

def prepro_decode(textlist, vocab):
    for idx, text in enumerate(textlist):
        textlist[idx] = text[:-1] + [vocab.pad_id]
    textlist = tl.prepro.pad_sequences(textlist, maxlen=None, dtype='int64', padding='post', truncating='pre', value=vocab.pad_id)
    return np.array(textlist)

def prepro_target(textlist, vocab):
    for idx, text in enumerate(textlist):
        textlist[idx] = text[1:] + [vocab.pad_id]
    textlist = tl.prepro.pad_sequences(textlist, maxlen=None, dtype='int64', padding='post', truncating='pre', value=vocab.pad_id)
    return np.array(textlist)

def load_data(filename, column="text"):
    df = pd.read_csv(data_filename, index_col=0)
    full_text_list = df[column].tolist()
    full_text_list = full_text_list[:2 * config.batch_size]

    full_text_list, vocab = preprocess(full_text_list)

    return full_text_list, vocab


if __name__ == "__main__":
    text_seqs, vocab = load_data(data_filename)
    print(prepro_encode(text_seqs.copy(), vocab))
    print(prepro_decode(text_seqs.copy(), vocab))
    print(prepro_target(text_seqs.copy(), vocab))
    pass