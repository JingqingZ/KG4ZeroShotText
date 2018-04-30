
import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorlayer as tl
import progressbar

import config

data_filename = "../data/wiki/simple_wiki_2.csv"

vocabulary_path = "../data/wiki/vocab.txt"
processed_text_path = "../data/wiki/processed_text.txt"

START_ID = '<START_ID>'
END_ID = '<END_ID>'
PAD_ID = '<PAD_ID>'
UNK_ID = '<UNK_ID>'

def preprocess(textlist, vocab_path=vocabulary_path):
    print("Preprocessing ...")
    with progressbar.ProgressBar(max_value=len(textlist)) as bar:
        for idx, text in enumerate(textlist):
            # textlist[idx].replace(",", " ")
            # textlist[idx].replace(".", " ")
            textlist[idx] = re.sub(r'[\W_]+', ' ', textlist[idx])
            textlist[idx] = tl.nlp.process_sentence(textlist[idx], start_word=START_ID, end_word=END_ID)
            # textlist[idx] = textlist[idx].split() # no empty string in the list
            bar.update(idx + 1)

    # create dictionary
    tl.nlp.create_vocab(textlist, word_counts_output_file=vocab_path, min_word_count=5)
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

def load_data(filename, column="text", vocab_file=vocabulary_path, processed_file=processed_text_path, force_process=False):
    print("Loading data ...")

    if not force_process and os.path.exists(processed_text_path) and os.path.exists(vocab_file):
        print("Processed data found in local files. Loading ...")
        with open(processed_file, "r") as f:
            full_text_list = eval(f.read())
        vocab = tl.nlp.Vocabulary(vocabulary_path, start_word=START_ID, end_word=END_ID, unk_word=UNK_ID)
    else:
        df = pd.read_csv(data_filename, index_col=0)

        full_text_list = df[column].tolist()
        # full_text_list = full_text_list[:2 * config.batch_size]
        full_text_list, vocab = preprocess(full_text_list, vocab_path=vocab_file)

        with open(processed_file, "w") as f:
            f.write(str(full_text_list))
        print("Processed data saved to %s" % processed_file)

    print("Data loaded: num of seqs %s" % len(full_text_list))
    return full_text_list, vocab


if __name__ == "__main__":
    text_seqs, vocab = load_data(data_filename, force_process=True)
    # print(prepro_encode(text_seqs.copy(), vocab))
    # print(prepro_decode(text_seqs.copy(), vocab))
    # print(prepro_target(text_seqs.copy(), vocab))
    pass