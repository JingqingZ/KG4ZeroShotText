
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

def preprocess(textlist, vocab_path):
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
    tl.nlp.create_vocab(textlist, word_counts_output_file=vocab_path, min_word_count=config.prepro_min_word_count)
    vocab = tl.nlp.Vocabulary(vocab_path, start_word=START_ID, end_word=END_ID, unk_word=UNK_ID)

    for idx, text in enumerate(textlist):
        textlist[idx] = [vocab.word_to_id(word) for word in text]


    return textlist, vocab

def prepro_encode(textlist, vocab):
    for idx, text in enumerate(textlist):
        textlist[idx] = text[1:-1] + [vocab.pad_id]
    textlist = tl.prepro.pad_sequences(textlist, maxlen=150 - 1, dtype='int64', padding='post', truncating='post', value=vocab.pad_id)
    for idx, text in enumerate(textlist):
        textlist[idx] = text[:-1] + [vocab.pad_id]
    return np.array(textlist)

def prepro_decode(textlist, vocab):
    for idx, text in enumerate(textlist):
        textlist[idx] = text[:-1] + [vocab.pad_id]
    textlist = tl.prepro.pad_sequences(textlist, maxlen=150, dtype='int64', padding='post', truncating='post', value=vocab.pad_id)
    for idx, text in enumerate(textlist):
        textlist[idx] = text[:-1] + [vocab.pad_id]
    return np.array(textlist)

def prepro_target(textlist, vocab):
    for idx, text in enumerate(textlist):
        textlist[idx] = text[1:] + [vocab.pad_id]
    textlist = tl.prepro.pad_sequences(textlist, maxlen=150, dtype='int64', padding='post', truncating='post', value=vocab.pad_id)
    for idx, text in enumerate(textlist):
        if text[-1] != vocab.pad_id:
            textlist[idx] = text[:-2] + [vocab.end_id, vocab.pad_id]
    return np.array(textlist)

def load_data(filename, vocab_file, processed_file, column, force_process=False):
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

        full_text_list = df[column].tolist()
        # full_text_list = full_text_list[:2 * config.batch_size]
        full_text_list, vocab = preprocess(full_text_list, vocab_path=vocab_file)

        if processed_file.endswith(".pkl"):
            with open(processed_file, "wb") as f:
                pickle.dump(full_text_list, f)
        else:
            with open(processed_file, "w") as f:
                f.write(str(full_text_list))
        print("Processed data saved to %s" % processed_file)

    print("Data loaded: num of seqs %s" % len(full_text_list))
    return full_text_list, vocab


if __name__ == "__main__":
    # text_seqs, vocab = load_data(config.wiki_data_path, config.wiki_vocab_path, config.wiki_processed_path, column="text", force_process=True)

    text_seqs, vocab = load_data(config.arxiv_data_path, config.arxiv_vocab_path, config.arxiv_processed_path, column="abstract", force_process=True)

    length = np.array([len(text) for text in text_seqs])
    print(np.max(length))
    print(np.median(length))
    print(np.mean(length))
    print(np.min(length))

    # print(prepro_encode(text_seqs.copy(), vocab))
    # print(prepro_decode(text_seqs.copy(), vocab))
    # print(prepro_target(text_seqs.copy(), vocab))
    pass