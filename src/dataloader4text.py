
import re
import pandas as pd
import tensorflow as tf
import tensorlayer as tl

data_filename = "../data/wiki/simple_wiki_2.csv"

START_ID = '<START_ID>'
END_ID = '<END_ID>'
PAD_ID = '<PAD_ID>'

def preprocess(textlist):
    for idx, text in enumerate(textlist):
        # textlist[idx].replace(",", " ")
        # textlist[idx].replace(".", " ")
        textlist[idx] = re.sub(r'[\W_]+', ' ', textlist[idx])
        textlist[idx] = textlist[idx].split() # no empty string in the list
    return textlist

def load_data(filename, column="text"):
    df = pd.read_csv(data_filename, index_col=0)
    full_text_list = df[column].tolist()
    full_text_list = full_text_list[:2]
    full_text_list = preprocess(full_text_list)

    full_encode_seqs = tl.prepro.sequences_add_end_id(full_text_list, end_id=PAD_ID)

    full_decode_seqs = tl.prepro.sequences_add_end_id(full_text_list, end_id=PAD_ID)
    full_decode_seqs = tl.prepro.sequences_add_start_id(full_decode_seqs, start_id=START_ID)

    full_target_seqs = tl.prepro.sequences_add_end_id(full_text_list, end_id=END_ID)
    full_target_seqs = tl.prepro.sequences_add_end_id(full_target_seqs, end_id=PAD_ID)

    #  TODO: prepro encode, decode, target, sequence
    full_mask_seqs = tl.prepro.sequences_get_mask(full_target_seqs, pad_val=PAD_ID)

    print(full_text_list)
    print(full_encode_seqs)
    print(full_decode_seqs)
    print(full_target_seqs)
    print(full_mask_seqs)

    return full_encode_seqs, full_decode_seqs, full_target_seqs, full_mask_seqs


if __name__ == "__main__":
    load_data(data_filename)
    pass