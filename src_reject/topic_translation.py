import pickle, json, requests, csv, copy, os, re, sys, math, random
import numpy as np
import pprint as pp
import pandas as pd
import os.path
import gensim.downloader as api
from gensim.models import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
import config
import utils
import dataloader 
import progressbar
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
import language_check

maxInt = sys.maxsize
decrement = True

while decrement:
    decrement = False
    try:
        csv.field_size_limit(maxInt)
    except OverflowError:
        maxInt = int(maxInt/10)
        decrement = True

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


if not os.path.isfile(config.word_embed_gensim_file_path):
    _ = glove2word2vec(config.word_embed_file_path, config.word_embed_gensim_file_path)
glove_model = KeyedVectors.load_word2vec_format(config.word_embed_gensim_file_path)

tool = language_check.LanguageTool('en-US')

pos_dict = {'JJ': 'a', 'JJR': 'a', 'JJS': 'a',
           'NN': 'n', 'NNP': 'n', 'NNPS': 'n', 'NNS': 'n',
           'RB': 'r', 'RBR': 'r', 'RBS': 'r',
           'VB': 'v', 'VBD': 'v', 'VBG': 'v', 'VBN': 'v', 'VBP': 'v', 'VBZ': 'v'}

POS_OF_WORD = dict()
WORD_TOPIC_TRANSLATION = dict()


def pos_list_of(word):
    global POS_OF_WORD
    if word not in POS_OF_WORD:
        POS_OF_WORD[word] = [ss.pos() for ss in wn.synsets(word)]
    return POS_OF_WORD[word]

def word_list_translation(word, from_class, to_class):
    global WORD_TOPIC_TRANSLATION
    word = word.lower()
    key = from_class+'-'+to_class
    if key not in WORD_TOPIC_TRANSLATION:
        WORD_TOPIC_TRANSLATION[key] = dict()
    if word not in WORD_TOPIC_TRANSLATION[key]:
        WORD_TOPIC_TRANSLATION[key][word] = [x[0] for x in glove_model.most_similar_cosmul(positive=[to_class, word], negative=[from_class], topn = 20)]
    return WORD_TOPIC_TRANSLATION[key][word]

def topic_transfer(text, from_class, to_class):
    from_class = from_class.lower()
    to_class = to_class.lower()

    original_tokens = word_tokenize(text)
    pos_original_tokens = nltk.pos_tag(original_tokens)
    
    transferred_tokens = []
    replace_dict = dict()
    for token in pos_original_tokens:
        if token[0].lower() in stop_words or token[1] not in pos_dict or token[0].lower() not in glove_model.vocab:
            transferred_tokens.append(token[0])
        elif token[0].lower() in replace_dict:
            replacement = replace_dict[token[0].lower()]
            if token[0][0].lower() != token[0][0]: # Begins with an upper-case letter
                replacement = replacement[0].upper() + replacement[1:]
            transferred_tokens.append(replacement)
        else:
            candidates = word_list_translation(token[0].lower(), from_class, to_class)
            find_replacement = False
            for cand in candidates:
                if pos_dict[token[1]] in pos_list_of(cand) and cand not in replace_dict.values():
                    replacement = cand
                    replace_dict[token[0].lower()] = cand
                    
                    if token[0][0].lower() != token[0][0]: # Begins with an upper-case letter
                        replacement = replacement[0].upper() + replacement[1:]
                    transferred_tokens.append(replacement)

                    find_replacement = True
                    break
            if not find_replacement:
                transferred_tokens.append(token[0])

    ans_sentence = ' '.join(transferred_tokens)
    matches = tool.check(ans_sentence)
    ans_sentence = language_check.correct(ans_sentence, matches)
    
    return ans_sentence

def augment_train(class_label_path, train_augmented_path, train_path, nott):
    global POS_OF_WORD, WORD_TOPIC_TRANSLATION
    if os.path.isfile(config.POS_OF_WORD_path):
        POS_OF_WORD = pickle.load(open(config.POS_OF_WORD_path, "rb"))

    if os.path.isfile(config.WORD_TOPIC_TRANSLATION_path):
        WORD_TOPIC_TRANSLATION = pickle.load(open(config.WORD_TOPIC_TRANSLATION_path, "rb"))

    class_dict = dataloader.load_class_dict(
        class_file=class_label_path,
        class_code_column="ClassCode",
        class_name_column="ClassWord"
    )
    
    fieldnames = ['No.','from_class', 'to_class', 'text']
    csvwritefile = open(train_augmented_path, 'w', encoding="latin-1", newline='')
    writer = csv.DictWriter(csvwritefile, fieldnames=fieldnames)
    writer.writeheader()


    with open(train_path, encoding="latin-1") as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
        random.shuffle(rows)
        if nott is not None: # no. of texts to be translated
            rows = rows[:min(nott, len(rows))]
        count = 0
        with progressbar.ProgressBar(max_value=len(rows)) as bar:
            for idx, row in enumerate(rows):
                text = row['text']
                class_id = int(row['class'])
                class_name = class_dict[class_id]

                for cidx in class_dict:
                    if cidx != int(row['class']):
                        try:
                            writer.writerow({'No.':count, 'from_class': class_id, 'to_class': cidx, 'text':topic_transfer(text, from_class = class_name, to_class = class_dict[cidx])})
                            count += 1
                        except:
                            continue
                bar.update(idx)    
                if idx % 100 == 0:
                    pickle.dump(POS_OF_WORD, open(config.POS_OF_WORD_path, "wb"))
                    pickle.dump(WORD_TOPIC_TRANSLATION, open(config.WORD_TOPIC_TRANSLATION_path, "wb"))
    csvwritefile.close()

if __name__ == "__main__":
    print(config.dataset, config.args.nott)
    if config.dataset == "dbpedia":
        augment_train(config.zhang15_dbpedia_class_label_path, config.zhang15_dbpedia_train_augmented_aggregated_path, config.zhang15_dbpedia_train_path, config.args.nott)
    elif config.dataset == "20news":
        augment_train(config.news20_class_label_path, config.news20_train_augmented_aggregated_path, config.news20_train_path, config.args.nott)
    else:
        raise Exception("config.dataset %s not found" % config.dataset)
    pass
