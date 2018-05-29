# all code here are just for fun

import numpy as np
import pandas as pd

import config

def combine_zhang15_dbpedia_train_test():
    # df_train = pd.read_csv(config.zhang15_dbpedia_train_path, names=["class", "title", "text"])
    # df_test = pd.read_csv(config.zhang15_dbpedia_test_path, names=["class", "title", "text"])
    df_train = pd.read_csv(config.zhang15_dbpedia_train_path, index_col=0)
    df_test = pd.read_csv(config.zhang15_dbpedia_test_path, index_col=0)
    # df_train.to_csv(config.zhang15_dbpedia_train_path)
    # df_test.to_csv(config.zhang15_dbpedia_test_path)
    df_full = pd.concat([df_train, df_test], ignore_index=True)
    df_full.to_csv(config.zhang15_dbpedia_full_data_path)

def combine_zhang15_yahoo_train_test():
    # df_train = pd.read_csv(config.zhang15_yahoo_train_path, names=["class", "question_title", "question_content", "best_answer"])
    # df_test = pd.read_csv(config.zhang15_yahoo_test_path, names=["class", "question_title", "question_content", "best_answer"])
    # df_train = pd.read_csv(config.zhang15_yahoo_train_path, index_col=0)
    df_test = pd.read_csv(config.zhang15_yahoo_test_path, index_col=0)
    # df_train.to_csv(config.zhang15_yahoo_train_path)
    # df_test.to_csv(config.zhang15_yahoo_test_path)
    # df_full = pd.concat([df_train, df_test], ignore_index=True)
    # df_full.to_csv(config.zhang15_yahoo_full_data_path)
    # print(df_test[["question_title", "question_content", "best_answer"]])
    # df_test["text"] = df_test[["question_title", "question_content", "best_answer"]].apply(lambda x: ' '.join([item if type(item) == str else ' ' for item in x]), axis=1)
    # print(df_test)
    # print(df_test["text"])


def load_wiki_npz():
    data = np.load(config.wiki_train_state_npz_path)
    print(data["state"].shape)
    data = np.load(config.wiki_test_state_npz_path)
    print(data["state"].shape)

def load_zhang15_dbpedia_npz():
    data = np.load(config.zhang15_dbpedia_train_state_npz_path)
    print(data["state"].shape)
    data = np.load(config.zhang15_dbpedia_test_state_npz_path)
    print(data["state"].shape)

def load_logs():
    data = np.load("../results/key_zhang15_dbpedia_4of4/logs/test_1_kg.npz")
    print(data["kg_vector_seen"].shape)

if __name__ == "__main__":
    # combine_zhang15_dbpedia_train_test()
    # load_zhang15_dbpedia_npz()
    # combine_zhang15_yahoo_train_test()
    load_logs()
    pass
