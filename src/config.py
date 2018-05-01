

vocab_size = int(60e3)
train_epoch = 100
batch_size = 64


# prepro_min_word_count = 5 # wiki
prepro_min_word_count = 20 # arxiv


wiki_dir = "../data/wiki/"

wiki_data_path = wiki_dir + "simple_wiki_2.csv"
wiki_vocab_path = wiki_dir + "vocab.txt"
wiki_processed_path = wiki_dir + "processed_text.pkl"

arxiv_dir = "../data/arxiv/"
arxiv_data_path = arxiv_dir + "arxiv-clean-formatted.csv"
arxiv_vocab_path = arxiv_dir + "vocab.txt"
arxiv_processed_path = arxiv_dir + "processed_text.pkl"

