

vocab_size = int(30000)
train_epoch = 100
batch_size = 64

embedding_dim = 512
hidden_dim = 512

# prepro_min_word_count = 5 # wiki
prepro_min_word_count = 100 # arxiv
prepro_max_sentence_length = 100


##################################

wiki_dir = "../data/wiki/"

wiki_full_data_path = wiki_dir + "simple_wiki_type.csv"

wiki_train_data_path = wiki_dir + "train-wiki.csv"
wiki_train_processed_path = wiki_dir + "processed_train_text.pkl"

wiki_test_data_path = wiki_dir + "test-wiki.csv"
wiki_test_processed_path = wiki_dir + "processed_test_text.pkl"

wiki_vocab_path = wiki_dir + "vocab.txt"

##################################

arxiv_dir = "../data/arxiv/"

arxiv_full_data_path = arxiv_dir + "arxiv-clean-formatted.csv"

arxiv_train_data_path = arxiv_dir + "train-arxiv.csv"
arxiv_train_processed_path = arxiv_dir + "processed_train_text.pkl"

arxiv_test_data_path = arxiv_dir + "test-arxiv.csv"
arxiv_test_processed_path = arxiv_dir + "processed_test_text.pkl"

arxiv_vocab_path = arxiv_dir + "vocab.txt"


