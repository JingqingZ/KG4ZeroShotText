
##################################
# default setting
# subject to changes for cases

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


wiki_train_state_npz_path = wiki_dir + "train_wiki_state.npz"
wiki_test_state_npz_path = wiki_dir + "test_wiki_state.npz"

##################################

arxiv_dir = "../data/arxiv/"

arxiv_full_data_path = arxiv_dir + "arxiv-clean-formatted.csv"

arxiv_train_data_path = arxiv_dir + "train-arxiv.csv"
arxiv_train_processed_path = arxiv_dir + "processed_train_text.pkl"

arxiv_test_data_path = arxiv_dir + "test-arxiv.csv"
arxiv_test_processed_path = arxiv_dir + "processed_test_text.pkl"

arxiv_vocab_path = arxiv_dir + "vocab.txt"

arxiv_train_state_npz_path = arxiv_dir + "train_arxiv_state.npz"
arxiv_test_state_npz_path = arxiv_dir + "test_arxiv_state.npz"

##################################

zhang15_dir = "../data/zhang15/"

zhang15_dbpedia_dir = zhang15_dir + "dbpedia_csv/"

zhang15_dbpedia_full_data_path = zhang15_dbpedia_dir + "full.csv"

zhang15_dbpedia_train_path = zhang15_dbpedia_dir + "train.csv"
zhang15_dbpedia_train_processed_path = zhang15_dbpedia_dir + "processed_train_text.pkl"

zhang15_dbpedia_test_path = zhang15_dbpedia_dir + "test.csv"
zhang15_dbpedia_test_processed_path = zhang15_dbpedia_dir + "processed_test_text.pkl"

zhang15_dbpedia_vocab_path = zhang15_dbpedia_dir + "vocab.txt"

zhang15_dbpedia_train_state_npz_path = zhang15_dbpedia_dir + "train_zhang15_dbpedia_state.npz"
zhang15_dbpedia_test_state_npz_path = zhang15_dbpedia_dir + "test_zhang15_dbpedia_state.npz"

##################################

zhang15_yahoo_dir = zhang15_dir + "yahoo_answers_csv/"

zhang15_yahoo_full_data_path = zhang15_yahoo_dir + "full.csv"

zhang15_yahoo_train_path = zhang15_yahoo_dir + "train.csv"
zhang15_yahoo_train_processed_path = zhang15_yahoo_dir + "processed_train_text.pkl"

zhang15_yahoo_test_path = zhang15_yahoo_dir + "test.csv"
zhang15_yahoo_test_processed_path = zhang15_yahoo_dir + "processed_test_text.pkl"

zhang15_yahoo_vocab_path = zhang15_yahoo_dir + "vocab.txt"

zhang15_yahoo_train_state_npz_path = zhang15_yahoo_dir + "train_zhang15_yahoo_state.npz"
zhang15_yahoo_test_state_npz_path = zhang15_yahoo_dir + "test_zhang15_yahoo_state.npz"
