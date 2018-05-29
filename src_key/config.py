
##################################
# default setting
# subject to changes for cases

vocab_size = int(30000)
train_epoch = 100
batch_size = 32
max_length = 50

negative_sample = 9

word_embedding_dim = 200
hidden_dim = 256

kg_embedding_dim = 400 # kg_vector hop=2
# kg_embedding_dim = 8 # kg_vector hop=1

# prepro_min_word_count = 5 # wiki
prepro_min_word_count = 100 # arxiv
prepro_max_sentence_length = max_length

cstep_print = 500

##################################


pos_dict = {'JJ': 'a', 'JJR': 'a', 'JJS': 'a',

           'NN': 'n', 'NNP': 'n', 'NNPS': 'n', 'NNS': 'n',

           'RB': 'r', 'RBR': 'r', 'RBS': 'r',

           'VB': 'v', 'VBD': 'v', 'VBG': 'v', 'VBN': 'v', 'VBP': 'v', 'VBZ': 'v'}


##################################

# kg_vector_dir = "../wordEmbeddings/"

# kg_vector_data_path = kg_vector_dir + "KG_VECTORS_1.pickle"
# kg_vector_data_path = kg_vector_dir + "KG_VECTORS_2.pickle"

word_embed_file_path = "../data/glove/glove.6B.200d.txt"


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

zhang15_dbpedia_class_label_path = zhang15_dbpedia_dir + "classLabelsDBpedia.csv"

zhang15_dbpedia_kg_vector_train_processed_path = zhang15_dbpedia_dir + "kg_vector_train_processed.pkl"
zhang15_dbpedia_kg_vector_test_processed_path = zhang15_dbpedia_dir + "kg_vector_test_processed.pkl"

zhang15_dbpedia_kg_vector_dir = zhang15_dbpedia_dir + "KG_VECTOR_3/"
zhang15_dbpedia_kg_vector_prefix = "KG_VECTORS_3_"

zhang15_dbpedia_word_embed_matrix_path = zhang15_dbpedia_dir + "word_embed_matrix.npz"

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


##################################

chen14_dir = "../data/chen14/clean/"

chen14_class_label_path = chen14_dir + "classLabelsChen14.csv"

chen14_full_data_path = chen14_dir + "full.csv"

chen14_train_path = chen14_dir + "train.csv"
chen14_train_processed_path = chen14_dir + "processed_train_text.pkl"

chen14_test_path = chen14_dir + "test.csv"
chen14_test_processed_path = chen14_dir + "processed_test_text.pkl"

chen14_vocab_path = chen14_dir + "vocab.txt"

# chen14_kg_vector_dir = chen14_dir + "KG_VECTOR_3/"
# chen14_kg_vector_prefix = "KG_VECTORS_3_"
chen14_kg_vector_dir = chen14_dir + "KG_VECTOR_3_Lem/"
chen14_kg_vector_prefix = "lemmatised_KG_VECTORS_3_"

chen14_word_embed_matrix_path = chen14_dir + "word_embed_matrix.npz"

##################################

news20_dir = "../data/20-newsgroups/clean/"

news20_class_label_path = news20_dir + "classLabels20news.csv"

news20_full_data_path = news20_dir + "full.csv"

news20_train_path = news20_dir + "train.csv"
news20_train_processed_path = news20_dir + "processed_train_text.pkl"

news20_test_path = news20_dir + "test.csv"
news20_test_processed_path = news20_dir + "processed_test_text.pkl"

news20_vocab_path = news20_dir + "vocab.txt"

news20_kg_vector_dir = news20_dir + "KG_VECTOR_3_Lem/"
news20_kg_vector_prefix = "lemmatised_KG_VECTORS_3_"

news20_word_embed_matrix_path = news20_dir + "word_embed_matrix.npz"
