
import argparse

parser = argparse.ArgumentParser(description='configurations')
parser.add_argument("--data",  type=str, required=False, help="dataset: dbpedia or 20news")
parser.add_argument("--unseen", type=float, required=False, help="unseen rate: 0.25 0.5 0.75")
# parser.add_argument("--aug", type=int, required=False, help="augmentation: 0 4000 8000 12000 16000 20000")
parser.add_argument("--model", type=str, required=False, help="model: vwvcvkg vwvc vwvkg vcvkg kgonly cnnfc rnnfc")
parser.add_argument("--ns", type=int, default=2, required=False, help="negative samples: integer, the ratio of positive and negative samples, the higher the more negative samples")
parser.add_argument("--ni", type=int, default=2, required=False, help="negative increase: integer, the speed of increasing negative samples during training per epoch")
parser.add_argument("--sepoch", type=int, required=False, help="small epoch: integer, repeat training of each epoch for several times so that the ratio of posi/negative, learning rate both keep the same")
parser.add_argument("--rgidx", type=int, default=1, required=False, help="random group starting index: e.g. if 5, the training will start from the 5th random group, by default 1")
parser.add_argument("--train", type=int, required=False, help="train or not")
parser.add_argument("--gpu", type=float, default=1.0, required=False, help="gpu occupation percentage")
parser.add_argument("--baseepoch", type=int, required=False, help="base epoch for testing")
parser.add_argument("--fulltest", type=int, required=False, help="full test or not")
parser.add_argument("--threshold", type=float, required=False, help="threshold for seen")
parser.add_argument("--nott", type=int, required=False, help="no. of original texts to be translated")
args = parser.parse_args()
print(args)

global_gpu_occupation = args.gpu if args.gpu is not None else 1.0
global_is_train = bool(args.train)
global_full_test = bool(args.fulltest)
global_threshold_for_seen = args.threshold
print("GPU percentage %s" % global_gpu_occupation)
print("Training %s" % global_is_train)
global_test_base_epoch = args.baseepoch

##################################
# default setting
# subject to changes for cases

random_group_start_idx = args.rgidx

dataset = args.data
# dataset = "dbpedia"
# dataset = "20news"

vocab_size = int(30000)
train_epoch = 100
max_length = 50

batch_size = 32

# unseen_rate = 0.25
unseen_rate = args.unseen

augmentation = 0
assert augmentation >= 0

model = args.model
# model = "vwvcvkg"
# model = "vwvc"
# model = "vwvkg"
# model = "vcvkg"
# model = "kgonly"

negative_sample = args.ns
negative_increase = args.ni
small_epoch = args.sepoch
# dbpedia
# negative_sample = 5
# negative_increase = 3
# small_epoch = 2

# 20news
# negative_sample = 1
# negative_increase = 1
# small_epoch = 10

# chen14_elec amazon review
# negative_sample = 1
# negative_increase = 1
# small_epoch = 10

word_embedding_dim = 200
hidden_dim = 256

# TODO if updating kg_vector file, you may need to change this also
# kg_embedding_dim = 10 # kg_vector cluster allgroup
kg_embedding_dim = 30 # kg_vector cluster 3group
# kg_embedding_dim = 400 # kg_vector hop=2
# kg_embedding_dim = 8 # kg_vector hop=1

# prepro_min_word_count = 5 # wiki
prepro_min_word_count = 100 # arxiv
prepro_max_sentence_length = max_length

cstep_print = 100
cstep_print_unseen = 50

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
word_embed_gensim_file_path = '../data/glove/glove.6B.200d.gensim.txt'
conceptnet_path = "../data/conceptnet-assertions-en-5.6.0.csv"
POS_OF_WORD_path = "../data/POS_OF_WORD.pickle"
WORD_TOPIC_TRANSLATION_path = "../data/WORD_TOPIC_TRANSLATION.pickle"

# TODO by Peter: how to get these rejector files
if dataset == "dbpedia" and unseen_rate == 0.25:
    rejector_file = "./dbpedia_unseen0.25_augmented12000.pickle"
elif dataset == "dbpedia" and unseen_rate == 0.5:
    rejector_file = "./dbpedia_unseen0.50_augmented8000.pickle"
elif dataset == "20news" and unseen_rate == 0.25:
    rejector_file = "./20news_unseen0.25_augmented4000.pickle"
elif dataset == "20news" and unseen_rate == 0.5:
    rejector_file = "./20news_unseen0.50_augmented3000.pickle"
else:
    rejector_file = None


##################################

# wiki_dir = "../data/wiki/"

# wiki_full_data_path = wiki_dir + "simple_wiki_type.csv"

# wiki_train_data_path = wiki_dir + "train-wiki.csv"
# wiki_train_processed_path = wiki_dir + "processed_train_text.pkl"

# wiki_test_data_path = wiki_dir + "test-wiki.csv"
# wiki_test_processed_path = wiki_dir + "processed_test_text.pkl"

# wiki_vocab_path = wiki_dir + "vocab.txt"


# wiki_train_state_npz_path = wiki_dir + "train_wiki_state.npz"
# wiki_test_state_npz_path = wiki_dir + "test_wiki_state.npz"

##################################

# arxiv_dir = "../data/arxiv/"

# arxiv_full_data_path = arxiv_dir + "arxiv-clean-formatted.csv"

# arxiv_train_data_path = arxiv_dir + "train-arxiv.csv"
# arxiv_train_processed_path = arxiv_dir + "processed_train_text.pkl"

# arxiv_test_data_path = arxiv_dir + "test-arxiv.csv"
# arxiv_test_processed_path = arxiv_dir + "processed_test_text.pkl"

# arxiv_vocab_path = arxiv_dir + "vocab.txt"

# arxiv_train_state_npz_path = arxiv_dir + "train_arxiv_state.npz"
# arxiv_test_state_npz_path = arxiv_dir + "test_arxiv_state.npz"

##################################

zhang15_dir = "../data/zhang15/"

zhang15_dbpedia_dir = zhang15_dir + "dbpedia_csv/"

zhang15_dbpedia_full_data_path = zhang15_dbpedia_dir + "full.csv"

zhang15_dbpedia_train_path = zhang15_dbpedia_dir + "train.csv"
zhang15_dbpedia_train_processed_path = zhang15_dbpedia_dir + "processed_train_text.pkl"

zhang15_dbpedia_train_augmented_path = zhang15_dbpedia_dir + "train_augmented.csv"
zhang15_dbpedia_train_augmented_aggregated_path = zhang15_dbpedia_dir + "train_augmented_aggregated.csv"
zhang15_dbpedia_train_augmented_processed_path = zhang15_dbpedia_dir + "processed_train_augmented_text.pkl"

zhang15_dbpedia_test_path = zhang15_dbpedia_dir + "test.csv"
zhang15_dbpedia_test_processed_path = zhang15_dbpedia_dir + "processed_test_text.pkl"

zhang15_dbpedia_vocab_path = zhang15_dbpedia_dir + "vocab.txt"

# zhang15_dbpedia_train_state_npz_path = zhang15_dbpedia_dir + "train_zhang15_dbpedia_state.npz"
# zhang15_dbpedia_test_state_npz_path = zhang15_dbpedia_dir + "test_zhang15_dbpedia_state.npz"

zhang15_dbpedia_class_label_path = zhang15_dbpedia_dir + "classLabelsDBpedia.csv"
zhang15_dbpedia_class_random_group_path = zhang15_dbpedia_dir + "dbpedia_random_group_%s.txt" % unseen_rate
# zhang15_dbpedia_class_random_group_path = zhang15_dbpedia_dir + "dbpedia_random_group_0.5.txt"

# zhang15_dbpedia_kg_vector_train_processed_path = zhang15_dbpedia_dir + "kg_vector_train_processed.pkl"
# zhang15_dbpedia_kg_vector_test_processed_path = zhang15_dbpedia_dir + "kg_vector_test_processed.pkl"

# zhang15_dbpedia_kg_vector_dir = zhang15_dbpedia_dir + "KG_VECTOR_3/"
# zhang15_dbpedia_kg_vector_prefix = "KG_VECTORS_3_"
zhang15_dbpedia_kg_vector_node_data_path = zhang15_dbpedia_dir + 'NODES_DATA.pickle'
zhang15_dbpedia_kg_vector_dir = zhang15_dbpedia_dir + "KG_VECTOR_CLUSTER_3GROUP/"
zhang15_dbpedia_kg_vector_prefix = "VECTORS_CLUSTER_3_"

zhang15_dbpedia_word_embed_matrix_path = zhang15_dbpedia_dir + "word_embed_matrix.npz"

##################################

# zhang15_yahoo_dir = zhang15_dir + "yahoo_answers_csv/"

# zhang15_yahoo_full_data_path = zhang15_yahoo_dir + "full.csv"

# zhang15_yahoo_train_path = zhang15_yahoo_dir + "train.csv"
# zhang15_yahoo_train_processed_path = zhang15_yahoo_dir + "processed_train_text.pkl"

# zhang15_yahoo_test_path = zhang15_yahoo_dir + "test.csv"
# zhang15_yahoo_test_processed_path = zhang15_yahoo_dir + "processed_test_text.pkl"

# zhang15_yahoo_vocab_path = zhang15_yahoo_dir + "vocab.txt"

# zhang15_yahoo_train_state_npz_path = zhang15_yahoo_dir + "train_zhang15_yahoo_state.npz"
# zhang15_yahoo_test_state_npz_path = zhang15_yahoo_dir + "test_zhang15_yahoo_state.npz"


##################################

# chen14_dir = "../data/chen14/clean/"

# chen14_class_label_path = chen14_dir + "classLabelsChen14.csv"

# chen14_full_data_path = chen14_dir + "full.csv"

# chen14_train_path = chen14_dir + "train.csv"
# chen14_train_processed_path = chen14_dir + "processed_train_text.pkl"

# chen14_test_path = chen14_dir + "test.csv"
# chen14_test_processed_path = chen14_dir + "processed_test_text.pkl"

# chen14_vocab_path = chen14_dir + "vocab.txt"

# chen14_kg_vector_dir = chen14_dir + "KG_VECTOR_3/"
# chen14_kg_vector_prefix = "KG_VECTORS_3_"
# chen14_kg_vector_dir = chen14_dir + "KG_VECTOR_3_Lem/"
# chen14_kg_vector_prefix = "lemmatised_KG_VECTORS_3_"

# chen14_word_embed_matrix_path = chen14_dir + "word_embed_matrix.npz"

##################################

# chen14_elec_dir = "../data/chen14/clean_elec/"

# chen14_elec_class_label_path = chen14_elec_dir + "classLabelsChen14Elec.csv"
# chen14_elec_class_random_group_path = chen14_elec_dir + "chen14_elec_random_group_%s.txt" % unseen_rate

# chen14_elec_full_data_path = chen14_elec_dir + "full.csv"

# chen14_elec_train_path = chen14_elec_dir + "train.csv"
# chen14_elec_train_processed_path = chen14_elec_dir + "processed_train_text.pkl"

# chen14_elec_test_path = chen14_elec_dir + "test.csv"
# chen14_elec_test_processed_path = chen14_elec_dir + "processed_test_text.pkl"

# chen14_elec_vocab_path = chen14_elec_dir + "vocab.txt"

# chen14_kg_vector_dir = chen14_dir + "KG_VECTOR_3/"
# chen14_kg_vector_prefix = "KG_VECTORS_3_"
# chen14_elec_kg_vector_dir = chen14_elec_dir + "KG_VECTOR_3_Lem/"
# chen14_elec_kg_vector_prefix = "lemmatised_KG_VECTORS_3_elec_"
# chen14_elec_kg_vector_dir = chen14_elec_dir + "KG_VECTOR_CLUSTER_3GROUP/"
# chen14_elec_kg_vector_prefix = "VECTORS_CLUSTER_3_"

# chen14_elec_word_embed_matrix_path = chen14_elec_dir + "word_embed_matrix.npz"


##################################

news20_dir = "../data/20-newsgroups/clean/"

news20_class_label_path = news20_dir + "classLabels20news.csv"
news20_class_random_group_path = news20_dir + "20news_random_group_%s.txt" % unseen_rate
# news20_class_random_group_path = news20_dir + "20news_random_group_0.5.txt"

news20_full_data_path = news20_dir + "full.csv"

news20_train_path = news20_dir + "train.csv"
news20_train_processed_path = news20_dir + "processed_train_text.pkl"

news20_test_path = news20_dir + "test.csv"
news20_test_processed_path = news20_dir + "processed_test_text.pkl"

news20_train_augmented_path = news20_dir + "train_augmented.csv"
news20_train_augmented_aggregated_path = news20_dir + "train_augmented_aggregated.csv"
news20_train_augmented_processed_path = news20_dir + "processed_train_augmented_text.pkl"


news20_vocab_path = news20_dir + "vocab.txt"

news20_kg_vector_node_data_path = news20_dir + 'NODES_DATA.pickle'
news20_kg_vector_dir = news20_dir + "KG_VECTOR_CLUSTER_3GROUP/"
news20_kg_vector_prefix = "VECTORS_CLUSTER_3_"

news20_word_embed_matrix_path = news20_dir + "word_embed_matrix.npz"

# news20_class_cluster_path = news20_dir + "class_clusters_20news.pickle"

