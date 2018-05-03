import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np
import time
import csv

# --------------------- Global Variables ----------------------
batch_size = 50
v_t_dim = 300
v_c_dim = 300
lr = 0.0001 # Learning rate for Adam optimizer
n_epoch = 500

# --------------------- Model ---------------------------------
g = tf.Graph()
with g.as_default() as graph:
	v_t = tf.placeholder(dtype = tf.float32, shape = [None, v_t_dim], name = "text_vectors")
	v_c = tf.placeholder(dtype = tf.float32, shape = [None, v_c_dim], name = "class_vectors")
	y = tf.placeholder(dtype = tf.int64, shape = [None, None], name = "answer_vectors")
	M = tf.Variable(tf.random_uniform([v_t_dim, v_c_dim], dtype=tf.float32), name = "bilinear_matrix")
	h = tf.matmul(tf.matmul(v_t, M) , tf.transpose(v_c))
	prob_sigmoid = tf.sigmoid(h)
	predicted_answer = tf.round(prob_sigmoid)

# --------------------- Optimizer -----------------------------
with g.as_default() as graph:
	loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = y, logits = h)
	train_op = tf.train.AdamOptimizer(learning_rate = lr).minimize(loss)
	print_all_variables(train_only=True)
	sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
	tl.layers.initialize_global_variables(sess)

# --------------------- Setup model ---------------------------
def load_model_parameter(filename, path = ''):
	with g.as_default() as graph:
		c = tl.files.load_npz(path = path, name = filename)
		op_assign = M.assign(c[0])
		sess.run(op_assign)

# --------------------- Training ------------------------------
def train_model(V_T_train, V_C_train, Y_train, dataset_name):
	with g.as_default() as graph:
		n_step = int(len(V_T_train)/batch_size)
		for epoch in range(n_epoch):
			epoch_time = time.time()

			# Train an epoch
			total_err, n_iter = 0, 0
			for X, Y in tl.iterate.minibatches(inputs = V_T_train, targets = Y_train, batch_size = batch_size, shuffle = True):
				step_time = time.time()
				_, err = sess.run([train_op, loss],
								{v_t: X,
								v_c: V_C_train,
								y: Y})

				if n_iter % 10 == 0:
					print("Epoch[%d/%d] step:[%d/%d] loss:%f took:%.5fs" % (epoch, n_epoch, n_iter, n_step, err, time.time() - step_time))
				
				total_err += err
				n_iter += 1
				
			print("Epoch[%d/%d] averaged loss:%f took:%.5fs" % (epoch, n_epoch, total_err/n_iter, time.time()-epoch_time))
			
			# Save trained parameters after running each epoch
			param = get_variables_with_name(name='bilinear_matrix:0')
			tl.files.save_npz(param, name='bilinear_matrix_' + dataset_name + '.npz', sess=sess)

# --------------------- Testing -------------------------------
def predict(V_T_test, V_C_test):
	with g.as_default() as graph:
		h_predict = sess.run(predicted_answer, {v_t: V_T_test, v_c: V_C_test})
		return h_predict

def test_model(V_T_test, V_C_test, Y_test, dataset_name):
	h_predict = predict(V_T_test, V_C_test)
	stats = get_statistics(h_predict, Y_test)
	print(stats)
	return stats

def get_statistics(prediction, ground_truth):
	assert prediction.shape == ground_truth.shape
	num_instance = prediction.shape[0]
	num_class = prediction.shape[1]

	# Accuracy
	accuracy = np.sum(prediction == ground_truth) / (num_instance*num_class)

	# Micro-average
	microP, microR, microF1 = get_precision_recall_f1(np.ravel(prediction), np.ravel(ground_truth))

	# Macro-average
	precisionList = []
	recallList = []
	for j in range(num_class): # Calculate Precision and Recall for class j
		p, r, _ = get_precision_recall_f1(prediction[:,j], ground_truth[:,j]) 
		precisionList.append(p)
		recallList.append(r)
	macroP = np.mean(np.array(precisionList))
	macroR = np.mean(np.array(recallList))
	macroF1 = 2 * macroP * macroR / (macroP + macroR)

	# Return stats results
	stats = {'accuracy': accuracy,
			'micro-precision': microP,
			'micro-recall': microR,
			'micro-F1': microF1,
			'macro-precision': macroP,
			'macro-recall': macroR,
			'macro-F1': macroF1,}
	return stats

def get_precision_recall_f1(prediction, ground_truth): # 1D data
	assert prediction.shape == ground_truth.shape and prediction.ndim == 1
	TP, FP, FN = 0, 0, 0
	for i in range(len(prediction)):
		if prediction[i] == 1 and ground_truth[i] == 1:
			TP += 1
		elif prediction[i] == 1 and ground_truth[i] == 0:
			FP += 1
		elif prediction[i] == 0 and ground_truth[i] == 1:
			FN += 1
	P = TP / (TP + FP)
	R = TP / (TP + FN)
	F1 = 2 * P * R / (P + R)
	return P, R, F1

# --------------------- Cross Validation ----------------------
def load_dataset(dataset_name, knowledge_graph):
	# requires class embeddings and text embeddings
	if dataset_name == 'arxiv':
		train_data, header = read_CSV_rows('../data/arxiv/train-arxiv.csv', have_header = True)
		Y_train_all = np.array([row[5:] for row in train_data])
		V_T_train = np.array([text_to_vector(row[2]) for row in train_data]) # row[2] = title, row[3] = abstract

		test_data, header = read_CSV_rows('../data/arxiv/test-arxiv.csv', have_header = True)
		Y_test_all = np.array([row[5:] for row in test_data])
		V_T_test = np.array([text_to_vector(row[2]) for row in test_data]) # row[2] = title, row[3] = abstract
		
		classCodes = header[5:]
		classList = read_CSV_dict('../data/arxiv/classLabelsWithManualLinking.csv') 
		if knowledge_graph == 'DBpedia':
			V_C_all = np.array([get_vector_by_uri('DBpedia', row['DBpediaManual']) for row in classList])
		elif knowledge_graph == 'ConceptNet':
			class_labels = [row['ClassLabel'].strip() for row in classList]
			V_C_all = np.array([get_vector_of_class(c, '', 'ConceptNet', corpus = class_labels)[1] for c in class_labels]) 
		else:
			assert False, "Unsupported knowledge_graph"
		return V_T_train, Y_train_all, V_T_test, Y_test_all, V_C_all, classCodes 

	elif dataset_name == 'wiki':
		train_data, header = read_CSV_rows('../data/wiki/train-wiki.csv', have_header = True)
		Y_train_all = np.array([row[5:] for row in train_data])
		V_T_train = np.array([text_to_vector(row[2]) for row in train_data]) # row[2] = abstract

		test_data, header = read_CSV_rows('../data/wiki/test-wiki.csv', have_header = True)
		Y_test_all = np.array([row[5:] for row in test_data])
		V_T_test = np.array([text_to_vector(row[2]) for row in test_data]) # row[2] = abstract
		
		classCodes = header[5:]
		classList = read_CSV_dict('../data/wiki/classLabelsWiki.csv') 
		if knowledge_graph == 'DBpedia':
			V_C_all = np.array([get_vector_by_uri('DBpedia', row['DBpediaManual']) for row in classList])
		elif knowledge_graph == 'ConceptNet':
			class_labels = [row['ClassLabel'].strip() for row in classList]
			V_C_all = np.array([get_vector_of_class(c, '', 'ConceptNet', corpus = class_labels)[1] for c in class_labels]) 
		else:
			assert False, "Unsupported knowledge_graph"
		return V_T_train, Y_train_all, V_T_test, Y_test_all, V_C_all, classCodes 

	else:
		assert False, "Unsupported dataset_name"

def cross_class_validation():
	pass

# --------------------- Helper functions ----------------------
def get_pseudo_data(num_instance = 1000, num_class = 10, with_answer = True):
	positive_rate = 0.3
	V_T = np.random.rand(num_instance, v_t_dim)
	V_C = np.random.rand(num_class, v_c_dim)
	if with_answer: 
		Y_train = np.array([[1 if np.random.rand(1) < positive_rate else 0 for j in range(num_class)] for i in range(num_instance)])
		return V_T, V_C, Y_train
	else:
		return V_T, V_C

def read_CSV_dict(filename):
	input_file = csv.DictReader(open(filename, encoding = "utf8"))
	return [row for row in input_file]

def read_CSV_rows(filename, have_header = False):
	with open(filename, 'r') as csvfile:
		lines = csv.reader(csvfile)
		results = []
		header = None
		for i, line in enumerate(lines):
			if have_header and i == 0:
				header = line
			else:
				results.append(line)
	return results, header

# --------------------- Main Operation ------------------------
if __name__ == "__main__":
	# V_T_train, V_C_train, Y_train = get_pseudo_data()
	# train_model(V_T_train, V_C_train, Y_train, 'pseudo')
	# V_T_test, V_C_test = get_pseudo_data(num_instance = 20, num_class = 5, with_answer = False)
	# print(predict(V_T_test, V_C_test))
	# test_model(V_T_train, V_C_train, Y_train, 'pseudo') # Test with training data
	# load_model_parameter('bilinear_matrix_pseudo.npz')
	# V_T_test, V_C_test, Y_test = get_pseudo_data()
	# test_model(V_T_test, V_C_test, Y_test, 'pseudo') # Test with testing data
	# load_dataset('arxiv')
	pass