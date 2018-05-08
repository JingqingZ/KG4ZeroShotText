import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np
import time
import csv

from class_embeddings import *

# --------------------- Global Variables ----------------------
dataset_name = 'dbpedia'
knowledge_graph = 'DBpedia'
model = 'bilinear deep'
training_method = 'simple' # simple, devise, l2-regularisation
single_label = True
beta = 0.00001 # l2-regularisation factor
n_fold = 4
v_c_dim_of = {'ConceptNet': 300, 'DBpedia': 100}
batch_size = 50
v_t_dim = 512
v_c_dim = v_c_dim_of[knowledge_graph] # ConceptNet = 300, DBpedia = 100
lr = 0.0001 # Learning rate for Adam optimizer
n_epoch = 20

# --------------------- Model ---------------------------------
g = tf.Graph()
with g.as_default() as graph:
	v_t = tf.placeholder(dtype = tf.float32, shape = [None, v_t_dim], name = "text_vectors")
	v_c = tf.placeholder(dtype = tf.float32, shape = [None, v_c_dim], name = "class_vectors")
	y = tf.placeholder(dtype = tf.int64, shape = [None, None], name = "answer_vectors")
	if 'deep' in model:
		network = tl.layers.InputLayer(v_t, name='input_layer')
		network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1')
		network = tl.layers.DenseLayer(network, n_units=800, act = tf.nn.tanh, name='tanh')
		network = tl.layers.DropoutLayer(network, keep=0.8, name='drop2')
		network = tl.layers.DenseLayer(network, n_units=v_t_dim, act = tl.activation.identity, name='output_v_t_layer')
		transformed_v_t = network.outputs
		print(network.all_params)
	M = tf.Variable(tf.random_uniform([v_t_dim, v_c_dim], dtype=tf.float32), name = "bilinear_matrix")
	if 'deep' in model:
		h = tf.matmul(tf.matmul(transformed_v_t, M) , tf.transpose(v_c))
	else:
		h = tf.matmul(tf.matmul(v_t, M) , tf.transpose(v_c))
	prob_sigmoid = tf.sigmoid(h)
	predicted_answer = tf.round(prob_sigmoid)
	single_label_answer = tf.argmax(prob_sigmoid, axis = 1)

# --------------------- Optimizer -----------------------------
with g.as_default() as graph:
	if 'simple' in training_method:
		loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = y, logits = h)
	elif 'devise' in training_method:
		label = tf.to_float(y)
		masked_sigmoid = tf.concat([tf.multiply(label, prob_sigmoid), tf.ones([tf.shape(label)[0], 1])], axis = 1)
		avg_prob_sigmoid_label = tf.expand_dims(tf.divide(tf.reduce_sum(masked_sigmoid, 1), tf.count_nonzero(label, axis = 1, dtype=tf.float32) + 1.0), 1)
		ones_vector = tf.ones([1, tf.shape(label)[1]])
		ones_matrix = tf.ones_like(label)
		inside_relu = ones_matrix - tf.matmul(avg_prob_sigmoid_label, ones_vector) + prob_sigmoid
		inside_relu = tf.multiply(inside_relu, ones_matrix - label)
		loss = tf.reduce_mean(tf.nn.relu(inside_relu))
	if 'l2-regularisation' in training_method:
		regularizer = beta * tf.nn.l2_loss(M)
		if 'deep' in model:
			regularizer = regularizer + tf.contrib.layers.l2_regularizer(beta)(network.all_params[0]) + tf.contrib.layers.l2_regularizer(beta)(network.all_params[2])
		loss = tf.reduce_mean(loss + regularizer)

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

def reset_model_parameter():
	with g.as_default() as graph:
		c = tf.random_uniform([v_t_dim, v_c_dim], dtype=tf.float32)
		op_assign = M.assign(c)
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
				feed_dict = {v_t: X, v_c: V_C_train, y: Y}
				feed_dict.update(network.all_drop) # Enable dropout
				_, err = sess.run([train_op, loss], feed_dict)

				if n_iter % 10 == 0:
					print("Epoch[%d/%d] step:[%d/%d] loss:%f took:%.5fs" % (epoch, n_epoch, n_iter, n_step, err, time.time() - step_time))
				
				total_err += err
				n_iter += 1
				
			print("Epoch[%d/%d] averaged loss:%f took:%.5fs" % (epoch, n_epoch, total_err/n_iter, time.time()-epoch_time))
			
			# Save trained parameters after running each epoch
			param = get_variables_with_name(name='bilinear_matrix:0')
			tl.files.save_npz(param, name='bilinear_matrix_' + dataset_name + '.npz', sess=sess)
			if 'deep' in model:
				tl.files.save_npz(network.all_params, name='network_' + dataset_name + '.npz', sess=sess)

# --------------------- Testing -------------------------------
def predict(V_T_test, V_C_test):
	with g.as_default() as graph:
		# Disable dropout
		dp_dict = tl.utils.dict_to_one(network.all_drop)
		feed_dict = {v_t: V_T_test, v_c: V_C_test}
		feed_dict.update(dp_dict)

		h_predict = sess.run(predicted_answer, feed_dict)
		return h_predict

def single_label_predict(V_T_test, V_C_test):
	with g.as_default() as graph:
		# Disable dropout
		dp_dict = tl.utils.dict_to_one(network.all_drop)
		feed_dict = {v_t: V_T_test, v_c: V_C_test}
		feed_dict.update(dp_dict)

		ans = sess.run(single_label_answer, {v_t: V_T_test, v_c: V_C_test})
		return ans

def test_model(V_T_test, V_C_test, Y_test, dataset_name, single_label = False):
	h_predict = predict(V_T_test, V_C_test)
	single_label_pred = None
	if single_label:
		single_label_pred = single_label_predict(V_T_test, V_C_test)
	stats = get_statistics(h_predict, Y_test, single_label_pred = single_label_pred)
	print(stats)
	return stats

def get_statistics(prediction, ground_truth, single_label_pred = None):
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
		if p is not None:
			precisionList.append(p)
		if r is not None:
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

	if single_label_pred is not None:
		single_label_ground_truth = np.argmax(ground_truth, axis = 1)
		single_label_error = np.mean(single_label_pred != single_label_ground_truth)
		stats['single-label-error'] = single_label_error
	
	return stats

def get_precision_recall_f1(prediction, ground_truth): # 1D data
	assert prediction.shape == ground_truth.shape and prediction.ndim == 1
	# print(prediction)
	# print(ground_truth)
	TP, FP, FN = 0, 0, 0
	for i in range(len(prediction)):
		if prediction[i] == 1 and ground_truth[i] == 1:
			TP += 1
		elif prediction[i] == 1 and ground_truth[i] == 0:
			FP += 1
		elif prediction[i] == 0 and ground_truth[i] == 1:
			FN += 1
	if (TP, FP, FN) == (0, 0, 0):
		return None, None, None
	P = TP / (TP + FP) if TP + FP != 0 else 0
	R = TP / (TP + FN) if TP + FN != 0 else 0
	F1 = 2 * P * R / (P + R) if P + R != 0 else 0
	return P, R, F1

# --------------------- Cross Validation ----------------------
def load_dataset(dataset_name, knowledge_graph):
	# requires class embeddings and text embeddings
	if dataset_name == 'arxiv':
		train_data, header = read_CSV_rows('../data/arxiv/train-arxiv.csv', have_header = True)
		Y_train_all = np.array([[float(x) for x in row[5:]] for row in train_data])
		# V_T_train = np.array([text_to_vector(row[2]) for row in train_data]) # row[2] = title, row[3] = abstract
		V_T_train = np.load('../data/arxiv/train_arxiv_state.npz')['state']

		test_data, header = read_CSV_rows('../data/arxiv/test-arxiv.csv', have_header = True)
		Y_test_all = np.array([[float(x) for x in row[5:]] for row in test_data])
		# V_T_test = np.array([text_to_vector(row[2]) for row in test_data]) # row[2] = title, row[3] = abstract
		V_T_test = np.load('../data/arxiv/test_arxiv_state.npz')['state']
		
		classCodes = header[5:]
		classList = read_CSV_dict('../data/arxiv/classLabelsWithManualLinking.csv') 
		if knowledge_graph == 'DBpedia':
			# V_C_all = np.array([get_vector_by_uri('DBpedia', row['DBpediaManual']) for row in classList])
			V_C_all = np.load('../data/arxiv/V_C_arxiv_DBpedia.npz')['arr_0']
		elif knowledge_graph == 'ConceptNet':
			class_labels = [row['ClassLabel'].strip() for row in classList]
			# V_C_all = np.array([get_vector_of_class(c, '', 'ConceptNet', corpus = class_labels)[1] for c in class_labels]) 
			V_C_all = np.load('../data/arxiv/V_C_arxiv_ConceptNet.npz')['arr_0']
		else:
			assert False, "Unsupported knowledge_graph"
		return V_T_train, Y_train_all, V_T_test, Y_test_all, V_C_all, classList 

	elif dataset_name == 'wiki':
		train_data, header = read_CSV_rows('../data/wiki/train-wiki.csv', have_header = True)
		Y_train_all = np.array([[float(x) for x in row[5:]] for row in train_data])
		# V_T_train = np.array([text_to_vector(row[2]) for row in train_data]) # row[2] = abstract
		V_T_train = np.load('../data/wiki/train_wiki_state.npz')['state']

		test_data, header = read_CSV_rows('../data/wiki/test-wiki.csv', have_header = True)
		Y_test_all = np.array([[float(x) for x in row[5:]] for row in test_data])
		# V_T_test = np.array([text_to_vector(row[2]) for row in test_data]) # row[2] = abstract
		V_T_test = np.load('../data/wiki/test_wiki_state.npz')['state']

		classCodes = header[5:]
		classList = read_CSV_dict('../data/wiki/classLabelsWiki.csv') 
		if knowledge_graph == 'DBpedia':
			# V_C_all = np.array([get_vector_by_uri('DBpedia', row['DBpediaManual']) for row in classList])
			V_C_all = np.load('../data/wiki/V_C_wiki_DBpedia.npz')['arr_0']
			# print(V_C_all)
		elif knowledge_graph == 'ConceptNet':
			class_labels = [row['ClassLabel'].strip() for row in classList]
			# V_C_all = np.array([get_vector_of_class(c, '', 'ConceptNet', corpus = class_labels)[1] for c in class_labels]) 
			V_C_all = np.load('../data/wiki/V_C_wiki_ConceptNet.npz')['arr_0']
		else:
			assert False, "Unsupported knowledge_graph"
		return V_T_train, Y_train_all, V_T_test, Y_test_all, V_C_all, classList  

	elif dataset_name == 'dbpedia':
		V_T_train = np.load('../data/dbpedia/train_zhang15_dbpedia_state.npz')['state']
		V_T_test = np.load('../data/dbpedia/test_zhang15_dbpedia_state.npz')['state']
		Y_train_all = np.array([[1 if i//40000 == j else 0 for j in range(14)] for i in range(560000)])
		Y_test_all = np.array([[1 if i//5000 == j else 0 for j in range(14)] for i in range(70000)])
		classList = read_CSV_dict('../data/dbpedia/classLabelsDBpedia.csv') 
		if knowledge_graph == 'DBpedia':
			V_C_all = np.load('../data/dbpedia/V_C_dbpedia_DBpedia.npz')['arr_0']
		elif knowledge_graph == 'ConceptNet':
			V_C_all = np.load('../data/dbpedia/V_C_dbpedia_ConceptNet.npz')['arr_0']
		else:
			assert False, "Unsupported knowledge_graph"
		return V_T_train, Y_train_all, V_T_test, Y_test_all, V_C_all, classList  

	elif dataset_name == 'pseudo':
		V_T, V_C, Y = get_pseudo_data(num_instance = 1000, num_class = 20, with_answer = True)
		V_T_train = V_T[:800]
		V_T_test = V_T[800:]
		Y_train_all = Y[:800]
		Y_test_all = Y[800:]
		V_C_all = V_C
		classList = []
		for i in range(20):
			classList.append({
				'ClassCode': 'A'+str(i),
				'ClassLabel': 'A'+str(i),
				'ClassDescription': '',
				'DBpediaManual': None,
				'ConceptNet': None,
				'Count': np.sum(Y[:, i])
				})
		return V_T_train, Y_train_all, V_T_test, Y_test_all, V_C_all, classList  

	else:
		assert False, "Unsupported dataset_name"

def cross_class_validation(V_T_train, Y_train_all, V_T_test, Y_test_all, V_C_all, classList, dataset_name, n_fold = 4, single_label = False):
	# print(classList)
	tempClasses = [(row['ClassCode'], row['Count']) for row in classList] 
	sortedClasses = [pair[0] for pair in sorted(tempClasses, reverse = True, key = lambda x: x[1])] # Sort class codes by counts
	rank = dict()
	for row in classList:
		rank[row['ClassCode']] = sortedClasses.index(row['ClassCode'])
	
	statseen = []
	statunseen = []
	stats = []
	for fold in range(n_fold):
		reset_model_parameter()
		train_this_fold = []
		unseen_this_fold = []
		for i in range(len(classList)):
			if rank[classList[i]['ClassCode']] % (2*n_fold) not in [fold, 2*n_fold-fold-1]:
				train_this_fold.append(i)
			else:
				unseen_this_fold.append(i)
		Y_train = Y_train_all[:, tuple(train_this_fold)]

		V_C_train = V_C_all[tuple(train_this_fold), :]
		train_model(V_T_train, V_C_train, Y_train, dataset_name + str(fold))	
		stats.append(test_model(V_T_test, V_C_all, Y_test_all, dataset_name + str(fold), single_label)) # generalised zero-shot

		V_C_seen = V_C_train
		Y_test_seen = Y_test_all[:, tuple(train_this_fold)]
		V_T_test_seen = V_T_test
		if single_label:
			positive_seen = [i for i in range(len(Y_test_seen)) if 1 in Y_test_seen[i]] # This row has an answer in seen classes
			Y_test_seen = Y_test_seen[tuple(positive_seen), :]
			V_T_test_seen = V_T_test[tuple(positive_seen), :]
		statseen.append(test_model(V_T_test_seen, V_C_seen, Y_test_seen, dataset_name + str(fold), single_label))

		V_C_unseen = V_C_all[tuple(unseen_this_fold), :]
		Y_test_unseen = Y_test_all[:, tuple(unseen_this_fold)]
		V_T_test_unseen = V_T_test
		if single_label:
			positive_unseen = [i for i in range(len(Y_test_unseen)) if 1 in Y_test_unseen[i]] # This row has an answer in seen classes
			Y_test_unseen = Y_test_unseen[tuple(positive_unseen), :]
			V_T_test_unseen = V_T_test[tuple(positive_unseen), :]
		statunseen.append(test_model(V_T_test_unseen, V_C_unseen, Y_test_unseen, dataset_name + str(fold), single_label))


	averageStats = dict([(key, np.mean(np.array([s[key] for s in stats]))) for key in stats[0]])
	averageSeenStats = dict([(key, np.mean(np.array([s[key] for s in statseen]))) for key in statseen[0]])
	averageUnseenStats = dict([(key, np.mean(np.array([s[key] for s in statunseen]))) for key in statunseen[0]])

	print_stats(stats, statseen, statunseen, averageStats, averageSeenStats, averageUnseenStats, n_fold)

def random_guess(V_T_train, Y_train_all, V_T_test, Y_test_all, V_C_all, classList, dataset_name, n_fold = 4):
	tempClasses = [(row['ClassCode'], row['Count']) for row in classList] 
	sortedClasses = [pair[0] for pair in sorted(tempClasses, reverse = True, key = lambda x: x[1])] # Sort class codes by counts
	rank = dict()
	for row in classList:
		rank[row['ClassCode']] = sortedClasses.index(row['ClassCode'])
	
	statseen = []
	statunseen = []
	stats = []
	for fold in range(n_fold):
		# reset_model_parameter()
		train_this_fold = []
		unseen_this_fold = []
		for i in range(len(classList)):
			if rank[classList[i]['ClassCode']] % (2*n_fold) not in [fold, 2*n_fold-fold-1]:
				train_this_fold.append(i)
			else:
				unseen_this_fold.append(i)
		Y_train = Y_train_all[:, tuple(train_this_fold)]

		positive_rate = np.sum(Y_train) / (Y_train.shape[0] * Y_train.shape[1])
		Y_predict = np.array([[1 if np.random.rand(1) < positive_rate else 0 for j in range(Y_test_all.shape[1])] for i in range(Y_test_all.shape[0])])

		stats.append(get_statistics(Y_predict, Y_test_all)) # generalised zero-shot

		Y_test_seen = Y_test_all[:, tuple(train_this_fold)]
		Y_predict_seen = Y_predict[:, tuple(train_this_fold)]
		statseen.append(get_statistics(Y_predict_seen, Y_test_seen))

		Y_test_unseen = Y_test_all[:, tuple(unseen_this_fold)]
		Y_predict_unseen = Y_predict[:, tuple(unseen_this_fold)]
		statunseen.append(get_statistics(Y_predict_unseen, Y_test_unseen))


	averageStats = dict([(key, np.mean(np.array([s[key] for s in stats]))) for key in stats[0]])
	averageSeenStats = dict([(key, np.mean(np.array([s[key] for s in statseen]))) for key in statseen[0]])
	averageUnseenStats = dict([(key, np.mean(np.array([s[key] for s in statunseen]))) for key in statunseen[0]])

	print_stats(stats, statseen, statunseen, averageStats, averageSeenStats, averageUnseenStats, n_fold)

def print_stats(stats, statseen, statunseen, averageStats, averageSeenStats, averageUnseenStats, n_fold):
	print('------------Testing results------------')
	text_out = ['seen + unseen classes', 'only seen classes', 'only unseen classes']
	for fold in range(n_fold):
		for i, s in enumerate([stats, statseen, statunseen]):
			print('Fold', fold, text_out[i])
			for key in s[fold]:
				print(key, s[fold][key])
			print('')
		print('---------------------------')

	for i, avs in enumerate([averageStats, averageSeenStats, averageUnseenStats]):
		print('Average', text_out[i],'results:')
		for key in avs:
			print(key, avs[key])
		print('')

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
	# V_C_all = np.load('../data/wiki/V_C_wiki_ConceptNet.npz')
	# print(V_C_all.keys())
	V_T_train, Y_train_all, V_T_test, Y_test_all, V_C_all, classList = load_dataset(dataset_name, knowledge_graph)
	# random_guess(V_T_train, Y_train_all, V_T_test, Y_test_all, V_C_all, classList, dataset_name, n_fold = n_fold)
	cross_class_validation(V_T_train, Y_train_all, V_T_test, Y_test_all, V_C_all, classList, dataset_name, n_fold = n_fold, single_label = single_label)
	pass