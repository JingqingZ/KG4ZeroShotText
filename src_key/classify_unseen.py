import numpy as np

def adjust_unseen_prob(prob_matrix, unseen_class_id, class_distance_matrix):
	total_class = len(prob_matrix[0])
	
	adjusted_unseen_prob = []
	for usid in unseen_class_id:
		seen_class_id = [i for i in range(total_class) if i not in unseen_class_id or usid == i] # including current unseen
		# print(seen_class_id)
		seen_prob = prob_matrix[:, seen_class_id]

		weight_vector = 1/class_distance_matrix[usid,:]
		# print(weight_vector)
		weight_vector = weight_vector[seen_class_id]
		weight_vector = normalise(weight_vector)
		# print(weight_vector)

		ans = np.zeros(len(prob_matrix))
		for i in range(len(seen_class_id)):
			ans += weight_vector[i] * seen_prob[:, i]

		# print(ans)
		adjusted_unseen_prob.append(ans)

	return np.array(adjusted_unseen_prob).T

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def normalise(x):
	return x / np.sum(x, axis = 0)

# print(softmax([1/4, 1/4, 1/2, 1/1, 1/2, 1/4, 1/5, 1/5, 1/5, 1/5, 1/5, 1/5, 1/5, 1/5]))
# print(normalise([1/4, 1/4, 1/2, 1/1, 1/2, 1/4, 1/5, 1/5, 1/5, 1/5, 1/5, 1/5, 1/5, 1/5]))
class_distance_matrix = np.loadtxt('../data/dbpedia/class_distance.txt')
# print(class_distance_matrix)
print(adjust_unseen_prob(np.random.rand(10,14), [2,8,11], class_distance_matrix))