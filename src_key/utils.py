
import os
import numpy as np

def make_dirlist(dirlist):
    for dir in dirlist:
        if not os.path.exists(dir):
            os.makedirs(dir)

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

def get_statistics(prediction, ground_truth, single_label_pred=False):
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

    if single_label_pred:
        single_label_ground_truth = np.argmax(ground_truth, axis = 1)
        single_label_prediction = np.argmax(prediction, axis = 1)
        single_label_error = np.mean(single_label_prediction != single_label_ground_truth)
        stats['single-label-error'] = single_label_error

    return stats


def dict_to_string_4_print(dict):
   return ', '.join(['%s: %.3f' % (key, dict[key]) for key in dict])


if __name__ == "__main__":
    print(dict_to_string_4_print({"accuracy": 1.213234, "f1": 1.2323232}))
    pass
