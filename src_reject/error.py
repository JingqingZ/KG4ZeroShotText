import numpy as np

import utils
import config
import dataloader
import os

def calculate_error(filename):
    data = np.load(filename)

    stats_seen = utils.get_statistics(data["pred_seen"], data["gt_seen"], True)
    stats_unseen = utils.get_statistics(data["pred_unseen"], data["gt_unseen"], True)

    unseen_classes = np.sum(data["gt_unseen"], axis=0)
    unseen_classes[unseen_classes > 0] = 1
    print(unseen_classes)

    unseen_pred_sum = np.sum(data["pred_unseen"], axis=0)
    print(unseen_pred_sum)
    unseen_pred_sum = np.sum(unseen_pred_sum * unseen_classes)
    print(unseen_pred_sum / data["pred_unseen"].shape[0])

    print(utils.dict_to_string_4_print(stats_seen))
    print(utils.dict_to_string_4_print(stats_unseen))

# classifying multiple_label
def classify_multiple_label(filename):
    print("multi label")
    data = np.load(filename)
    # data = np.load("../results/key_zhang15_dbpedia_kg3_%dof4/logs/test_5.npz" % (f + 1))
    # data = np.load("../results/key_zhang15_dbpedia_kg3_random%d_unseen0.25/logs/test_2.npz" % (f + 1))
    # data = np.load("../results/key_zhang15_dbpedia_kg3_random%d_unseen0.25_max100/logs/test_5.npz" % (f + 1))
    # data = np.load("../results/key_news20_kg3_random%d_unseen0.25/logs/test_5_max300_full.npz" % (f + 1))

    for i in range(50000, 100000, 1000):
        threshold = i / 100000
        pred_unseen = data["pred_unseen"]
        pred_unseen[pred_unseen > threshold] = 1
        pred_unseen[pred_unseen <= threshold] = 0
        stats = utils.get_statistics(pred_unseen, data["gt_unseen"], single_label_pred=False)
        try:
            print("uns threshold: %.5f: %s" % (threshold, utils.dict_to_string_4_print(stats)))
        except:
            print("uns threshold: %.5f: error" % (threshold))
        break

    for i in range(50000, 100000, 1000):
        threshold = i / 100000
        pred_seen = data["pred_seen"]
        pred_seen[pred_seen > threshold] = 1
        pred_seen[pred_seen <= threshold] = 0
        stats = utils.get_statistics(pred_seen, data["gt_seen"], single_label_pred=False)
        try:
            print("see threshold: %.5f: %s" % (threshold, utils.dict_to_string_4_print(stats)))
        except:
            print("see threshold: %.5f: error" % (threshold))
        break

    for i in range(50000, 100000, 1000):
        threshold = i / 100000

        gt_both = np.concatenate([data["gt_seen"], data["gt_unseen"]], axis=0)
        pred_both = np.concatenate([data["pred_seen"], data["pred_unseen"]], axis=0)

        pred_both[pred_both > threshold] = 1
        pred_both[pred_both <= threshold] = 0
        stats = utils.get_statistics(pred_both, gt_both, single_label_pred=False)
        try:
            print("all threshold: %.5f: %s" % (threshold, utils.dict_to_string_4_print(stats)))
        except:
            print("all threshold: %.5f: error" % (threshold))
        break

def classify_single_label(filename):
    print("single label")
    data = np.load(filename)
    # data = np.load("../results/key_zhang15_dbpedia_kg3_random%d_unseen0.25/logs/test_2.npz" % (f + 1))
    # data = np.load("../results/key_news20_kg3_random%d_unseen0.25/logs/test_5_max300_full.npz" % (f + 1))
    # data = np.load("../results/key_zhang15_dbpedia_kg3_random%d_unseen0.25_max100/logs/test_5.npz" % (f + 1))
    # data = np.load("../results/key_zhang15_dbpedia_kg3_random%d_unseen0.25_max100_cnn/logs/test_4.npz" % (f + 1))

    # print(data["pred_seen"].shape)
    # print(data["pred_unseen"].shape)

    seen_class = np.nonzero(np.sum(data["gt_seen"], axis=0))[0]
    unseen_class = np.nonzero(np.sum(data["gt_unseen"], axis=0))[0]
    print(seen_class, unseen_class)

    pred_unseen = data["pred_unseen"]
    for pidx, pred in enumerate(pred_unseen):
        maxconf = -1
        argmax = -1
        for class_idx in range(len(pred)):
            if pred[class_idx] > maxconf and class_idx in unseen_class:
                argmax = class_idx
                maxconf = pred[class_idx]
        assert argmax in unseen_class
        pred_unseen[pidx] = 0
        pred_unseen[pidx, argmax] = 1
    out_pred_unseen = pred_unseen

    stats = utils.get_statistics(pred_unseen, data["gt_unseen"], single_label_pred=True)
    try:
        print("uns tra: %s" % (utils.dict_to_string_4_print(stats)))
    except:
        print("uns tra: error")

    pred_seen = data["pred_seen"]
    for pidx, pred in enumerate(pred_seen):
        maxconf = -1
        argmax = -1
        for class_idx in range(len(pred)):
            if pred[class_idx] > maxconf and class_idx in seen_class:
                argmax = class_idx
                maxconf = pred[class_idx]
        assert argmax in seen_class
        pred_seen[pidx] = 0
        pred_seen[pidx, argmax] = 1
    out_pred_seen = pred_seen

    stats = utils.get_statistics(pred_seen, data["gt_seen"], single_label_pred=True)
    try:
        print("see tra: %s" % (utils.dict_to_string_4_print(stats)))
    except:
        print("see tra: error")


    pred_unseen = data["pred_unseen"]
    for pidx, pred in enumerate(pred_unseen):
        class_idx = np.argmax(pred)
        pred_unseen[pidx] = 0
        pred_unseen[pidx, class_idx] = 1

    stats = utils.get_statistics(pred_unseen, data["gt_unseen"], single_label_pred=True)
    try:
        print("uns gen: %s" % (utils.dict_to_string_4_print(stats)))
    except:
        print("uns gen: error")

    pred_seen = data["pred_seen"]
    for pidx, pred in enumerate(pred_seen):
        class_idx = np.argmax(pred)
        pred_seen[pidx] = 0
        pred_seen[pidx, class_idx] = 1

    stats = utils.get_statistics(pred_seen, data["gt_seen"], single_label_pred=True)
    try:
        print("see gen: %s" % (utils.dict_to_string_4_print(stats)))
    except:
        print("see gen: error")

    gt_both = np.concatenate([data["gt_seen"], data["gt_unseen"]], axis=0)
    pred_both = np.concatenate([pred_seen, pred_unseen], axis=0)

    stats = utils.get_statistics(pred_both, gt_both, single_label_pred=True)
    try:
        print("all gen: %s" % (utils.dict_to_string_4_print(stats)))
    except:
        print("all gen: error")

    return out_pred_seen, out_pred_unseen, pred_both, gt_both

def classify_single_label2(filename):
    print("single label")
    data = np.load(filename)
    # data = np.load("../results/key_zhang15_dbpedia_kg3_random%d_unseen0.25/logs/test_2.npz" % (f + 1))
    # data = np.load("../results/key_news20_kg3_random%d_unseen0.25/logs/test_5_max300_full.npz" % (f + 1))
    # data = np.load("../results/key_zhang15_dbpedia_kg3_random%d_unseen0.25_max100/logs/test_5.npz" % (f + 1))
    # data = np.load("../results/key_zhang15_dbpedia_kg3_random%d_unseen0.25_max100_cnn/logs/test_4.npz" % (f + 1))

    # print(data["pred_seen"].shape)
    # print(data["pred_unseen"].shape)

    seen_class = np.nonzero(np.sum(data["gt_seen"], axis=0))[0]
    unseen_class = np.nonzero(np.sum(data["gt_unseen"], axis=0))[0]
    print(seen_class, unseen_class)

    pred_unseen = data["pred_unseen"]
    for pidx, pred in enumerate(pred_unseen):
        maxconf = -1
        argmax = -1
        for class_idx in range(len(pred)):
            if pred[class_idx] > maxconf and class_idx in unseen_class:
                argmax = class_idx
                maxconf = pred[class_idx]
        assert argmax in unseen_class
        pred_unseen[pidx] = 0
        pred_unseen[pidx, argmax] = 1
    out_pred_unseen = pred_unseen

    stats1 = utils.get_statistics(pred_unseen, data["gt_unseen"], single_label_pred=True)
    try:
        print("uns tra: %s" % (utils.dict_to_string_4_print(stats1)))
    except:
        print("uns tra: error")

    pred_seen = data["pred_seen"]
    for pidx, pred in enumerate(pred_seen):
        maxconf = -1
        argmax = -1
        for class_idx in range(len(pred)):
            if pred[class_idx] > maxconf and class_idx in seen_class:
                argmax = class_idx
                maxconf = pred[class_idx]
        assert argmax in seen_class
        pred_seen[pidx] = 0
        pred_seen[pidx, argmax] = 1
    out_pred_seen = pred_seen

    stats2 = utils.get_statistics(pred_seen, data["gt_seen"], single_label_pred=True)
    try:
        print("see tra: %s" % (utils.dict_to_string_4_print(stats2)))
    except:
        print("see tra: error")


    pred_unseen = data["pred_unseen"]
    for pidx, pred in enumerate(pred_unseen):
        class_idx = np.argmax(pred)
        pred_unseen[pidx] = 0
        pred_unseen[pidx, class_idx] = 1

    stats3 = utils.get_statistics(pred_unseen, data["gt_unseen"], single_label_pred=True)
    try:
        print("uns gen: %s" % (utils.dict_to_string_4_print(stats3)))
    except:
        print("uns gen: error")

    pred_seen = data["pred_seen"]
    for pidx, pred in enumerate(pred_seen):
        class_idx = np.argmax(pred)
        pred_seen[pidx] = 0
        pred_seen[pidx, class_idx] = 1

    stats4 = utils.get_statistics(pred_seen, data["gt_seen"], single_label_pred=True)
    try:
        print("see gen: %s" % (utils.dict_to_string_4_print(stats4)))
    except:
        print("see gen: error")

    gt_both = np.concatenate([data["gt_seen"], data["gt_unseen"]], axis=0)
    pred_both = np.concatenate([pred_seen, pred_unseen], axis=0)

    stats5 = utils.get_statistics(pred_both, gt_both, single_label_pred=True)
    try:
        print("all gen: %s" % (utils.dict_to_string_4_print(stats5)))
    except:
        print("all gen: error")

    return pred_both, stats1, stats2, stats3, stats4, stats5

# TODO: visualise distribution of confidence predicted
def classify_single_label_vis(filename):
    print("single label vis")
    data = np.load(filename)
    # data = np.load("../results/key_zhang15_dbpedia_kg3_random%d_unseen0.25/logs/test_2.npz" % (f + 1))
    # data = np.load("../results/key_news20_kg3_random%d_unseen0.25/logs/test_5_max300_full.npz" % (f + 1))
    # data = np.load("../results/key_zhang15_dbpedia_kg3_random%d_unseen0.25_max100/logs/test_5.npz" % (f + 1))
    # data = np.load("../results/key_zhang15_dbpedia_kg3_random%d_unseen0.25_max100_cnn/logs/test_4.npz" % (f + 1))

    # print(data["pred_seen"].shape)
    # print(data["pred_unseen"].shape)

    seen_class = np.nonzero(np.sum(data["gt_seen"], axis=0))[0]
    unseen_class = np.nonzero(np.sum(data["gt_unseen"], axis=0))[0]
    print(seen_class, unseen_class)

    pred_unseen = data["pred_unseen"]
    for pidx, pred in enumerate(pred_unseen):
        maxconf = -1
        argmax = -1
        for class_idx in range(len(pred)):
            if pred[class_idx] > maxconf and class_idx in unseen_class:
                argmax = class_idx
                maxconf = pred[class_idx]
        assert argmax in unseen_class
        pred_unseen[pidx] = 0
        pred_unseen[pidx, argmax] = 1

    stats1 = utils.get_statistics(pred_unseen, data["gt_unseen"], single_label_pred=True)
    try:
        print("uns tra: %s" % (utils.dict_to_string_4_print(stats1)))
    except:
        print("uns tra: error")

    pred_seen = data["pred_seen"]
    for pidx, pred in enumerate(pred_seen):
        maxconf = -1
        argmax = -1
        for class_idx in range(len(pred)):
            if pred[class_idx] > maxconf and class_idx in seen_class:
                argmax = class_idx
                maxconf = pred[class_idx]
        assert argmax in seen_class
        pred_seen[pidx] = 0
        pred_seen[pidx, argmax] = 1

    stats2 = utils.get_statistics(pred_seen, data["gt_seen"], single_label_pred=True)
    try:
        print("see tra: %s" % (utils.dict_to_string_4_print(stats2)))
    except:
        print("see tra: error")

def rejection_single_label(filename, printopt=False):
    np.set_printoptions(threshold=np.nan)
    print("rejecting")
    # data = np.load("../results/key_zhang15_dbpedia_kg3_random%d_unseen0.25/logs/test_2.npz" % (f + 1))
    # data = np.load("../results/key_chen14_kg3_random%d_unseen0.25/logs/test_6.npz" % (f + 1))
    # data = np.load("../results/key_news20_kg3_random%d_unseen0.25/logs/test_5_max300_full.npz" % (f + 1))
    # data = np.load("../results/key_news20_kg3_random%d_unseen0.25/logs/test_20.npz" % (f + 1))
    # data = np.load("../results/key_zhang15_dbpedia_kg3_random%d_unseen0.25_max100/logs/test_5.npz" % (f + 1))
    data = np.load(filename)

    seen_class = np.nonzero(np.sum(data["gt_seen"], axis=0))[0]
    unseen_class = np.nonzero(np.sum(data["gt_unseen"], axis=0))[0]
    # print("seen", seen_class, seen_class.shape)
    # print("unseen", unseen_class, unseen_class.shape)

    num_seen_class = seen_class.shape[0] + 1

    all_class = np.concatenate([seen_class, unseen_class], axis=0)

    print(seen_class, unseen_class)

    for t in range(50000, 100000, 1000):
        threshold = t / 100000

        pred_reject = np.zeros((data["pred_seen"].shape[0] + data["pred_unseen"].shape[0], num_seen_class + 1))
        gt_reject = np.zeros((data["gt_seen"].shape[0] + data["gt_unseen"].shape[0], num_seen_class + 1))

        for didx, seen_data in enumerate(data["gt_seen"]):
            seen_index = np.where(all_class==np.argmax(seen_data))[0][0]
            gt_reject[didx, seen_index] = 1
        gt_reject[data["gt_seen"].shape[0]:, num_seen_class] = 1
        for g in gt_reject:
            assert np.sum(g) == 1


        pred_seen = data["pred_seen"]
        # print("seen")
        # print(pred_seen[2000:2003])
        # pred_seen[pred_seen >= threshold] = 1
        pred_seen[pred_seen < threshold] = 0
        # print(pred_seen[2000:2003])

        pred_unseen = data["pred_unseen"]
        if printopt:
            print("unseen")
            print(pred_unseen[1000:1010])
        pred_unseen[pred_unseen >= threshold] = 1
        pred_unseen[pred_unseen < threshold] = 0
        # print(pred_unseen[1000:1003])

        all_pred = np.concatenate([pred_seen, pred_unseen])
        assert all_pred.shape[0] == pred_reject.shape[0]

        for i in range(pred_reject.shape[0]):
            num_seen_class_high_confidence = 0
            for cidx, class_confidence in enumerate(all_pred[i]):
                if class_confidence >= threshold and cidx in seen_class:
                    num_seen_class_high_confidence += 1
                else:
                    all_pred[i, cidx] = 0
            if num_seen_class_high_confidence == 0:
                pred_reject[i, -1] = 1
            else:
                seen_index = np.argmax(all_pred[i])
                assert seen_index in seen_class
                # print(seen_index)
                seen_index = np.where(all_class==seen_index)[0][0]
                # print(seen_index)
                pred_reject[i, seen_index] = 1

        if printopt:
            # print("reject")
            # print(pred_reject[2000:2003])
            # print(pred_reject[pred_seen.shape[0]+1000:1000+pred_seen.shape[0] + 10])

            # print("gt reject")
            # print(gt_reject[2000:2003])
            # print(gt_reject[data["gt_seen"].shape[0]+1000:1000+data["gt_seen"].shape[0] + 10])
            # print("gt")
            # print(data["gt_seen"][2000: 2003, :])
            print(data["gt_unseen"][1000: 1010, :])
            # exit()

        stats = utils.get_statistics(pred_reject, gt_reject, True)
        stats_seen = utils.get_statistics(pred_reject[:data["gt_seen"].shape[0]], gt_reject[:data["gt_seen"].shape[0]], True)
        stats_unseen = utils.get_statistics(pred_reject[data["gt_seen"].shape[0] :], gt_reject[data["gt_seen"].shape[0] :], True)
        try:
            print("uns threshold: %.5f:  %s" % (threshold, utils.dict_to_string_4_print(stats_unseen)))
            print("see threshold: %.5f:  %s" % (threshold, utils.dict_to_string_4_print(stats_seen)))
            print("all threshold: %.5f:  %s" % (threshold, utils.dict_to_string_4_print(stats)))
        except:
            print("threshold: %.5f: error" % (threshold))
        return pred_reject, stats, stats_seen, stats_unseen

def reject_then_classify_single_label(filename, pred_reject):

    data = np.load(filename)

    pred_raw = np.concatenate([data["pred_seen"], data["pred_unseen"]], axis=0)
    gt_both = np.concatenate([data["gt_seen"], data["gt_unseen"]], axis=0)

    assert pred_raw.shape == gt_both.shape

    seen_class = np.nonzero(np.sum(data["gt_seen"], axis=0))[0]
    unseen_class = np.nonzero(np.sum(data["gt_unseen"], axis=0))[0]
    print(seen_class, unseen_class)

    pred_both = np.zeros(gt_both.shape)
    for idx in range(pred_reject.shape[0]):
        assert np.sum(pred_reject[idx]) == 1

        if pred_reject[idx][-1] == 1:
            maxconf = -1
            argmax = -1
            for class_idx in range(len(pred_raw[idx])):
                if pred_raw[idx][class_idx] > maxconf and class_idx in unseen_class:
                    argmax = class_idx
                    maxconf = pred_raw[idx][class_idx]
            assert argmax in unseen_class
            pred_both[idx, argmax] = 1
        else:
            maxconf = -1
            argmax = -1
            for class_idx in range(len(pred_raw[idx])):
                if pred_raw[idx][class_idx] > maxconf and class_idx in seen_class:
                    argmax = class_idx
                    maxconf = pred_raw[idx][class_idx]
            assert argmax in seen_class
            pred_both[idx, argmax] = 1

    stats = utils.get_statistics(pred_both, gt_both, True)
    stats_seen = utils.get_statistics(pred_both[:data["gt_seen"].shape[0]], gt_both[:data["gt_seen"].shape[0]], True)
    stats_unseen = utils.get_statistics(pred_both[data["gt_seen"].shape[0] :], gt_both[data["gt_seen"].shape[0] :], True)
    assert pred_both.shape[0] == data["gt_seen"].shape[0] + data["gt_unseen"].shape[0]
    try:
        print("uns %s" % (utils.dict_to_string_4_print(stats_unseen)))
        print("see %s" % (utils.dict_to_string_4_print(stats_seen)))
        print("all %s" % (utils.dict_to_string_4_print(stats)))
    except:
        print("all new: error")
    return None, stats, stats_seen, stats_unseen

def classify_adjust_single_label(filename, class_distance_matrix):
    print("adjust single label")
    data = np.load(filename)

    seen_class = np.nonzero(np.sum(data["gt_seen"], axis=0))[0]
    unseen_class = np.nonzero(np.sum(data["gt_unseen"], axis=0))[0]
    # print("seen", seen_class, seen_class.shape)
    # print("unseen", unseen_class, unseen_class.shape)

    num_seen_class = seen_class.shape[0] + 1

    all_class = np.concatenate([seen_class, unseen_class], axis=0)

    print(seen_class, unseen_class)

    for t in range(50000, 100000, 1000):
        threshold = t / 100000

        pred_seen = data['pred_seen']
        pred_unseen = data['pred_unseen']
        pred = np.concatenate([pred_seen, pred_unseen], axis = 0)

        gt_seen = data['gt_seen']
        gt_unseen = data['gt_unseen']
        gt = np.concatenate([gt_seen, gt_unseen], axis=0)


        pred_choose_ans_from = np.max(pred[:, seen_class], axis = 1) > threshold
        # pred_choose_ans_from[data["gt_seen"].shape[0]:] = False
        pred_ans_seen_original = seen_class[np.argmax(pred[:, seen_class], axis = 1)]
        pred_ans_unseen_adjusted = unseen_class[np.argmax(adjust_unseen_prob(pred, unseen_class, class_distance_matrix), axis=1)]
        pred_final = (pred_choose_ans_from * pred_ans_seen_original) + ((1-pred_choose_ans_from) * pred_ans_unseen_adjusted)

        pred_matrix = np.zeros(pred.shape)
        for i in range(len(pred_matrix)):
            pred_matrix[i][pred_final[i]] = 1



        stats = utils.get_statistics(pred_matrix, gt, True)
        stats_seen = utils.get_statistics(pred_matrix[:data["gt_seen"].shape[0]], gt[:data["gt_seen"].shape[0]],
                                          True)
        stats_unseen = utils.get_statistics(pred_matrix[data["gt_seen"].shape[0]:],
                                            gt[data["gt_seen"].shape[0]:], True)
        try:
            print("uns threshold: %.5f:  %s" % (threshold, utils.dict_to_string_4_print(stats_unseen)))
            print("see threshold: %.5f:  %s" % (threshold, utils.dict_to_string_4_print(stats_seen)))
            print("all threshold: %.5f:  %s" % (threshold, utils.dict_to_string_4_print(stats)))
        except:
            print("threshold: %.5f: error" % (threshold))
        return pred_matrix, stats, stats_seen, stats_unseen

def classify_without_adjust_single_label(filename, class_distance_matrix):
    print("without adjust single label")
    data = np.load(filename)

    seen_class = np.nonzero(np.sum(data["gt_seen"], axis=0))[0]
    unseen_class = np.nonzero(np.sum(data["gt_unseen"], axis=0))[0]
    # print("seen", seen_class, seen_class.shape)
    # print("unseen", unseen_class, unseen_class.shape)

    num_seen_class = seen_class.shape[0] + 1

    all_class = np.concatenate([seen_class, unseen_class], axis=0)

    print(seen_class, unseen_class)

    for t in range(50000, 100000, 1000):
        threshold = t / 100000

        pred_seen = data['pred_seen']
        pred_unseen = data['pred_unseen']
        pred = np.concatenate([pred_seen, pred_unseen], axis = 0)

        gt_seen = data['gt_seen']
        gt_unseen = data['gt_unseen']
        gt = np.concatenate([gt_seen, gt_unseen], axis=0)


        pred_choose_ans_from = np.max(pred[:, seen_class], axis = 1) > threshold
        pred_ans_seen_original = seen_class[np.argmax(pred[:, seen_class], axis = 1)]
        pred_ans_unseen_adjusted = unseen_class[np.argmax(pred[:, unseen_class], axis=1)]
        pred_final = (pred_choose_ans_from * pred_ans_seen_original) + ((1-pred_choose_ans_from) * pred_ans_unseen_adjusted)

        pred_matrix = np.zeros(pred.shape)
        for i in range(len(pred_matrix)):
            pred_matrix[i][pred_final[i]] = 1



        stats = utils.get_statistics(pred_matrix, gt, True)
        stats_seen = utils.get_statistics(pred_matrix[:data["gt_seen"].shape[0]], gt[:data["gt_seen"].shape[0]],
                                          True)
        stats_unseen = utils.get_statistics(pred_matrix[data["gt_seen"].shape[0]:],
                                            gt[data["gt_seen"].shape[0]:], True)
        try:
            print("uns threshold: %.5f:  %s" % (threshold, utils.dict_to_string_4_print(stats_unseen)))
            print("see threshold: %.5f:  %s" % (threshold, utils.dict_to_string_4_print(stats_seen)))
            print("all threshold: %.5f:  %s" % (threshold, utils.dict_to_string_4_print(stats)))
        except:
            print("threshold: %.5f: error" % (threshold))
        return pred_matrix, stats, stats_seen, stats_unseen

def adjust_unseen_prob(prob_matrix, unseen_class_id, class_distance_matrix):
    total_class = len(prob_matrix[0])

    adjusted_unseen_prob = []
    for usid in unseen_class_id:
        seen_class_id = [i for i in range(total_class) if
                         i not in unseen_class_id or usid == i]  # including current unseen

        seen_prob = prob_matrix[:, seen_class_id]

        weight_vector = 1 / class_distance_matrix[usid, :]
        weight_vector = weight_vector[seen_class_id]
        weight_vector = normalise(weight_vector)


        ans = np.zeros(len(prob_matrix))
        for i in range(len(seen_class_id)):
            ans += weight_vector[i] * seen_prob[:, i]
        adjusted_unseen_prob.append(ans)

    return np.array(adjusted_unseen_prob).T

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def normalise(x):
    return x / np.sum(x, axis=0)

def error_deprecated():
    reject_list = list()
    num = 10
    for i in range(num):
        # calculate_error("../results/key_zhang15_dbpedia_4of4/logs/test_5_att.npz")
        # filename = "../results/key_zhang15_dbpedia_kg3_random%d_unseen0.25_max100_cnn_negative%d/logs/test_%d.npz" \
        #            % (5, 7, 10)
        # filename = "../results/selected_zhang15_dbpedia_kg3_random%d_unseen0.25_max50_cnn_negative9increase2_randomtext/logs/test_5.npz" \
        # filename = "../results/selected_zhang15_dbpedia_noclasslabel_random%d_unseen0.25_max50_cnn_negative9increase2_randomtext/logs/test_5.npz" \
        # filename = "../results//selected_news20_kg3_random%d_unseen0.25_max100_cnn_negative2increase2_randomtext/logs/test_10.npz" \
        # filename = "../results/selected_zhang15_dbpedia_kg3_random%d_unseen0.25_max50_cnn_negative9increase2_randomtext/logs/test_5.npz" \
        # filename = "../results/selected_news20_kg3_random%d_unseen0.25_max100_cnn_negative2increase2_randomtext/logs/test_10.npz" \
        # filename = "../results/selected_zhang15_dbpedia_nokg_random%d_unseen0.25_max50_cnn_negative-1_randomtext/logs/test_10.npz" \
        # filename = "../results/selected_news20_kg3_random%d_unseen0.25_max100_cnn_negative9increase2_randomtext/logs/test_10.npz" \
        # filename = "../results/selected_news20_kg3_random%d_unseen0.25_max100_cnn_negative9increase2_randomtext/logs/test_10.npz" \
        # filename = "../results/selected_chen14_elec_kg3_random%d_unseen0.25_max200_cnn_negative9increase2_randomtext/logs/test_20.npz" \
        # filename ="../results/selected_zhang15_dbpedia_kg3_cluster_allgroup_random%d_unseen0.25_max50_cnn_negative5increase2_randomtext/logs/test_5.npz" \
        # filename ="../results/full_zhang15_dbpedia_kg3_cluster_allgroup_only_random%d_unseen0.25_max50_cnn_negative9increase2_randomtext/logs/test_5.npz" \
        # filename ="../results/selected_zhang15_dbpedia_kg3_cluster_3group_random%d_unseen0.25_max50_cnn_negative5increase2_randomtext/logs/test_5.npz" \
        # filename = "../results/selected_zhang15_dbpedia_kg3_random%d_unseen0.25_max50_cnn_negative9increase3_randomtext/logs/test_5.npz" \
        # filename = "../results/selected_zhang15_dbpedia_nokg_random%d_unseen0.25_max50_cnn_negative-1_randomtext/logs/test_10.npz" \
        # filename = "../results/selected_zhang15_dbpedia_noclasslabel_random%d_unseen0.25_max50_cnn_negative9increase2_randomtext/logs/test_5.npz" \
        # filename = "../results/selected_tfidf_zhang15_dbpedia_kg3_cluster_3group_random%d_unseen0.25_max50_cnn_negative5increase2_randomtext/logs/test_5.npz" \
        # filename = "../results/selected_tfidf_zhang15_dbpedia_kg3_cluster_3group_only_random%d_unseen0.25_max50_cnn_negative5increase2_randomtext/logs/test_5.npz" \
        # filename = "../results/full_zhang15_dbpedia_kg3_cluster_3group_only_random%d_unseen0.25_max50_cnn_negative5increase2_randomtext/logs/test_5.npz" \
        # filename = "../results/seen_full_zhang15_dbpedia_vwonly_random%d_unseen0.25_max50_cnn_negative5increase2_randomtext/logs/test_5.npz" \
        filename = "../results/unseen_full_zhang15_dbpedia_kg3_cluster_3group_random%d_unseen0.25_max50_cnn_negative5increase2_randomtext/logs/test_5.npz" \
                          % (i + 1)

        # pred_seen, pred_unseen, pred_both, gt_both = classify_single_label(filename)
        # classify_multiple_label(filename)
        # pred_reject = rejection_single_label(filename)
        # pred_reject = reject_then_classify_single_label(filename, pred_reject[0])
        # reject_list.append(pred_reject)

        class_distance_matrix = np.loadtxt('../data/zhang15/dbpedia_csv/class_distance.txt')
        # class_distance_matrix = np.loadtxt(config.news20_dir + 'class_distance_20newsgroups.txt')
        # class_distance_matrix = np.loadtxt(config.chen14_elec_dir + 'class_distance_elec.txt')
        # class_distance_matrix = np.loadtxt('../data/zhang15/dbpedia_csv/class_distance_glove.txt')

        # classify_adjust = classify_adjust_single_label(filename, class_distance_matrix)
        # reject_list.append(classify_adjust)

        # classify_noadjust = classify_without_adjust_single_label(filename, None)
        # reject_list.append(classify_noadjust)

        classify = classify_single_label2(filename)
        reject_list.append(classify)

    pred_dict = list()
    for sidx in range(1, len(reject_list[0])):
        pred_dict.append(dict())
    for reject in reject_list:
        for sidx in range(1, len(reject_list[0])):
            for mea in reject[sidx]:
                if mea not in pred_dict[sidx - 1]:
                    pred_dict[sidx - 1][mea] = 0
                pred_dict[sidx - 1][mea] += reject[sidx][mea]

    for sidx in range(len(pred_dict)):
        for mea in pred_dict[sidx]:
            pred_dict[sidx][mea] /= num
            pred_dict[sidx][mea] = pred_dict[sidx][mea]

    # pred_dict = [dict(), dict(), dict(), dict(), dict()]
    # for reject in reject_list:
    #     for sidx in range(1, 6):
    #         for mea in reject[sidx]:
    #             if mea not in pred_dict[sidx - 1]:
    #                 pred_dict[sidx - 1][mea] = 0
    #             pred_dict[sidx - 1][mea] += reject[sidx][mea]
    #
    # for sidx in range(5):
    #     for mea in pred_dict[sidx]:
    #         pred_dict[sidx][mea] /= 10

    # print("average overall ============================")
    # print(utils.dict_to_string_4_print(pred_dict[0]))
    # print("average seen ============================")
    # print(utils.dict_to_string_4_print(pred_dict[1]))
    # print("average unseen ============================")
    # print(utils.dict_to_string_4_print(pred_dict[2]))

    for pred in pred_dict:
        print(utils.dict_to_string_4_print(pred))

    # print("average uns tra ============================")
    # print(utils.dict_to_string_4_print(pred_dict[0]))
    # print("average see tra ============================")
    # print(utils.dict_to_string_4_print(pred_dict[1]))
    # print("average uns gen ============================")
    # print(utils.dict_to_string_4_print(pred_dict[2]))
    # print("average see gen ============================")
    # print(utils.dict_to_string_4_print(pred_dict[3]))
    # print("average all gen ============================")
    # print(utils.dict_to_string_4_print(pred_dict[4]))


    pass

def classify_single_label_for_seen(filename, rgroup=None):
    data = np.load(filename)

    seen_class = np.nonzero(np.sum(data["gt_seen"], axis=0))[0]
    seen_class += 1
    # unseen_class = data["unseen_class"]

    if rgroup is None:
        assert np.array_equal(data["seen_class"], seen_class)
    else:
        assert np.array_equal(seen_class, np.array(rgroup[0]))
    #  print("Seen classes:", seen_class)
    # print("Unseen classes:", unseen_class)

    stats = utils.get_statistics(data["pred_seen"], data["gt_seen"], single_label_pred=True)
    print("seen: %s" % (utils.dict_to_string_4_print(stats)))
    return stats

def error_seen():
    # random_group = dataloader.get_random_group(config.zhang15_dbpedia_class_random_group_path)
    random_group = dataloader.get_random_group(config.news20_class_random_group_path)
    # random_group = dataloader.get_random_group(config.chen14_elec_class_random_group_path)

    overall_stats = dict()
    print_string = ""
    for i, rgroup in enumerate(random_group):
        # filename = "../results/seen_full_zhang15_dbpedia_vwonly_random%d_unseen%s_max%d_cnn/logs/test_%d.npz" \
        # filename = "../results/seen_full_chen14_elec_vwonly_random%d_unseen%s_max%d_cnn/logs/test_%d.npz" \
        filename = "../results/seen_selected_tfidf_news20_vwonly_random%d_unseen%s_max%d_cnn/logs/test_%d.npz" \
                   % (i + 1, "-".join(str(_) for _ in rgroup[1]), 200, 30 if i < 5 else 100)

        classify_stats = classify_single_label_for_seen(filename)
        for k in classify_stats:
            v = classify_stats[k]
            if k not in overall_stats:
                overall_stats[k] = list()
            overall_stats[k].append(v)
        print_string += "%.3f/%.3f/%.3f," \
                        % (1 - classify_stats["single-label-error"],
                           classify_stats["micro-F1"],
                           classify_stats["macro-F1"])

    for k in overall_stats:
        overall_stats[k] = np.mean(overall_stats[k])

    print("=======")
    print("overall: %s" % (utils.dict_to_string_4_print(overall_stats)))
    print("for Google Sheets, split by comma")
    print_string += "%.3f/%.3f/%.3f" \
                    % (1 - overall_stats["single-label-error"],
                       overall_stats["micro-F1"],
                       overall_stats["macro-F1"])
    print(print_string)

def classify_single_label_for_unseen(filename, rgroup, printstats=True):
    data = np.load(filename)

    unseen_class = np.nonzero(np.sum(data["gt_unseen"], axis=0))[0]

    for class_id in unseen_class:
        assert class_id + 1 in rgroup[1]
    assert len(rgroup[1]) == unseen_class.shape[0]
    # print("Seen classes:", rgroup[0])
    # print("Unseen classes:", rgroup[1])

    pred_unseen = data["pred_unseen"]
    for pidx, pred in enumerate(pred_unseen):
        maxconf = -1
        argmax = -1
        for class_idx in range(len(pred)):
            if pred[class_idx] > maxconf and class_idx in unseen_class:
                argmax = class_idx
                maxconf = pred[class_idx]
        assert argmax in unseen_class
        pred_unseen[pidx] = 0
        pred_unseen[pidx, argmax] = 1

    stats = utils.get_statistics(pred_unseen, data["gt_unseen"], single_label_pred=True)
    if printstats:
        print("unseen: %s" % (utils.dict_to_string_4_print(stats)))
    return stats

def error_unseen():
    if config.dataset == "dbpedia":
        random_group = dataloader.get_random_group(config.zhang15_dbpedia_class_random_group_path)
    elif config.dataset == "20news":
        random_group = dataloader.get_random_group(config.news20_class_random_group_path)
    else:
        raise Exception("invalid dataset")
    # random_group = dataloader.get_random_group(config.chen14_elec_class_random_group_path)

    for epoch in range(11):
        # if epoch != 1:
        #     continue

        overall_stats = dict()
        print_string = ""

        for i, rgroup in enumerate(random_group):

            # filename = "../results/unseen_full_zhang15_dbpedia_kg3_cluster_3group_random%d_unseen%s_max%d_cnn_negative%dincrease3_randomtext/logs/test_%d.npz" \
            # filename = "../results/unseen_full_zhang15_dbpedia_kg3_cluster_3group_random%d_unseen%s_max%d_cnn_negative%dincrease3_randomtext/logs/test_full_%d.npz" \
            # filename = "../results/unseen_selected_tfidf_news20_kg3_cluster_3group_only_smallepoch5_random%d_unseen%s_max%d_cnn_negative%dincrease%d_randomtext/logs/test_%d.npz" \
            # filename = "../results/unseen_selected_tfidf_news20_kg3_cluster_3group_only_random%d_unseen%s_max%d_cnn_negative%dincrease%d_randomtext/logs/test_%d.npz" \
            # filename = "../results/unseen_selected_tfidf_news20_kg3_cluster_3group_only_random%d_unseen%s_max%d_cnn_negative%dincrease%d_randomtext/logs/test_full_%d.npz" \
            # filename = "../results/unseen_full_chen14_elec_kg3_cluster_3group_random%d_unseen%s_max%d_cnn_negative%dincrease%d_randomtext/logs/test_%d.npz" \
            #            % (i + 1, "-".join(str(_) for _ in rgroup[1]), 100, 1, 1, epoch)
                       # % (i + 1, "-".join(str(_) for _ in rgroup[1]), 50, 1, 1, epoch)
            if config.dataset == "dbpedia":
                filename = "../results/unseen_full_zhang15_dbpedia_kg3_cluster_3group_%s_random%d_unseen%s_max%d_cnn_negative%dincrease%d_randomtext_aug%d/logs/test_%d.npz" \
                           % (config.model, i + 1, "-".join(str(_) for _ in rgroup[1]), 80, config.negative_sample, config.negative_increase, config.augmentation, epoch)
            elif config.dataset == "20news":
                filename = "../results/unseen_selected_tfidf_news20_kg3_cluster_3group_%s_random%d_unseen%s_max%d_cnn_negative%dincrease%d_randomtext_aug%d/logs/test_%d.npz" \
                           % (config.model, i + 1, "-".join(str(_) for _ in rgroup[1]), 50, config.negative_sample, config.negative_increase, config.augmentation, epoch)
            else:
                raise Exception("invalid dataset")

            # print(filename)
            # print(os.path.exists(filename))
            # exit()
            if not os.path.exists(filename):
                continue

            classify_stats = classify_single_label_for_unseen(filename, rgroup, False)
            # classify_stats = classify_single_label_for_unseen(filename, rgroup, True)
            for k in classify_stats:
                v = classify_stats[k]
                if k not in overall_stats:
                    overall_stats[k] = list()
                overall_stats[k].append(v)
            print_string += "%.3f/%.3f/%.3f," \
                            % (1 - classify_stats["single-label-error"],
                               classify_stats["micro-F1"],
                               classify_stats["macro-F1"])

        for k in overall_stats:
            overall_stats[k] = np.mean(overall_stats[k])

        print("=======")
        print(epoch, "overall: %s" % (utils.dict_to_string_4_print(overall_stats)))
        print("for Google Sheets, split by comma")
        print_string += "%.3f/%.3f/%.3f" \
                        % (1 - overall_stats["single-label-error"],
                           overall_stats["micro-F1"],
                           overall_stats["macro-F1"])
        print(print_string)

def error_unseen_best():
    random_group = dataloader.get_random_group(config.zhang15_dbpedia_class_random_group_path)
    # random_group = dataloader.get_random_group(config.news20_class_random_group_path)
    # random_group = dataloader.get_random_group(config.chen14_elec_class_random_group_path)

    overall_stats = dict()
    print_string = ""

    for i, rgroup in enumerate(random_group):

        best_stats = None
        for epoch in range(11):
            filename = "../results/unseen_full_zhang15_dbpedia_kg3_cluster_3group_random%d_unseen%s_max%d_cnn_negative%dincrease3_randomtext/logs/test_%d.npz" \
                       % (i + 1, "-".join(str(_) for _ in rgroup[1]), 80, 5, epoch)

            if not os.path.exists(filename):
                continue

            classify_stats = classify_single_label_for_unseen(filename, rgroup)

            if best_stats == None or classify_stats["single-label-error"] < best_stats["single-label-error"]:
                best_stats = classify_stats

        for k in best_stats:
            v = best_stats[k]
            if k not in overall_stats:
                overall_stats[k] = list()
            overall_stats[k].append(v)
        print_string += "%.3f/%.3f/%.3f," \
                        % (1 - best_stats["single-label-error"],
                           best_stats["micro-F1"],
                           best_stats["macro-F1"])

    for k in overall_stats:
        overall_stats[k] = np.mean(overall_stats[k])

    print("=======")
    print("overall: %s" % (utils.dict_to_string_4_print(overall_stats)))
    # print("for Google Sheets, split by comma")
    print_string += "%.3f/%.3f/%.3f" \
                    % (1 - overall_stats["single-label-error"],
                       overall_stats["micro-F1"],
                       overall_stats["macro-F1"])
    print(print_string)


if __name__ == "__main__":
    # error_deprecated()
    # error_seen()
    error_unseen()
    # error_unseen_best()
    # classify_single_label_for_seen("../results/unseen_selected_tfidf_news20_kg3_cluster_3group_only_random2_unseen20-4-14-2-10_max200_cnn_negative5increase3_randomtext/logs/test_0.npz")
    pass
