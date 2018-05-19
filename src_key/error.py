import numpy as np

import utils


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
    try:
        print("all new: %s" % (utils.dict_to_string_4_print(stats)))
    except:
        print("all new: error")

if __name__ == "__main__":
    reject_list = list()
    for i in range(1, 11):
        # calculate_error("../results/key_zhang15_dbpedia_4of4/logs/test_5_att.npz")
        # filename = "../results/key_zhang15_dbpedia_kg3_random%d_unseen0.25_max100_cnn_negative%d/logs/test_%d.npz" \
        #            % (5, 7, 10)
        filename = "../results/selected_zhang15_dbpedia_kg3_random%d_unseen0.25_max50_cnn_negative-1_randomtext/logs/test_10.npz" \
                    % (i)

        # pred_seen, pred_unseen, pred_both, gt_both = classify_single_label(filename)
        # classify_multiple_label(filename)
        pred_reject = rejection_single_label(filename)
        # reject_then_classify_single_label(filename, pred_reject)
        reject_list.append(pred_reject)

    pred_dict = [dict(), dict(), dict()]
    for reject in reject_list:
        for sidx in range(1, 4):
            for mea in reject[sidx]:
                if mea not in pred_dict[sidx - 1]:
                    pred_dict[sidx - 1][mea] = 0
                pred_dict[sidx - 1][mea] += reject[sidx][mea]


    for sidx in range(3):
        for mea in pred_dict[sidx]:
            pred_dict[sidx][mea] /= 10

    print("average overall ============================")
    print(utils.dict_to_string_4_print(pred_dict[0]))
    print("average seen ============================")
    print(utils.dict_to_string_4_print(pred_dict[1]))
    print("average unseen ============================")
    print(utils.dict_to_string_4_print(pred_dict[2]))

    pass

