import pickle
import numpy as np


if __name__ == "__main__":
    # kg_vector_1 = pickle.load(open("../wordEmbeddings/KG_VECTORS_1.pickle", "rb"))
    # print(kg_vector_1.keys())
    # kg_vector_2 = pickle.load(open("../wordEmbeddings/KG_VECTORS_2.pickle", "rb"))
    # print(kg_vector_2["/c/en/building"]['/c/en/syriac'].shape)

    data = np.load("../results/key_zhang15_dbpedia_4of4/logs/test_5.npz")
    print(data["pred_unseen"].shape)
    print(data["gt_unseen"].shape)
    print(data["align_unseen"].shape)
    print(data["pred_seen"].shape)
    print(data["gt_seen"].shape)
    print(data["align_seen"].shape)


