# all code here are just for fun

import numpy as np

import config

if __name__ == "__main__":
    data = np.load(config.wiki_train_state_npz_path)
    print(data["state"].shape)
    data = np.load(config.wiki_test_state_npz_path)
    print(data["state"].shape)
