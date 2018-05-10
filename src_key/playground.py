import pickle


if __name__ == "__main__":
    kg_vector_1 = pickle.load(open("../wordEmbeddings/KG_VECTORS_1.pickle", "rb"))
    print(kg_vector_1)
