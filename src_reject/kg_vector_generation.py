import pickle, json, requests, csv, copy, os, re
import numpy as np
import pprint as pp
import urllib.request, urllib.parse
from sklearn.metrics.pairwise import cosine_similarity
from json import JSONDecodeError
from text_to_uri import *

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

import config

## Global variables initialisation

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

pos_dict = {'JJ': 'a', 'JJR': 'a', 'JJS': 'a',
           'NN': 'n', 'NNP': 'n', 'NNPS': 'n', 'NNS': 'n',
           'RB': 'r', 'RBR': 'r', 'RBS': 'r',
           'VB': 'v', 'VBD': 'v', 'VBG': 'v', 'VBN': 'v', 'VBP': 'v', 'VBZ': 'v'}

NODES_DATA = dict()
lemmatise_dict = dict()


## Functions

### Category (Class) related functions

class Category:
    
    def __init__(self, label, description, hierarchy): # Create an empty path
        self.label = label.strip().lower()
        self.description = description.strip().lower()
        self.hierarchy = hierarchy.strip().lower().split(';')
        self.nodes = {'the_class':[],
                      'super_class':[],
                      'description':[],
                     }
        self.find_nodes()
    
    def __repr__(self):
        return self.label + ' => \n' + '\n'.join([key + ': ' + str(val) for key, val in self.nodes.items()]) + '\n'
    
    def find_nodes(self):
        # The class
        self.nodes['the_class'] = get_all_nodes_from_label(self.label)
                
        # Super class
        for a_super_class in self.hierarchy:
            self.nodes['super_class'].extend(get_all_nodes_from_label(a_super_class))
        self.nodes['super_class'] = list(set(self.nodes['super_class']))
        
        # Description
        text = nltk.pos_tag(word_tokenize(self.description))
        for token in text:
            if token[1].startswith('NN') and token[0] not in stop_words: # Noun and not stop words
                self.nodes['description'].extend(get_all_nodes_from_label(token[0]))
        self.nodes['description'] = list(set(self.nodes['description']))
       
    def get_all_nodes(self):
        ans = []
        for key, val in self.nodes.items():
            ans.extend(val)
        return set(ans)


def get_class_info(filename):
    with open(filename, encoding = 'utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        ans = [row for row in reader]
        print("No. of classes =", len(ans))
        print("Header =", ans[0].keys())
        return ans

def get_all_nodes_from_label(label):
    ans = []
    if standardized_uri('en', label) in NODES_DATA:
        ans.append(standardized_uri('en', label))
    for token in label.split():
        if token not in stop_words:
            token = lemmatise_ConceptNet_label(token)
            if standardized_uri('en', token) in NODES_DATA and standardized_uri('en', token) not in ans:
                ans.append(standardized_uri('en', token))
    return ans

### ConceptNet (nodes) related functions

class ConceptNet_node:
    
    def __init__(self, uri): # Create a node
        self.uri = remove_word_sense(uri)
        self.label = uri[uri.rfind('/')+1:]
        self.neighbors = {0: set([self.uri]),
                          1: set()}
        
    def find_neighbors(self, hop):
        if hop not in self.neighbors:
            one_hop_less = self.find_neighbors(hop-1)
            ans = set()
            for n in one_hop_less:
                ans = ans.union(NODES_DATA[n].find_neighbors(1))
            ans = ans.difference(self.find_neighbors_within(hop-1))
            self.neighbors[hop] = ans
            print('Finish finding neighbors of ', self.uri, 'hop =', hop)
        return self.neighbors[hop]
    
    def find_neighbors_within(self, hop):
        assert hop >= 0, 'Hop number must be non-negative'
        if hop == 0:
            return self.neighbors[0]
        else:
            return self.find_neighbors(hop).union(self.find_neighbors_within(hop-1))


def get_neighbors_of_cluster(node_set, hop):
    ans = set()
    for n in node_set:
        assert n in NODES_DATA, "Invalid node " + n
        ans = ans.union(NODES_DATA[n].find_neighbors_within(hop))
    return ans

def remove_word_sense(sub):
    if sub.count('/') > 3:
        if sub.count('/') > 4:
            print(sub)
            assert False, "URI error (with more than 4 slashes)"
        sub = sub[:sub.rfind('/')]
    return sub

def get_label_from_uri(uri):
    uri = remove_word_sense(uri)
    return uri[uri.rfind('/')+1:]

def lemmatise_ConceptNet_label(label):
    if '_' in label:
        return label
    else:
        tag = nltk.pos_tag([label])[0][1]
        if tag not in pos_dict:
            return label
        else:
            return lemmatizer.lemmatize(label, pos_dict[tag])

def lemmatise_ConceptNet_uri(uri):
    label = get_label_from_uri(uri)
    lemmatised_label = lemmatise_ConceptNet_label(label)
    return standardized_uri('en', lemmatised_label)

def create_lemmatised_dict(ns): # ns is a set of nodes from read_all_nodes()
    nodes = dict()
    for n in tqdm(ns):
        nodes[n] = lemmatise_ConceptNet_uri(n)
    return nodes

### Loading ConceptNet functions

def read_all_nodes(filename): # get all distinct uri in conceptnet (without part of speech)
    nodes = set()
    with open(filename, 'r', encoding = "utf8") as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for line in tqdm(reader):
            if not line[2].startswith('/c/en/') or not line[3].startswith('/c/en/'): # only relationships with english nodes
                continue
            sub = remove_word_sense(line[2])
            obj = remove_word_sense(line[3])
            nodes.add(sub)
            nodes.add(obj)
    return nodes


def load_one_hop_data(filename, NODES_DATA, rel_list):
    count_edges = 0
    with open(filename, 'r', encoding = "utf8") as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for line in tqdm(reader):
            rel = line[1].strip()
            if rel_list is None or rel in rel_list:
                details = json.loads(line[4])
                w = details['weight']
                if w < 1.0:
                    continue
                if not line[2].startswith('/c/en/') or not line[3].startswith('/c/en/'): # only relationships with english nodes
                    continue
                sub = lemmatise_dict[remove_word_sense(line[2])]
                obj = lemmatise_dict[remove_word_sense(line[3])]
                if sub != obj:
                    NODES_DATA[sub].neighbors[1].add(obj)
                    NODES_DATA[obj].neighbors[1].add(sub)
                    count_edges += 1
    print("Total no. of registered edges =", count_edges)


def load_ConceptNet():
    global lemmatise_dict, NODES_DATA
    
    filename = config.conceptnet_path
    
    # Find all lemmatised nodes
    print("Reading all nodes from ConceptNet")
    ALL_NODES = read_all_nodes(filename)
    print('Before lemmatising, no. of all nodes = ', len(ALL_NODES))
    lemmatise_dict = create_lemmatised_dict(ALL_NODES)
    ALL_NODES = set(lemmatise_dict.values())
    print('After lemmatising, no. of all nodes = ', len(ALL_NODES))
    
    # Create all lemmatised nodes objects in the process
    for n in ALL_NODES:
        NODES_DATA[n] = ConceptNet_node(n)
    del ALL_NODES
    print('Finish creating lemmatised nodes')
    
    # Load one hop data from ConceptNet
    rel_list = ['/r/IsA', '/r/PartOf', '/r/AtLocation', '/r/RelatedTo']
    load_one_hop_data(filename, NODES_DATA, rel_list)    
    print('Finish loading one hop data')

### Creating KG vector function

def get_vector_of(n, all_c_nodes, hop): # n = uri, c = Category_node
    v = np.zeros(3 * hop + 1)
    v[0] = 1.0 if n in all_c_nodes else 0.0
    for i in range(hop):
        have_hops = [n in NODES_DATA[c].find_neighbors(i+1) for c in all_c_nodes]
        if len(have_hops) > 0:
            v[3 * i + 1] = float(any(have_hops))
            v[3 * i + 2] = float(sum(have_hops))
            v[3 * i + 3] = float(np.mean(have_hops))
        else:
            v[3 * i + 1] = 0.0
            v[3 * i + 2] = 0.0
            v[3 * i + 3] = 0.0
    return v


## Main Program
def main_program(class_filename, node_data_filename, kg_vector_dir, kg_vector_prefix):
    # - Load conceptnet
    load_ConceptNet()

    # - Load class data and form a cluster of nodes for each class
    class_nodes = set()
    class_info = get_class_info(class_filename)
    classes = [Category(c['ConceptNet'], c['ClassDescription'], c['Hierarchy']) for c in class_info]
    for c in classes:
        class_nodes = class_nodes.union(c.get_all_nodes())
    print(len(class_nodes), class_nodes)

    for c in classes:
        print(c)

    class_clusters = dict()
    for c in classes:
        class_clusters[c.label] = c.get_all_nodes()
    print(class_clusters)

    # - Find neighbors of nodes in each cluster 

    for c in tqdm(class_nodes):
        print('Processing class', c)
        NODES_DATA[c].find_neighbors(3)

    pickle.dump(NODES_DATA, open(node_data_filename, "wb"))


    # - Calculate KG vectors for each class

    for c in tqdm(classes):
        all_c_nodes = c.get_all_nodes()
        all_neighbors = get_neighbors_of_cluster(all_c_nodes, hop = 3)
        print(c, len(all_neighbors))
        
        vectors = dict()
        for n in all_neighbors:
            # Consider each partition of nodes separately
            vectors[n] = np.concatenate((get_vector_of(n, c.nodes['the_class'], hop = 3), get_vector_of(n, c.nodes['super_class'], hop = 3), get_vector_of(n, c.nodes['description'], hop = 3)), axis = 0) 

        pickle.dump(vectors, open(kg_vector_dir + kg_vector_prefix + c.label + ".pickle", "wb"))
        print('Finish calculating vectors for', c.label)

if __name__ == "__main__":
    print(config.dataset)
    if config.dataset == "dbpedia":
        main_program(config.zhang15_dbpedia_class_label_path, config.zhang15_dbpedia_kg_vector_node_data_path, config.zhang15_dbpedia_kg_vector_dir, config.zhang15_dbpedia_kg_vector_prefix)
    elif config.dataset == "20news":
        main_program(config.news20_class_label_path, config.news20_kg_vector_node_data_path, config.news20_kg_vector_dir, config.news20_kg_vector_prefix)
    else:
        raise Exception("config.dataset %s not found" % config.dataset)
    pass
