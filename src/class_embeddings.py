from text_to_uri import *
import numpy as np
import re
import csv
import urllib.request, urllib.parse
import xml.etree.ElementTree as ET

# --------------------- Global Variables ---------------------
TransE_entity2id = dict()
TransE_id2entity = dict()
ConceptNet_entity2id = dict()
ConceptNet_id2entity = dict()

# --------------------- Preparation --------------------------
def prepare():
	prepare_TransE()
	prepare_ConceptNet()

def prepare_TransE():
	global TransE_entity2id, TransE_id2entity
	# Read ids of all entities
	with open("../wordEmbeddings/entity2id.txt", "r", encoding="utf8") as f1:
		content = f1.readlines()
		content = [x.strip() for x in content]
		entity_num = float(len(content))
		for line in content:
			st = line.split()[0][1:-1]
			x = int(line.split()[1])
			TransE_entity2id[st]=x
			TransE_id2entity[x]=st			
	print("Finish preparing TransE indices")

def prepare_ConceptNet():
	global ConceptNet_entity2id, ConceptNet_id2entity
	with open("../wordEmbeddings/numberbatch-en.txt", "r", encoding="utf8") as f1:
		for i, line in enumerate(f1):
			if i == 0: # Skip the first line
				continue	
			entity = line.split()[0]
			ConceptNet_entity2id[entity] = i
			ConceptNet_id2entity[i] = entity
	print("Finish preparing ConceptNet indices")

# --------------------- Main (router) -------------------------
def get_vector_of_class(class_label, class_description, method): 
	# Return the corresponding uri and its vector
	if method == 'DBpedia':
		return DBpedia_vector(class_label, class_description)
	elif method == 'ConceptNet':
		return ConceptNet_vector(class_label, class_description)
	else:
		assert False, "Unsupported method '" + str(method) + "'"

def get_vector_by_uri(method, uri):
	if method == 'DBpedia':
		if uri in TransE_entity2id:
			return TransE_vector(TransE_entity2id[uri])
		else:
			return None
	elif method == 'ConceptNet':
		if uri in ConceptNet_entity2id:
			return ConceptNet_lookup_vector(ConceptNet_entity2id[uri])
		else:
			return None
	else:
		assert False, "Unsupported method '" + str(method) + "'"

# --------------------- DBpedia Vector ------------------------
def DBpedia_vector(class_label, class_description):
	# Replace special characters in string using the %xx escape. 
	# E.g., changing from "châteu" to "ch%C3%A2teu" for using in a url string
	quoted_query = urllib.parse.quote(class_label.encode('utf8'))

	# DBpedia lookup
	results = DBPLookup(query_string = quoted_query)
	for res in results:
		# res[0] is a uri, res[1] is a number of reference counts.
		uri = urllib.parse.unquote(res[0].strip()).replace('"','').replace('|','')
		# print(uri.encode("utf-8"), res[1])
		if uri in TransE_entity2id:
			# print("DBpedia Match:", class_label, "-", uri)
			return uri, TransE_vector(TransE_entity2id[uri])
	# print("LinkingError: Cannot find the corresponding entity for", class_label,"in DBpedia")
	return None, None

def DBPLookup(query_string, maxHits = 10, query_class = None):
	# Request a result via DBpedia lookup API
	if query_class is not None:
		url_string = "http://lookup.dbpedia.org/api/search.asmx/KeywordSearch?QueryClass=%s&QueryString=%s&MaxHits=%d" % (query_class, query_string, maxHits)
	else:
		url_string = "http://lookup.dbpedia.org/api/search.asmx/KeywordSearch?QueryString=%s&MaxHits=%d" % (query_string, maxHits)
	xml = urllib.request.urlopen(url_string).read()
	
	# Parse the result string
	root = ET.fromstring(xml)
	results = []
	for res in root.iter('{http://lookup.dbpedia.org/}Result'):
		uri = res.find('{http://lookup.dbpedia.org/}URI').text
		ref_count = int(res.find('{http://lookup.dbpedia.org/}Refcount').text)
		results.append((uri, ref_count))
	return results 

def TransE_vector(id):
	# Lookup the corresponding vector from the embedding file
	with open("../wordEmbeddings/entity2vec.bern","r") as f1:
		for i, line in enumerate(f1):
			if i == id:
				return np.array([float(x) for x in line.strip().split('\t')])
		assert False, "TransE_vector(id): ID is out of bound"

# --------------------- ConceptNet Vector ---------------------
def ConceptNet_vector(class_label, class_description):
	expected_uri = standardized_uri('en', class_label)
	expected_uri = re.sub(r"/c/en/", '', expected_uri)
	if expected_uri in ConceptNet_entity2id:
			# print("ConceptNet Match:", class_label, "-", expected_uri)
			return expected_uri, ConceptNet_lookup_vector(ConceptNet_entity2id[expected_uri])
	# print("LinkingError: Cannot find the corresponding entity for", class_label,"in ConceptNet")
	return None, None

def ConceptNet_lookup_vector(id):
	# Lookup the corresponding vector from the embedding file
	with open("../wordEmbeddings/numberbatch-en.txt", "r", encoding="utf8") as f1:
		for i, line in enumerate(f1):
			if i == id:
				return np.array([float(x) for x in line.strip().split()[1:]])
		assert False, "ConceptNet_lookup_vector(id): ID is out of bound"

# --------------------- Main Operation ------------------------
if __name__ == "__main__":
	prepare()
	print(get_vector_of_class("Functional Analysis", "", "DBpedia"))
	print(get_vector_of_class("Functional Analysis", "", "ConceptNet"))
	print(get_vector_of_class("Pricing of Securities", "", "DBpedia"))
	print(get_vector_of_class("Pricing of Securities", "", "ConceptNet"))

	# input_file = csv.DictReader(open("../data/arxiv/classLabels.csv"))
	# for row in input_file:
	# 	c_label = row['ClassLabel']
	# 	uri, _ = get_vector_of_class(c_label, '', 'DBpedia')
	# 	print(str(uri))

