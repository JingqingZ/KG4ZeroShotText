from text_to_uri import *
import numpy as np
import urllib.request, urllib.parse
import xml.etree.ElementTree as ET

# --------------------- Global Variables ---------------------
entity2id = dict()
id2entity = dict()

# --------------------- Preparation --------------------------
def prepare():
	prepare_TransE()

def prepare_TransE():
	global entity2id, id2entity
	# Read ids of all entities
	with open("../wordEmbeddings/entity2id.txt", "r", encoding="utf8") as f1:
		content = f1.readlines()
		content = [x.strip() for x in content]
		entity_num = float(len(content))
		for line in content:
			st = line.split()[0][1:-1]
			x = int(line.split()[1])
			entity2id[st]=x
			id2entity[x]=st			
	print("Finish preparing TransE indices")

# --------------------- Main (router) -------------------------
def get_vector_of_class(class_label, class_description, method):
	if method == 'DBpedia':
		return DBpedia_vector(class_label, class_description)
	elif method == 'ConceptNet':
		return ConceptNet_vector(class_label, class_description)
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
		if uri in entity2id:
			print("Match with:", uri)
			return TransE_vector(entity2id[uri])
	print("LinkingError: Cannot find the corresponding entity in DBpedia")
	return None

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

# --------------------- Main Operation ------------------------
if __name__ == "__main__":
	prepare()
	print(get_vector_of_class("Mathematics", "", "DBpedia"))

