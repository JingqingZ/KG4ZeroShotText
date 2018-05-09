import pickle, json, csv
import urllib.request, urllib.parse

class_uri = ['/c/en/company',
			'/c/en/education',
			'/c/en/artist',
			'/c/en/athlete',
			'/c/en/officer',
			'/c/en/transport',
			'/c/en/building',
			'/c/en/nature',
			'/c/en/village',
			'/c/en/animal',
			'/c/en/plant',
			'/c/en/album',
			'/c/en/film',
			'/c/en/writing']

def get_edges_of(uri, rel = None):
	url_string = 'http://api.conceptnet.io/query?node=' + uri + '&other=/c/en'
	if rel is not None:
		url_string += '&rel=' + rel
	json_text = urllib.request.urlopen(url_string).read()
	print(json_text)

# print(len(class_uri))
# get_edges_of('/c/en/cat', '/r/IsA')

# write_csvfile = open('../wordEmbeddings/conceptnet-assertions-en-5.6.0.csv', 'w', encoding = "utf8", newline= '')
# writer = csv.writer(write_csvfile, delimiter='\t')
# with open('../wordEmbeddings/conceptnet-assertions-5.6.0.csv', 'r', encoding = "utf8") as csvfile:
# 	reader = csv.reader(csvfile, delimiter='\t')
# 	for line in reader:
# 		if '/c/en/' in line[2] and '/c/en/' in line[3]:
# 			# print(line)
# 			writer.writerow(line)
# write_csvfile.close()



write_csvfile = open('../wordEmbeddings/conceptnet-assertions-en-filter-5.6.0.csv', 'w', encoding = "utf8", newline= '')
writer = csv.writer(write_csvfile, delimiter='\t')
with open('../wordEmbeddings/conceptnet-assertions-en-5.6.0.csv', 'r', encoding = "utf8") as csvfile:
	reader = csv.reader(csvfile, delimiter='\t')
	for line in reader:
		if line[1] in ['/r/IsA', '/r/PartOf', '/r/AtLocation', '/r/RelatedTo']:
			writer.writerow(line)
write_csvfile.close()
