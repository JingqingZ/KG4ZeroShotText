import time
import operator
import numpy as np
import pandas as pd
import progressbar

from SPARQLWrapper import SPARQLWrapper, JSON

def get_type(query):
    try:
        sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        sparql.setReturnFormat(JSON)

        sparql.setQuery(query)  # the previous query as a literal string

        response = sparql.query().convert()

        results = [item["o"]["value"].split("/")[-1] for item in response["results"]["bindings"]]
    except:
        results = []

    return results


def get_all_wiki_type(infilename, outfilename, typefile):
    df = pd.read_csv(infilename, index_col=0)

    df["type"] = ""

    typedict = dict()

    with progressbar.ProgressBar(max_value=len(df.index)) as bar:
        for index, row in df.iterrows():
            title = row["title"].replace(" ", "_")
            query = """
                SELECT ?o
                WHERE {
                  <http://dbpedia.org/resource/%s> a ?o.
                  FILTER regex(?o, "http://dbpedia.org/ontology/")
                }
            """ % title
            typelist = get_type(query)

            for dbtype in typelist:
                if dbtype not in typedict:
                    typedict[dbtype] = 0
                typedict[dbtype] += 1

            # print(title, typelist)
            df.at[index, "type"] = ';'.join(typelist)
            bar.update(index + 1)

            if index % 10 == 0:
                time.sleep(1)

            if index % 1000 == 0:
                time.sleep(10)

            if index % 100 == 0:
                df.to_csv(outfilename)

                sorted_type_dict = sorted(typedict.items(), key=operator.itemgetter(1), reverse=True)
                with open(typefile, 'w') as f:
                    f.write(str(sorted_type_dict))

    df = df[df["type"] != ""]
    df = df.reset_index(drop=True)
    df.to_csv(outfilename)

    sorted_type_dict = sorted(typedict.items(), key=operator.itemgetter(1), reverse=True)
    with open(typefile, 'w') as f:
        f.write(str(sorted_type_dict))

    return df

if __name__ == "__main__":
    get_all_wiki_type("simple_wiki_2.csv", "simple_wiki_type.csv", "all_type.txt")



