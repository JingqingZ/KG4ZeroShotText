
import re
import time
import operator
import progressbar
import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

########################################
# configuration settings
########################################

datapath_dbpedia = "../../data/dbpedia/csv/"
datapath_wikipedia = "../../data/wikipedia/"

datafile_wikipedia_full = datapath_wikipedia + 'enwiki-20170820-pages-articles.xml'
datafile_wikipedia_simple = datapath_wikipedia + 'simplewiki-20170820-pages-meta-current.xml'
xml_ns = "{http://www.mediawiki.org/xml/export-0.10/}"

clean_wiki_file = "simple_wiki.csv"
clean_wiki_file2 = "simple_wiki_2.csv"

########################################
# preprocessing stage 1
########################################

def choose_first(matchobj):
    content = matchobj.group(0)
    content = content[2:-2]
    content = content.split("|")
    return content[0]

def get_fields(text):

    pattern = re.compile('field = (.*)')
    result = pattern.search(text)

    if result is None:
        raise Exception("The fields are not specified in the wiki.")

    fields_str = result.group(1)
    fields_list = re.findall("\[\[([^\]]*)\]\]", fields_str)

    if len(fields_list) == 0:
        raise Exception("The wiki does not belong to any field.")

    fields_list = [f.lower() for f in fields_list]
    return fields_list

def wiki_cleaner_full(text):
    text = text.replace("\n", "")
    # print(text)

    # remove content inside {{ }}
    text = re.sub(r"(\{\{.*?\}\})", '', text)

    # find the single }}
    loc = text.find("}}")
    text = text[loc + 2:]

    # regex = re.compile("(\[\[.*?\]\])")
    # result = re.findall(regex, text)
    # print(result)


    text = re.sub(r"<ref>.*?</ref>", '', text)
    text = re.sub(r"\{\{.*?\}\}", '', text)
    text = re.sub(r"==.*?\}\}", '', text)
    text = re.sub(r"==.*?==", '', text)
    text = re.sub(r"<.*?>", '', text)
    text = re.sub(r"([=])", '', text)

    # remove web link
    text = re.sub(r"(http.*?\s)", '', text)

    # replace content in [[ A | B ]] with A
    text = re.sub(r"(\[\[.*?\]\])", choose_first, text)

    # replace content in []
    text = re.sub(r"(\[.*?\])", '', text)

    # remove ' " #
    text = re.sub(r"[\"\'#]", '', text)

    # remove content in ()
    text = re.sub(r"(\(.*?\))", '', text)

    # deduplicate multiple space into one
    text = re.sub(r"\s+", ' ', text)

    # keep main body only
    text = text.split("*")[0]

    # split into sentences
    # sentences = text.split(".")
    # print(sentences)

    return text

def wiki_cleaner(text):
    # very tricky
    # find the start of articles
    loc = text.find("\'\'\'")
    text = text[loc:]

    text = text.replace("\n", "")
    # print(text)

    # remove content inside {{ }}
    text = re.sub(r"(\{\{.*?\}\})", '', text)

    # regex = re.compile("(\[\[.*?\]\])")
    # result = re.findall(regex, text)
    # print(result)


    text = re.sub(r"<ref>.*?</ref>", '', text)
    text = re.sub(r"\{\{.*?\}\}", '', text)
    text = re.sub(r"==.*?\}\}", '', text)
    text = re.sub(r"==.*?==", '', text)
    text = re.sub(r"<.*?>", '', text)
    text = re.sub(r"([=])", '', text)

    # remove web link
    text = re.sub(r"(http.*?\s)", '', text)

    # remove content in [[File:]]
    text = re.sub(r"(\[\[File:.*\]\])", '', text)

    # replace content in [[ A | B ]] with A
    text = re.sub(r"(\[\[.*?\]\])", choose_first, text)

    # replace content in []
    text = re.sub(r"(\[.*?\])", '', text)

    # remove ' " #
    text = re.sub(r"[\"\'#]", '', text)

    # remove content in ()
    text = re.sub(r"(\(.*?\))", '', text)

    # deduplicate multiple space into one
    text = re.sub(r"\s+", ' ', text)

    # keep main body only
    text = text.split("*")[0]

    text = re.sub(r"([\[\]])", '', text)

    # split into sentences
    # sentences = text.split(".")
    # print(sentences)

    return text

def wiki_stat(xmlfilename):

    iter_entity = 0
    with progressbar.ProgressBar(max_value=progressbar.UnknownLength) as bar:
        for event, content in ET.iterparse(xmlfilename):
            if not (event == "end" and content.tag == xml_ns + "page"):
                continue

            iter_entity += 1
            if iter_entity % 10000 == 0:
                bar.update(iter_entity)

    print("Num of Entities: %d" % iter_entity)
    return iter_entity

def fields_stat(xmlfilename, num_entities=progressbar.UnknownLength):

    fields_dict = dict()

    iter_entity = 0

    with progressbar.ProgressBar(max_value=num_entities) as bar:
        for event, content in ET.iterparse(xmlfilename):
            if not (event == "end" and content.tag == xml_ns + "page"):
                continue

            try:
                titleelem = content.find(xml_ns + "title")
                title = ''.join(titleelem.itertext())

                textelem = content.find(xml_ns + "revision")

                raw_text = ''.join(textelem.itertext())

                fields = get_fields(raw_text)
                for f in fields:
                    if f not in fields_dict:
                        fields_dict[f] = 0
                    fields_dict[f] += 1

                iter_entity += 1
                if iter_entity % 100 == 0:
                    bar.update(iter_entity)
                if iter_entity > 10000:
                    break

            except:
                pass

    sorted_fields_dict = sorted(fields_dict.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_fields_dict

def simple_prepro(xmlfilename, num_entities=progressbar.UnknownLength):

    # columns = ['title', 'fields', 'text']
    columns = ['title', 'text']
    df = pd.DataFrame(columns=columns)

    fields_dict = dict()

    iter_entity = 0
    pending_data = list()

    with progressbar.ProgressBar(max_value=num_entities) as bar:
        for event, content in ET.iterparse(xmlfilename):
            if not (event == "end" and content.tag == xml_ns + "page"):
                continue

            try:
                titleelem = content.find(xml_ns + "title")
                title = ''.join(titleelem.itertext())

                # if title != "Alan Turing":
                #      continue

                textelem = content.find(xml_ns + "revision")
                textelem = textelem.find(xml_ns + "text")
                raw_text = ''.join(textelem.itertext())

                '''
                fields = get_fields(raw_text)
                for f in fields:
                    if f not in fields_dict:
                        fields_dict[f] = 0
                    fields_dict[f] += 1
                '''

                clean_text = wiki_cleaner(raw_text)

                # pending_data.append([title, "|".join([str(item) for item in fields]), clean_text])
                pending_data.append([title, clean_text])

                iter_entity += 1
                if iter_entity % 1000 == 0:
                    bar.update(iter_entity)

            except:
                pass

    df = pd.DataFrame(pending_data, columns=columns)
    df.to_csv(clean_wiki_file)

########################################
# preprocessing stage 2
########################################

def csv_stat(csvfilename):
    df = pd.read_csv(csvfilename, index_col=0)
    df = df.dropna(how="any")
    df = df[~df["title"].str.contains(":")]
    df["len_text"] = df["text"].apply(len)
    upperbound = df["len_text"].quantile(q=0.9)
    lowerbound = df["len_text"].quantile(q=0.1)
    df = df[df["len_text"] > lowerbound]
    df = df[df["len_text"] < upperbound]
    '''
    print(df)
    print(df.loc[df["len_text"].idxmax()])
    print(df.loc[df["len_text"].idxmin()])
    print("max", df["len_text"].max())
    print("90", df["len_text"].quantile(q=0.9))
    print("60", df["len_text"].quantile(q=0.6))
    print("30", df["len_text"].quantile(q=0.3))
    print("10", df["len_text"].quantile(q=0.1))
    print("median", df["len_text"].median())
    print("mean", df["len_text"].mean())
    print("min", df["len_text"].min())
    df["len_text"].hist(bins=100)
    plt.savefig("wiki_len_text.png")
    '''
    df = df.reset_index(drop=True)
    df.to_csv(clean_wiki_file2)
    return df

if __name__ == "__main__":

    ########################################
    # stage 1: preprocessing simple wiki
    # num_entities = wiki_stat(datafile_wikipedia_simple)
    # all_fields = fields_stat(datafile_wikipedia_full, num_entities=num_entities)
    # print(all_fields)
    # simple_prepro(datafile_wikipedia_simple, num_entities=430040)

    ########################################
    # stage 2: preprocessing simplewiki.csv
    csv_stat(clean_wiki_file)

    pass
