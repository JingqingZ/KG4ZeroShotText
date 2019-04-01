# Integrating Semantic Knowledge to Tackle Zero-shot Text Classification. NAACL-HLT 2019. Oral (Accepted)
### Jingqing Zhang, Piyawat Lertvittayakumjorn, Yike Guo

##### Jingqing and Piyawat contributed equally to this project.

Paper link: [arXiv:1903.12626](https://arxiv.org/abs/1903.12626)

## Contents
1. [Abstract](#Abstract)
2. [Code](#Code)
3. [Acknowledgement](#Acknowledgement)
4. [Citation](#Citation)

<h2 id="Abstract">Abstract</h2>
Insufficient or even unavailable training data of emerging classes 
is a big challenge of many classification tasks, including text 
classification. Recognising text documents of classes that have 
never been seen in the learning stage, so-called zero-shot text 
classification, is therefore difficult and only limited previous 
works tackled this problem. In this paper, we propose a two-phase 
framework together with data augmentation and feature augmentation 
to solve this problem. Four kinds of semantic knowledge 
(word embeddings, class descriptions, class hierarchy, and a general 
knowledge graph) are incorporated into the proposed framework to 
deal with instances of unseen classes effectively. Experimental 
results show that each and the combination of the two phases 
clearly outperform baseline and recent approaches in classifying 
real-world texts under the zero-shot scenario.

<h2 id="Code">Code</h2>

### Checklist

In order to run the code, please check the following issues.

- [x] Package dependencies:
    - Python 3.5
    - TensorFlow 1.11.0
    - [TensorLayer] 1.11.0
    - Numpy 1.14.5
    - Pandas 0.21.0
    - NLTK 3.2.5
    - tqdm 2.2.3
    - [gensim](https://pypi.org/project/gensim/) 3.7.1
    - [language-check](https://pypi.org/project/language-check/) 1.1
- [x] Download original datasets
    - [GloVe.6B.200d](https://nlp.stanford.edu/projects/glove/)
    - [ConceptNet v5.6.0](https://github.com/commonsense/conceptnet5/wiki/Downloads)
    - [DBpedia ontology dataset](https://github.com/zhangxiangxiao/Crepe)
    - [20 Newsgroups original 19997 docs](http://qwone.com/~jason/20Newsgroups/)
- [x] Check [config.py] and update the locations of data files accordingly. The [config.py] also defines the locations of intermediate files.
- [x] The intermediate files already provided in this repo
    - [classLabelsDBpedia.csv](data/zhang15/dbpedia_csv/classLabelsDBpedia.csv): A summary of classes in DBpedia and linked nodes in ConceptNet.
    - [classLabels20news.csv](data/20-newsgroups/clean/classLabels20news.csv): A summary of classes in 20news and linked nodes in ConceptNet.
    - Random selection of seen/unseen classes in DBpedia with unseen rate [0.25](data/zhang15/dbpedia_csv/dbpedia_random_group_0.25.txt) and [0.5](data/zhang15/dbpedia_csv/dbpedia_random_group_0.5.txt).
    - Random selection of seen/unseen classes in 20news with unseen rate [0.25](data/20-newsgroups/clean/20news_random_group_0.25.txt) and [0.5](data/20-newsgroups/clean/20news_random_group_0.5.txt).
    - Note: seen/unseen classes were randomly selected for 10 times. You may randomly generate another 10 groups of seen/unseen classes.
- [x] Some intermediate files have been uploaded to [supplementary].
- [x] Other intermediate files should be generated automatically when they are needed.

Please feel free to raise an issue if you find any difficulty to run the code or get the intermediate files.

[supplementary]: https://imperiallondon-my.sharepoint.com/:f:/g/personal/jz9215_ic_ac_uk/EmI6JPnwLW5Lu33WbQ4nuEUBSkoc9DI9EcHWassF-QxSZQ?e=oxuXI0
[TensorLayer]: https://github.com/tensorlayer/tensorlayer
[config.py]: src_reject/config.py
[playground.py]: src_reject/playground.py

### How to perform data augmentation

An example:
```bash
python3 topic_translation.py \
        --data dbpedia \
        --nott 100
```

The arguments of the command represent
* `data`: Dataset, either `dbpedia` or `20news`.
* `nott`: No. of original texts to be translated into all classes except the original class. If `nott` is not given, all the texts in the training dataset will be translated. 

The location of the result file is specified by config.\{zhang15_dbpedia, news20\}_train_augmented_aggregated_path.


### How to perform feature augmentation / create v_{w,c}

An example:
```bash
python3 kg_vector_generation.py --data dbpedia 
```
The argument of the command represents
* `data`: Dataset, either `dbpedia` or `20news`.

The locations of the result files are specified by config.\{zhang15_dbpedia, news20\}_kg_vector_dir.

### How to train / test Phase 1

- Without data augmentation: an example
```bash
python3 train_reject.py \
        --data dbpedia \
        --unseen 0.5 \
        --model vw \
        --nepoch 3 \
        --rgidx 1 \
        --train 1
```

- With data augmentation: an example
```bash
python3 train_reject_augmented.py \
        --data dbpedia \
        --unseen 0.5 \
        --model vw \
        --nepoch 3 \
        --rgidx 1 \
        --naug 100 \
        --train 1
```

The arguments of the command represent
* `data`: Dataset, either `dbpedia` or `20news`.
* `unseen`: Rate of unseen classes, either `0.25` or `0.5`.
* `model`: The model to be trained. This argument can only be
    * `vw`: the inputs are embedding of words (from text)
* `nepoch`: The number of epochs for training
* `train`: In Phase 1, this argument does not affect the program. The program will run training and testing together.
* `rgidx`: Optional, Random group starting index: e.g. if 5, the training will start from the 5th random group, by default `1`. This argument is used when the program is accidentally interrupted.
* `naug`: The number of augmented data per unseen class

The location of the result file (pickle) is specified by config.rejector_file. The pickle file is actually a list of 10 sublists (corresponding to 10 iterations). Each sublist contains predictions of each test case (1 = predicted as seen, 0 = predicted as unseen).

### How to train / test the traditional classifier in Phase 2

An example:
```bash
python3 train_seen.py \
        --data dbpedia \
        --unseen 0.5 \
        --model vw \
        --sepoch 1 \
        --train 1
```

The arguments of the command represent
* `data`: Dataset, either `dbpedia` or `20news`.
* `unseen`: Rate of unseen classes, either `0.25` or `0.5`.
* `model`: The model to be trained. This argument can only be
    * `vw`: the inputs are embedding of words (from text)
* `sepoch`: Repeat training of each epoch for several times. The ratio of positive/negative samples and learning rate will keep consistent in one epoch no matter how many times the epoch is repeated.
* `train`: For the traditional classifier, this argument does not affect the program. The program will run training and testing together.
* `rgidx`: Optional, Random group starting index: e.g. if 5, the training will start from the 5th random group, by default `1`. This argument is used when the program is accidentally interrupted.
* `gpu`: Optional, GPU occupation percentage, by default `1.0`, which means full occupation of available GPUs.
* `baseepoch`: Optional, you may want to specify which epoch to test.

### How to train / test the zero-shot classifier in Phase 2

An example:
```bash
python3 train_unseen.py \
        --data 20news \
        --unseen 0.5 \
        --model vwvcvkg \
        --ns 2 --ni 2 --sepoch 10 \
        --rgidx 1 --train 1
```

The arguments of the command represent
* `data`: Dataset, either `dbpedia` or `20news`.
* `unseen`: Rate of unseen classes, either `0.25` or `0.5`.
* `model`: The model to be trained. This argument can be (correspond with Table 6 in the paper)
    * `kgonly`: the inputs are the relationship vectors which are extracted from knowledge graph (KG).
    * `vcvkg`: the inputs contain the embedding of class labels and the relationship vectors.
    * `vwvkg`: the inputs contain the embedding of words (from text) and the relationship vectors.
    * `vwvc`: the inputs contain the embedding of words and class labels.
    * `vwvcvkg`: all three kinds of inputs mentioned above.
* `train`: 1 for training, 0 for testing.
* `sepoch`: Repeat training of each epoch for several times. The ratio of positive/negative samples and learning rate will keep consistent in one epoch no matter how many times the epoch is repeated.
* `ns`: Optional, Integer, the ratio of positive and negative samples, the higher the more negative samples, by default `2`. 
* `ni`: Optional, Integer, the speed of increasing negative samples during training per epoch, by default `2`.
* `rgidx`: Optional, Random group starting index: e.g. if 5, the training will start from the 5th random group, by default `1`. This argument is used when the program is accidentally interrupted.
* `gpu`: Optional, GPU occupation percentage, by default `1.0`, which means full occupation of available GPUs.
* `baseepoch`: Optional, you may want to specify which epoch to test.

<h2 id="Acknowledgement">Acknowledgement</h2>
We would like to thank Douglas McIlwraith, Nontawat Charoenphakdee, 
and three anonymous reviewers for helpful suggestions. Jingqing and 
Piyawat would also like to thank the support from 
LexisNexis&reg; Risk Solutions HPCC Systems&reg; academic program
and Anandamahidol 
Foundation, respectively.

<h2 id="Citation">Citation</h2>

    @inproceedings{zhangkumjornZeroShot,
        title = "Integrating Semantic Knowledge to Tackle Zero-shot Text Classification",
        author = "Zhang, Jingqing and
        Lertvittayakumjorn, Piyawat and 
        Guo, Yike",
        booktitle = "Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Long Papers)",
        month = jun,
        year = "2019",
        address = "Minneapolis, USA",
        publisher = "Association for Computational Linguistics",
    }



