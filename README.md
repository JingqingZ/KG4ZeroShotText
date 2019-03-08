# Integrating Semantic Knowledge to Tackle Zero-shot Text Classification. NAACL-HLT 2019. (Accepted)
### Jingqing Zhang, Piyawat Lertvittayakumjorn, Yike Guo

##### Jingqing and Piyawat contributed equally to this project.

Paper link: Pending

## Contents
1. [Abstract](#Abstract)
2. [Code](#Code)
3. [Acknowledgement](#Acknowledgement)
4. [Citation](#Citation)

<h2 id="Abstract">Abstract</h2>

<h2 id="Code">Code</h2>

### Quick start
```bash
cd src_reject
sh run.sh
```

### Checklist

In order to run the code, please check the following issues.

- [x] Package Dependencies:
    - Python 3.5
    - TensorFlow 1.11.0
    - [TensorLayer] 1.11.3
    - Numpy 1.14.5
    - Pandas 0.21.0
    - NLTK 3.2.5
- [x] Original Dataset
- [x] Intermediate files (after preprocessing)

[TensorLayer]: https://github.com/tensorlayer/tensorlayer

### How to train / test Phase 1

An example:
```bash
python3 train_seen.py \
        --data dbpedia \
        --unseen 0.5 \
        --model vw \
        --ns 0 --ni 0 --sepoch 1 \
        --rgidx 1 --train 1
```

The arguments of the commands represent
* `data`: Dataset, either `dbpedia` or `20news`.
* `unseen`: Rate of unseen classes, either 0.25 or 0.5.
* `model`: The model specified to train the model. This argument can only be
    * `vwonly`: the inputs are embedding of words (from text)
* `ns`: Integer, the ratio of positive and negative samples, the higher the more negative samples
* `ni`: Integer, the speed of increasing negative samples during training per epoch
* `sepoch`: Repeat training of each epoch for several times. The ratio of positive/negative samples and learning rate will keep consistent in one epoch no mather how many times the epoch is repeated.
* `rgidx`: Random group starting index: e.g. if 5, the training will start from the 5th random group, by default 1. This argument is used when the program is accidentally interrupted.
* `train`: In Phase 1, this argument does not affect the program. The program will run training and testing together.
* `gpu`: Optional, GPU occupation percentage, by default 1.0, which means full occupation of available GPUs.
* `baseepoch`: Optional, you may want to specify which epoch to test.

### How to train / test Phase 2

An example:
```bash
python3 train_unseen.py \
        --data 20news \
        --unseen 0.5 \
        --model vwvcvkg \
        --ns 2 --ni 2 --sepoch 10 \
        --rgidx 1 --train 1
```

The arguments of the commands represent
* `data`: Dataset, either `dbpedia` or `20news`.
* `unseen`: Rate of unseen classes, either 0.25 or 0.5.
* `model`: The model specified to train the model. This argument can be (correspond with Table 6 in the paper)
    * `kgonly`: the inputs are the relationship vectors which are extracted from knowledge graph (KG).
    * `vcvkg`: the inputs contain the embedding of class labels and the relationship vectors.
    * `vwvkg`: the inputs contain the embedding of words (from text) and the relationship vectors.
    * `vwvc`: the inputs contain the embedding of words and class labels.
    * `vwvcvkg`: all three kinds of inputs mentioned above.
* `ns`: Integer, the ratio of positive and negative samples, the higher the more negative samples
* `ni`: Integer, the speed of increasing negative samples during training per epoch
* `sepoch`: Repeat training of each epoch for several times. The ratio of positive/negative samples and learning rate will keep consistent in one epoch no mather how many times the epoch is repeated.
* `rgidx`: Random group starting index: e.g. if 5, the training will start from the 5th random group, by default 1. This argument is used when the program is accidentally interrupted.
* `train`: 1 for training, 0 for testing.
* `gpu`: Optional, GPU occupation percentage, by default 1.0, which means full occupation of available GPUs.
* `baseepoch`: Optional, you may want to specify which epoch to test.

<h2 id="Acknowledgement">Acknowledgement</h2>
We would like to thank Douglas McIlwraith, Nontawat Charoenphakdee, and three anonymous reviewers for helpful suggestions. Jingqing and Piyawat would also like to thank the support from [LexisNexis HPCC Systems Academic Program] and Anandamahidol Foundation, respectively.

[LexisNexis HPCC Systems Academic Program]: https://hpccsystems.com/community/academics

<h2 id="Citation">Citation</h2>
Pending



