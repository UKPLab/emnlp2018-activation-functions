# Is it Time to Swish? Comparing Deep Learning Activation Functions Across NLP tasks 

This repository contains selected code and data for our EMNLP paper on [Comparing Deep Learning Activation Functions Across NLP tasks](http://aclweb.org/anthology/D18-1472). 

## Citation 

```
@inproceedings{Eger:2018:EMNLP,
	title = {Is it Time to Swish? Comparing Deep Learning Activation Functions Across NLP tasks},
	author = {Eger, Steffen and Youssef, Paul and Gurevych, Iryna},
        booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
        year = {2018},
        pages = {4415--4424},
        month = {October},
        location = {Brussels, Belgium},
        publisher = {Association for Computational Linguistics}
}
```
> **Abstract:** Activation functions play a crucial role in neural networks because they are the nonlinearities which have been attributed to the success story of deep learning. One of the currently most popular activation functions is ReLU, but several competitors have recently been proposed or ‘discovered’, including LReLU functions and swish. While most works compare newly proposed activation functions on few tasks (usually from image classification) and against few competitors (usually ReLU), we perform the first large-scale comparison of 21 activation functions across eight different NLP tasks. We find that a largely unknown activation function performs most stably across all tasks, the so-called penalized tanh function. We also show that it can successfully replace the sigmoid and tanh gates in LSTM cells, leading to a 2 percentage point (pp) improvement over the standard choices on a challenging NLP task. 


Contact person: Steffen Eger, eger@ukp.informatik.tu-darmstadt.de

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/


Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

## Data

### Sentence Classification

Exemplarily, we provide the Argumentation Mining Persuasive Essays (PE) dataset used in the Sentence Classification tasks in ``data/ArgMin/infersent/dataset``. We also called it ``AM`` in the paper.
The other sentence classification task (MR, SUBJ, TREC) are available from [SentEval](https://github.com/facebookresearch/SentEval). For space reasons, we only provide the vectorized input for the dev set. For train and test set, you should run [InferSent](https://github.com/facebookresearch/InferSent) (or any other sentence embedding technique) on the provided text files. It is safer to also vectorize the dev set in this case using the same technique as for train and test.

### Sequence Tagging

We provide the same AM data on token-level for the sequence tagging tasks in ``data/ArgMin/sequenceTagging/data``. This data is also available from [here](https://github.com/UKPLab/acl2017-neural_end2end_am) and [here](https://github.com/UKPLab/coling2018-xling_argument_mining) in CONLL format; the original goes back to [this paper](https://www.informatik.tu-darmstadt.de/ukp/research_6/data/argumentation_mining_1/argument_annotated_essays_version_2/index.en.jsp).


## Code

### Sentence Classification

To run a sample script, go e.g. to `PE-infersent/scripts/tanh`. In this folder, all activation functions in the hidden layers use the tanh activation function. In each script, ``meta.py`` is invoked with specific random hyperparameters. ``meta.py`` can be found in ``progs``. This loads the data and invokes ``MLPs.py`` from the ``neuralnets`` directory, which runs the code and does the evaluation. 

**NB** You need to adapt all paths to your local machine. This concerns multiple python files. You may also wish to copy the file ``progs/activations.py`` to your local anaconda path, e.g., ``.local/lib/python3.5/site-packages/keras/activations.py``


### Sequence Tagging

To run a sample script, go e.g., to `runs_seqtagging/scripts/penalized_tanh/`. In this folder, all activation functions in the hidden layers use the penalized tanh activation function. In each script, ``PE-my.py`` is invoked with specific random hyperparameters.  ``PE-my.py`` can be found in ``progs``. This loads the data and invokes ``BiLSTM.py`` from the ``neuralnets`` directory, which runs the code and does the evaluation.

The original BiLSTM code is available from [here](https://github.com/UKPLab/emnlp2017-bilstm-cnn-crf).

**NB** You need to adapt all paths to your local machine. This concerns multiple python files. You may also wish to copy the file ``progs/activations.py`` to your local anaconda path, e.g., ``.local/lib/python3.5/site-packages/keras/activations.py``

