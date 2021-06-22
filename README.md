# 🐠<sup>🐠</sup> NEMO<sup>2</sup> - Neural Modeling for (Hebrew) Named Entities and Morphology


Table of Contents
=================
* [Introduction](#introduction)
* [Main Features](#main-features)
* [Requirements](#requirements)
* [Setup](#setup)
* [Basic Usage](#basic-usage)
* [Models and Scenarios](#models-and-scenarios)
* [Important Notes](#important-notes)
* [Training Your Own Model](#training-your-own-model)
* [Morpheme and Word Embeddings](#morpheme-and-word-embeddings)
* [Evaluation](#evaluation)
* [Ben-Mordecai Corpus](#ben-mordecai-corpus)
* [Citations](#citations)


## Introduction
Code and models for neural modeling of Hebrew NER. Described in the TACL paper ["*Neural Modeling for Named Entities and Morphology (NEMO<sup>2</sup>)"*](https://arxiv.org/abs/2007.15620) along with extensive experiments on the different modeling scenarios provided in this repository.


## Main Features
1. Trained on the Hebrew NER and Morphology [NEMO corpus](https://github.com/OnlpLab/NEMO-Corpus) of gold annotated Modern Hebrew news articles. 
1. Multiple modeling options to go from raw Hebrew text to morpheme and/or token-level NER boundaries.
1. Neural model implementation of [NCRF++](https://github.com/jiesutd/NCRFpp)
1. [bclm](https://github.com/OnlpLab/bclm) is used for reading and transforming morpho-syntactic information layers.


## Requirements
1. `python>=3.6`
1. `torch=1.0`
1. `networkx`
1. `yap`: https://github.com/OnlpLab/yap (don't forget `export GOPATH=</path/to/yapproj>`)
1. `bclm>=1.0.0`: http://github.com/OnlpLab/bclm 


## Setup
1. Install all requirements, preferably in a virtual env.
1. Clone the repo.
1. Change to the repo directory: `cd NEMO`
1. Unpack model files: `gunzip data/*.gz`
1. Change `YAP_PATH` in `config.py` to the path of your local `yap` executable.


## Basic Usage
1. All you need to do is run `nemo.py` with a specific command (scenario), on a text file of Hebrew sentences separated by a line-break.
1. You can run a neural NER model directly, or choose a full end-to-end scenario that includes morphological segmentation and alignments (described fully in the [next section](#models-and-scenarios)). e.g.:
    * the `run_ner_model` command with the `token-single` model will tokenize sentences and run the `token-single` model: 
        - ```python nemo.py run_ner_model token-single example.txt example_output.txt```
    * the `morph_hybrid` command runs the end-to-end segmentation and NER pipeline which provided our best performing morpheme-level NER boundaries:  
        - ```python nemo.py morph_yap morph example.txt example_output_MORPH.txt```
1. You can find outputs of different commands on [example.txt](./example.txt) in: [example_output_MORPH_HYBRID_ALIGN_TOKENS.txt](./example_output_MORPH_HYBRID_ALIGN_TOKENS.txt), [example_output_MORPH_HYBRID.txt](./example_output_MORPH_HYBRID.txt), [example_output_MORPH_YAP.txt](./example_output_MORPH_YAP.txt), [example_output_MULTI_ALIGN.txt](./example_output_MULTI_ALIGN.txt), [example_output_SINGLE.txt](./example_output_SINGLE.txt)
1. For a full list of the available commands please consult the [next section](#models-and-scenarios) and the inline documentation at the end of `nemo.py`. 
1. Please use only the regular and not the `*_oov` models (which contain embeddings only for words that appear in the NEMO corpus). In other words, unless you use the model to replicate our results on the Hebrew treebank, always use e.g. `token-multi` and not `token-multi_oov`. 


## Models and Scenarios
Models are all standard Bi-LSTM-CRF with char encoding (LSTM/CNN) of [NCRFpp](https://github.com/jiesutd/NCRFpp) with pre-trained fastText embeddings. Differences between models lay in:
1. **Input units**: morphemes `morph` vs. tokens `token-*`
1. **Output label set**: `token-single` single sequence labels  (e.g. `B-ORG`) vs. `token-multi` multi-labels (atomic labels, e.g. `O-ORG^B-ORG^I-ORG`) that predict, in order, the labels for the morphemes the token is made of.

Token-based Models         |  Morpheme-based Model
:-------------------------:|:---------------------:
<img src="./docs/token_models.svg" alt="token-based models" width="345" />  |  <img src="./docs/morph_model.svg" alt="morpheme-based Model" width="345" /> 

Morphemes must be predicted. This is done by performing morphological disambiguation (*MD*). We offer two options to do so: 
1. **Standard pipeline**: MD using YAP. This is used in the `morph_yap` command, which runs our `morph` NER model on the output of YAP joint segmentation.
1. **Hybrid pipeline**: MD using our best performing *Hybrid* approach, which uses the output of the `token-multi` model to reduce the MD option space. This is used in `morph_hybrid`, `multi_align_hybrid` and `morph_hybrid_align_tokens`. We will explain these scenarios next.

|MD Approach        |  Commands |
|                --:|:---------------------:|
| Standard <img src="./docs/standard_diagram.png" alt="Standard MD" width="345" /> |  `morph_yap`
| Hybrid <img src="./docs/hybrid_diagram.png" alt="Hybrid MD" width="345" /> <br> <img src="./docs/lattice_pruning.png" alt="Hybrid MD" width="345" /> | `morph_hybrid`,<br>`multi_align_hybrid`,<br>`morph_hybrid_align_tokens`|

Finally, to get our desired output (tokens/morphemes), we can choose between different scenarios, some involving extra post-processing alignments:
1. To get morpheme-level labels we have two options:
    * Run our `morph` NER model on predicted morphemes: Commands: `morph_yap` or `morph_hybrid` (better). 
    * `token-multi` labels can be aligned with predicted morphemes to get morpheme-level boundaries. Command: `multi_align_hybrid`.
    
|Run `morph` NER on Predicted Morphemes        |  Multi Predictions Aligned with Predicted Morpheme  |
| :--:|:---------------------:
|<img src="./docs/morph_ner.png" alt="Morph NER on Predicted Morphemes" width="175" /> |  <img src="./docs/multi_align_morph.png" alt="Multi Predictions Aligned with Predicted Morpheme" width="345" /> |
|`morph_yap`,`morph_hybrid` | `multi_align_hybrid` |

2. To get token-level labels we have three options:
    *  `run_ner_model` command with `token-single` model.
    * the predicted labels of the `token-multi` can be mapped to `token-single` labels to get standard token-single output. The command `multi_to_single` does this end-to-end.
    * Morpheme-level output can be aligned back to token-level boundaries. Command: `morph_hybrid_align_tokens` (this achieved best token-level results in our experiments). 
    
| Run `token-single`        |  Map `token-multi` to `token-single` | Align `morph` NER with Tokens  | 
|:--:|:---------------------:|:---:|
|<img src="./docs/token_single.png" alt="Run token-single" width="175" /> |  <img src="./docs/multi_to_single.png" alt="Map token-multi to token-single" width="345" /> | <img src="./docs/morph_align_tokens.png" alt="Align morph NER with Tokens" width="345" /> |
|`run_ner_model token-single` | `multi_to_single` | `morph_hybrid_align_tokens` |

* Note: while the `morph_hybrid*` scenarios offer the best performance, they are less efficient since they requires running both `morph` and `token-multi` NER models.


## Important Notes
1. NCRFpp was great for our experiments on the NEMO corpus (which is given, constant, data), but it holds some caveats for real life scenarios of arbitrary text:
    * fastText is not used on the fly to obtain vectors for OOV words (i.e. those that were not seen in our Wikipedia corpus). Instead, it is used as a regular embedding matrix. Hence the full generalization capacities of fastText, as shown in our experiments, are not available in the currently provided models, which will perform slightly worse than they could on arbitrary text. In our experiments we created such a matrix in advance with all the words in the NEMO corpus and used it during training. Information regarding training your own model with your own vocabulary in the [next section](#training-your-own-model).
    * We currently do not provide an API, only file input/outputs. The pipeline works in the background through temp files, you can choose to delete these by default using the `DELETE_TEMP_FILES` config parameter.  
1. In the near future we plan to publish a cleaner end-to-end implementation, including use of our new [AlephBERT](https://github.com/OnlpLab/AlephBERT) pre-trained Transformer models. 
1. For archiving and reproducibility purposes, our original code used for experiments and analysis can be found in the following repos: https://github.com/cjer/NCRFpp, https://github.com/cjer/NER (beware - 2 years of Jupyter notebooks).


## Training your own model
We provide template NCRF++ config files. These files already contain the hyperparameters we used in our training. To train your own model:
1. Copy the config for the variant (token-multi, token-single, morph) you wish to use from the [ncrf_train_configs](./ncrf_train_configs) folder.
1. Change the parameter `word_emb_dir`  to that of an embedding vectors file in standard word2vec textual format. You can use the fastText bin models we make available (in the [next section](#morpheme-and-word-embeddings)) or any other embedding vectors of your choice.
1. Run the following in your shell: 
```bash
python ncrf_main.py --config <path_to_config> --device <gpu_device_number>
```
4. For more information, please consult [NCRF++](https://github.com/jiesutd/NCRFpp) documentation.
5. To evaluate your trained models, please consult the [evaluation section](#evaluation).

## Morpheme and Word Embeddings 
The word embeddings we trained and used in our models are available:
1. Space-delimited tokens (traditional word embeddings): fastText ([bin](https://nlp.biu.ac.il/~danb/heb_embeds/wikipedia.alt_tok.tokenized.fasttext_skipgram.model.bin.gz), [text](https://nlp.biu.ac.il/~danb/heb_embeds/wikipedia.alt_tok.tokenized.fasttext_skipgram.txt.gz)), [GloVe](https://nlp.biu.ac.il/~danb/heb_embeds/wikipedia.alt_tok.tokenized.glove.txt.gz), [word2vec](https://nlp.biu.ac.il/~danb/heb_embeds/wikipedia.alt_tok.tokenized.word2vec_skipgram.txt.gz)
1. Morphemes: fastText ([bin](https://nlp.biu.ac.il/~danb/heb_embeds/wikipedia.alt_tok.yap_form.fasttext_skipgram.model.bin.gz), [text](https://nlp.biu.ac.il/~danb/heb_embeds/wikipedia.alt_tok.yap_form.fasttext_skipgram.txt.gz)), [GloVe](https://nlp.biu.ac.il/~danb/heb_embeds/wikipedia.alt_tok.yap_form.glove.txt.gz), [word2vec](https://nlp.biu.ac.il/~danb/heb_embeds/wikipedia.alt_tok.yap_form.word2vec_skipgram.txt.gz)

These were trained on a [2013 Wiki dump corpus by Yoav Goldberg](https://u.cs.biu.ac.il/~yogo/hebwiki/), which we re-tokenized and then re-parsed using YAP:
1. [Space-delimited tokens](https://nlp.biu.ac.il/~danb/wiki_2013_corpus/wikipedia.alt_tok.tokenized.txt.gz)
1. [Morphemes](https://nlp.biu.ac.il/~danb/wiki_2013_corpus/wikipedia.alt_tok.yap_form.txt.gz), automatic YAP segmentation (using the morpheme FORM as the unit for embedding)
1. [CONLL files](https://nlp.biu.ac.il/~danb/wiki_2013_corpus/wiki_joint.tar.gz) of full morpho-syntactic output of YAP


## Evaluation
To evaluate your predictions against gold use the [ne_evaluate_mentions.py](./ne_evaluate_mentions.py) script. Evaluation looks for exact match of string and entity category, but is slightly different than the standard CoNLL2003 evaluation commonly used for NER. The reason is that predicted segmentation differs from gold, so positional indexes of sequence labels cannot be used. What we do instead, is extract multi-sets of entity mentions and use set operations to compute precision, recall and F1-score. You can find more detailed discussion of evaluation in the NEMO<sup>2</sup> paper.  

To evaluate an output prediction file against a gold file use:
```bash
python ne_evaluate_mentions.py <path_to_gold_ner> <path_to_predicted_ner>
```
If you're within python, just call `ne_evaluate_mentions.evaluate_files(...)` with the same parameters.


## Ben-Mordecai Corpus
In our NEMO<sup>2</sup> paper we also evaluate our models on the [Ben-Mordecai Hebrew NER Corpus](https://www.cs.bgu.ac.il/~elhadad/nlpproj/naama/) (BMC). The 3 random splits we used can be found [here](https://github.com/OnlpLab/HebrewResources/tree/master/BMCNER). 
## Citations

If you use any of the NEMO<sup>2</sup> code, models, embeddings or the NEMO corpus, please cite the NEMO<sup>2</sup> paper:
```bibtex
@article{DBLP:journals/corr/abs-2007-15620,
  author    = {Dan Bareket and
               Reut Tsarfaty},
  title     = {Neural Modeling for Named Entities and Morphology (NEMO{\^{}}2)},
  journal   = {CoRR},
  volume    = {abs/2007.15620},
  year      = {2020},
  url       = {https://arxiv.org/abs/2007.15620},
  archivePrefix = {arXiv},
  eprint    = {2007.15620},
  timestamp = {Mon, 03 Aug 2020 14:32:13 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2007-15620.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

If you use the NEMO<sup>2</sup>'s NER models please also cite NCRF++:
```bibtex
@inproceedings{yang2018ncrf,  
 title={{NCRF}++: An Open-source Neural Sequence Labeling Toolkit},  
 author={Yang, Jie and Zhang, Yue},  
 booktitle={Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics},
 Url = {http://aclweb.org/anthology/P18-4013},
 year={2018}  
}
 ```
 


