# 🐠 NEMO^2 - Neural Modeling for (Hebrew) Named Entities and Morphology
## Introduction

## Main Features
1. Multiple modeling options to go from raw text to morpheme and/or token-level NER boundaries.
1. Neural model implementation of [NCRFpp](https://github.com/jiesutd/NCRFpp)
1. [bclm](https://github.com/OnlpLab/bclm) is used for reading and transforming morpho-syntactic files.


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
1. All you need to do is run `nemo.py` with specific command (scenario), with a text file of Hebrew sentences separated by a linebreak as input.
    * For `token-single`, which will give you token-level boundaries only: `python nemo.py run_ner_model token-single example.txt example_output.txt`
    * For `morph hybrid`, which provided our best performing morpheme-level boundaries:  `python nemo.py morph_yap morph example.txt example_output_MORPH.txt`
1. For a full description of the available scenarios please consult the inline documentation at the end of `nemo.py`. 
1. Please use only the regular and not the `*_oov` models (which contain embeddings only for words that appear in the NEMO corpus). Unless you use the model to replicate our results on the Hebrew treebank, always use e.g. `token-multi` and not `token-multi_oov`. 

## Important Notes
1. NCRFpp was great for our experiments on the NEMO corpus (which is given, constant, data), but it holds some caveats for real life scenarios of arbitrary text:
    * fastText is not used on the fly to obtain vectors for OOV words (which were not seen in our Wikipedia corpus). Instead, it is used as a regular embedding matrix. In our experiments we created such a matrix in advance with all the words of our , and used during training. Hence the full generalization capacities of fastText, as shown in our experiments, are not available in the currently provided models, which will perform slightly worse than they could on arbitrary text.  
    * We currently do not provide an API, only file input/outputs. The pipeline works in the background through temp files, you can choose to delete these by default using the `DELETE_TEMP_FILES` config parameter.  
1. In the near future we plan to publish a cleaner end-to-end implementation, including use of our new [AlephBERT](https://github.com/OnlpLab/AlephBERT) pre-trained Transformer models. 
1. For archiving and reproducibility, our original code used for experiments and analysis can be found in the following repos: https://github.com/cjer/NCRFpp, https://github.com/cjer/NER.


## Citations

If you use NEMO² or the NEMO corpus, please cite the NEMO² paper:
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