from fastapi import FastAPI, Query
from pydantic import BaseModel
import pandas as pd
from typing import Optional, List
from tempfile import mkstemp
import atexit
import os
from config import *
import nemo
import requests
import json
import networkx as nx
import bclm
from ne_evaluate_mentions import fix_multi_biose
from enum import Enum
from io import StringIO

os.environ['CUDA_VISIBLE_DEVICES'] = ''

## NCRF stuff
from utils.data import Data
import torch
from model.seqlabel import SeqLabel
from ncrf_main import evaluate

if 'YAP_API_HOST' in os.environ and os.environ['YAP_API_HOST']:
    YAP_API_HOST = os.environ['YAP_API_HOST']
if 'YAP_API_PORT' in os.environ and os.environ['YAP_API_PORT']:
    YAP_API_PORT = os.environ['YAP_API_PORT']

def get_ncrf_data_object(model_name): #, input_path, output_path):
    data = Data()
    model = MODEL_PATHS[model_name]
    data.dset_dir = model['dset']
    data.load(data.dset_dir)
    data.HP_gpu = False
    #data.raw_dir = input_path
    #data.decode_dir = output_path
    data.load_model_dir = model['model']
    data.nbest = 1
    return data


def load_ncrf_model(data):
    model = SeqLabel(data)
    print('loading model:', data.load_model_dir)
    model.load_state_dict(torch.load(data.load_model_dir, map_location=torch.device('cpu')))
    return model


def ncrf_decode(model, data, temp_input):
    data.raw_dir = temp_input
    #data.decode_dir = temp_output
    data.generate_instance('raw')
    _, _, _, _, _, preds, _ = evaluate(data, model, 'raw', data.nbest)
    if data.nbest==1:
        preds = [sent[0] for sent in preds]
    return preds
    
    
def get_sents(text, tokenized):
    if not tokenized:
        sents = nemo.tokenize_text(text)
    else:
        sents = [sent.split(' ') for sent in text.split('\n')]
    return sents
    
    
def create_input_file(text, path, tokenized):
    sents = get_sents(text, tokenized)
    nemo.write_tokens_file(sents, path, dummy_o=True)
    return sents


## YAP stuff
def yap_request(route, data, host=YAP_API_HOST,
                port=YAP_API_PORT, headers=YAP_API_HEADERS):
    url = YAP_API_URL_TEMPLATE.format(host=host, port=port)
    return requests.get(url+route, data=data, headers=headers).json()


def run_yap_hebma(tokenized_sentences):
    text = "  ".join([" ".join(sent) for sent in tokenized_sentences])
    data = json.dumps({"text": f"{text}  "})
    resp = yap_request('/yap/heb/ma', data)
    return resp['ma_lattice']
    
    
def run_yap_md(ma_lattice):
    data = json.dumps({'amblattice': ma_lattice})
    resp = yap_request('/yap/heb/md', data)
    return resp['md_lattice']
    
    
def run_yap_joint(tokenized_sentences):
    text = "  ".join([" ".join(sent) for sent in tokenized_sentences])
    data = json.dumps({"text": f"{text}  "})
    resp = yap_request('/yap/heb/joint', data)
    return resp


def run_yap_dep(md_lattice):
    data = json.dumps({'disamblattice': md_lattice})
    resp = yap_request('/yap/heb/dep', data)
    return resp['dep_tree']


def get_biose_count(ner_multi_preds):
    bc = []
    for i, sent in enumerate(ner_multi_preds):
        for j, bio in enumerate(sent):
            bc.append([i+1, j+1, bio, len(bio.split('^'))])

    bc = pd.DataFrame(bc, columns=['sent_id', 'token_id',
                                   'biose', 'biose_count'])
    return bc


def prune_lattice(ma_lattice, ner_multi_preds):
    bc = get_biose_count(ner_multi_preds)
    lat = bclm.read_lattices(StringIO(ma_lattice))
    valid_edges = nemo.get_valid_edges(lat, bc, non_o_only=False, keep_all_if_no_valid=True)
    cols = ['sent_id', 'token_id', 'ID1', 'ID2']
    pruned_lat = lat[lat[cols].apply(lambda x: tuple(x), axis=1).isin(valid_edges)]
    pruned_lat = to_lattices_str(pruned_lat)
    return pruned_lat


def to_lattices_str(df, cols = ['ID1', 'ID2', 'form', 'lemma', 'upostag', 'xpostag', 'feats', 'token_id']):
    lat = ''
    for _, sent in df.groupby('sent_id'):
        for _, row in sent[cols].iterrows():
            lat += '\t'.join(row.astype(str).tolist())+'\n'
        lat += '\n'
    return lat
            
    
def soft_merge_bio_labels(ner_multi_preds, md_lattices):
    multitok_sents = bclm.get_sentences_list(get_biose_count(ner_multi_preds), ['biose'])
    md_sents = bclm.get_sentences_list(bclm.get_token_df(bclm.read_lattices(StringIO(md_lattices)), fields=['form'], token_fields=['sent_id', 'token_id'], add_set=False), ['token_id', 'form'])
    new_sents = []
    for (i, mul_sent), (sent_id, md_sent) in zip(multitok_sents.iteritems(), md_sents.iteritems()):
        new_sent = []
        for (bio,), (token_id, forms) in zip(mul_sent, md_sent):
            forms = forms.split('^')
            bio = bio.split('^')
            if len(forms) == len(bio):
                new_forms = (1, list(zip(forms,bio)))
            elif len(forms)>len(bio):
                dif = len(forms) - len(bio)
                new_forms = (2, list(zip(forms[:dif],['O']*dif)) + list(zip(forms[::-1], bio[::-1]))[::-1])
            else:
                new_forms = (3, list(zip(forms[::-1], bio[::-1]))[::-1])
            new_sent.extend(new_forms[1])
        new_sents.append(new_sent)
    return new_sents


def align_multi_md(ner_multi_preds, md_lattice):
    aligned_sents = soft_merge_bio_labels(ner_multi_preds, md_lattice) 
    labels = [[t[1] for t in sent] for sent in aligned_sents]
    return labels
    
    
def temporary_filename(suffix='tmp', dir=None, text=False, remove_on_exit=True):
    """Returns a temporary filename that, like mkstemp(3), will be secure in
    its creation.  The file will be closed immediately after it's created, so
    you are expected to open it afterwards to do what you wish.  The file
    will be removed on exit unless you pass removeOnExit=False.  (You'd think
    that amongst the myriad of methods in the tempfile module, there'd be
    something like this, right?  Nope.)"""
    
    
    (file_handle, path) = mkstemp(suffix=suffix, dir=dir, text=text)
    os.close(file_handle)

    def remove_file(path):
        os.remove(path)

    if remove_on_exit:
        atexit.register(remove_file, path)

    return path
    

description = """
NEMO API helps you do awesome stuff with Hebrew named entities and morphology 🐠

All endpoints get Hebrew sentences split by a linebreak char, and return different combinations of neural NER and morpho-syntactic parsing.\\
API schema served at [openapi.json](openapi.json)

Have fun and use responsibly 😊
"""

app = FastAPI(
    title="NEMO",
    description=description,
    version="0.1.0",
    terms_of_service="https://github.com/OnlpLab/NEMO",
    contact={
        "name": "Dan Bareket",
        "email": "dbareket@gmail.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)


available_commands = ['run_ner_model', 'multi_align_hybrid', 'multi_to_single',
                      'morph_yap', 'morph_hybrid', 'morph_hybrid_align_tokens']


class ModelName(str, Enum):
    token_single = 'token-single'
    token_multi = "token-multi"
    morph = "morph"

class MultiModelName(str, Enum):
    token_multi = "token-multi"

class MorphModelName(str, Enum):
    morph = "morph"


#query objects for FastAPI documentation
# sent_query = Query( None,
#                     description="Hebrew sentences seprated by '\\n'",
#                     example="עשרות אנשים מגיעים מתאילנד לישראל.\nתופעה זו התבררה אתמול בוועדת העבודה והרווחה של הכנסת.",
#                   )


# tokenized_query = Query( False,
#                     description="Are sentences pre-tokenized? If so, we split each sentence by space char. Else, we use a built in tokenizer."
#                   )


multi_model_query = Query(MultiModelName.token_multi,
                          description="Name of an available toke-multi model.",
                  )


morph_model_query = Query(MorphModelName.morph,
                          description="Name of an available morph model.",
                  )

class NEMOQuery(BaseModel):
    sentences: str
    tokenized: Optional[bool]= False

    class Config:
        schema_extra = {
            "example": {
                "sentences": "עשרות אנשים מגיעים מתאילנד לישראל.\nתופעה זו התבררה אתמול בוועדת העבודה והרווחה של הכנסת.",
                "tokenized": False,
            }
        }


#response models
class NEMODoc(BaseModel):
    tokenized_text: List[str]

class NCRFPreds(NEMODoc):
    ncrf_preds: List[str]

class TokenMultiDoc(NEMODoc):
    multi_ncrf_preds: List[str]
    multi_ncrf_preds_align_single: List[str]

class MDDoc(NEMODoc):
    ma_lattice: str
    md_lattice: str
    morph_forms: List[str]
    dep_tree: Optional[str] = None

class MorphNERDoc(MDDoc):
    morph_ncrf_preds: List[str]
    morph_ncrf_preds_align_tokens: Optional[List[str]] = None

class HybridDoc(TokenMultiDoc, MDDoc):
    pruned_lattice: str
    multi_ncrf_preds_align_morph:  List[str]

class MorphHybridDoc(HybridDoc, MorphNERDoc):
    pass


@app.get("/",
         summary="Get list of available command endpoints"   
        )
def list_commands():
    return {"message": "Please specify command in URL path.",
            "available_commands": available_commands}


# load all models on app startup
@app.on_event("startup")
def load_all_models():
    global loaded_models
    loaded_models = {}
    for model in ModelName:
        m = {}
        m['data'] = get_ncrf_data_object(model)
        m['model'] = load_ncrf_model(m['data'])
        loaded_models[model] = m


@app.post("/run_ner_model",
         response_model=List[NCRFPreds],
         summary="Get NER sequence label predictions, no morphological segmentation"
        )
def run_ner_model(q: NEMOQuery, 
                   model_name: Optional[ModelName]=ModelName.token_single,
                   ):
    model = loaded_models[model_name]
    temp_input = temporary_filename()
    tok_sents = create_input_file(q.sentences, temp_input, q.tokenized)
    preds = ncrf_decode(model['model'], model['data'], temp_input)
    response = []
    for t, p in zip(tok_sents, preds):
        response.append( NCRFPreds( tokenized_text=t,
                                    ncrf_preds=p))
    return response


@app.post("/multi_to_single", response_model=List[TokenMultiDoc],
         summary="Use token-multi model to get token-level NER labels. No morphological segmentation."
        )
def multi_to_single(q: NEMOQuery,
                    multi_model_name: Optional[MultiModelName]=multi_model_query,
                    ):
    model_out = run_ner_model(q, multi_model_name)
    tok_sents, ner_multi_preds = zip(*[(x.tokenized_text, x.ncrf_preds) for x in model_out])
    ner_single_preds = [[fix_multi_biose(label) for label in sent] for sent in ner_multi_preds]

    response = []
    for t, nm, ns in zip(tok_sents, ner_multi_preds, ner_single_preds):
        response.append( TokenMultiDoc( tokenized_text=t,
                                        multi_ncrf_preds=nm,
                                        multi_ncrf_preds_align_single=ns,
                                    ))
    return response


@app.post("/multi_align_hybrid",
         response_model=List[HybridDoc],
         summary="Use token-multi model for MD and NER labels"
        )
def multi_align_hybrid(q: NEMOQuery,
                       multi_model_name: Optional[MultiModelName]=multi_model_query,
                       ):
    model_out = run_ner_model(q, multi_model_name)
    tok_sents, ner_multi_preds = zip(*[(x.tokenized_text, x.ncrf_preds) for x in model_out])
    ner_single_preds = [[fix_multi_biose(label) for label in sent] for sent in ner_multi_preds]
    ma_lattice = run_yap_hebma(tok_sents)
    pruned_lattice = prune_lattice(ma_lattice, ner_multi_preds)
    md_lattice = run_yap_md(pruned_lattice) #TODO: this should be joint, but there is currently no joint on MA in yap api
    dep_tree = run_yap_dep(md_lattice) # instead, we run yap as pipeline md->dep
    md_sents = (bclm.get_sentences_list(bclm.read_lattices(StringIO(md_lattice)), ['form']).apply(lambda x: [t[0] for t in x] )).to_list()
    morph_aligned_preds = align_multi_md(ner_multi_preds, md_lattice)
    
    response = []
    for t, nm, ns, ma, pr, md, mf, al in zip(tok_sents, ner_multi_preds, ner_single_preds,
                                             ma_lattice.split('\n\n'), pruned_lattice.split('\n\n'), md_lattice.split('\n\n'),
                                             md_sents, morph_aligned_preds):
        response.append( HybridDoc( tokenized_text=t,
                                    multi_ncrf_preds=nm,
                                    multi_ncrf_preds_align_single=ns,
                                    ma_lattice=ma,
                                    pruned_lattice=pr,
                                    md_lattice=md,
                                    dep_tree=dep_tree,
                                    morph_forms=mf,
                                    multi_ncrf_preds_align_morph=al,
                                    ))
    return response


@app.post("/morph_yap",
         response_model=List[MorphNERDoc],
         summary="Standard pipeline - use yap for morpho-syntax, then use NER morph model for NER labels"
        )
def morph_yap(q: NEMOQuery,
              morph_model_name: Optional[MorphModelName]=morph_model_query,
              ):
    tok_sents = get_sents(q.sentences, q.tokenized)
    yap_out = run_yap_joint(tok_sents)
    md_sents = (bclm.get_sentences_list(bclm.read_lattices(StringIO(yap_out['md_lattice'])), ['form']).apply(lambda x: [t[0] for t in x] )).to_list()
    model = loaded_models[morph_model_name]
    temp_input = temporary_filename()
    nemo.write_tokens_file(md_sents, temp_input, dummy_o=True)
    morph_preds = ncrf_decode(model['model'], model['data'], temp_input)

    response = []
    for t, ma, md, dep, mf, mp in zip(tok_sents, yap_out['ma_lattice'].split('\n\n'),
                                      yap_out['md_lattice'].split('\n\n'), yap_out['dep_tree'].split('\n\n'),
                                      md_sents, morph_preds):
        response.append( MorphNERDoc( tokenized_text=t,
                                      ma_lattice=ma,
                                      md_lattice=md,
                                      dep_tree=dep,
                                      morph_forms=mf,
                                      morph_ncrf_preds=mp,
                                    ))
    return response


flatten = lambda l: [item for sublist in l for item in sublist]


@app.post("/morph_hybrid",
         response_model=List[MorphHybridDoc] ,
         summary="Segment using hybrid method (w/ token-multi). Then get NER labels with morph model.")
def morph_hybrid(q: NEMOQuery,
                 multi_model_name: Optional[MultiModelName]=multi_model_query,
                 morph_model_name: Optional[MorphModelName]=morph_model_query,
                 align_tokens: Optional[bool] = False):
    model_out = run_ner_model(q, multi_model_name)
    tok_sents, ner_multi_preds = zip(*[(x.tokenized_text, x.ncrf_preds) for x in model_out])
    ner_single_preds = [[fix_multi_biose(label) for label in sent] for sent in ner_multi_preds]
    ma_lattice = run_yap_hebma(tok_sents)
    pruned_lattice = prune_lattice(ma_lattice, ner_multi_preds)
    md_lattice = run_yap_md(pruned_lattice) #TODO: this should be joint, but there is currently no joint on MA in yap api
    dep_tree = run_yap_dep(md_lattice) # instead, we run yap as pipeline md->dep
    morph_aligned_preds = align_multi_md(ner_multi_preds, md_lattice)
    md_sents = (bclm.get_sentences_list(bclm.read_lattices(StringIO(md_lattice)), ['form']).apply(lambda x: [t[0] for t in x] )).to_list()
    model = loaded_models[morph_model_name]
    temp_input = temporary_filename()
    nemo.write_tokens_file(md_sents, temp_input, dummy_o=True)
    morph_preds = ncrf_decode(model['model'], model['data'], temp_input)
    r = {
            't': tok_sents,
            'nm': ner_multi_preds,
            'ns': ner_single_preds,
            'ma': ma_lattice,
            'pr': pruned_lattice,
            'md': md_lattice,
            'dep': dep_tree,
            'mf': md_sents,
            'al': morph_aligned_preds,
            'mor': morph_preds,
        } 
    
    if align_tokens:
        md_sents_for_align = (bclm.get_sentences_list(bclm.read_lattices(StringIO(md_lattice)), ['token_id']).apply(lambda x: [t[0] for t in x] )).to_list()
        tok_aligned_sents = flatten([[(sent_id, m, p) for (m,p) in zip(m_sent, p_sent)] for sent_id, (m_sent, p_sent) in enumerate(zip(md_sents_for_align, morph_preds))])
        tok_aligned_df = pd.DataFrame(tok_aligned_sents, columns=['sent_id', 'token_id', 'biose'])
        new_toks = bclm.get_token_df(tok_aligned_df, fields=['biose'], token_fields=['sent_id', 'token_id'])
        new_toks['fixed_bio'] = new_toks.biose.apply(lambda x: nemo.get_fixed_bio_sequence(tuple(x.split('^'))))
        tok_aligned = (bclm.get_sentences_list(new_toks, ['fixed_bio']).apply(lambda x: [t[0] for t in x] )).to_list()
        r['moral'] = tok_aligned
    else: r['moral'] = [None, ]*len(r['t'])
    
    response = []
    for t, nm, ns, ma, pr, md, dep, mf, al, mor, moral in zip(r['t'], r['nm'], r['ns'],
                                             r['ma'].split('\n\n'), r['pr'].split('\n\n'), r['md'].split('\n\n'), r['dep'].split('\n\n'),
                                             r['mf'], r['al'], r['mor'], r['moral']):
        response.append( MorphHybridDoc( tokenized_text=t,
                                    multi_ncrf_preds=nm,
                                    multi_ncrf_preds_align_single=ns,
                                    ma_lattice=ma,
                                    pruned_lattice=pr,
                                    md_lattice=md,
                                    dep_tree=dep,
                                    morph_forms=mf,
                                    multi_ncrf_preds_align_morph=al,
                                    morph_ncrf_preds=mor,
                                    morph_ncrf_preds_align_tokens=moral,
                                    ))
    return response


@app.post("/morph_hybrid_align_tokens",
         response_model=List[MorphHybridDoc] ,
         summary="Segment using hybrid method (w/ token-multi). Then get NER labels with morph model + align with tokens to get token-level NER.")
def morph_hybrid_align_tokens(q: NEMOQuery,
                              multi_model_name: Optional[MultiModelName]=multi_model_query,
                              morph_model_name: Optional[MorphModelName]=morph_model_query,
                              ):
    return morph_hybrid(q, multi_model_name, morph_model_name, align_tokens=True)


#
# @app.post("/run_separate_nemo/")
# def run_separate_nemo(command: str, model_name: str, sentence: str):
#     if command in available_commands:
#         if command == 'run_ner_model':
#             with Temp('r', encoding='utf8') as temp_output:
#                 nemo.run_ner_model(model_name, None, temp_output.name, text_input=sentence)
#                 output_text = temp_output.read()
#             return { 'nemo_output': output_text }
#     else: 
#         return {'error': 'command not supported'}
