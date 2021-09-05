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
from operator import itemgetter
from itertools import groupby
import iobes
if SUPPRESS_IOBES_WARNINGS:
    from logging import ERROR
    iobes.LOGGER.setLevel(level=ERROR)

os.environ['CUDA_VISIBLE_DEVICES'] = ''

## NCRF stuff
from utils.data import Data
import torch
from model.seqlabel import SeqLabel
from ncrf_main import evaluate

#fastapi stuff
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
import asyncio
from schema import * 

# deal with exploding thread count
# taken from https://github.com/tiangolo/fastapi/issues/603#issuecomment-545075929
try: 
    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=MAX_THREADS_FASTAPI))
except:
    print("No running asyncio event loop...")


#get yap location from env vars
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
    data.nbest = None
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
    _, _, _, _, _, preds, _ = evaluate(data, model, 'raw', nbest=data.nbest, calc_fmeasure=False)
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
    try:
        return requests.get(url+route, data=data, headers=headers).json()
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=500, detail="YAP API unavailable. If you just started it, have a drink and give it some time to load :)")


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


def hybrid_md(tok_sents, ner_multi_preds):
    ma_lattice = run_yap_hebma(tok_sents)
    pruned_lattice = prune_lattice(ma_lattice, ner_multi_preds)
    md_lattice = run_yap_md(pruned_lattice) #TODO: this should be joint, but there is currently no joint on MA in yap api
    return ma_lattice, pruned_lattice, md_lattice


from itertools import zip_longest


def doc_add_yap_outputs(docs, ma_lattice='', pruned_lattice='', md_lattice=''):
    for doc, ma, pr, md in zip_longest(
                                docs, 
                                ma_lattice.split('\n\n'),
                                pruned_lattice.split('\n\n'), 
                                md_lattice.split('\n\n')
                                ):
        if ma:
            doc.ma_lattice = ma
        if pr:
            doc.pruned_lattice = pr
        if md:
            doc.md_lattice = md
    return docs


def doc_set_token_attr(docs, name, values):
    for doc, doc_val in zip(docs, values):
        for tok, tok_val in zip(doc, doc_val):
            setattr(tok, name, tok_val)
    return docs

def doc_add_multi_align_tok(docs, ner_multi_preds):
    mul_align_tok = [[fix_multi_biose(label) for label in sent] for sent in ner_multi_preds]
    doc_set_token_attr(docs, 'nemo_multi_align_token', mul_align_tok)
    return docs, mul_align_tok


def docs_init_tokens(tok_sents, sents):
    docs = []
    for s, t in zip(tok_sents, sents):
        tokens = [Token(text=t) for t in s]
        docs.append(Doc(text=t, tokens=tokens))
    return docs


def doc_init_morphs(docs, tok_morph_sents):
    for doc, tm_sent  in zip(docs, tok_morph_sents):
        for tok, tok_mor in zip(doc, tm_sent):
            morphs = [ Morpheme(form=form, lemma=lemma, pos=xpostag, feats=feats) 
                        for form, lemma, xpostag, feats
                        in tok_mor]
            tok.morphs = morphs
    return docs


def doc_set_morph_attr(docs, name, values):
    for doc, mor_doc in zip(docs, values):
        for tok, mor_tok in zip(doc, mor_doc):
            for mor, mor_val in zip(tok, mor_tok):
                if len(mor_val)==1:
                    setattr(mor, name, mor_val[0])


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
        for row in sent[cols].astype(str).itertuples(index=False):
            lat += '\t'.join(row)+'\n'
        lat += '\n'
    return lat
            
    
def soft_merge_bio_labels(ner_multi_preds, md_lattices):
    multitok_sents = bclm.get_sentences_list(get_biose_count(ner_multi_preds), ['biose'])
    md_sents = bclm.get_sentences_list(
                                        _get_token_df(bclm.read_lattices(StringIO(md_lattices)), 
                                                            fields=['form'], token_fields=['sent_id', 'token_id'], add_set=False),
                                        ['token_id', 'form']
                                    )
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


def _get_token_df(df, fields=None, biose=None, token_fields = bclm.TOK_FIELDS, sep='^', fill_value='', add_set=True):
    tok_dfs = []
    
    if biose is not None:
        for col in biose:
            tok_dfs.append(bclm.get_token_biose(df, col))
        
    if fields is not None:
        for field in fields:
            tok_fields = (df.fillna(fill_value)
                    .groupby(token_fields)[field]
                    .apply(sep.join))
            tok_dfs.append(tok_fields)
    tok_df = pd.concat(tok_dfs, axis=1)

    if add_set and 'set' in df.columns:
            tok_df = tok_df.assign(set = lambda x: (x.index
                                                     .get_level_values('sent_id')
                                                     .map(df[['sent_id', 'set']]
                                                     .drop_duplicates()
                                                     .set_index('sent_id')['set'])))
            
    tok_df = tok_df.sort_index().reset_index()
    
    return tok_df


def get_token_morphs_list(md_sents):
    sents = []
    for sent in md_sents:
        glo = [[x[1:] for x in g]
                for k,g in  groupby(sent,key=itemgetter(0))]
        sents.append(glo)
    return sents


def align_token_morph_list(md_sents, morph_attrs):
    return get_token_morphs_list([
                                    [(tid, p) for (tid, *_), p 
                                        in zip(md_sent, attrs_sent)] 
                                    for md_sent, attrs_sent 
                                    in zip(md_sents, morph_attrs) 
                                ])


def get_md_sents(md_lattice, cols):
    md_sents = (bclm.get_sentences_list(
                            bclm.read_lattices(StringIO(md_lattice)), cols)
                            .to_list())
    return md_sents


def get_dep_sents(dep_tree, cols):
    dep_sents = (bclm.get_sentences_list(
                            bclm.read_conll(StringIO(dep_tree)), cols)
                            .to_list())
    return dep_sents


def add_dep_info(docs, md_sents, dep_tree, include_yap_outputs):
    dep_sents = get_dep_sents(dep_tree, ['id', 'head', 'deprel'])
    dep_sents = [[(tid, *dep) for (tid, *_), dep in zip(md_sent, dep_sent)] 
                for md_sent, dep_sent in zip(md_sents, dep_sents)]
    tok_dep_sents = get_token_morphs_list(dep_sents)
    for doc, tds, dep in zip(docs, tok_dep_sents, dep_tree.split('\n\n')):
        for tok, td in zip(doc, tds):
            for morph, (id, head, deprel) in zip(tok, td):
                morph.id = id
                morph.head = head
                morph.deprel = deprel
                #morph.deps = deps
        if include_yap_outputs:
            doc.dep_tree = dep


def iter_token_attrs(doc, attr):
    for token in doc:
        yield getattr(token, attr)


def iter_morph_attrs(doc, attr):
    for token in doc:
        for morph in token:
            yield getattr(morph, attr)


NEMO_FIELDS_TOKEN = ['nemo_single', 'nemo_multi_align_token', 'nemo_morph_align_token']
NEMO_FIELDS_MORPH = ['nemo_morph', 'nemo_multi_align_morph']


def to_dict(span, text):
    return {
        'text': ' '.join(text[span[1]:span[2]]),
        'label': span[0],
        'start': span[1],
        'end': span[2]
    }


def get_spans(doc, token_fields=None, morph_fields=None):
    all_spans = {}
    if morph_fields:
        try: 
            morph_text = list(iter_morph_attrs(doc, 'form'))
        except KeyError:
            pass
        morph_spans = {}
        for f in morph_fields:
            try:
                labels = list(iter_morph_attrs(doc, f))
                spans = [to_dict(x, morph_text) 
                        for x in iobes.parse_spans_iobes(labels)]
                morph_spans[f] = spans
            except KeyError:
                pass
        all_spans['morph'] = morph_spans
    if token_fields:
        tok_text = list(iter_token_attrs(doc, 'text'))
        tok_spans = {}
        for f in token_fields:
            try:
                labels = list(iter_token_attrs(doc, f))
                spans = [to_dict(x, tok_text) 
                        for x in iobes.parse_spans_iobes(labels)]
                tok_spans[f] = spans
            except KeyError:
                pass
        all_spans['token'] = tok_spans
      
    return all_spans


description = """
NEMO API helps you do awesome stuff with Hebrew named entities and morphology 🐠

* All endpoints expect an HTTP POST request:    
    - Request body contains a JSON with Hebrew `sentences` and optional `tokenized` flag for signaling whether they are pre-tokenized or not
    - Request URL may include further optional path parameters for choosing models/scenarios (in all but `run_ncrf_model` there is no need to touch these)
    - `verbose` parameter (more info &rarr; longer runtime):
        - `0`: tokens, morphemes (if morph endpoint) and final requested nemo preds
        - `1`: adds intermediate nemo preds 
            - for example when running `morph_hybrid`, you also get NER preds `nemo_multi`, `nemo_multi_align_token`, `nemo_multi_align_morph`
        - `2` adds syntactic dependency tree features (SPMRL): `head', `deprel`
    - `include_yap_outputs` flag - adds yap raw output to each sentence
* Results are in JSON form in HTTP response body:
    - The response is a list of `Doc` objects, one for each sentence in the input
    - Each `Doc` contains: 
        - Predicted entity spans in the `ents` attribute (organized by scenarios)
        - `tokens` - a list of `Token` objects, each containing:
            - `text` - the surface form of the token
            - Predicted BIOSE labels for the requested scenarios: `nemo_multi` / `nemo_multi_align_token` / `nemo_morph_align_token` 
            - `morphs` - a list of `Morpheme` objects, each containing:
                - Surface `form` and other predicted morphological features: `lemma`, `pos`, `feats`
                - Predicted BIOSE labels for the requested scenarios: `nemo_morph` / `nemo_multi_align_morph`
                - If requested, predicted dependency syntax features: `head`, `deprel` 
        - If requested, raw YAP outputs - `ma_lattice`, `pruned_lattice` (using the hybrid method), `md_lattice`, `dep_tree`


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


available_commands = ['run_ncrf_model', 'multi_align_hybrid', 'multi_to_single',
                      'morph_yap', 'morph_hybrid', 'morph_hybrid_align_tokens']


#query objects for FastAPI documentation
multi_model_query = Query(MultiModelName.token_multi,
                          description="Name of an available toke-multi model.",
                  )


morph_model_query = Query(MorphModelName.morph,
                          description="Name of an available morph model.",
                  )


verbosity_query = Query(Verbosity.BASIC,
                          description="0: token, morph info & final NER prediction; 1: add intermediate NER preds; 2: add syntactic info",
                  )


class NEMOQuery(BaseModel):
    sentences: str
    tokenized: Optional[bool] = False

    class Config:
        schema_extra = {
            "example": {
                "sentences": "עשרות אנשים מגיעים מתאילנד לישראל.\nתופעה זו התבררה אתמול בוועדת העבודה והרווחה של הכנסת.",
                "tokenized": False,
            }
        }


@app.get("/",
         summary="Get list of available command endpoints"   
        )
def list_commands():
    return {"message": "Please specify command in URL path in a POST request and provide some input text in the request body.",
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


@app.post("/run_ncrf_model",
         response_model=List[NCRFPreds],
         summary="Get NER sequence label predictions, no morphological segmentation",
         response_model_exclude_unset=True
        )
def run_ncrf_model(q: NEMOQuery, 
                   model_name: Optional[ModelName]=ModelName.token_single,
                   ):
    if not q.sentences.strip():
        return []
    model = loaded_models[model_name]
    temp_input = temporary_filename()
    tok_sents = create_input_file(q.sentences, temp_input, q.tokenized)
    preds = ncrf_decode(model['model'], model['data'], temp_input)
    response = []
    for t, p in zip(tok_sents, preds):
        response.append( NCRFPreds( tokenized_text=t,
                                    ncrf_preds=p))
    return response


@app.post("/multi_to_single", response_model=List[Doc],
         summary="Use token-multi model to get token-level NER labels. No morphological segmentation.",
         response_model_exclude_unset=True
        )
def multi_to_single(
                    q: NEMOQuery,
                    multi_model_name: Optional[MultiModelName]=multi_model_query,
                    verbose: Optional[Verbosity]=verbosity_query
                    ):
    if not q.sentences.strip():
        return []
    sents = q.sentences.split('\n')
    nemo_token_fields = ['nemo_multi_align_token']
    model_out = run_ncrf_model(q, multi_model_name)
    tok_sents, ner_multi_preds = zip(*[(x.tokenized_text, x.ncrf_preds) for x in model_out])
    docs = docs_init_tokens(tok_sents, sents)

    if verbose>=Verbosity.INTERMID:
        doc_set_token_attr(docs, 'nemo_multi', ner_multi_preds)
        nemo_token_fields.append('nemo_multi')
    
    doc_add_multi_align_tok(docs, ner_multi_preds)

    for doc in docs:
        doc.ents = get_spans(doc, token_fields=nemo_token_fields)
    return docs


@app.post("/morph_yap",
         response_model=List[Doc],
         summary="Standard pipeline - use yap for morpho-syntax, then use NER morph model for NER labels",
         response_model_exclude_unset=True
        )
def morph_yap(q: NEMOQuery,
              morph_model_name: Optional[MorphModelName]=morph_model_query,
              verbose: Optional[Verbosity]=verbosity_query,
              include_yap_outputs: Optional[bool]=False):
    if not q.sentences.strip():
        return []
    
    nemo_morph_fields = ['nemo_morph']

    sents = q.sentences.split('\n')
    tok_sents = get_sents(q.sentences, q.tokenized)
    docs = docs_init_tokens(tok_sents, sents)

    yap_out = run_yap_joint(tok_sents)
    md_lattice = yap_out['md_lattice']
    dep_tree = yap_out['dep_tree']
    if include_yap_outputs:
        doc_add_yap_outputs(docs, ma_lattice = yap_out['ma_lattice'], md_lattice = md_lattice)

    model = loaded_models[morph_model_name]
    temp_input = temporary_filename()
    md_sents = get_md_sents(md_lattice, ['token_id', 'form', 'lemma', 'xpostag', 'feats'])
    md_sents_for_yap = [[form for _, form, *_ in sent] for sent in md_sents]
    nemo.write_tokens_file(md_sents_for_yap, temp_input, dummy_o=True)
    morph_preds = ncrf_decode(model['model'], model['data'], temp_input)

    tok_md_sents = get_token_morphs_list(md_sents)
    doc_init_morphs(docs, tok_md_sents)
    tok_morph_preds = align_token_morph_list(md_sents, morph_preds)
    doc_set_morph_attr(docs, 'nemo_morph', tok_morph_preds)

    if verbose>=Verbosity.SYNTAX:
        add_dep_info(docs, md_sents, dep_tree, include_yap_outputs)
    for doc in docs:
        doc.ents = get_spans(doc, morph_fields=nemo_morph_fields)
    return docs


@app.post("/multi_align_hybrid",
         response_model=List[Doc],
         summary="Use token-multi model for MD and morpheme NER labels",
         response_model_exclude_unset=True
        )
def multi_align_hybrid(q: NEMOQuery,
                       multi_model_name: Optional[MultiModelName]=multi_model_query,
                       verbose: Optional[Verbosity]=verbosity_query,
                       include_yap_outputs: Optional[bool]=False):
    if not q.sentences.strip():
        return []
        
    nemo_morph_fields = ['nemo_multi_align_morph']
    nemo_token_fields = []

    sents = q.sentences.split('\n')
    model_out = run_ncrf_model(q, multi_model_name)
    tok_sents, ner_multi_preds = zip(*[(x.tokenized_text, x.ncrf_preds) for x in model_out])
    docs = docs_init_tokens(tok_sents, sents)

    if verbose>=Verbosity.INTERMID:
        doc_set_token_attr(docs, 'nemo_multi', ner_multi_preds)
        doc_add_multi_align_tok(docs, ner_multi_preds)
        nemo_token_fields.append('nemo_multi_align_token')

        
    ma_lattice, pruned_lattice, md_lattice = hybrid_md(tok_sents, ner_multi_preds)
    if include_yap_outputs:
        doc_add_yap_outputs(docs, ma_lattice, pruned_lattice, md_lattice)
    
    morph_aligned_preds = align_multi_md(ner_multi_preds, md_lattice)

    md_sents = get_md_sents(md_lattice, ['token_id', 'form', 'lemma', 'xpostag', 'feats'])
    tok_md_sents = get_token_morphs_list(md_sents)
    doc_init_morphs(docs, tok_md_sents)

    tok_morph_aligned_preds = align_token_morph_list(md_sents, morph_aligned_preds)
    doc_set_morph_attr(docs, 'nemo_multi_align_morph', tok_morph_aligned_preds)

    if verbose>=Verbosity.SYNTAX:
        dep_tree = run_yap_dep(md_lattice)
        add_dep_info(docs, md_sents, dep_tree, include_yap_outputs)

    for doc in docs:
        doc.ents = get_spans(doc, morph_fields=nemo_morph_fields, token_fields=nemo_token_fields)

    return docs


flatten = lambda l: [item for sublist in l for item in sublist]


@app.post("/morph_hybrid",
         response_model=List[Doc] ,
         summary="Segment using hybrid method (w/ token-multi). Then get NER labels with morph model.",
         response_model_exclude_unset=True)
def morph_hybrid(
                 q: NEMOQuery,
                 multi_model_name: Optional[MultiModelName]=multi_model_query,
                 morph_model_name: Optional[MorphModelName]=morph_model_query,
                 align_tokens: Optional[bool] = False,
                 verbose: Optional[Verbosity]=verbosity_query,
                 include_yap_outputs: Optional[bool]=False
                 ):
    if not q.sentences.strip():
        return []
    nemo_morph_fields = []
    nemo_token_fields = []

    sents = q.sentences.split('\n')
    model_out = run_ncrf_model(q, multi_model_name)
    tok_sents, ner_multi_preds = zip(*[(x.tokenized_text, x.ncrf_preds) for x in model_out])
    docs = []
    for s, t in zip(tok_sents, sents):
        tokens = [Token(text=t) for t in s]
        docs.append(Doc(text=t, tokens=tokens))

    if verbose>=Verbosity.INTERMID:
        doc_set_token_attr(docs, 'nemo_multi', ner_multi_preds)
        doc_add_multi_align_tok(docs, ner_multi_preds)
        nemo_token_fields.append('nemo_multi_align_token')

    ma_lattice, pruned_lattice, md_lattice = hybrid_md(tok_sents, ner_multi_preds)
    if include_yap_outputs:
        doc_add_yap_outputs(docs, ma_lattice, pruned_lattice, md_lattice)
    
    md_sents = get_md_sents(md_lattice, ['token_id', 'form', 'lemma', 'xpostag', 'feats'])
    model = loaded_models[morph_model_name]
    temp_input = temporary_filename()
    md_sents_for_ncrf = [[form for _, form, *_ in sent] for sent in md_sents]
    nemo.write_tokens_file(md_sents_for_ncrf, temp_input, dummy_o=True)
    morph_preds = ncrf_decode(model['model'], model['data'], temp_input)

    if verbose>=Verbosity.INTERMID or align_tokens==False:
        tok_md_sents = get_token_morphs_list(md_sents)
        doc_init_morphs(docs, tok_md_sents)
        tok_morph_preds = align_token_morph_list(md_sents, morph_preds)
        doc_set_morph_attr(docs, 'nemo_morph', tok_morph_preds)
        nemo_morph_fields.append('nemo_morph')

    if verbose>=Verbosity.INTERMID:
        morph_aligned_preds = align_multi_md(ner_multi_preds, md_lattice)
        tok_morph_aligned_preds = align_token_morph_list(md_sents, morph_aligned_preds)
        doc_set_morph_attr(docs, 'nemo_multi_align_morph', tok_morph_aligned_preds)
        nemo_morph_fields.append('nemo_multi_align_morph')
    if verbose>=Verbosity.SYNTAX:
        dep_tree = run_yap_dep(md_lattice)
        add_dep_info(docs, md_sents, dep_tree, include_yap_outputs)
    
    if align_tokens:
        md_sents_for_align = (bclm.get_sentences_list(bclm.read_lattices(StringIO(md_lattice)), ['token_id']).apply(lambda x: [t[0] for t in x] )).to_list()
        tok_aligned_sents = flatten([[(sent_id, m, p) for (m,p) in zip(m_sent, p_sent)] for sent_id, (m_sent, p_sent) in enumerate(zip(md_sents_for_align, morph_preds))])
        tok_aligned_df = pd.DataFrame(tok_aligned_sents, columns=['sent_id', 'token_id', 'biose'])
        new_toks = _get_token_df(tok_aligned_df, fields=['biose'], token_fields=['sent_id', 'token_id'])
        new_toks['fixed_bio'] = new_toks.biose.apply(lambda x: nemo.get_fixed_bio_sequence(tuple(x.split('^'))))
        tok_aligned = (bclm.get_sentences_list(new_toks, ['fixed_bio']).apply(lambda x: [t[0] for t in x] )).to_list()
        for doc, tok_preds in zip(docs, tok_aligned):
            for tok, tok_pred in zip(doc, tok_preds):
                tok.nemo_morph_align_token = tok_pred
        nemo_token_fields.append('nemo_morph_align_token')

    for doc in docs:
        doc.ents = get_spans(doc, morph_fields=nemo_morph_fields, token_fields=nemo_token_fields)

    return docs


@app.post("/morph_hybrid_align_tokens",
         response_model=List[Doc] ,
         summary="Segment using hybrid method (w/ token-multi). Then get NER labels with morph model + align with tokens to get token-level NER.",
         response_model_exclude_unset=True)
def morph_hybrid_align_tokens(q: NEMOQuery,
                              multi_model_name: Optional[MultiModelName]=multi_model_query,
                              morph_model_name: Optional[MorphModelName]=morph_model_query,
                              verbose: Optional[Verbosity]=verbosity_query,
                              include_yap_outputs: Optional[bool]=False):
    return morph_hybrid(q, multi_model_name, morph_model_name, align_tokens=True, 
                        verbose=verbose, include_yap_outputs=include_yap_outputs)
                        