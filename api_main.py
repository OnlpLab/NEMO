from fastapi import FastAPI
from typing import Optional
from tempfile import NamedTemporaryFile as Temp
import os
from config import *
import nemo


## NCRF stuff
from utils.data import Data
import torch
from model.seqlabel import SeqLabel
from ncrf_main import evaluate

def get_ncrf_data_object(model_name): #, input_path, output_path):
    data = Data()
    model = MODEL_PATHS[model_name]
    data.dset_dir = model['dset']
    data.load(data.dset_dir)
    #data.raw_dir = input_path
    #data.decode_dir = output_path
    data.load_model_dir = model['model']
    data.nbest = 1
    return data

def load_ncrf_model(data):
    model = SeqLabel(data)
    print('loading model:', data.load_model_dir)
    model.load_state_dict(torch.load(data.load_model_dir))
    return model

def ncrf_decode(model, data, temp_input):
    data.raw_dir = temp_input
    #data.decode_dir = temp_output
    data.generate_instance('raw')
    _, _, _, _, _, preds, _ = evaluate(data, model, 'raw', data.nbest)
    if data.nbest==1:
        preds = [sent[0] for sent in preds]
    return preds
    
    
def create_input_file(text, path, tokenized):
    if not tokenized:
        sents = nemo.tokenize_text(text)
    else:
        sents = [sent.split(' ') for sent in text.split('\n')]
    nemo.write_tokens_file(sents, path, dummy_o=True)
    return sents
    
# load all models
available_models = ['token-single', 'token-multi', 'morph']
loaded_models = {}
for model in available_models:
    m = {}
    m['data'] = get_ncrf_data_object(model)
    m['model'] = load_ncrf_model(m['data'])
    loaded_models[model] = m

available_commands = ['run_ner_model']

app = FastAPI()

@app.get("/")
def home():
    return {"error": "Please specify command"}

@app.get("/run_nemo/")
def run_nemo(command: str, model_name: str, sentences: str, tokenized: str = False):
    if command in available_commands:
        if command == 'run_ner_model':
            if model_name in available_models:
                model = loaded_models[model_name]
                with Temp() as temp_input:
                    tok_sents = create_input_file(sentences, temp_input.name, tokenized)
                    preds = ncrf_decode(model['model'], model['data'], temp_input.name)
                    #output_text = temp_output.read()
                return { 
                    'tokenized_text': tok_sents,
                    'nemo_predictions': preds 
                }
            else:
                return {'error': f'model name "{model_name}" unavailable'}
    else: 
        return {'error': f'command "{command}" not supported'}
    
# @app.get("/run_separate_nemo/")
# def run_separate_nemo(command: str, model_name: str, sentence: str):
#     if command in available_commands:
#         if command == 'run_ner_model':
#             with Temp('r', encoding='utf8') as temp_output:
#                 nemo.run_ner_model(model_name, None, temp_output.name, text_input=sentence)
#                 output_text = temp_output.read()
#             return { 'nemo_output': output_text }
#     else: 
#         return {'error': 'command not supported'}
