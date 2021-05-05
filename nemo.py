import os
import datetime
import bclm
import subprocess
import sys
import networkx as nx
import traceback

DEVICE = 3
LOCAL_TEMP_FOLDER = 'temp'
DELETE_TEMP_FILES = False

MODEL_PATHS = {
    'token-single_oov': { 'model': 'data/token.char_lstm.ft_oov_tok.44_seed.146.model',
                          'dset': 'data/token.char_lstm.ft_oov_tok.44_seed.dset'},
    'token-single': { 'model': 'data/token.char_cnn.ft_tok.46_seed.104.model',
                      'dset': 'data/token.char_cnn.ft_tok.46_seed.dset'},
    'token-multi_oov': { 'model': 'data/multitok.char_cnn.ft_oov_tok.48_seed.123.model', 
                     'dset': 'data/multitok.char_cnn.ft_oov_tok.48_seed.dset'},
    'token-multi': {'model': 'data/multitok.char_cnn.ft_tok.52_seed.173.model',
                        'dset': 'data/multitok.char_cnn.ft_tok.52_seed.dset'},
    'morph_oov': {'model': 'data/morph.char_cnn.ft_oov_yap.49_seed.87.model',
                  'dset': 'data/morph.char_cnn.ft_oov_yap.49_seed.dset'},
    'morph': { 'model': 'data/morph.char_cnn.ft_yap.50_seed.80.model',
              'dset': 'data/morph.char_cnn.ft_yap.50_seed.dset'}
}
MULTI_MODEL_FOR_HYBRID = 'multi'

YAP_PATH = '../yapproj/src/yap/yap'


def read_text_file(path):
    sents = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            toks = bclm.tokenize(line.rstrip())
            sents.append(toks)
    return sents
        

def write_tokens_file(sents, file_path, dummy_o=False, only_tokens=False):
    with open(file_path, 'w', encoding='utf8') as of:
        for sent in sents:
            for fields in sent: 
                if type(fields) is str:
                    word = fields
                else:
                    word = fields[0]
                if only_tokens:
                    line = word
                elif dummy_o:
                    line = word + ' O'
                else:
                    line = word + ' ' + fields[-1]
                of.write(line + '\n')
            of.write('\n')


def write_ncrf_conf(conf_path, input_path, output_path, model, dset):
            params = { 'status': 'decode' }

            params['load_model_dir'] = model
            params['dset_dir'] = dset
            params['decode_dir'] = output_path
            params['raw_dir'] = input_path

            if not os.path.exists(conf_path):
                with open(conf_path, 'w', encoding='utf8') as of:
                    for k, v in params.items():
                        of.write(k+'='+str(v)+'\n')        
    
            
def run_yap_hebma(tokens_path, output_path, log_path):
    result = subprocess.run([YAP_PATH, 'hebma', '-raw', tokens_path, 
                    '-out', output_path], capture_output=True)
    with open(log_path, 'wb') as of:
        of.write(result.stdout)
    if len(result.stderr)>0:
        print(result.stderr) 

def run_yap_joint(lattices_path, seg_path, map_path, conll_path, log_path):
    result = subprocess.run([YAP_PATH, 'joint', '-in', lattices_path, 
                    '-os', seg_path, '-om', map_path, '-oc', conll_path], capture_output=True)
    with open(log_path, 'wb') as of:
        of.write(result.stdout)
    if len(result.stderr)>0:
        print(result.stderr) 
        
        
def run_ncrf_main(conf_path, device, log_path):
    result = subprocess.run(['python', 'ncrf_main.py', '--config', conf_path, 
                             '--device', str(device)], capture_output=True)
    with open(log_path, 'wb') as of:
        of.write(result.stdout)
    if len(result.stderr)>0:
        print(result.stderr) 
        

def run_ner_model(model_name, input_path, output_path):
    temp_input_path = os.path.join(LOCAL_TEMP_FOLDER, datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')+'.txt')
    temp_conf_path = temp_input_path.replace('.txt','.conf')
    temp_log_path = temp_input_path.replace('.txt','.log')
    try:
        sents = read_text_file(input_path)
        write_tokens_file(sents, temp_input_path, dummy_o=True)

        write_ncrf_conf(temp_conf_path, temp_input_path, output_path, MODEL_PATHS[model_name]['model'], MODEL_PATHS[model_name]['dset'])

        run_ncrf_main(temp_conf_path, DEVICE, temp_log_path)
    except Exception as e:
        print(e)
    if DELETE_TEMP_FILES:
        for path in [temp_input_path, temp_conf_path, temp_log_path]:
            if os.path.exists(path):
                os.remove(path)
                
def run_morph_yap(model_name, input_path, output_path):
    temp_tokens_path = os.path.join(LOCAL_TEMP_FOLDER, datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')+'.txt')
    temp_conf_path = temp_tokens_path.replace('.txt','_ncrf.conf')
    temp_yap_log_path = temp_tokens_path.replace('.txt','_yap.log')
    temp_lattices_path = temp_tokens_path.replace('.txt','.lattices')
    temp_seg_path = temp_tokens_path.replace('.txt','.seg')
    temp_map_path = temp_tokens_path.replace('.txt','.map')
    temp_conll_path = temp_tokens_path.replace('.txt','.conll')
    temp_ncrf_morph_input = temp_tokens_path.replace('.txt', '_yapform.txt')
    temp_ncrf_log_path = temp_tokens_path.replace('.txt','_ncrf.log')
    try:
        # read file and tokenize
        sents = read_text_file(input_path)
        #write temporary tokens file for yap
        write_tokens_file(sents, temp_tokens_path, only_tokens=True)
        #run yap hebma to create ambiguous lattices
        run_yap_hebma(temp_tokens_path, temp_lattices_path, temp_yap_log_path)
        #run yap joint
        run_yap_joint(temp_lattices_path, temp_seg_path, temp_map_path, temp_conll_path, temp_yap_log_path)
        #create temp morph input with dummy o
        lat = bclm.read_conll(temp_conll_path)
        form_sents = lat.groupby('sent_id').form.apply(lambda x: x.tolist()).tolist()
        write_tokens_file(form_sents, temp_ncrf_morph_input, dummy_o=True)
        #run ncrf morph model
        write_ncrf_conf(temp_conf_path, temp_ncrf_morph_input, output_path, MODEL_PATHS[model_name]['model'], MODEL_PATHS[model_name]['dset'])
        run_ncrf_main(temp_conf_path, DEVICE, temp_ncrf_log_path)
    except Exception as e:
        print(traceback.format_exc())
    #delete all temp files
    if DELETE_TEMP_FILES:
        for path in [temp_tokens_path, temp_conf_path, temp_yap_log_path, temp_lattices_path, 
                     temp_seg_path, temp_map_path, temp_conll_path,
                     temp_ncrf_morph_input, temp_ncrf_log_path]:
            if os.path.exists(path):
                os.remove(path)

                
def run_morph_hybrid(model_name, input_path, output_path):
    pass
        
def run_multi_hybrid(model_name, input_path, output_path):
    pass
    
    
def run_morph_tok(model_name, input_path, output_path):
    pass

 
    
    
if __name__=='__main__':
    command = sys.argv[1]
    model_name = sys.argv[2]
    input_path = sys.argv[3]
    output_path = sys.argv[4]
    
    if not os.path.exists(LOCAL_TEMP_FOLDER):
        os.makedirs(LOCAL_TEMP_FOLDER)
    
    #just run a ner model on a file of space delimited sentences
    if command=='run_ner_model':
        run_ner_model(model_name, input_path, output_path)
    
    #run multi model and transform multi-labels to single label per token
    if command=='token_multi':
        run_morph_yap(model_name, input_path, output_path)
        
    #run morph model on yap segmented output
    if command=='morph_yap':
        run_morph_yap(model_name, input_path, output_path)
    
    #run morph model on hybrid segmented output (includes an initial run of multi model)
    if command=='morph_hybrid':
        run_morph_hybrid(model_name, input_path, output_path)
        
    #run multi model and align hybrid segmented output 
    if command=='multi_hybrid':
        run_multi_hybrid(model_name, input_path, output_path)
        
    #run morph model on hybrid segmented output and align back with tokens
    if command=='morph_hybrid_token':
        run_morph_hybrid_token(model_name, input_path, output_path)

