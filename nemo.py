import os
import datetime
import bclm
import pandas as pd
import subprocess
import sys
import networkx as nx
import traceback
import re

from config import *
from ne_evaluate_mentions import fix_multi_biose, read_file_sents

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


def get_biose_count(path, sent_id_shift=1):
    sents = read_file_sents(path, fix_multi_tag=False, sent_id_shift=sent_id_shift)
    bc = []
    for i, sent in sents.iteritems():
        for j, (tok, bio) in enumerate(sent):
            bc.append([i, j+1, tok, bio, len(bio.split('^'))])

    bc = pd.DataFrame(bc, columns=['sent_id', 'token_id', 'token_str', 
                                   'biose', 'biose_count'])
    return bc


def get_valid_edges(lattices, bc,
                    non_o_only=True, keep_all_if_no_valid=True):
    valid_edges = []
    for (i, df), (_, biose, biose_count) in zip(lattices.groupby(['sent_id', 'token_id']), 
                                                bc[['biose', 'biose_count']].itertuples()):
        el = df[['ID1', 'ID2']].rename(columns={'ID1': 'source', 'ID2': 'target'})
        #min_node = [n for n,v in G.nodes(data=True) if v['since'] == 'December 2008'][0]

        g = nx.from_pandas_edgelist(el, create_using=nx.DiGraph)
        min_node = el.source.min()
        max_node = el.target.max()
        #print(min_node,max_node)
        #print(biose_count)
        if non_o_only and not '-' in biose:
            vp = list(nx.all_simple_paths(g, min_node, max_node))
        else:
            vp = [path for path in nx.all_simple_paths(g, min_node, max_node, cutoff=biose_count+1) if len(path)==biose_count+1]
        if keep_all_if_no_valid and len(vp)==0:
             vp = nx.all_simple_paths(g, min_node, max_node)
        for path in vp:
            for source, target in zip(path[:-1], path[1:]):
                valid_edges.append((i[0], i[1], source, target))
                
    return valid_edges


def to_lattices(df, path, cols = ['ID1', 'ID2', 'form', 'lemma', 'upostag', 'xpostag', 'feats', 'token_id']):
    with open(path, 'w', encoding='utf8') as of:
        for _, sent in df.groupby('sent_id'):
            for _, row in sent[cols].iterrows():
                of.write('\t'.join(row.astype(str).tolist())+'\n')
            of.write('\n')
            
    
def prune_lattices(lattices_path, ner_pred_path, output_path, keep_all_if_no_valid=True):
    lat = bclm.read_lattices(lattices_path)
    bc = get_biose_count(ner_pred_path, sent_id_shift=1)
    valid_edges = get_valid_edges(lat, bc, non_o_only=False, keep_all_if_no_valid=keep_all_if_no_valid)
    cols = ['sent_id', 'token_id', 'ID1', 'ID2']
    pruned_lat = lat[lat[cols].apply(lambda x: tuple(x), axis=1).isin(valid_edges)]
    to_lattices(pruned_lat, output_path)
    

def soft_merge_bio_labels(multitok_sents, tokmorph_sents, verbose=False):
    new_sents = []
    for (i, mt_sent), (sent_id, mor_sent) in zip(multitok_sents.iteritems(), tokmorph_sents.iteritems()):
        new_sent = []
        for (form, bio), (token_id, token_str, forms) in zip(mt_sent, mor_sent):
            forms = forms.split('^')
            bio = bio.split('^')
            if len(forms) == len(bio):
                new_forms = (1, list(zip(forms,bio)))
            elif len(forms)>len(bio):
                dif = len(forms) - len(bio)
                new_forms = (2, list(zip(forms[:dif],['O']*dif)) + list(zip(forms[::-1], bio[::-1]))[::-1])
                if verbose:
                    print(new_forms)
            else:
                new_forms = (3, list(zip(forms[::-1], bio[::-1]))[::-1])
                if verbose:
                    print(new_forms)
            new_sent.extend(new_forms[1])
        new_sents.append(new_sent)
    return new_sents


def align_multitok(ner_pred_path, tokens_path, conll_path, map_path, output_path):
    x = read_file_sents(ner_pred_path, fix_multi_tag=False)
    prun_yo = bclm.read_yap_output(treebank_set=None, tokens_path=tokens_path, dep_path=conll_path, map_path=map_path)
    prun_yo = bclm.get_token_df(prun_yo, fields=['form'])
    prun_sents = bclm.get_sentences_list(prun_yo, fields=['token_id', 'token_str', 'form'])
    new_sents = soft_merge_bio_labels(x, prun_sents, verbose=False)

    with open(output_path, 'w') as of:
        for sent in new_sents:
            for form, bio in sent:
                of.write(form+' '+bio+'\n')
            of.write('\n')


def get_fixed_for_valid_biose(bio_seq):
    o_re = re.compile('^O+$') 
    s_re = re.compile('^O*SO*$|^O*BI*EO*$')
    b_re = re.compile('^O*BI*$')
    i_re = re.compile('^I+$')
    e_re = re.compile('^I*EO*$')
    if o_re.match(bio_seq):
        return 'O'
    if s_re.match(bio_seq):
        return 'S'
    if b_re.match(bio_seq):
        return 'B'
    if i_re.match(bio_seq):
        return 'I'
    if e_re.match(bio_seq):
        return 'E'
    raise ValueError
    

def get_fixed_for_invalid_biose(parts):
    bio = 'O'
    if 'S' in parts:
        bio = 'S'
    elif 'B' in parts and 'E' in parts:
        bio='S'
    elif 'E' in parts:
        bio = 'E'
    elif 'B' in parts:
        bio = 'B'
    elif 'I' in parts:
        bio = 'I'
    return bio



def validate_biose_sequence(full_bio_seq):
    valid_bio_re = re.compile('^O*BI*$|^O*BI*EO*$|^I+$|^I*EO*$|^O*SO*$')
    bio_seq, type_seq = zip(*[('O', None) if b=='O' else b.split('-') for b in full_bio_seq])
    bio_seq = ''.join(bio_seq)
    valid_bio = valid_bio_re.match(bio_seq)
    type_seq = list(filter(lambda x: x is not None, type_seq))
    type_seq_set = set(type_seq)

    if valid_bio:
        fixed_bio = get_fixed_for_valid_biose(bio_seq)
        if fixed_bio!='O':
            fixed_bio += '-' + type_seq[0]
            
    else:
        #take the first BIOSE tag which is not O:
        #fixed_bio = list(filter(lambda x: x!='O', full_bio_seq))[0]
        #rough BIOSE and first category:
        fixed_bio = get_fixed_for_invalid_biose(bio_seq)
        if fixed_bio!='O':
            fixed_bio += '-' + type_seq[0]
        
    return valid_bio is not None, len(type_seq_set)<=1, fixed_bio


def get_fixed_bio_sequence(full_bio_seq):
    return validate_biose_sequence(full_bio_seq)[2]


def get_fixed_tok(path, orig_sents):
    x = read_file_sents(path, fix_multi_tag=False)
    new_sents = []
    for (i, ner_sent), (sent_id, yap_sent) in zip(x.iteritems(), orig_sents.iteritems()):
        for (form, bio), (token_id, token_str) in zip(ner_sent, yap_sent):
            new_sents.append((sent_id, token_id, token_str, form, bio))
    new_sents = pd.DataFrame(new_sents, columns=['sent_id', 'token_id', 'token_str', 'form', 'bio'])
    new_toks = bclm.get_token_df(new_sents, fields=['bio'])
    new_toks['fixed_bio'] = new_toks.bio.apply(lambda x: get_fixed_bio_sequence(tuple(x.split('^'))))
    return new_toks


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
    temp_input_path = os.path.join(LOCAL_TEMP_FOLDER, datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')+'_run_ner_model_'+model_name+'.txt')
    temp_conf_path = temp_input_path.replace('.txt','.conf')
    temp_log_path = temp_input_path.replace('.txt','.log')
    try:
        sents = read_text_file(input_path)
        write_tokens_file(sents, temp_input_path, dummy_o=True)

        write_ncrf_conf(temp_conf_path, temp_input_path, output_path, MODEL_PATHS[model_name]['model'], MODEL_PATHS[model_name]['dset'])

        run_ncrf_main(temp_conf_path, DEVICE, temp_log_path)
    except Exception as e:
        print(traceback.format_exc())
    if DELETE_TEMP_FILES:
        for path in [temp_input_path, temp_conf_path, temp_log_path]:
            if os.path.exists(path):
                os.remove(path)
             
            
def run_morph_yap(model_name, input_path, output_path):
    temp_tokens_path = os.path.join(LOCAL_TEMP_FOLDER, datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')+'_morph_yap_'+model_name+'.txt')
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

                
def run_morph_hybrid(model_name, input_path, output_path, align_tokens=False):
    temp_tokens_path = os.path.join(LOCAL_TEMP_FOLDER, datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')+'_morph_hybrid_'+model_name+'.txt')
    if align_tokens:
        temp_tokens_path = temp_tokens_path.replace('_morph_hybrid_', '_morph_hybrid_align_tokens_')
        temp_morph_ner_output_path = temp_tokens_path.replace('.txt','_morph_ner.txt')
    else: 
        temp_morph_ner_output_path = output_path
    temp_conf_path = temp_tokens_path.replace('.txt','_ncrf.conf')
    temp_yap_log_path = temp_tokens_path.replace('.txt','_yap.log')
    temp_lattices_path = temp_tokens_path.replace('.txt','.lattices')
    temp_seg_path = temp_tokens_path.replace('.txt','.seg')
    temp_map_path = temp_tokens_path.replace('.txt','.map')
    temp_conll_path = temp_tokens_path.replace('.txt','.conll')
    temp_ncrf_morph_input = temp_tokens_path.replace('.txt', '_yapform.txt')
    temp_ncrf_log_path = temp_tokens_path.replace('.txt','_ncrf.log')
    temp_multi_output_path = temp_tokens_path.replace('.txt','_multi.txt')
    temp_pruned_lattices_path = temp_tokens_path.replace('.txt','_pruned.lattices')
    try:
        # read file and tokenize
        sents = read_text_file(input_path)
        #write temporary tokens file for yap
        write_tokens_file(sents, temp_tokens_path, only_tokens=True)
        #run yap hebma to create ambiguous lattices
        run_yap_hebma(temp_tokens_path, temp_lattices_path, temp_yap_log_path)
        #run multi
        run_ner_model(MULTI_MODEL_FOR_HYBRID, input_path, temp_multi_output_path)
        #prune lattices
        prune_lattices(temp_lattices_path, temp_multi_output_path, temp_pruned_lattices_path)
        #run yap joint on pruned lattices
        run_yap_joint(temp_pruned_lattices_path, temp_seg_path, temp_map_path, temp_conll_path, temp_yap_log_path)
        #create temp morph input with dummy o
        lat = bclm.read_conll(temp_conll_path)
        form_sents = lat.groupby('sent_id').form.apply(lambda x: x.tolist()).tolist()
        write_tokens_file(form_sents, temp_ncrf_morph_input, dummy_o=True)
        #run ncrf morph model
        write_ncrf_conf(temp_conf_path, temp_ncrf_morph_input, temp_morph_ner_output_path, MODEL_PATHS[model_name]['model'], MODEL_PATHS[model_name]['dset'])
        run_ncrf_main(temp_conf_path, DEVICE, temp_ncrf_log_path)
        
        if align_tokens:
            prun_yo = bclm.read_yap_output(treebank_set=None,
                                       tokens_path=temp_tokens_path,
                                       dep_path=temp_conll_path,
                                       map_path=temp_map_path,
                                        )
            prun_sents = bclm.get_sentences_list(prun_yo, fields=['token_id', 'token_str'])
            new_toks = get_fixed_tok(temp_morph_ner_output_path, orig_sents=prun_sents)
            new_sents = bclm.get_sentences_list(new_toks, fields=['token_str', 'fixed_bio'])
            write_tokens_file(new_sents, output_path)

    except Exception as e:
        print(traceback.format_exc())
    #delete all temp files
    if DELETE_TEMP_FILES:
        for path in [temp_tokens_path, temp_conf_path, temp_yap_log_path, temp_lattices_path, 
                     temp_seg_path, temp_map_path, temp_conll_path,
                     temp_ncrf_morph_input, temp_ncrf_log_path,
                     temp_multi_output_path, temp_pruned_lattices_path, temp_morph_ner_output_path]:
            if os.path.exists(path):
                os.remove(path)

                     
def multi_to_single(model_name, input_path, output_path):
    temp_multi_output_path = os.path.join(LOCAL_TEMP_FOLDER, datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')+'_multi_to_single_'+model_name+'.txt')
    try:
        run_ner_model(model_name, input_path, temp_multi_output_path)
        sents = read_file_sents(temp_multi_output_path, fix_multi_tag=True).tolist()
        write_tokens_file(sents, output_path)
    except Exception as e:
        print(traceback.format_exc())
    #delete all temp files
    if DELETE_TEMP_FILES:
        for path in [temp_multi_output_path]:
            if os.path.exists(path):
                os.remove(path)          
                     
def run_multi_align_hybrid(model_name, input_path, output_path):
    temp_tokens_path = os.path.join(LOCAL_TEMP_FOLDER, datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')+'_multi_align_hybrid_'+model_name+'.txt')
    temp_conf_path = temp_tokens_path.replace('.txt','_ncrf.conf')
    temp_yap_log_path = temp_tokens_path.replace('.txt','_yap.log')
    temp_lattices_path = temp_tokens_path.replace('.txt','.lattices')
    temp_seg_path = temp_tokens_path.replace('.txt','.seg')
    temp_map_path = temp_tokens_path.replace('.txt','.map')
    temp_conll_path = temp_tokens_path.replace('.txt','.conll')
    temp_ncrf_morph_input = temp_tokens_path.replace('.txt', '_yapform.txt')
    temp_ncrf_log_path = temp_tokens_path.replace('.txt','_ncrf.log')
    temp_multi_output_path = temp_tokens_path.replace('.txt','_multi.txt')
    temp_pruned_lattices_path = temp_tokens_path.replace('.txt','_pruned.lattices')
    try:
        # read file and tokenize
        sents = read_text_file(input_path)
        #write temporary tokens file for yap
        write_tokens_file(sents, temp_tokens_path, only_tokens=True)
        #run yap hebma to create ambiguous lattices
        run_yap_hebma(temp_tokens_path, temp_lattices_path, temp_yap_log_path)
        #run multi
        run_ner_model(MULTI_MODEL_FOR_HYBRID, input_path, temp_multi_output_path)
        #prune lattices
        prune_lattices(temp_lattices_path, temp_multi_output_path, temp_pruned_lattices_path)
        #run yap joint on pruned lattices
        run_yap_joint(temp_pruned_lattices_path, temp_seg_path, temp_map_path, temp_conll_path, temp_yap_log_path)
        align_multitok(temp_multi_output_path, 
                   temp_tokens_path, 
                   temp_conll_path,
                   temp_map_path,
                   output_path
                  )
    except Exception as e:
        print(traceback.format_exc())
    #delete all temp files
    if DELETE_TEMP_FILES:
        for path in [temp_tokens_path, temp_conf_path, temp_yap_log_path, temp_lattices_path, 
                     temp_seg_path, temp_map_path, temp_conll_path,
                     temp_ncrf_morph_input, temp_ncrf_log_path,
                     temp_multi_output_path, temp_pruned_lattices_path]:
            if os.path.exists(path):
                os.remove(path)    

    
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
    if command=='multi_to_single':
        multi_to_single(model_name, input_path, output_path)
        
    #run morph model on yap segmented output
    if command=='morph_yap':
        run_morph_yap(model_name, input_path, output_path)
    
    #run morph model on hybrid segmented output (includes an initial run of multi model)
    if command=='morph_hybrid':
        run_morph_hybrid(model_name, input_path, output_path)
        
    #run multi model and align hybrid segmented output 
    if command=='multi_align_hybrid':
        run_multi_align_hybrid(model_name, input_path, output_path)
        
    #run morph model on hybrid segmented output and result align back to tokens
    if command=='morph_hybrid_align_tokens':
        run_morph_hybrid(model_name, input_path, output_path, align_tokens=True)

