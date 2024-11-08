import transformers
from typing import Callable, Iterable, List, Optional, Tuple, Union
from transformers import T5ForConditionalGeneration, T5Tokenizer
import argparse
import os
import sys
import torch
import numpy as np
import json
from transformers import BitsAndBytesConfig
#from peft import PeftModel, PeftConfig
import pandas as pd
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.big_modeling import infer_auto_device_map, init_empty_weights
from accelerate.utils import get_balanced_memory
import pickle
from optimum.bettertransformer import BetterTransformer
from torch import nn



def get_parser():
    """
    Generate a parameter parser
    """
    parser = argparse.ArgumentParser(description="Rerank and Pick Translations with LLM's")
    parser.add_argument("--input_file", type=str, default="")
    parser.add_argument("--hyp_file", type=str, default="")
    parser.add_argument("--output_file", type=str, default="")
    parser.add_argument("--best_hyp_file", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--model", type=str, default="google/madlad400-10b-mt")
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0)
    parser.add_argument("--beta", type=float, default=1)

    return parser

def read_data(filepath):
    data = open(filepath, mode='r', encoding='utf-8', newline='\n').readlines()
    data = [x.rstrip() for x in data]
    return data

def write_list(data, fname):
    with open(fname, 'w', encoding='utf-8') as (f):
        for item in data:
            try:
                f.write('%s\n' % item)
            except:
                item = item.encode('utf-8')
                f.write('%s\n' % item)

def process_sent(gen_sent, prompt):
    prompt_len = len(prompt)
    gen_sent = gen_sent[prompt_len:]
    gen_sent = gen_sent.replace("\n", ".")
    return gen_sent

def divide_chunks(l, n):
         for i in range(0, len(l), n):
             yield l[i:i + n]

def main(params):
    
    transformers.set_seed(0)

    src = read_data(params.input_file)
    rank_hyp = read_data(params.hyp_file)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = T5Tokenizer.from_pretrained(params.model, cache_dir=params.cache_dir)
    #model = AutoModelForSeq2SeqLM.from_pretrained(params.model,
	#    cache_dir=params.cache_dir, 
	#    device_map=device_map,
	#    load_in_8bit=True, offload_folder='/project/OML/skoneru/iwslt23/scripts/bloom/cache/',)
    model_hf = T5ForConditionalGeneration.from_pretrained(params.model, device_map="auto", load_in_8bit=True)
    model = BetterTransformer.transform(model_hf, keep_original_model=False)
    
    src_batches = list(divide_chunks(src,params.batch_size))
    #lm_head_device = model.hf_device_map["model.encoder.layer_norm"]
    hyp_llm = []
    hyp_best = []

#


    cnt = 0
    
    model.eval()
    sent_probs = []
    for j in range(len(src_batches)):
        src_batch = src_batches[j]
        src_batch = ["<2de> " + x for x in src_batch]
        src_batch = [src_batch[0]]*5
        rank_hyps_batch = rank_hyp[j*5: (j+1)*5]
        #rank_hyps_batch = ["<unk> " + x + tokenizer.eos_token for x in rank_hyps_batch]
        rank_hyps_batch = ["<unk> " + x  for x in rank_hyps_batch]
        only_hyps = rank_hyps_batch
        hyp_best.append(rank_hyps_batch[0])

        encoder_inputs = tokenizer(src_batch,return_tensors='pt', padding=True, return_token_type_ids=False).to(model.device)
        decoder_inputs = tokenizer(rank_hyps_batch,return_tensors='pt', padding=True, return_token_type_ids=False).to(model.device)

        outputs = model(**encoder_inputs, decoder_input_ids=decoder_inputs.input_ids,return_dict=True, use_cache=False)
        for l in range(len(src_batch)):
            elem_input = decoder_inputs.input_ids[l]
            #assert decoder_inputs.input_ids[0][-1] == 1
            pad_tokens = torch.sum(elem_input == tokenizer.convert_tokens_to_ids(tokenizer.pad_token))
            logprobs = nn.functional.log_softmax(outputs.logits[l], dim=-1)
            logprobs = logprobs[:logprobs.shape[0] - pad_tokens]
            tok_probs = logprobs[range(0,len(elem_input) - pad_tokens - 1), elem_input[1:len(elem_input)-pad_tokens]]
            sent_prob = torch.mean(tok_probs)
            sent_probs.append(str(sent_prob.cpu().item()))

        cnt+=len(src_batch)
        print(cnt)

    write_list(sent_probs, params.output_file)

    return

if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()

    main(params)

