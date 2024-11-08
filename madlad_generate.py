import transformers
from typing import Callable, Iterable, List, Optional, Tuple, Union
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel, AutoConfig, LogitsProcessorList, BeamSearchScorer, MinLengthLogitsProcessor, LogitsProcessor, LlamaTokenizer
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
from peft import PeftModel
from madlad_alma_new_logits_processor import LLMLogitsProcessor
#from madlad_debug_beam_scores import LLMLogitsProcessor



def get_parser():
    """
    Generate a parameter parser
    """
    parser = argparse.ArgumentParser(description="Evaluate Translation for LLM's")
    parser.add_argument("--input_file", type=str, default="")
    parser.add_argument("--output_file", type=str, default="")
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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_name = params.model
    config = AutoConfig.from_pretrained(model_name)
    with init_empty_weights():
        model = AutoModelForSeq2SeqLM.from_config(config)
    model.tie_weights()
    device_map = infer_auto_device_map(
        model,
        # Force splits model.encoder into separate layers and devices
        dtype="int8",
    )


    for module, device in device_map.items():
        device_map[module] = 0
    
    tokenizer = T5Tokenizer.from_pretrained(params.model, cache_dir="/project/OML/skoneru/iwslt23/scripts/bloom/cache/",use_fast=True)
    model_hf = T5ForConditionalGeneration.from_pretrained(model_name, device_map=device_map)
    model = BetterTransformer.transform(model_hf, keep_original_model=False)
    #model = model_hf
    
    src_batches = list(divide_chunks(src,params.batch_size))
    #lm_head_device = model.hf_device_map["model.encoder.layer_norm"]
    hyp_llm = []


    cnt = 0
    
    for j in range(len(src_batches)):
        src_batch = src_batches[j]
        src_batch = ["<2de> " + x for x in src_batch]
        inputs = tokenizer(src_batch,return_tensors='pt', padding=True).to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=256,num_beams=params.num_beams, early_stopping=False, num_return_sequences=params.num_beams, do_sample=False, output_scores=True, return_dict_in_generate=True)

        hyps = llm_tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

        hyp_llm.extend(hyps)
        scores.extend(outputs.sequences_scores)
        cnt+=len(src_batch)
        print(hyps)
        print(cnt)


    write_list(scores, params.output_file + ".scores")
    write_list(hyp_llm, params.output_file)

    return

if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()

    main(params)

