import transformers
from typing import Callable, Iterable, List, Optional, Tuple, Union
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel, AutoConfig, LogitsProcessorList, BeamSearchScorer, MinLengthLogitsProcessor, LogitsProcessor, LlamaTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoProcessor, SeamlessM4Tv2Model, SeamlessM4Tv2ForSpeechToText
from transformers import SeamlessM4TTokenizer
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
from madlad_seamless_logits_processor import LLMLogitsProcessor
import torchaudio
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#from madlad_debug_beam_scores import LLMLogitsProcessor



def get_parser():
    """
    Generate a parameter parser
    """
    parser = argparse.ArgumentParser(description="Evaluate Joint Decoded Translation for LLM's")
    parser.add_argument("--input_file", type=str, default="")
    parser.add_argument("--wav_input", type=str, default="")
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

def append_data(data, fname):
    with open(fname, 'a', encoding='utf-8') as (f):
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

def read_wavs(wav_paths):
    audio_data = []
    sample_rates = []
    noaudio = []
    
    for filename in wav_paths:
        try:
            waveform, sample_rate = torchaudio.load(filename)
            audio_data.append(waveform)
            sample_rates.append(sample_rate)
        except:
            print(filename)
            noaudio.append(filename)

    return audio_data, sample_rates, noaudio

def main(params):
    
    transformers.set_seed(0)

    src = read_data(params.input_file)
    
    #src = src[11:]

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
    
    tokenizer = T5Tokenizer.from_pretrained(params.model, cache_dir=params.cache_dir,use_fast=True)
    model_hf = T5ForConditionalGeneration.from_pretrained(model_name, device_map=device_map, cache_dir=params.cache_dir)
    model = BetterTransformer.transform(model_hf, keep_original_model=False)

    
    wav_files = read_data(params.wav_input)
    wav_files = ["./data/MuST-SHE-v1.2.1-data/wav/" + str(x) + ".wav" for x in wav_files]
    src_audio,src_rates, noaudio = read_wavs(wav_files)

    src_batches = list(divide_chunks(src,params.batch_size))
    src_audio_batches = list(divide_chunks(src_audio,params.batch_size))
    
    hyp_llm = []

    llm_model = SeamlessM4Tv2ForSpeechToText.from_pretrained("facebook/seamless-m4t-v2-large",cache_dir=params.cache_dir, device_map=device_map, torch_dtype=torch.float16).to("cuda:1")
    llm_model = PeftModel.from_pretrained(llm_model, "skoneru/seamless_bal/",is_trainable=False).to("cuda:1")
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    llm_tokenizer = SeamlessM4TTokenizer.from_pretrained("facebook/seamless-m4t-v2-large")

    cnt = 0

    with torch.inference_mode():
        for j in range(len(src_batches)):
            src_batch = src_batches[j]
            src_batch = ["<2fr> " + x for x in src_batch]
            inputs = tokenizer(src_batch,return_tensors='pt', padding=True).to(model.device)

            src_audio_batch_single = src_audio_batches[j]
            
            try:
                src_audio_batch = [element for element in src_audio_batch_single for _ in range(params.num_beams)]
                src_audio_batch = [element for element in src_audio_batch for _ in range(params.topk)]
                audio_features = processor(audios=src_audio_batch, return_tensors="pt", sampling_rate=src_rates[0], use_fast=False).to("cuda:1")
                logits_processor = LogitsProcessorList(
                    [
                        LLMLogitsProcessor(llm_model, audio_features, processor,llm_tokenizer,sent_tokenizer = tokenizer, top_k=params.topk, start_prefix_split=1, batch_size=params.batch_size, num_beams=params.num_beams, alpha=params.alpha, beta=params.beta),
                    ]
                )
                
                outputs = model.generate(**inputs, max_new_tokens=128,num_beams=params.num_beams, early_stopping=False, length_penalty=0,logits_processor=logits_processor, num_return_sequences=params.num_beams, do_sample=False)
            except:
                append_data(["--------------"], params.output_file)
                src_audio_batch = [element for element in src_audio_batch_single for _ in range(params.num_beams)]
                src_audio_batch = [element for element in src_audio_batch for _ in range(1)]
                audio_features = processor(audios=src_audio_batch, return_tensors="pt", sampling_rate=src_rates[0], use_fast=False).to("cuda:1")
                logits_processor = LogitsProcessorList(
                    [
                        LLMLogitsProcessor(llm_model, audio_features, processor,llm_tokenizer,sent_tokenizer = tokenizer, top_k=1, start_prefix_split=1, batch_size=params.batch_size, num_beams=params.num_beams, alpha=params.alpha, beta=params.beta),
                    ]
                )
                
                outputs = model.generate(**inputs, max_new_tokens=128,num_beams=params.num_beams, early_stopping=False, length_penalty=0,logits_processor=logits_processor, num_return_sequences=params.num_beams, do_sample=False)

            
            #outputs = model.generate(**inputs, max_new_tokens=256,num_beams=params.num_beams, early_stopping=False,logits_processor=logits_processor, num_return_sequences=1)
            hyps = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            hyp_llm.extend(hyps)
            cnt+=len(src_batch)
            print(hyps)
            print(cnt)
            append_data(hyps, params.output_file)
            
            #print(outputs.sequences_scores)
            #breakpoint()


        #write_list(hyp_llm, params.output_file)

    return

if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()

    main(params)

