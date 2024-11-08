import transformers
import torch
import argparse
import os
import sys
import torchaudio
from transformers import AutoProcessor, SeamlessM4Tv2Model, SeamlessM4Tv2ForSpeechToText
import pandas as pd
from peft import PeftModel, PeftConfig
from torch import nn

def get_parser():
    """
    Generate a parameter parser
    """
    parser = argparse.ArgumentParser(description="ASR using SeamlessM4T")
    parser.add_argument("--wav_input", type=str, default="")
    parser.add_argument("--text_input", type=str, default="")
    parser.add_argument("--score_output", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--list_size", type=int, default=25)
    parser.add_argument("--model", type=str, default="facebook/seamless-m4t-v2-large")
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--language", type=str, default="fra")

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

def read_wav(file_path):
    audio_data = []
    sample_rates = []

    waveform, sample_rate = torchaudio.load(file_path)
    audio_data.append(waveform)
    sample_rates.append(sample_rate)

    return audio_data, sample_rates


def main(params):
    

    transformers.set_seed(0)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    wav_files = read_data(params.wav_input)

    
    wav_files = ["./data/MuST-SHE-v1.2.1-data/wav/" + str(x) + ".wav" for x in wav_files]
    src_audio,src_rates, noaudio = read_wavs(wav_files)
    hyps = read_data(params.text_input)

    print(len(noaudio))
    assert len(noaudio) == 0

    model = SeamlessM4Tv2ForSpeechToText.from_pretrained("facebook/seamless-m4t-v2-large",cache_dir=params.cache_dir).to(device)
    model = PeftModel.from_pretrained(model, "skoneru/seamless_bal",is_trainable=False).to(device)
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")

    
    src_batches = list(divide_chunks(src_audio,params.batch_size))
    hyp_batches = list(divide_chunks(hyps,params.batch_size*params.list_size))
    #lm_head_device = model.hf_device_map["model.encoder.layer_norm"]
    hyp = []

    cnt = 0
    sent_probs = []
   
    with torch.no_grad():
        for j in range(len(src_batches)):
            src_batch = src_batches[j]
            src_batch = [element for element in src_batch for _ in range(params.list_size)]

            hyps_batch = hyp_batches[j]
            hyps_batch = ["</s>__fra__ " + x + "</s>" for x in hyps_batch]
            #hyp_batch = [element for element in hyp_batch for _ in range(params.list_size)]


            
            for k in range(5):
                src_batch_small = src_batch[k*5: (k+1)*5]
                audio_inputs = processor(audios=src_batch_small, return_tensors="pt", sampling_rate=src_rates[0])

                for key in audio_inputs.keys():
                    audio_inputs[key] = audio_inputs[key].to(device)
                
                hyps_batch_small = hyps_batch[k*5: (k+1)*5]
                decoder_inputs = processor.tokenizer(hyps_batch_small ,return_tensors='pt', padding=True, return_token_type_ids=False, add_special_tokens=False).to(model.device)

                outputs = model(**audio_inputs, decoder_input_ids=decoder_inputs.input_ids,return_dict=True, use_cache=False)
                
                for l in range(len(src_batch_small)):
                    elem_input = decoder_inputs.input_ids[l]
                    #assert decoder_inputs.input_ids[0][-1] == 1
                    pad_tokens = torch.sum(elem_input == processor.tokenizer.pad_token_id)
                    logprobs = nn.functional.log_softmax(outputs.logits[l], dim=-1)
                    logprobs = logprobs[1:logprobs.shape[0] - pad_tokens]
                    tok_probs = logprobs[range(0,len(elem_input) - pad_tokens - 2), elem_input[2:len(elem_input)-pad_tokens]]
                    sent_prob = torch.mean(tok_probs)
                    sent_probs.append(str(sent_prob.cpu().item()))
                    print(sent_prob)   
                 

           

            cnt+=params.batch_size
            print(cnt)
            del outputs
       
        
    write_list(sent_probs, params.score_output)
    return

if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()

    main(params)

