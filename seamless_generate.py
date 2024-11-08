import transformers
import torch
import argparse
import os
import sys
import torchaudio
from transformers import AutoProcessor, SeamlessM4Tv2Model, SeamlessM4Tv2ForSpeechToText
import pandas as pd
from peft import PeftModel, PeftConfig

def get_parser():
    """
    Generate a parameter parser
    """
    parser = argparse.ArgumentParser(description="ASR using SeamlessM4T")
    parser.add_argument("--tsv_input", type=str, default="./data//MuST-SHE-v1.2.1-data/tsv/MONOLINGUAL.fr_v1.2.1.tsv")
    parser.add_argument("--text_output", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--model", type=str, default="facebook/seamless-m4t-v2-large")
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--language", type=str, default="fra")

    parser.add_argument("--num_beams", type=int, default=5)

    return parser

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
            if "fr-" in filename:
                filename+=".wav"
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

    df = pd.read_csv(params.tsv_input, sep="\t")
    
    wav_files = ["./data/mustshe/MuST-SHE-v1.2.1-data/wav/" + str(x) for x in df['ID'].tolist()]
    
    src_audio,src_rates, noaudio = read_wavs(wav_files)

    print(len(noaudio))
    assert len(noaudio) == 0

    model = SeamlessM4Tv2ForSpeechToText.from_pretrained("facebook/seamless-m4t-v2-large",cache_dir=params.cache_dir).to(device)
    model = PeftModel.from_pretrained(model, "skoneru/seamless_bal",is_trainable=False).to(device)
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")

    
    src_batches = list(divide_chunks(src_audio,params.batch_size))
    #lm_head_device = model.hf_device_map["model.encoder.layer_norm"]
    hyp = []

    cnt = 0
   
    for j in range(len(src_batches)):
        src_batch = src_batches[j]

        audio_inputs = processor(audios=src_batch, return_tensors="pt", sampling_rate=src_rates[0])

        for key in audio_inputs.keys():
            audio_inputs[key] = audio_inputs[key].to(device)

        outputs = model.generate(**audio_inputs, tgt_lang=params.language, max_new_tokens=256, num_beams=25, num_return_sequences=25)


        hyps = processor.batch_decode(outputs, skip_special_tokens=True)

        print(hyps,cnt)


        hyp.extend(hyps)
        cnt+=len(src_batch)
        
    write_list(hyp, params.text_output)

    return

if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()

    main(params)

