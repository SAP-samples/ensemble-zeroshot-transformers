import transformers
import copy
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel, AutoConfig, LogitsProcessorList, BeamSearchScorer, MinLengthLogitsProcessor, LogitsProcessor
import argparse
import os
import sys
import torch
import numpy as np
from transformers import BitsAndBytesConfig
#from peft import PeftModel, PeftConfig
from torch import nn
import logging
import string

class LLMLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] enforcing a min-length by setting EOS probability to 0.

    Args:
        model (`hf.model`):
            LLM Model for rescoring.
        tokenizer:
            LLM Tokenizer.
        sent_tokenizer:
            NMT Tokenizer.
        top_k:
            How many top tokens in each beam to use for rescoring.
        start_prefix_split:
            How many space split tokens during the starting of the decoding are prefix tokens which we dont pass to LLM for rescoring. E.g for NLLB its 1 <s><DE>
        src_text:
            The source file we want to translate to set the prefix of LLM while rescoring
        batch_size:
            Batch size during model.generate() to keep track of which sentence we are decoding
        num_beams:
            Number of beams passed to model.generate()
        alpha:
            Fixed Linear weight for NMT scores
        beta:
            Fixed Linear weight for LLM scores
    """

    def __init__(self, model, audio_features, processor, llm_tokenizer, sent_tokenizer, top_k, start_prefix_split, batch_size, num_beams, alpha, beta):

        self.model = model
        self.audio_features = audio_features
        self.start_prefix_split = start_prefix_split
        self.topk = top_k
        self.processor = processor
        self.tokenizer = llm_tokenizer
        self.sent_tokenizer = sent_tokenizer
        self.batch_size = batch_size
        self.num_beams = num_beams
        self.counter = -1
        self.alpha = alpha
        self.beta = beta

        ### Cached Past values for prompt to speed up the forward pass
        self.past_key_values = None
        self.past_logits = None
        self.past_attention_mask = None

        self.start_cnt = 1
        self.num_prefix_tokens = 1
        self.step = 1 # Use for Debugging

        self.llm_lenpen=1






    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:


        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename='debug.log', filemode='w')
        #logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        if input_ids.shape[1] < self.num_prefix_tokens:
            return scores

        if input_ids.shape[1] == self.num_prefix_tokens:# We get input tokens with only prefix tokens. This is start of generating new batch
            logging.debug("Current batch number: %f",self.counter)
            self.counter +=1
            self.step = 1
            self.past_key_values = None
            self.past_logits=None
            self.llm_beam_scores = [0] * (self.num_beams*self.batch_size*self.topk)
            self.sent_beam_scores = [0] * (self.num_beams*self.batch_size*self.topk)
            self.full_beam_scores = [0] * (self.num_beams*self.batch_size*self.topk)
            self.output_sent_scores = [] #Scores for sentence-nmt at each time step: First index time step, Second index beam id
            self.prev_true_beam_scores = [0] * (self.num_beams*self.batch_size*self.topk)
            self.prev_input_ids = []
            self.prev_last_word_tokens = []
            self.prev_word_ending = []
            self.prev_llm_scores = []



        logging.debug("Step number: %d", self.step)
        logging.debug("*"*100)


        cur_len = input_ids.shape[-1]
        nll_scores = nn.functional.log_softmax(scores, dim=-1)
        topk_values, topk_indices = torch.topk(nll_scores, self.topk, dim=1) #Get top k tokens for each beam 2D (batch_size*beam_size,top_k)

        
        rerank_prefix_tokens = []
        rerank_full_new_tokens = [] # The full sequence with new token joined
        rerank_new_tokens = [] # Only the last token for this step
        new_input_ids = [] # Just in case the first token is a single meta space, we keep track and use that token id for llama instead of empty
        add_prefix_space = []
        new_sent_scores = []
        sent_input_ids = []

        for row_idx in range(topk_indices.size(0)):
            top_idx = topk_indices[row_idx,:]

            top_tokens = self.sent_tokenizer.batch_decode(top_idx) 
            prev_tokens = self.sent_tokenizer.batch_decode(input_ids) 

            prev_tokens = [x.split("<unk>")[1] for x in prev_tokens]


            for k in range(self.topk):
                concat_input = torch.cat((input_ids[row_idx],top_idx[k].unsqueeze(0)),dim = 0) 
                sent_input_ids.append(concat_input)
                rerank_full_new_tokens.append(self.sent_tokenizer.decode(concat_input))
                new_input_ids.append(concat_input)
                new_sent_scores.append(topk_values[row_idx][k])

            rerank_new_tokens.extend(top_tokens)

        # Madlad starts with <unk> as start, so get the remaining hyp for reranking
        rerank_full_new_tokens = [x.split("<unk>")[1] for x in rerank_full_new_tokens]
        rerank_full_new_tokens = [x.lstrip() if x!="" else "" for x in rerank_full_new_tokens]

        cxt_rerank_prompt_tokens = ["</s>__fra__ "   for x in rerank_full_new_tokens]
        cxt_rerank_full_tokens = ["</s>__fra__ " + x  for x in rerank_full_new_tokens]

        if self.past_key_values == None:
            llm_prompt_inputs = self.tokenizer(cxt_rerank_prompt_tokens,return_tensors='pt', padding=True,  return_token_type_ids=False, add_special_tokens=False).to(self.model.device)
            llm_prompt_outputs = self.model(**self.audio_features, decoder_input_ids=llm_prompt_inputs.input_ids,return_dict=True, use_cache=True)

            past_full_logits = llm_prompt_outputs.logits

            self.past_logits = nn.functional.log_softmax(past_full_logits,dim=-1)[:,-1:,:]
            self.past_key_values = llm_prompt_outputs.past_key_values
            self.past_attention_mask = llm_prompt_inputs.attention_mask
            last_word_tokens = rerank_full_new_tokens

        else:    

            align_beam_idx = self.beam_align(self.prev_input_ids,input_ids)
            self.prev_last_word_tokens = [self.prev_last_word_tokens[idx] for idx in align_beam_idx for _ in range(self.topk)]
            self.prev_llm_scores = [self.prev_llm_scores[idx] for idx in align_beam_idx for _ in range(self.topk)]
            self.prev_word_ending = [self.prev_word_ending[idx] for idx in align_beam_idx for _ in range(self.topk)]
            self.llm_beam_scores = [self.llm_beam_scores[idx] for idx in align_beam_idx for _ in range(self.topk)]
            self.sent_beam_scores = [self.sent_beam_scores[idx] for idx in align_beam_idx for _ in range(self.topk)]
            self.prev_true_beam_scores = [self.prev_true_beam_scores[idx] for idx in align_beam_idx for _ in range(self.topk)]

            realigned_sent_scores = []
            for step in range(len(self.output_sent_scores)):
                realigned_sent_scores.append([self.output_sent_scores[step][idx] for idx in align_beam_idx for _ in range(self.topk)])
            self.output_sent_scores = realigned_sent_scores


            last_word_tokens = [x.split(" ")[-1] if len(x.split(" ")) > 1 else x for x in rerank_full_new_tokens]
            #last_word_tokens = [x if x!="<unk> " else " " for x in last_word_tokens]
            #last_word_tokens = [x if x!="" else " " for x in last_word_tokens]
            logging.debug("Current Last Word Tokens")
            logging.debug(last_word_tokens)
            logging.debug("="*100)


        llm_inputs = self.tokenizer(cxt_rerank_full_tokens,return_tensors='pt', padding=True,  return_token_type_ids=False, add_special_tokens=False).to(self.model.device)
        #llm_inputs['input_ids'] = llm_inputs.input_ids[:,1:]
        #llm_inputs['attention_mask'] = torch.cat((self.past_attention_mask, llm_inputs.attention_mask[:,1:]), dim=1)

        if self.step == 1:
            mask = torch.all(llm_inputs['input_ids'][:,2:] == self.tokenizer.pad_token_id, dim=1)
            llm_inputs['input_ids'][mask,2] = self.tokenizer.sp_model.piece_to_id("▁") + 1
            llm_inputs['attention_mask'][mask,2] = 1


        
        llm_outputs = self.model(**self.audio_features, labels=llm_inputs.input_ids, return_dict=True, past_key_values = self.past_key_values)
        loss = llm_outputs.loss
        llm_logits = llm_outputs.logits
        
        llm_scores = []

        alpha = []
        beta = []
        prev_word_ending = []
        correct_error_scores = [0]*len(cxt_rerank_full_tokens)
        is_new_words = []

        llm_beam_scores= [0] * (self.num_beams*self.batch_size*self.topk)
        true_beam_scores = [0] * (self.num_beams*self.batch_size*self.topk)

        
        for elem in range(len(cxt_rerank_full_tokens)):

            self.sent_beam_scores[elem] = new_sent_scores[elem] + self.sent_beam_scores[elem]

            tok_entries = self.sent_tokenizer.convert_ids_to_tokens(sent_input_ids[elem])
            tok_spaces = []
            for tok_idx in range(len(tok_entries)-1,0,-1):
                if "▁" == tok_entries[tok_idx][0]:
                    break
            
            # if len(tok_spaces) == 1:
            #     last_word_sent_hyp_ids = sent_input_ids[elem][tok_spaces[0]:]
            # elif tok_spaces[-1] == len(tok_entries) - 1:
            #     last_word_sent_hyp_ids = sent_input_ids[elem][tok_spaces[-2]:]
            # else:
            #     last_word_sent_hyp_ids = sent_input_ids[elem][tok_spaces[-1]:]
            if tok_idx == 0: 
                last_word_sent_hyp_ids = sent_input_ids[elem][1:]
            else:
                last_word_sent_hyp_ids = sent_input_ids[elem][tok_idx:]
            #last_word_sent_hyp_ids = self.sent_tokenizer.encode(self.tokenizer.decode(last_word_inputs), add_special_tokens=False)
            #full_hyp_sent_ids = self.sent_tokenizer.encode(cxt_rerank_full_tokens[elem], add_special_tokens=False)
            full_hyp_sent_ids = sent_input_ids[elem][1:]


            elem_input = llm_inputs.input_ids[elem][2:]
            pad_tokens = torch.sum(elem_input == self.tokenizer.convert_tokens_to_ids('<pad>'))

            last_word_inputs = self.tokenizer.encode(last_word_tokens[elem],return_tensors='pt', padding=False,return_token_type_ids=False, add_special_tokens=False)[0].to(self.model.device)

            if rerank_new_tokens[elem] == "</s>":
                last_word_inputs = torch.tensor([self.tokenizer.encode("</s>")[1]],device=self.model.device)
                pass
            else:
                if len(last_word_inputs) == 0:
                    last_word_inputs = torch.tensor([self.tokenizer.sp_model.piece_to_id("▁") + 1], device=self.model.device)
 

            nll_new_logits = nn.functional.log_softmax(llm_logits[elem],dim=-1)
            nll_full_logits = nll_new_logits[2:len(elem_input) - pad_tokens + 2]
            #nll_full_logits = torch.cat((self.past_logits[0],nll_new_logits), dim=0)
            
            llm_next_outputs = torch.argmax(nll_full_logits[-1,:], dim=-1)
            #nll_full_logits = nll_full_logits[pad_tokens:]
            if len(elem_input) - pad_tokens == len(last_word_inputs):
                nll_logits = nll_full_logits
                prev_word_score = 0
            else:
                nll_logits = nll_full_logits[len(elem_input) - pad_tokens  - len(last_word_inputs):len(elem_input)  - pad_tokens]

                ## Get score for words before the last word
                nll_prev_word_logits = nll_full_logits[:len(elem_input) - pad_tokens  - len(last_word_inputs)]
                prev_word_score = torch.sum(nll_prev_word_logits[range(len(elem_input) - pad_tokens  - len(last_word_inputs)), elem_input[:len(elem_input) - pad_tokens  - len(last_word_inputs)]])

            
            if self.step == 1:
                is_new_word = True
            else:
                is_new_word = False
                if rerank_new_tokens[elem] != "</s>":
                    is_new_word = self.check_last_word(cxt_rerank_full_tokens[elem], self.prev_last_word_tokens[elem], last_word_tokens[elem])
                else:
                    is_new_word = True
                if self.prev_word_ending[elem] == True and is_new_word == False:
                    correct_error_scores[elem] = -1 * self.prev_llm_scores[elem]
                if self.prev_word_ending[elem] == False and is_new_word == True:
                    correct_error_scores[elem] = self.prev_llm_scores[elem]
                    

            
            elem_score = torch.sum(nll_logits[range(len(last_word_inputs)),last_word_inputs])
            is_new_words.append(is_new_word)


            elem_score = torch.nan_to_num(elem_score, nan=-100)
            llm_scores.append(elem_score)


            gen_tok = self.tokenizer.sp_model.id_to_piece(llm_next_outputs.item() - 1)
            #if gen_tok[0] == "▁" or rerank_new_tokens[elem]=="</s>":
            if gen_tok == "</s>" or gen_tok[0] == "▁" or rerank_new_tokens[elem] == "</s>":
                alpha.append(self.alpha)
                beta.append(self.beta)
                prev_word_ending.append(True)
                self.llm_beam_scores[elem] = self.llm_beam_scores[elem] + elem_score + correct_error_scores[elem] 
            else:
                alpha.append(1)
                beta.append(0)
                prev_word_ending.append(False)
                self.llm_beam_scores[elem] = self.llm_beam_scores[elem] + correct_error_scores[elem] 


            if self.step!=1:
                prev_word_sent_score, last_word_sent_score = self.get_sent_scores(self.output_sent_scores, new_sent_scores, full_hyp_sent_ids, last_word_sent_hyp_ids, elem)
                
                assert (len(full_hyp_sent_ids) -1) == (len(self.output_sent_scores)), "{},{}".format(len(full_hyp_sent_ids), len(self.output_sent_scores))

            else:
                prev_word_sent_score = 0
                last_word_sent_score = new_sent_scores[elem]

                if len(last_word_sent_hyp_ids) == 0:
                    last_word_sent_hyp_ids = [805]
            
            sent_full_alpha_score = ((alpha[elem] * (prev_word_sent_score + last_word_sent_score))/(len(input_ids[int(elem/self.topk)]))) ## No need to add 1 more to mean since this includes extra <unk> token

            sent_last_score = (last_word_sent_score/len(last_word_sent_hyp_ids))
            sent_last_score_unnorm = (last_word_sent_score)

            prev_hyp_sent_len = len(full_hyp_sent_ids) - len(last_word_sent_hyp_ids)
            assert prev_hyp_sent_len >= 0
            if (len(full_hyp_sent_ids) - len(last_word_sent_hyp_ids)) != 0:
                sent_prev_alpha_score = ((self.alpha * (prev_word_sent_score))/(len(full_hyp_sent_ids) - len(last_word_sent_hyp_ids)))
                sent_prev_alpha_score_unnorm = self.alpha * (prev_word_sent_score)
            else:
                sent_prev_alpha_score = 0
                sent_prev_alpha_score_unnorm = 0

            llm_full_beta_score = self.beta*(prev_word_score + elem_score)/((len(elem_input) - pad_tokens)**self.llm_lenpen) ### use beta for multiplication
            llm_beam_scores[elem] = llm_full_beta_score

            if len(elem_input) - len(last_word_inputs) - pad_tokens != 0:
                llm_prev_beta_score = ((self.beta * (prev_word_score))/((len(elem_input) - pad_tokens -  len(last_word_inputs))**self.llm_lenpen))
                llm_prev_beta_score_unnorm = self.beta * (prev_word_score)
                llm_last_beta_score =  ((self.beta * (elem_score))/(len(last_word_inputs)**self.llm_lenpen))
            else:
                llm_prev_beta_score = torch.tensor(0)
                llm_prev_beta_score_unnorm = torch.tensor(0)
                llm_last_beta_score =  ((self.beta * (elem_score))/(len(last_word_inputs)**self.llm_lenpen))
            

            try:
                sent_prev_alpha_score = sent_prev_alpha_score.cpu()
                sent_prev_alpha_score_unnorm = sent_prev_alpha_score_unnorm.cpu()
            except:
                pass
            try:
                sent_last_score = sent_last_score.cpu()
                sent_last_score_unnorm = sent_last_score_unnorm.cpu()
            except:
                pass

            if prev_word_ending[-1]:
                self.full_beam_scores[elem] =  llm_full_beta_score.cpu() + sent_full_alpha_score.cpu() - self.prev_true_beam_scores[elem]
                curr_score  = llm_full_beta_score.cpu() + sent_full_alpha_score.cpu() 
                true_beam_scores[elem] =  curr_score
            else:
                #self.full_beam_scores[elem] =  (sent_full_alpha_score.cpu())/(alpha[elem]) - self.prev_true_beam_scores[elem]
                #curr_score  = (sent_full_alpha_score.cpu())/(alpha[elem])
                #true_beam_scores[elem] =  curr_score      
                #self.full_beam_scores[elem] = ((sent_prev_alpha_score_unnorm + llm_prev_beta_score_unnorm + sent_last_score_unnorm)/len(full_hyp_sent_ids)) - self.prev_true_beam_scores[elem]
                #curr_score = ((sent_prev_alpha_score_unnorm + llm_prev_beta_score_unnorm + sent_last_score_unnorm)/len(full_hyp_sent_ids))
                self.full_beam_scores[elem] = ((sent_prev_alpha_score + llm_prev_beta_score)*prev_hyp_sent_len + (sent_last_score_unnorm))/len(full_hyp_sent_ids) - (self.prev_true_beam_scores[elem])
                curr_score = ((sent_prev_alpha_score + llm_prev_beta_score)*prev_hyp_sent_len + (sent_last_score_unnorm))/len(full_hyp_sent_ids) 
                #curr_score =  (sent_prev_alpha_score + llm_prev_beta_score) + (sent_last_score)
                true_beam_scores[elem] =  curr_score
            
            try:
                assert curr_score < 0
            except:
                breakpoint()
           
            
            

        # for i in range(len(llm_beam_scores)):
        #     print(rerank_full_new_tokens[i])
        #     print(last_word_tokens[i])
        #     print(llm_beam_scores[i], self.sent_beam_scores[i], prev_word_ending[i], true_beam_scores[i])
        
        #print("###"*100)
        #top_5_indices = sorted(range(len(true_beam_scores)), key=lambda i: true_beam_scores[i], reverse=True)[:5]
        #for x in top_5_indices:
        #    print(rerank_full_new_tokens[x], true_beam_scores[x], )
        #print("###"*100)

        reranked_scores = torch.full_like(scores, -float("inf")).to(input_ids.device)
        altered_beam_scores = torch.tensor(self.full_beam_scores).view(input_ids.shape[0],-1).contiguous().to(input_ids.device)
        for i in range(reranked_scores.shape[0]):
            #reranked_scores[i,topk_indices[i,:]] = alpha_vals[i]*scores[i,topk_indices[i,:]].to(torch.float16) + beta_vals[i]*llm_scores[i].to(torch.float16) + self.beta*correct_offset_scores[i].to(torch.float16)
            reranked_scores[i,topk_indices[i,:]] = altered_beam_scores[i]


        
        self.output_sent_scores.append(new_sent_scores)

        self.prev_last_word_tokens = last_word_tokens
        self.prev_input_ids = torch.stack(new_input_ids)
        self.prev_word_ending = prev_word_ending
        self.prev_llm_scores = llm_scores
        self.prev_true_beam_scores = true_beam_scores

        
        self.step+=1
        
        

        return reranked_scores

    def beam_align(self, tensor_old, tensor_new):
        
        row_mapping = [] # First elem indicates which row tensor_new corresponds with old

        for i in range(tensor_new.size(0)):
            for j in range(tensor_old.size(0)):
                if torch.all(tensor_old[j,:] == tensor_new[i,:]):
                    row_mapping.append(j)
                    break

        return row_mapping

    def join_detok(self,llm_prefix,llm_suffix):
        joined_tokens = {}
        joined_tokens['input_ids'] = torch.cat((llm_prefix['input_ids'],llm_suffix['input_ids']),dim=1)
        joined_tokens['attention_mask'] = torch.cat((llm_prefix['attention_mask'],llm_suffix['attention_mask']),dim=1)
        join_detok = self.tokenizer.batch_decode(joined_tokens['input_ids'], skip_special_tokens=True)
        return join_detok 

    def check_last_word(self, full_sent, prev_last_word, last_word):

        if full_sent == prev_last_word:
            return True

        joined_word = prev_last_word + " " + last_word
        if full_sent[-len(joined_word):] == joined_word:
            return True
        else:
            last_word = last_word.strip()
            prev_last_word = prev_last_word.strip()
            if last_word == "" or prev_last_word == "" or "</s>" in last_word:
                return True
            else:
                return False

    def get_sent_scores(self, output_sent_scores, new_sent_scores, full_hyp_sent_ids, last_word_sent_hyp_ids, elem):
        r""""
        Output_sent_scores: scores at each time step realigned with elem as index
        new_sent_scores: scores for the current time
        sent_hyp_ids: Tokenized input ids for the full sentence
        last_word_sent_ids: Tokenized input ids only for the last word
        elem: index of the current beam
        """
        curr_scores = []

        for step in range(len(output_sent_scores)):
            curr_scores.append(output_sent_scores[step][elem])
        curr_scores.append(new_sent_scores[elem])

        if len(full_hyp_sent_ids) == len(last_word_sent_hyp_ids):
            return 0, sum(curr_scores)
        
        else:
            return sum(curr_scores[0:len(curr_scores)-len(last_word_sent_hyp_ids)]), sum(curr_scores[len(curr_scores)-len(last_word_sent_hyp_ids):])
