import sys
import os
#os.chdir('../MQuAKE/')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
import random
from tqdm import tqdm
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from llm import gptj_interface, vllm_gptj_interface
from retriever import get_sent_embeddings, retrieve_facts, batch_retrieve_facts
from util import get_all_facts_cf
from mello import MelloContext


def load_gptj_huggingface():
    print('Loading Huggingface GPT-J...')
    gptj_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", device_map='cuda:0').eval()
    gptj_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    return gptj_model, gptj_tokenizer


def test_gptj_huggingface(gptj_model, gptj_tokenizer):
    clue_config = {
        'repetition_penalty' : 1.05,
        'temperature' : 0.3,
        'top_k' : 5,
        'top_p' : 0.85,
        'max_new_tokens' : 32,
        'do_sample' : True,
    }
    tmp_stoppers = ['.']

    gptj_api = gptj_interface(gptj_tokenizer, gptj_model, tmp_stoppers, clue_config)
    # for ix in range(1000):
    #     output_sents = gptj_api.call_gptj_local('Ellie Kemper is a citizen of')
    #     for sent in output_sents:
    #         print('[Generated Sample]\t', sent, '\n')
    
    # import time
    # time.sleep(1000000)


def load_contriever():
    print('Loading Contriever...')
    contriever = AutoModel.from_pretrained("facebook/contriever-msmarco").cuda(1)
    ct_tokenizer = AutoTokenizer.from_pretrained("facebook/contriever-msmarco")
    return contriever, ct_tokenizer


def test_contriever(contriever, ct_tokenizer, new_facts, edit_embs):
    fact_ids = retrieve_facts("Who is the president of the US?", edit_embs, contriever, ct_tokenizer)
    print(new_facts[fact_ids[0]])


def run_mello_batch(cf_dataset, llm_api, task_prompt, contriever, ct_tokenizer, new_facts, edit_embs, BSZ, log_fn='./mello.log'):
    if log_fn:
        fout = open(log_fn, 'a')
    else:
        fout = sys.stdout
    str_line = '===================================================='
    
    all_ques, qid2ans = [], {}
    for entry in cf_dataset:
        all_ques.append([entry['questions'][0], entry['case_id']])
        all_ques.append([entry['questions'][1], entry['case_id']])
        all_ques.append([entry['questions'][2], entry['case_id']])
        qid2ans[entry['case_id']] = set([entry['answer']] + entry['answer_alias'])
    random.shuffle(all_ques)

    pbar = tqdm(total=len(all_ques))
    is_correct, is_complete = set(), defaultdict(lambda : 0)
    get_complete_num = lambda is_com : len([x for x in is_com.values() if x == 3])
    next_ix = BSZ
    cur_batch = [MelloContext(qi[1], qi[0], task_prompt) for qi in all_ques[:BSZ]]

    while cur_batch:
        batched_inputs = [qi_obj.make_prompt() for qi_obj in cur_batch]
        batched_outputs = llm_api.call_gptj_batch(batched_inputs)

        subques_to_retrieve, new_batch = [], []
        for qi_obj, llm_respi in zip(cur_batch, batched_outputs):
            qi_obj.update_llm_response(llm_respi[0])
            if qi_obj.status == 'Running':  # Need fact-retrieval & next iter
                if qi_obj.qid not in is_correct:
                    subques_to_retrieve.append(qi_obj.sub_qu)
                    new_batch.append(qi_obj)    # later add retriever results to the new batch
                else:   # early-stopped
                    is_complete[qi_obj.qid] += 1
                    print('[Early stopped #{}]\n{}\n{}'.format(qi_obj.qid, qi_obj.make_context_log(), str_line), file=fout, flush=True)

            else:   # Ended, calculate correctness & leave room for the new batch entries
                if qi_obj.ans in qid2ans[qi_obj.qid]:
                    is_correct.add(qi_obj.qid)
                is_complete[qi_obj.qid] += 1
                print('[{} #{}]\n{}\n{}'.format(qi_obj.status, qi_obj.qid, qi_obj.make_context_log(), str_line), file=fout, flush=True)
            
        # Batched fact retrieval
        fact_ids = batch_retrieve_facts(subques_to_retrieve, edit_embs, contriever, ct_tokenizer)
        for nix, fid in enumerate(fact_ids):
            new_batch[nix].update_retrieved_fact(new_facts[fid])
        
        pbar.set_postfix_str('Est. {}/{} = {}'.format(len(is_correct), next_ix//3, len(is_correct)*3 / next_ix))
        pbar.update(len(cur_batch) - len(new_batch))
        
        # Complement batch
        while len(new_batch) < BSZ and next_ix < len(all_ques):
            new_qi = all_ques[next_ix]
            new_batch.append(MelloContext(new_qi[1], new_qi[0], task_prompt))
            next_ix += 1
        cur_batch = new_batch   # Terminate if new_batch == []
    
    complete_num = get_complete_num(is_complete)
    assert(complete_num == len(cf_dataset)), (is_complete, complete_num)
    print(f'Multi-hop acc = {len(is_correct) / len(cf_dataset)} ({len(is_correct)} / {len(cf_dataset)})')


def run_mello(cf_dataset, llm_api, task_prompt, contriever, ct_tokenizer, new_facts, edit_embs, log_fn='./mello.log'):
    # Run mello-baseline for K=3000 (full edit space)
    if log_fn:
        fout = open(log_fn, 'a')
    else:
        fout = sys.stdout
    
    cor, tot = 0, 0
    #cor, tot = 339, 2829

    # NOTE shuffle MQUAKE-CF to fix distribution shift @~Q#1000
    random.shuffle(cf_dataset)
    for d in tqdm(cf_dataset[tot:]):
        tot += 1
        for q in d["questions"]:
            found_ans = False
            prompt = task_prompt + "\n\nQuestion: " + q
            print('======================================\n[Question]', q, file=fout)
            for ix in range(4):
                # prompt the model to generate a subquestion and a tentative answer
                #gen = call_gpt(prompt, mquake_stop)
                llm_output = llm_api.call_gptj_local(prompt)
                # 直接选择top-1
                gen = llm_output[0]
                print('\n--------~~~~~~~~--------\n', gen[len(task_prompt)+2 : ], end='\n------------------------\n', file=fout, flush=True)
                
                # if final answer is there, get the answer and exit
                # NOTE GPTJ不会结束生成, 因此将下一个Question的生成也作为finalize触发条件
                last_sent, prev_sent = gen.strip().split('\n')[-1], gen.strip().split('\n')[-3]
                if last_sent.startswith('Final answer: '):
                    ans = last_sent[len("Final answer: "):]
                    found_ans = True
                if last_sent.startswith('Question:'):
                    assert(prev_sent.startswith('Final answer: '))
                    ans = prev_sent[len("Final answer: "):]
                    found_ans = True
                if found_ans:
                    print('[Found Answer]', ans, file=fout)
                    break
                
                # otherwise, extract the generated subquestion
                if len(gen.strip().split('\n')) < 2:
                    print('[Generation Error] Only one line', file=fout)
                    break # failed case

                # NOTE StoppingCriteria会保留stop words, 此处更新逻辑以跳过最后Retrieved fact行
                subquestion = gen.strip().split('\n')[-3]
                if not subquestion.startswith('Subquestion: '):
                    print('[Subquestion Prefix Error]', subquestion, file=fout)
                    break # failed case
                subquestion = subquestion[len("Subquestion: "):]
                
                # retrieve an edited fact using the generated subquestion
                fact_ids = retrieve_facts(subquestion, edit_embs, contriever, ct_tokenizer)
                fact_sent = new_facts[fact_ids[0]]
                
                # put the retrieved fact at the end of the prompt, the model self-checks if it contradicts
                #prompt = prompt + gen + 'Retrieved fact: ' + fact_sent + '.'
                # NOTE transformers的generate结果会保留input, 此处fix prompt更新逻辑
                # 此外, 也移除额外生成的retrieved fact
                prompt = gen.strip()[:-len('\nRetrieved fact:')]
                prompt += '\nRetrieved fact: ' + fact_sent + '.'
                        
            if not found_ans:
                continue
            # if the answer is correct
            if ans == d["new_answer"] or ans in d["new_answer_alias"]:
                cor += 1
                break
        
        print('Running acc = {} / {} = {}'.format(cor, tot, cor/tot), file=fout)

    print(f'Multi-hop acc = {cor / tot} ({cor} / {tot})', file=fout)
    fout.close()


def main(framework):
    # Dataset loading & resolving edit statements
    with open('datasets/MQuAKE-CF-3k.json', 'r') as f:
        cf_dataset = json.load(f)
    new_facts = get_all_facts_cf(cf_dataset)
    
    # Embedding-based fact retriever
    contriever, ct_tokenizer = load_contriever()    
    edit_embs = get_sent_embeddings(new_facts, contriever, ct_tokenizer)
    test_contriever(contriever, ct_tokenizer, new_facts, edit_embs)

    # LLM API
    # mquake_stop = ["Retrieved fact:", "Question:"]
    mquake_stop = ["Retrieved fact:", "\n\n"]
    if framework == 'huggingface':
        gptj_model, gptj_tokenizer = load_gptj_huggingface()
        test_gptj_huggingface(gptj_model, gptj_tokenizer)
        clue_config = {
            'repetition_penalty' : 1.05,
            'temperature' : 0.3,
            'top_k' : 5,
            'top_p' : 0.85,
            'max_new_tokens' : 64,
            'do_sample' : True,
        }
        gptj_api = gptj_interface(gptj_tokenizer, gptj_model, mquake_stop, clue_config)
        log_fn = 'mello.log'
    
    elif framework == 'vllm':
        import ray
        ray.init(num_cpus=24)
        vllm_config = {
            'repetition_penalty' : 1.05,
            'temperature' : 0.3,
            'top_k' : 5,
            'top_p' : 0.85,
            'max_tokens' : 64,
            'best_of' : 3,
            'use_beam_search' : False,
        }
        gptj_api = vllm_gptj_interface(mquake_stop, vllm_config)
        log_fn = 'vllm.log'

    # In-context demonstration
    with open('prompts/MeLLo-prompt.txt', 'r', encoding='utf-8') as f:
        mello_prompt = f.read()

    # run_mello(cf_dataset, gptj_api, mello_prompt, contriever, ct_tokenizer, new_facts, edit_embs,
    #     log_fn = log_fn,
    # )
    run_mello_batch(cf_dataset, gptj_api, mello_prompt, contriever, ct_tokenizer, new_facts, edit_embs,
        BSZ = 6,
        log_fn = log_fn,
    )


if __name__ == '__main__':
    main(framework='vllm')
