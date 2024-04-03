import sys
import os
#os.chdir('../MQuAKE/')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
import random
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from llm import gptj_interface
from retriever import get_sent_embeddings, retrieve_facts
from util import get_all_facts_cf


def load_gptj_huggingface():
    print('Loading Huggingface GPT-J...')
    gptj_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", device_map='auto')
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
    output_sents = gptj_api.call_gptj_local('Ellie Kemper is a citizen of')
    for sent in output_sents:
        print(sent, '\n')


def load_contriever():
    print('Loading Contriever...')
    contriever = AutoModel.from_pretrained("facebook/contriever-msmarco").cuda()
    ct_tokenizer = AutoTokenizer.from_pretrained("facebook/contriever-msmarco")
    return contriever, ct_tokenizer


def test_contriever(contriever, ct_tokenizer, new_facts, edit_embs):
    fact_ids = retrieve_facts("Who is the president of the US?", edit_embs, contriever, ct_tokenizer)
    print(new_facts[fact_ids[0]])


def run_mello_huggingface(cf_dataset, llm_api, task_prompt, contriever, ct_tokenizer, new_facts, edit_embs, log_fn='./mello.log'):
    # Run mello-baseline for K=3000 (full edit space)
    if log_fn:
        fout = open(log_fn, 'a')
    else:
        fout = sys.stdout
    cor, tot = 0, 0

    # NOTE shuffle MQUAKE-CF to fix distribution shift @~Q#1000
    random.shuffle(cf_dataset)
    for d in tqdm(cf_dataset):
        tot += 1
        for q in d["questions"]:
            found_ans = False
            prompt = task_prompt + "\n\nQustion: " + q
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


def main_huggingface():
    # Dataset loading & resolving edit statements
    with open('datasets/MQuAKE-CF-3k.json', 'r') as f:
        cf_dataset = json.load(f)
    new_facts = get_all_facts_cf(cf_dataset)
    
    # Embedding-based fact retriever
    contriever, ct_tokenizer = load_contriever()    
    edit_embs = get_sent_embeddings(new_facts, contriever, ct_tokenizer)
    test_contriever(contriever, ct_tokenizer, new_facts, edit_embs)

    # LLM API
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
    mquake_stop = ["Retrieved fact:", "Question:"]
    gptj_api = gptj_interface(gptj_tokenizer, gptj_model, mquake_stop, clue_config)

    # In-context demonstration
    with open('prompts/MeLLo-prompt.txt', 'r') as f:
        mello_prompt = f.read()

    run_mello_huggingface(cf_dataset, gptj_api, mello_prompt, contriever, ct_tokenizer, edit_embs)


if __name__ == '__main__':
    main_huggingface()
