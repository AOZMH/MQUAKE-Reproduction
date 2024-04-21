import sys
import os
#os.chdir('../MQuAKE/')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaTokenizerFast

from transformers import AutoTokenizer, AutoModel

# from util import nethook

import os
import sys
import json
import random

from tqdm import tqdm

def load_model(args):
    print(f"Loading model {args.model_name}")
    if args.model_name == 'llama-7B':
        from transformers import LlamaForCausalLM, LlamaTokenizer
        tok = LlamaTokenizer.from_pretrained('/u/zzhong/nlpzzhong/repos/LLaMA/llama-7B')
        tok.pad_token = tok.eos_token
        model = LlamaForCausalLM.from_pretrained('/u/zzhong/nlpzzhong/repos/LLaMA/llama-7B').cuda()
        model.config._name_or_path = 'llama-7B'
    elif args.model_name == 'vicuna-7B':
        from transformers import LlamaForCausalLM, LlamaTokenizer
        tok = LlamaTokenizer.from_pretrained('/u/zzhong/nlpzzhong/repos/LLaMA/vicuna-7B')
        tok.pad_token = tok.eos_token
        model = LlamaForCausalLM.from_pretrained('/u/zzhong/nlpzzhong/repos/LLaMA/vicuna-7B').cuda()
        model.config._name_or_path = 'vicuna-7B'
    else:
        tok = AutoTokenizer.from_pretrained(args.model_name)
        tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.model_name, pad_token_id=tok.eos_token_id).cuda()
    
    return model, tok

def greedy_generate(model, tok, prompts):
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model.eval()
    
    input_dict = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    max_length = input_dict["input_ids"].shape[-1] + 100
    with torch.no_grad():
        outputs = model.generate(**input_dict, max_length=max_length)
    generations = [tok.decode(output, skip_special_tokens=True, pad_token_id=tok.eos_token_id) for output in outputs]
    
    tok.padding_side = "right"
    
    return generations

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings
def get_sent_embeddings(sents, contriever, tok, BSZ=32):    
    all_embs = []
    for i in tqdm(range(0, len(sents), BSZ)):
        sent_batch = sents[i:i+BSZ]
        inputs = tok(sent_batch, padding=True, truncation=True, return_tensors='pt').to("cuda")
        with torch.no_grad():
            outputs = contriever(**inputs)
            embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
        all_embs.append(embeddings.cpu())
    all_embs = torch.vstack(all_embs)
    return all_embs
def retrieve_facts(query, fact_embs, contriever, tok):
    inputs = tok([query], padding=True, truncation=True, return_tensors='pt').to("cuda")
    with torch.no_grad():
        outputs = contriever(**inputs)
        query_emb = mean_pooling(outputs[0], inputs['attention_mask']).cpu()
    sim = (query_emb @ fact_embs.T)[0]
    knn = sim.topk(1, largest=True)
    return knn.indices

def get_line(gen, prefix):
    # return the first line that starts with prefix
    lines = gen.split('\n')
    for l in lines:
        if l.startswith(prefix):
            return l
    raise Exception(f"No line starts with {prefix}.\n\n========\n{gen}")

def edit_hash(edit):
    return edit["question"]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ds_path",
        default="mqke/gptj_filtered/gptj_filtered_mqke.json.subset",
    )
    parser.add_argument(
        "--res_path",
        default="results/Mello/gptj_filtered_cf_subset_vicuna",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--dataset_start_point",
        type=int,
        default=0,
        help="where of the dataset to start evaluation. (0 means evaluating from the first sample)"
    )
    parser.add_argument(
        "--model_name",
        choices=["gpt2-medium", "gpt2-large", "gpt2-xl", "EleutherAI/gpt-j-6B", "llama-7B", "vicuna-7B"],
        type=str,
        default='vicuna-7B',
    )
    parser.add_argument(
        "--multi_samples",
        default=1,
        # required=True,
        type=int,
    )
    parser.add_argument(
        "--group_id",
        default=None,
        type=int,
    )
    args = parser.parse_args()

    # with open('prompts/QA-prompts.txt', 'r') as f:
    #     qa_prompts = f.read()
    # with open('prompts/QA-prompts_cloze.txt', 'r') as f:
    #     qa_cloze_prompts = f.read()
    # with open('prompts/QA-prompts_cot.txt', 'r') as f:
    #     cot_prompts = f.read()

    # with open('prompts/mem_prompt.txt', 'r') as f:
    with open('prompts/MeLLo-prompt.txt', 'r') as f:
        task_prompt = f.read()

    with open(args.ds_path, 'r') as f:
        dataset = json.load(f)

    assert args.multi_samples == 1

    # BSZ = 1
    if args.multi_samples is not None:
        if args.multi_samples > len(dataset):
            args.multi_samples = len(dataset)
        ids = list(range(len(dataset)))
        random.seed(0)
        random.shuffle(ids)
        group_id = [0 for i in range(len(dataset))]
        gid = 0
        for i in range(0, len(dataset), args.multi_samples):
            for j in range(i, i + args.multi_samples):
                group_id[ids[j]] = gid
            gid += 1
        num_group = gid

        edit_group = {i: {} for i in range(num_group)}
        for i in range(len(dataset)):
            d = dataset[i]
            for edit in d["requested_rewrite"]:
                eh = edit_hash(edit)
                edit_group[group_id[i]][eh] = edit
        
        total_edits = sum([len(edit_group[gid]) for gid in range(num_group)])
        print(f"Avg edits per group: {total_edits / num_group} ({total_edits} / {num_group})")

        all_data = dataset

        # dataset = [d for i, d in enumerate(dataset) if group_id[i] == args.group_id]
        # st = args.dataset_start_point
        # ed = len(dataset) if args.dataset_size_limit is None else min(st + args.dataset_size_limit, len(dataset))
        args.res_path = f"{args.res_path}_{args.multi_samples}"
        # output_file = f"{args.res_path}/g{args.group_id}-{st}_{ed}.json"
    else:
        st = args.dataset_start_point
        ed = len(dataset) if args.dataset_size_limit is None else st + args.dataset_size_limit
        output_file = f"{args.res_path}/{st}_{ed}.json"

    # Build retrieval index
    contriever = AutoModel.from_pretrained("facebook/contriever-msmarco").cuda()
    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever-msmarco")

    model, tok = None, None
    cor_global, tot_global = 0, 0

    for gid in tqdm(range(args.dataset_start_point, min(args.dataset_start_point + args.dataset_size_limit, len(all_data)))):
        dataset = [d for i, d in enumerate(all_data) if group_id[i] == gid]
        output_file = f"{args.res_path}/g{gid}-0_1.json"
        
        if os.path.exists(output_file):
            continue
                     
        print(f'The results will be saved to {output_file}')

        new_facts = set()
        for d in dataset:
            for r in d["requested_rewrite"]:
                new_facts.add(f'{r["prompt"].format(r["subject"])} {r["target_new"]["str"]}')
        print('# new facts:', len(new_facts))
        new_facts = list(new_facts)

        embs = get_sent_embeddings(new_facts, contriever, tokenizer)

        def retr_f(query):
            return retrieve_facts(query, embs, contriever, tokenizer)

        # load model
        if model is None:
            model, tok = load_model(args)

        def run_Mello(question, retr_f):
            prompt = task_prompt + '\n\nQuestions: ' + question
            for i in range(4):
                output = greedy_generate(model, tok, [prompt])[0]
                gen = output[len(prompt):].strip()
                subq = gen.find('Subquestion: ')
                retr = gen.find('Retrieved fact: ')
                fans = gen.find('Final answer: ')
                if fans != -1 and (subq == -1 or subq > fans):
                    fans_line = get_line(gen, 'Final answer: ')
                    ans = fans_line[len("Final answer: "):].strip()
                    return ans, output
                elif (subq != -1) and (retr != -1) and (subq < retr):
                    subq_line = get_line(gen, 'Subquestion: ')
                    subq = subq_line[len('Subquestion: '):].strip()
                    fact_ids = retr_f(subq)
                    fact_sent = new_facts[fact_ids[0]]
                    prompt = output[:(output[len(prompt):].find('Retrieved fact: ')+len(prompt))] + 'Retrieved fact: ' + fact_sent + '.'
                else:
                    break
                    raise Exception(f"Format error!\n{output}")
            # ans not found
            return '!!!!!', output

        # Run Memcode
        st = 0
        ed = 1
        cor = 0
        for i in range(st, ed):
            d = dataset[i]
            dataset[i]["gens"] = []
            dataset[i]["preds"] = []
            dataset[i]["succ"] = False
            for q in d["questions"]:
                ans, output = run_Mello(q, retr_f)
                dataset[i]["gens"].append(output)
                dataset[i]["preds"].append(ans)
                all_ans = set([d["answer"]] + d["answer_alias"])
                print('pred:', ans, 'gold:', all_ans)
                if ans in all_ans: # need to consider aliases  (fixed when gathering results at the end)
                    dataset[i]["succ"] = True
                    cor += 1
                    break
        
        cor = 0
        tot = 0
        for i in range(st, ed):
            if dataset[i].get("succ", False):
                cor += 1
            tot += 1
        cor_global += cor
        tot_global += tot
        print(f"{cor} / {tot}, {cor_global} / {tot_global} = {cor_global / tot_global}")

        print(args.res_path)
        if not os.path.exists(args.res_path):
            os.mkdir(args.res_path)

        print(output_file)
        with open(output_file, 'w') as f:
            f.write(json.dumps(dataset[st:ed], indent=2))
