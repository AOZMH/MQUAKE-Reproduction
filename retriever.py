# Retrieve edit-statement w.r.t. sub-question
from tqdm import tqdm
import torch


######## FAIR Contriever #########

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


def get_sent_embeddings(sents, contriever, tok, BSZ=32):    
    # Pre-calculate statement embedding via mean pooling over all sentence tokens
    all_embs = []
    contriever_device = next(iter(contriever.parameters())).device
    for i in tqdm(range(0, len(sents), BSZ)):
        sent_batch = sents[i:i+BSZ]
        inputs = tok(sent_batch, padding=True, truncation=True, return_tensors='pt').to(contriever_device)
        with torch.no_grad():
            outputs = contriever(**inputs)
            embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
        all_embs.append(embeddings.cpu())
    all_embs = torch.vstack(all_embs)
    return all_embs


def retrieve_facts(query, fact_embs, contriever, tok, k=1):
    # Match subquestion embedding with statements to get top-k
    contriever_device = next(iter(contriever.parameters())).device
    inputs = tok([query], padding=True, truncation=True, return_tensors='pt').to(contriever_device)
    with torch.no_grad():
        outputs = contriever(**inputs)
        query_emb = mean_pooling(outputs[0], inputs['attention_mask']).cpu()
    # Indexing [0] since inference BSZ==1
    sim = (query_emb @ fact_embs.T)[0]
    knn = sim.topk(k, largest=True)
    return knn.indices


def batch_retrieve_facts(queries, fact_embs, contriever, tok, k=1):
    # Differed from above only in queris = [query_1, query_2,...]
    if queries == []:
        return []
    contriever_device = next(iter(contriever.parameters())).device
    inputs = tok(queries, padding=True, truncation=True, return_tensors='pt').to(contriever_device)
    with torch.no_grad():
        # outputs[0] ~ <BSZ, seq_len, hid_dim>
        outputs = contriever(**inputs)
        # query_emb ~ <BSZ, hid_dim>
        query_emb = mean_pooling(outputs[0], inputs['attention_mask']).to(fact_embs.device)
    # sim ~ <BSZ, num_facts>
    sim = (query_emb @ fact_embs.T)
    knn = sim.topk(k, largest=True)
    # knn.indices ~ <BSZ, k> of type torch.Long
    return knn.indices

