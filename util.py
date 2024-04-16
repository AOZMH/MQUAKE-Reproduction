# MQUAKE dataset related utils, e.g. statement collector/allocator
import random

def get_all_facts_cf(cf_dataset):
    new_facts = set()
    for d in cf_dataset:
        for r in d["requested_rewrite"]:
            new_facts.add(f'{r["prompt"].format(r["subject"])} {r["target_new"]["str"]}')
    new_facts = list(new_facts)
    return new_facts


def split_edit_batch(cf_dataset, new_facts, BSZ=1):
    # Randomly split CF-dataset into batches to group edits
    edit2ix = {ed : ix for ix, ed in enumerate(new_facts)}
    d_indices = list(range(len(cf_dataset)))
    random.shuffle(d_indices)
    for st_ix in range(0, len(cf_dataset), BSZ):
        cur_batch = d_indices[st_ix : st_ix + BSZ]
        cur_edit_batch = set()
        for cix in cur_batch:
            for r in cf_dataset[cix]["requested_rewrite"]:
                cur_edit_batch.add(f'{r["prompt"].format(r["subject"])} {r["target_new"]["str"]}')
        cur_edit_batch = list(cur_edit_batch)
        cur_edit_ixs = [edit2ix[ce] for ce in cur_edit_batch]
        for cix in cur_batch:
            cf_dataset[cix]['edit_cand_ixs'] = cur_edit_ixs


