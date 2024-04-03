# MQUAKE dataset related utils, e.g. statement collector/allocator

def get_all_facts_cf(cf_dataset):
    new_facts = set()
    for d in cf_dataset:
        for r in d["requested_rewrite"]:
            new_facts.add(f'{r["prompt"].format(r["subject"])} {r["target_new"]["str"]}')
    new_facts = list(new_facts)
    return new_facts



