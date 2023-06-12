# This script demonstrates distributed LoRA (Low Rank Adaptation) fine-tuning on ensembles of LLMs.
# LoRA workers cooperatively operate on disjoint model regions, with independent input data streams.

# In the LoRA procedure, only a small fraction of the model parameter space varies, while the rest is held fixed.
# Multiple model parameter regions can be adapted independently and work can be assigned based on activation patterns.

# Workers can nucleate and prune model components, based on input data activation patterns or error distribution.

################################

import time, utils, objects


def make_workers(N=1):
    return [objects.Worker() for _ in range(N)]

def do_LORA(workers, finetuning_corpus, model):
    # function to run the update 
    return model

def _eval(model):
    # print out if the model is any good
    pass

if __name__ == "__main__":
    finetuning_corpus = utils.get_corpus(size=100)
    workers = make_workers(N=4)
    model = FullModel(base_models=[])
    while True:
        time.sleep(1)
        model = do_LORA(workers, finetuning_corpus, model)
        _eval(model)


