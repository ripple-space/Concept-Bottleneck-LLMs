from custom import args, device, get_checkpoint_path
import os
import torch
import numpy as np
import config as CFG
from transformers import LlamaConfig, LlamaModel, AutoTokenizer, RobertaTokenizerFast
from datasets import load_dataset, concatenate_datasets
from modules import CBL, Roberta_classifier
from utils import eos_pooling
import evaluate
import time
import gc


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    config = LlamaConfig.from_pretrained(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    roberta_tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')


    concept_set = CFG.concepts_from_labels[args.dataset]
    print("concept len: ", len(concept_set))

    print("preparing backbone")
    peft_path = get_checkpoint_path("from_pretained_llama3_lora_cbm/" + args.dataset.replace('/', '_') + "/llama3", args.epoch_index)
    cbl_path = get_checkpoint_path("from_pretained_llama3_lora_cbm/" + args.dataset.replace('/', '_') + "/cbl", args.epoch_index)
    preLM = LlamaModel.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(device)
    preLM.load_adapter(peft_path)
    preLM.eval()
    cbl = CBL(config, len(concept_set), tokenizer).to(device)
    cbl.load_state_dict(torch.load(cbl_path, map_location=device))
    cbl.eval()

    pred = []
    perplexity = evaluate.load("perplexity", module_type="metric")
    input_ids = torch.tensor([tokenizer.encode("")]).to(device)
    for i in range(100):
        print("example", str(i), end="\r")
        with torch.no_grad():
            text_ids, _ = cbl.generate(input_ids, preLM)
            pred.append(tokenizer.decode(text_ids[0]))
    perplexity.add_batch(predictions=pred)

    del preLM
    del cbl
    gc.collect()
    torch.cuda.empty_cache()

    print(perplexity.compute(model_id=args.model_id, max_length=100)['mean_perplexity'])
