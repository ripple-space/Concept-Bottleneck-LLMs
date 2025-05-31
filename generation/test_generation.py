from custom import args, device, get_checkpoint_path
import os
import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
import config as CFG
from transformers import LlamaConfig, LlamaModel, AutoTokenizer
from modules import CBL
from utils import eos_pooling
import evaluate
import time


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    config = LlamaConfig.from_pretrained(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token


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

    input_ids = torch.tensor([tokenizer.encode("")]).to(device)

    print("generation...")
    for j in range(len(concept_set)):
        v = [0] * len(concept_set)
        # For SST2 and Yelp, intervene v[0] to a large value will give you negative movie or Yelp review
        # For AGnews, there are four classes World news, Sport news, Business news and Tech news, intervene v[0], ..., v[3] accordingly will give you the sentences related to the concepts
        j = j if args.intervention_class is None else args.intervention_class
        v[j] = args.intervention_value
        v = None if args.intervention_value is None else v
        
        print()
        print(concept_set[j])
        print(f"intervention_value={v}")
        
        with torch.no_grad():
            text_ids, concept_activation = cbl.generate(input_ids, preLM, intervene=v)
        print(tokenizer.decode(text_ids[0]))
        # print(concept_activation)
