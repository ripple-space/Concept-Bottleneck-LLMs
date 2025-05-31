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
    classifier_path = get_checkpoint_path(args.dataset.replace('/', '_') + "_classifier", args.epoch_index)
    classifier = Roberta_classifier(len(concept_set)).to(device)
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))

    if args.dataset == "dbpedia_14":
        intervention_value = 150
    else:
        intervention_value = 100
    pred = []
    text = []
    acc = evaluate.load("accuracy")
    for i in range(100 // len(concept_set)):
        print("example", str(i), end="\r")
        with torch.no_grad():
            input_ids = torch.tensor([tokenizer.encode("")]).to(device)
            for j in range(len(concept_set)):
                v = [0] * len(concept_set)
                j = j if args.intervention_class is None else args.intervention_class
                v[j] = args.intervention_value
                v = None if args.intervention_value is None else v
                text_ids, _ = cbl.generate(input_ids, preLM, intervene=v)
                decoded_text_ids = tokenizer.decode(text_ids[0][~torch.isin(text_ids[0], torch.tensor([128000, 128001]).to(device))])
                text.append(decoded_text_ids)
                roberta_text_ids = torch.tensor([roberta_tokenizer.encode(decoded_text_ids)]).to(device)
                roberta_input = {"input_ids": roberta_text_ids, "attention_mask": torch.tensor([[1]*roberta_text_ids.shape[1]]).to(device)}
                logits = classifier(roberta_input)
                pred.append(logits)
    pred = torch.cat(pred, dim=0).detach().cpu()
    pred = np.argmax(pred.numpy(), axis=-1)
    acc.add_batch(predictions=pred, references=list(range(len(concept_set)))*(100 // len(concept_set)))

    print(acc.compute())
