import argparse
import os
import torch
from transformers import RobertaTokenizerFast, RobertaModel, GPT2TokenizerFast, GPT2Model
from datasets import load_dataset
import evaluate
import config as CFG
from modules import Roberta_Baseline, GPT2_Baseline, MLP
from utils import eos_pooling

parser = argparse.ArgumentParser()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--dataset", type=str, default="SetFit/sst2")
parser.add_argument("--model_path", type=str, default="baseline_models/roberta/backbone_finetuned_sst2.pt")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument('--tune_mlp_only', action=argparse.BooleanOptionalAction)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--projection_dim", type=int, default=256)
parser.add_argument("--dropout", type=float, default=0.1)

class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __getitem__(self, idx):
        t = {key: torch.tensor(values[idx]) for key, values in self.texts.items()}
        return t

    def __len__(self):
        return len(self.texts['input_ids'])

def build_loaders(texts, mode):
    dataset = ClassificationDataset(texts)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True if mode == "train" else False)
    return dataloader

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parser.parse_args()

    backbone = args.model_path.split("/")[1]

    print("loading data...")
    dataset = load_dataset(args.dataset, split='test')
    print("data len: ", len(dataset))
    print("tokenizing...")

    if 'roberta' in backbone:
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    elif 'gpt2' in backbone:
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise Exception("backbone should be roberta or gpt2")
    
    encoded_dataset = dataset.map(
        lambda e: tokenizer(e[CFG.example_name[args.dataset]], padding=True, truncation=True, max_length=args.max_length), batched=True,
        batch_size=len(dataset))
    encoded_dataset = encoded_dataset.remove_columns([CFG.example_name[args.dataset]])
    if args.dataset == 'SetFit/sst2':
        encoded_dataset = encoded_dataset.remove_columns(['label_text'])
    if args.dataset == 'dbpedia_14':
        encoded_dataset = encoded_dataset.remove_columns(['title'])
    encoded_dataset = encoded_dataset[:len(encoded_dataset)]

    print("creating loader...")
    test_loader = build_loaders(encoded_dataset, mode="test")

    if 'roberta' in backbone:
        if args.tune_mlp_only:
            print("loading MLP...")
            mlp = MLP(CFG.class_num[args.dataset], args.projection_dim, args.dropout).to(device)
            mlp.load_state_dict(torch.load(args.model_path, map_location=device))
            mlp.eval()
            preLM = RobertaModel.from_pretrained('roberta-base').to(device)
            preLM.eval()
        else:
            print("loading roberta...")
            LM = Roberta_Baseline(CFG.class_num[args.dataset], args.projection_dim, args.dropout).to(device)
            LM.load_state_dict(torch.load(args.model_path, map_location=device))
            LM.eval()
    elif 'gpt2' in backbone:
        if args.tune_mlp_only:
            print("loading MLP...")
            mlp = MLP(CFG.class_num[args.dataset], args.projection_dim, args.dropout).to(device)
            mlp.load_state_dict(torch.load(args.model_path, map_location=device))
            mlp.eval()
            preLM = GPT2Model.from_pretrained('gpt2').to(device)
            preLM.eval()
        else:
            print("loading gpt2...")
            LM = GPT2_Baseline(CFG.class_num[args.dataset], args.projection_dim, args.dropout).to(device)
            LM.load_state_dict(torch.load(args.model_path, map_location=device))
            LM.eval()
    else:
        raise Exception("backbone should be roberta or gpt2")

    print("start testing...")
    # add nec list
    NEC_list = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100)
    results = {}

    if args.tune_mlp_only:
        orig_fc = mlp.fc2.weight.data.clone() #fc2 is the final layer in mlp structure
    else:
        orig_fc = LM.fc.weight.data.clone() #fc is the final layer in Roberta_Baseline and GPT2_Baseline structure
    
    
    for NEC in NEC_list:
        # 2a. compute threshold so only the top_n magnitudes survive
        flat = orig_fc.abs().view(-1)
        threshold = torch.topk(flat, NEC, largest=True).values.min()

        # 2b. prune: zero out anything below threshold
        mask = orig_fc.abs() >= threshold
        if args.tune_mlp_only:
            mlp.fc2.weight.data = orig_fc * mask
        else:
            LM.fc.weight.data = orig_fc * mask

        # 2c. evaluate
        metric = evaluate.load("accuracy")
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                if args.tune_mlp_only:
                    LM_features = preLM(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
                    if 'roberta' in backbone:
                        LM_features = LM_features[:, 0, :]
                    elif 'gpt2' in backbone:
                        LM_features = eos_pooling(LM_features, batch["attention_mask"])
                    else:
                        raise Exception("backbone should be roberta or gpt2")
                    p = mlp(LM_features)
                else:
                    p = LM(batch)
            pred = torch.argmax(p, dim=-1)
            metric.add_batch(predictions=pred, references=batch["label"])
        acc = metric.compute()["accuracy"]
        results[NEC] = acc
        print(f"NEC {NEC:2d} weights → accuracy: {acc:.4f}")

        # 2d. restore original weights before next round
        if args.tune_mlp_only:
            mlp.fc2.weight.data = orig_fc.clone()
        else:
            LM.fc.weight.data = orig_fc.clone()

    # 3. (optionally) summarise all at once
    print("Summary of pruned‐fc2 accuracies:", results)