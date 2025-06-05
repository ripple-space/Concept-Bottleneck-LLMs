import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import RobertaTokenizerFast, RobertaModel, GPT2TokenizerFast, GPT2Model
from datasets import load_dataset
import config as CFG
from modules import CBL, RobertaCBL, GPT2CBL
from glm_local import IndexedTensorDataset, glm_saga
from torch.utils.data import DataLoader, TensorDataset
from utils import normalize, eos_pooling, weight_truncation, weight_truncation_by_concept # add weight_truncation

parser = argparse.ArgumentParser()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--cbl_path", type=str, default="mpnet_acs/SetFit_sst2/roberta_cbm/cbl.pt")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--saga_epoch", type=int, default=500)
parser.add_argument("--saga_batch_size", type=int, default=256)

parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--num_workers", type=int, default=0)
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
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                             shuffle=True if mode == "train" else False)
    return dataloader

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parser.parse_args()

    acs = args.cbl_path.split("/")[0]
    dataset = args.cbl_path.split("/")[1] if 'sst2' not in args.cbl_path.split("/")[1] else args.cbl_path.split("/")[1].replace('_', '/')
    backbone = args.cbl_path.split("/")[2]
    cbl_name = args.cbl_path.split("/")[-1]
    
    print("loading data...")
    train_dataset = load_dataset(dataset, split='train')
    if dataset == 'SetFit/sst2':
        val_dataset = load_dataset(dataset, split='validation')
    test_dataset = load_dataset(dataset, split='test')
    print("training data len: ", len(train_dataset))
    if dataset == 'SetFit/sst2':
        print("val data len: ", len(val_dataset))
    print("test data len: ", len(test_dataset))
    print("tokenizing...")

    if 'roberta' in backbone:
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    elif 'gpt2' in backbone:
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise Exception("backbone should be roberta or gpt2")

    # adjust batch size for dbpedia_14
    if dataset == 'dbpedia_14':
        encoded_train_dataset = train_dataset.map(lambda e: tokenizer(e[CFG.example_name[dataset]], padding="max_length", truncation=True, max_length=args.max_length),
                                                  batched=True, batch_size=1_000,            # or 2_000, 5_000… whatever your RAM can handle
                                                  num_proc=os.cpu_count())     # parallelize across your cores
        encoded_train_dataset = encoded_train_dataset.remove_columns([CFG.example_name[dataset]])
    else:
        encoded_train_dataset = train_dataset.map(lambda e: tokenizer(e[CFG.example_name[dataset]], padding=True, truncation=True, max_length=args.max_length), batched=True, batch_size=len(train_dataset))
        encoded_train_dataset = encoded_train_dataset.remove_columns([CFG.example_name[dataset]])
    if dataset == 'SetFit/sst2':
        encoded_train_dataset = encoded_train_dataset.remove_columns(['label_text'])
    if dataset == 'dbpedia_14':
        encoded_train_dataset = encoded_train_dataset.remove_columns(['title'])
    encoded_train_dataset = encoded_train_dataset[:len(encoded_train_dataset)]

    if dataset == 'SetFit/sst2':
        encoded_val_dataset = val_dataset.map(lambda e: tokenizer(e[CFG.example_name[dataset]], padding=True, truncation=True, max_length=args.max_length), batched=True, batch_size=len(val_dataset))
        encoded_val_dataset = encoded_val_dataset.remove_columns([CFG.example_name[dataset]])
        if dataset == 'SetFit/sst2':
            encoded_val_dataset = encoded_val_dataset.remove_columns(['label_text'])
        if dataset == 'dbpedia_14':
            encoded_val_dataset = encoded_val_dataset.remove_columns(['title'])
        encoded_val_dataset = encoded_val_dataset[:len(encoded_val_dataset)]

    #adjust batch size for dbpedia_14
    if dataset == 'dbpedia_14':
        encoded_test_dataset = test_dataset.map(lambda e: tokenizer(e[CFG.example_name[dataset]], padding="max_length", truncation=True, max_length=args.max_length), 
                                                batched=True, batch_size=1_000,            # or 2_000, 5_000… whatever your RAM can handle
                                                  num_proc=os.cpu_count())     # parallelize across your cores
        encoded_test_dataset = encoded_test_dataset.remove_columns([CFG.example_name[dataset]])
    else:
        encoded_test_dataset = test_dataset.map(lambda e: tokenizer(e[CFG.example_name[dataset]], padding=True, truncation=True, max_length=args.max_length), batched=True, batch_size=len(test_dataset))
        encoded_test_dataset = encoded_test_dataset.remove_columns([CFG.example_name[dataset]])
    if dataset == 'SetFit/sst2':
        encoded_test_dataset = encoded_test_dataset.remove_columns(['label_text'])
    if dataset == 'dbpedia_14':
        encoded_test_dataset = encoded_test_dataset.remove_columns(['title'])
    encoded_test_dataset = encoded_test_dataset[:len(encoded_test_dataset)]

    print("creating loader...")
    train_loader = build_loaders(encoded_train_dataset, mode="valid")
    if dataset == 'SetFit/sst2':
        val_loader = build_loaders(encoded_val_dataset, mode="valid")
    test_loader = build_loaders(encoded_test_dataset, mode="test")

    concept_set = CFG.concept_set[dataset]

    if 'roberta' in backbone:
        if 'no_backbone' in cbl_name:
            print("preparing CBL only...")
            cbl = CBL(len(concept_set), args.dropout).to(device)
            cbl.load_state_dict(torch.load(args.cbl_path, map_location=device))
            cbl.eval()
            preLM = RobertaModel.from_pretrained('roberta-base').to(device)
            preLM.eval()
        else:
            print("preparing backbone(roberta)+CBL...")
            backbone_cbl = RobertaCBL(len(concept_set), args.dropout).to(device)
            backbone_cbl.load_state_dict(torch.load(args.cbl_path, map_location=device))
            backbone_cbl.eval()
    elif 'gpt2' in backbone:
        if 'no_backbone' in cbl_name:
            print("preparing CBL only...")
            cbl = CBL(len(concept_set), args.dropout).to(device)
            cbl.load_state_dict(torch.load(args.cbl_path, map_location=device))
            cbl.eval()
            preLM = GPT2Model.from_pretrained('gpt2').to(device)
            preLM.eval()
        else:
            print("preparing backbone(gpt2)+CBL...")
            backbone_cbl = GPT2CBL(len(concept_set), args.dropout).to(device)
            backbone_cbl.load_state_dict(torch.load(args.cbl_path, map_location=device))
            backbone_cbl.eval()
    else:
        raise Exception("backbone should be roberta or gpt2")


    print("get concept features...")
    FL_train_features = []
    if dataset == 'SetFit/sst2':
        FL_val_features = []
    FL_test_features = []
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            if 'no_backbone' in cbl_name:
                train_features = preLM(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
                if 'roberta' in backbone:
                #change args.backbone to if 'roberta' in backbone, since we didn't set parser.add_argument('--backbone', type=str, default='roberta', help='Backbone model type')
                    train_features = train_features[:, 0, :]
                elif 'gpt2' in backbone: #change args.backbone to if 'gpt2' in backbone:
                    train_features = eos_pooling(train_features, batch["attention_mask"])
                else:
                    raise Exception("backbone should be roberta or gpt2")
                train_features = cbl(train_features)
            else:
                train_features = backbone_cbl(batch)
            FL_train_features.append(train_features)

    if dataset == 'SetFit/sst2':
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                if 'no_backbone' in cbl_name:
                    val_features = preLM(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
                    if 'roberta' in backbone: #change args.backbone to if 'roberta' in backbone:
                        val_features = val_features[:, 0, :]
                    elif 'gpt2' in backbone: #change args.backbone to if 'gpt2' in backbone:
                        val_features = eos_pooling(val_features, batch["attention_mask"])
                    else:
                        raise Exception("backbone should be roberta or gpt2")
                    val_features = cbl(val_features)
                else:
                    val_features = backbone_cbl(batch)
                FL_val_features.append(val_features)

    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            if 'no_backbone' in cbl_name:
                test_features = preLM(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
                if 'roberta' in backbone: #change args.backbone to if 'roberta' in backbone:
                    test_features = test_features[:, 0, :]
                elif 'gpt2' in backbone: #change args.backbone to if 'gpt2' in backbone:
                    test_features = eos_pooling(test_features, batch["attention_mask"])
                else:
                    raise Exception("backbone should be roberta or gpt2")
                test_features = cbl(test_features)
            else:
                test_features = backbone_cbl(batch)
            FL_test_features.append(test_features)

    train_c = torch.cat(FL_train_features, dim=0).detach().cpu()
    if dataset == 'SetFit/sst2':
        val_c = torch.cat(FL_val_features, dim=0).detach().cpu()
    test_c = torch.cat(FL_test_features, dim=0).detach().cpu()

    train_c, train_mean, train_std = normalize(train_c, d=0)
    train_c = F.relu(train_c)

    prefix = "./" + acs + "/" + dataset.replace('/', '_') + "/" + backbone + "/"
    model_name = cbl_name[3:]
    torch.save(train_mean, prefix + 'train_mean' + model_name)
    torch.save(train_std, prefix + 'train_std' + model_name)

    if dataset == 'SetFit/sst2':
        val_c, _, _ = normalize(val_c, d=0, mean=train_mean, std=train_std)
        val_c = F.relu(val_c)

    test_c, _, _ = normalize(test_c, d=0, mean=train_mean, std=train_std)
    test_c = F.relu(test_c)


    train_y = torch.LongTensor(encoded_train_dataset["label"])
    indexed_train_ds = IndexedTensorDataset(train_c, train_y)

    if dataset == 'SetFit/sst2':
        val_y = torch.LongTensor(encoded_val_dataset["label"])
        val_ds = TensorDataset(val_c, val_y)

    test_y = torch.LongTensor(encoded_test_dataset["label"])
    test_ds = TensorDataset(test_c, test_y)

    indexed_train_loader = DataLoader(indexed_train_ds, batch_size=args.saga_batch_size, shuffle=True)
    if dataset == 'SetFit/sst2':
        val_loader = DataLoader(val_ds, batch_size=args.saga_batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.saga_batch_size, shuffle=False)

    print("dim of concept features: ", train_c.shape[1])
    linear = torch.nn.Linear(train_c.shape[1], CFG.class_num[dataset])
    linear.weight.data.zero_()
    linear.bias.data.zero_()
    STEP_SIZE = 0.05
    ALPHA = 0.99
    metadata = {}
    metadata['max_reg'] = {}
    metadata['max_reg']['nongrouped'] = 0.0007
    MAX_GLM_STEP = 100
    GLM_STEP_SIZE = 2 ** 0.1
    measure_level=(5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100)
    max_sparsity = measure_level[-1] / len(concept_set)

    print("training final layer...")

    if dataset == 'SetFit/sst2':
        output_proj = glm_saga(linear, indexed_train_loader, STEP_SIZE, args.saga_epoch, ALPHA, 
                               epsilon=1 / (GLM_STEP_SIZE ** MAX_GLM_STEP), k=MAX_GLM_STEP,
                               val_loader=val_loader, test_loader=test_loader, do_zero=False,
                               n_classes=CFG.class_num[dataset], metadata=metadata, n_ex=train_c.shape[0],
                              max_sparsity=max_sparsity)
    else:
        output_proj = glm_saga(linear, indexed_train_loader, STEP_SIZE, args.saga_epoch, ALPHA, 
                               epsilon=1 / (GLM_STEP_SIZE ** MAX_GLM_STEP), k=MAX_GLM_STEP,
                               test_loader=test_loader, do_zero=False,
                               n_classes=CFG.class_num[dataset], metadata=metadata, n_ex=train_c.shape[0],
                              max_sparsity=max_sparsity)
    path = output_proj['path']
    sparsity_list = [(params['weight'].abs() > 1e-5).float().mean().item() for params in path]

    final_layer = torch.nn.Linear(train_c.shape[1], CFG.class_num[dataset]).to(device)
    accs = []
    weights = []

    for eff_concept_num in measure_level:
        target_sparsity = eff_concept_num / train_c.shape[1]
        for i, sparsity in enumerate(sparsity_list):
            if sparsity >= target_sparsity:
                break
        params = path[i]
        W_g, b_g, lam = params["weight"], params["bias"], params["lam"]
        print(eff_concept_num, lam, sparsity)
        print(f"Num of effective concept: {eff_concept_num}. Choose lambda={lam:.6f} with sparsity {sparsity:.4f}")

        W_g_trunc = weight_truncation(W_g, target_sparsity)
        weight_contribs = torch.sum(torch.abs(W_g_trunc), dim=0)
        print("Num concepts with outgoing weights:{}/{}".format(torch.sum(weight_contribs > 1e-5), len(weight_contribs)))
        print(target_sparsity, (W_g_trunc.abs() > 0).sum())

        final_layer.load_state_dict({"weight": W_g_trunc, "bias": b_g})
        final_layer.eval()

        correct = []
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = final_layer(x).argmax(dim=-1)
            correct.append(pred == y)
        correct = torch.cat(correct)
        accs.append(correct.float().mean().item())
        weights.append((W_g_trunc, b_g))
        print(f"Test Acc: {correct.float().mean():.4f}")

    print(f"Average acc: {sum(accs) / len(accs):.4f}")

    prefix = "./" + acs + "/" + dataset.replace('/', '_') + "/" + backbone + "_nec/"
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    for eff_concept_num, (W_g, b_g) in zip(measure_level, weights):
        torch.save(W_g, prefix + f'W_g_sparse_top{eff_concept_num}' + model_name)
        torch.save(b_g, prefix + f'b_g_sparse_top{eff_concept_num}' + model_name)

