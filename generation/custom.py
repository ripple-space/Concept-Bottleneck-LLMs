import os
cache_path = os.path.expanduser('~/teams/dsmlp/huggingface/hub')
os.environ['TRANSFORMERS_CACHE'] = cache_path
os.environ['HF_HOME'] = cache_path
os.environ['HF_HUB_CACHE'] = cache_path
os.environ['HF_DATASETS_CACHE'] = cache_path

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B")
parser.add_argument("--dataset", type=str, default="SetFit/sst2")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--max_length", type=int, default=350)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--device", type=str, default=device)
parser.add_argument("--epoch_index", type=int, default=-1)
parser.add_argument("--intervention_class", type=int, default=None)
parser.add_argument("--intervention_value", type=float, default=None)
parser.add_argument("--prompt", type=str, default="")
parser.add_argument("--n_train_samples", type=int, default=100000)
parser.add_argument("--n_eval_samples", type=int, default=100)
args = parser.parse_args()

device = torch.device(args.device)

def get_checkpoint_path(path, epoch_index=-1):
    import glob
    paths = glob.glob(f"{path}*")
    paths = sorted(paths, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    print(f"Loading from checkpoint: {paths[epoch_index]}")
    return paths[epoch_index]

# Recursively set chmod 777 (read, write, and execute for all users) for all files and directories
def safe_chmod(path, mode):
    try:
        os.chmod(path, mode)
    except PermissionError:
        print(f"Permission denied: {path}")        
def chmod_recursive(path, mode=0o777):
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files + dirs:
            safe_chmod(os.path.join(root, name), mode)
    safe_chmod(path, mode)
chmod_recursive(cache_path)

import torch
import numpy as np
import random
import os

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False