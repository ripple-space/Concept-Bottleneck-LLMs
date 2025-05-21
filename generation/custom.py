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
args = parser.parse_args()

device = torch.device(args.device)

def get_checkpoint_path(path, epoch_index=-1):
    import glob
    paths = glob.glob(f"{path}*")
    paths = sorted(paths, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    print(f"Loading from checkpoint: {paths[epoch_index]}")
    return paths[epoch_index]