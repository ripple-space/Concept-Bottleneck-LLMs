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
args = parser.parse_args()

device = torch.device(args.device)