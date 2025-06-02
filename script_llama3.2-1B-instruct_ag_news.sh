# Reproduces all scripts from https://github.com/Trustworthy-ML-Lab/CB-LLMs/tree/main

python generation/train_CBLLM.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="ag_news"
python generation/test_concepts.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="ag_news"
python generation/train_classifier.py --dataset="ag_news"
python generation/test_steerability.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="ag_news"
python generation/test_perplexity.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="ag_news"
python generation/test_weight.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="ag_news"
python generation/test_generation.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="ag_news"