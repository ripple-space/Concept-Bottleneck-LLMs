# Reproduces all scripts from https://github.com/Trustworthy-ML-Lab/CB-LLMs/tree/main

python Concept-Bottleneck-LLM/generation/train_CBLLM.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="SetFit/sst2"
python Concept-Bottleneck-LLM/generation/test_concepts.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="SetFit/sst2"
python Concept-Bottleneck-LLM/generation/train_classifier.py --dataset="SetFit/sst2"
python Concept-Bottleneck-LLM/generation/test_steerability.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="SetFit/sst2" --intervention_value=100
python Concept-Bottleneck-LLM/generation/test_perplexity.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="SetFit/sst2"
python Concept-Bottleneck-LLM/generation/test_weight.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="SetFit/sst2"
python Concept-Bottleneck-LLM/generation/test_generation.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="SetFit/sst2" --intervention_value=100