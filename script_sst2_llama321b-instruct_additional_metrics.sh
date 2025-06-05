# Records additional metrics after running script_llama321b-instruct_sst2.sh
# NOTE: This script does not test concept detection!

# Perplexity: when no intervention, when zero intervention, when non-zero intervention
python Concept-Bottleneck-LLM/generation/test_perplexity.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="SetFit/sst2"

python Concept-Bottleneck-LLM/generation/test_perplexity.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="SetFit/sst2" --intervention_value=0 --intervention_class=0
python Concept-Bottleneck-LLM/generation/test_perplexity.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="SetFit/sst2" --intervention_value=0 --intervention_class=1

python Concept-Bottleneck-LLM/generation/test_perplexity.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="SetFit/sst2" --intervention_value=100 --intervention_class=0
python Concept-Bottleneck-LLM/generation/test_perplexity.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="SetFit/sst2" --intervention_value=100 --intervention_class=1

# Steerability: when no intervention, when zero intervention, when non-zero intervention
python Concept-Bottleneck-LLM/generation/test_steerability.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="SetFit/sst2"
python Concept-Bottleneck-LLM/generation/test_steerability.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="SetFit/sst2" --intervention_class=0
python Concept-Bottleneck-LLM/generation/test_steerability.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="SetFit/sst2" --intervention_class=1

python Concept-Bottleneck-LLM/generation/test_steerability.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="SetFit/sst2" --intervention_value=0
python Concept-Bottleneck-LLM/generation/test_steerability.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="SetFit/sst2" --intervention_value=0 --intervention_class=0
python Concept-Bottleneck-LLM/generation/test_steerability.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="SetFit/sst2" --intervention_value=0 --intervention_class=1

python Concept-Bottleneck-LLM/generation/test_steerability.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="SetFit/sst2" --intervention_value=100
python Concept-Bottleneck-LLM/generation/test_steerability.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="SetFit/sst2" --intervention_value=100 --intervention_class=0
python Concept-Bottleneck-LLM/generation/test_steerability.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="SetFit/sst2" --intervention_value=100 --intervention_class=1

# Generation: when no intervention, when zero intervention, when non-zero intervention
python Concept-Bottleneck-LLM/generation/test_generation.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="SetFit/sst2"
python Concept-Bottleneck-LLM/generation/test_generation.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="SetFit/sst2" --intervention_value=0
python Concept-Bottleneck-LLM/generation/test_generation.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="SetFit/sst2" --intervention_value=100