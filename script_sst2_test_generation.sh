prompt=""
echo $prompt
python Concept-Bottleneck-LLM/generation/test_generation.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="SetFit/sst2" --intervention_class=0 --prompt="$prompt"
python Concept-Bottleneck-LLM/generation/test_generation.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="SetFit/sst2" --intervention_class=0 --intervention_value=0 --prompt="$prompt"
prompt="This movie was"
echo $prompt
python Concept-Bottleneck-LLM/generation/test_generation.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="SetFit/sst2" --intervention_class=0 --prompt="$prompt"
python Concept-Bottleneck-LLM/generation/test_generation.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="SetFit/sst2" --intervention_class=0 --intervention_value=0 --prompt="$prompt"
prompt="This movie was very good"
echo $prompt
python Concept-Bottleneck-LLM/generation/test_generation.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="SetFit/sst2" --intervention_class=0 --prompt="$prompt"
python Concept-Bottleneck-LLM/generation/test_generation.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="SetFit/sst2" --intervention_class=0 --intervention_value=0 --prompt="$prompt"
prompt="This movie was very bad"
echo $prompt
python Concept-Bottleneck-LLM/generation/test_generation.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="SetFit/sst2" --intervention_class=0 --prompt="$prompt"
python Concept-Bottleneck-LLM/generation/test_generation.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="SetFit/sst2" --intervention_class=0 --intervention_value=0 --prompt="$prompt"