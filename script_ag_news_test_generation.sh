prompt=""
echo $prompt
python Concept-Bottleneck-LLM/generation/test_generation.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="SetFit/sst2" --intervention_value=0 --prompt="$prompt"
python Concept-Bottleneck-LLM/generation/test_generation.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="SetFit/sst2" --intervention_value=100 --prompt="$prompt"