# Records additional metrics after running script_llama321b-instruct_sst2.sh
# NOTE: Since we are running interventions, concept detection will always result in the intervened concept

# Examples of steering with intervention
python generation/test_generation.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="SetFit/sst2" --intervention_value=100

# Examples of steering without intervention
python generation/test_generation.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="SetFit/sst2"

# Per-concept Perplexity (with intervention)
python generation/test_perplexity.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="SetFit/sst2" --intervention_value=100 --intervention_class=0
python generation/test_perplexity.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="SetFit/sst2" --intervention_value=100 --intervention_class=1

# Per-concept Steerability (with intervention)
python generation/test_steerability.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="SetFit/sst2" --intervention_value=100 --intervention_class=0
python generation/test_steerability.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="SetFit/sst2" --intervention_value=100 --intervention_class=1

# Zero-out all concepts and recompute metrics
python generation/test_perplexity.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="SetFit/sst2" --intervention_value=0 --intervention_class=0 # any class is fine here
python generation/test_steerability.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="SetFit/sst2" --intervention_value=0
python generation/test_generation.py --model_id="unsloth/Llama-3.2-1B-Instruct" --dataset="SetFit/sst2" --intervention_value=0