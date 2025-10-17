model_name=${args[0]}
language=${args[1]}

if [[ ${model_name} == *"xglm"* ]]; then
  full_model_name="facebook/${model_name}"
elif [[ ${model_name} == *"bloom"* ]]; then
  full_model_name="bigscience/${model_name}"
elif [[ ${model_name} == *"Llama-2"* ]]; then
  full_model_name="meta-llama/${model_name}"
else
  echo "Model name ${model_name} is not supported."
  exit
fi

cp config/default/compute_responses_config.yaml config/
python scripts/compute_responses.py model_name_or_path=${full_model_name} langs=sense/${language}