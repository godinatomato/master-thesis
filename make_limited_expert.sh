model_name=$1
language=$2
data_type=$3

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

cp config/default/make_limited_expert_config.yaml config/
python scripts/make_limited_expert.py model_name=${full_model_name} lang=${language} data_type=${data_type}
