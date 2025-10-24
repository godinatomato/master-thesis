MODEL_NAMES=("xglm-564M")
LANGUAGES=("de" "en" "es" "fr" "ja" "zh")
LABELS=("positive" "neutral" "negative")
LANGS_WITH_LABELS=()
for language in ${LANGUAGES[@]}
do
    for label in ${LABELS[@]}
    do
        LANGS_WITH_LABELS+=("${language}_${label}")
    done
done

data_type="Sentiment"

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

for model_name in ${MODEL_NAMES[@]}
do
    for language in ${LANGS_WITH_LABELS[@]}
    do
        python scripts/analyze_neurons.py model_name=${model_name} lang=${language} data_type=${data_type}
    done
done
