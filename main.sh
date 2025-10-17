MODEL_NAMES=("xglm-564M")
LANGUAGES=("de" "en" "es" "fr" "ja" "zh")

for model_name in ${MODEL_NAMES[@]}
do
    for language in ${LANGUAGES[@]}
    do
        echo "Start experiment for language ${language}."

        bash compute_responses.sh "${model_name} ${language}"
    done
done
