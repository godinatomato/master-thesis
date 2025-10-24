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

for model_name in ${MODEL_NAMES[@]}
do
    for language in ${LANGS_WITH_LABELS[@]}
    do
        echo "Start experiment for language ${language}."

        echo "-------------------"
        echo "compute_responses"
        bash compute_responses.sh "${model_name}" "${language}" "${data_type}"

        echo "-------------------"
        echo "compute_expertise"
        bash compute_expertise.sh "${model_name}" "${language}" "${data_type}"

        echo "-------------------"
        echo "make_limited_expert"
        bash make_limited_expert.sh "${model_name}" "${language}" "${data_type}"

        echo "-------------------"
        echo "generate_sequences"
        bash generate_sequences.sh "${model_name}" "${language}" "${data_type}"
    done
done
