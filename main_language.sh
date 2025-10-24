# MODEL_NAMES=("xglm-564M")
# MODEL_NAMES=("Llama-2-7b-hf")
# MODEL_NAMES=("Llama-2-13b-hf")
MODEL_NAMES=("bloom-1b7")
LANGUAGES=("de" "en" "es" "fr" "ja" "zh")

data_type="Language"

for model_name in ${MODEL_NAMES[@]}
do
    for language in ${LANGUAGES[@]}
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
