MODEL_NAMES=("xglm-564M")
LANGUAGES=("de" "en" "es" "fr" "zh")

data_type="Language"

for model_name in ${MODEL_NAMES[@]}
do
    for language in ${LANGUAGES[@]}
    do
        echo "Start experiment for language ${language}."

        echo "-------------------"
        echo "prepare_icl_dataset"
        bash prepare_icl_dataset.sh "${language}"

        echo "-------------------"
        echo "generate_icl"
        bash generate_icl.sh "${model_name}" "${language}" "${data_type}"
    done
done
