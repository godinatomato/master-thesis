MODEL_NAMES=("xglm-564M" "bloom-1b7")
LABELS=("positive" "neutral" "negative")

data_type="Sentiment" 

for model_name in ${MODEL_NAMES[@]}
do
    for label in ${LABELS[@]}
    do
        echo "Start experiment for label ${label}."

        echo "-------------------"
        echo "compute_responses"
        bash compute_responses.sh "${model_name}" "${label}" "${data_type}"

        echo "-------------------"
        echo "compute_expertise"
        bash compute_expertise.sh "${model_name}" "${label}" "${data_type}"

        echo "-------------------"
        echo "make_limited_expert"
        bash make_limited_expert.sh "${model_name}" "${label}" "${data_type}"

        echo "-------------------"
        echo "generate_sequences"
        bash generate_sequences.sh "${model_name}" "${label}" "${data_type}"
    done
done
