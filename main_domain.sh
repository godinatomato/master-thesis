MODEL_NAMES=("xglm-564M" "bloom-1b7")
DOMAINS=("daily" "financial" "literature" "mathematics" "medicine")

data_type="Domain" 

for model_name in ${MODEL_NAMES[@]}
do
    for domain in ${DOMAINS[@]}
    do
        echo "Start experiment for domain ${domain}."

        echo "-------------------"
        echo "compute_responses"
        bash compute_responses.sh "${model_name}" "${domain}" "${data_type}"

        echo "-------------------"
        echo "compute_expertise"
        bash compute_expertise.sh "${model_name}" "${domain}" "${data_type}"

        echo "-------------------"
        echo "make_limited_expert"
        bash make_limited_expert.sh "${model_name}" "${domain}" "${data_type}"

        echo "-------------------"
        echo "generate_sequences"
        bash generate_sequences.sh "${model_name}" "${domain}" "${data_type}"
    done
done
