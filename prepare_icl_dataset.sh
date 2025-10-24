language=$1

cp config/default/prepare_icl_dataset_config.yaml config/
python scripts/prepare_icl_dataset.py lang=${language}