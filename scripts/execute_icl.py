import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import random
from typing import List, Union

import hydra
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from neuron.icl import preprocess_for_icl
from neuron.generation import force_units_hooks, generate_sentence, set_seed
from neuron.data import ICLPromptDataset
from neuron.models import PytorchTransformersModel


# TODO: set seedしたい


def execute_icl(
    model_name_or_path: str,
    cache_dir: str,
    lang: str,
    max_samples: int,
    n_examples: int,
    expertise_path: str,
    metric: str,
    length: int,
    temperature: float,
    per_layer: bool = False,
    only_last_token: bool = False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    full_dataset = load_dataset("xnli", lang, split="validation")
    if len(full_dataset) > max_samples:
        indices = random.sample(range(len(full_dataset)), max_samples)
        hf_dataset_subset = full_dataset.select(indices)
    else:
        hf_dataset_subset = full_dataset

    processed_icl_data = preprocess_for_icl(hf_dataset_subset, n_examples, lang)

    icl_dataset = ICLPromptDataset(processed_icl_data)

    expertise = pd.read_csv(expertise_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    readable_model = PytorchTransformersModel(
        model_name=model_name_or_path,
        seq_len=128,
        cache_dir=cache_dir,
        device=device,
    )

    ###
    forcing_values = ["on_p50"]
    top_ns = []
    layer_names = (
        list(expertise.sort_values("layer").layer.unique())
        if per_layer
        else [
            None,
        ]
    )
    num_units_list = [2000, 0]
    ###
    icl_results = []

    for forcing_value in forcing_values:
        for top_n in top_ns:
            for force_layer in layer_names:
                for num_units in num_units_list:
                    pbar = tqdm(
                        total=len(icl_dataset),
                        desc=(
                            "Generating"
                            f" [force={forcing_value} units={num_units}/{len(expertise)} ({100 * num_units / len(expertise):0.3f}%)"
                            f" top_n={top_n} layers={force_layer}]"
                        ),
                    )

                    if num_units > 0:
                        model, df_force = force_units_hooks(
                            model=readable_model,
                            expertise=expertise,
                            value=forcing_value,
                            metric=metric,
                            num_units=num_units,
                            top_n=top_n,
                            use_layers=force_layer,
                            only_last_token=only_last_token,
                        )
                    else:
                        model = readable_model

                    for prompt in icl_dataset:
                        sentence = generate_sentence(
                            model=readable_model.module,
                            tokenizer=tokenizer,
                            prompt=prompt,
                            length=length,
                            temperature=temperature,
                        )
                        icl_results.append(
                            [forcing_value, num_units, top_n, sentence, force_layer]
                        )
                        pbar.update()

                    if num_units > 0:
                        readable_model.restore_units()

                    pbar.close()


@hydra.main(
    config_path="../config", config_name="execute_icl_config", version_base=None
)
def main(cfg):
    execute_icl(cfg.lang, cfg.max_samples, cfg.n_examples)


if __name__ == "__main__":
    main()
