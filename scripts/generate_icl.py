import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
from pathlib import Path
from typing import Union

import hydra
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from neuron.icl import create_icl_dataset
from neuron.generation import force_units_hooks, generate_sentence, set_seed
from neuron.models import PytorchTransformersModel


def generate(
    model_name: str,
    dataset_name: str,
    n_examples: int,
    forcing_value: str,
    top_n: int,
    num_units: int,
    generation_length: int,
    temperature: float,
    metric: str,
    data_path: Path,
    expertise_path: Path,
    results_path: Union[Path, None] = None,
    per_layer: bool = False,
    only_last_token: bool = False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    expertise = pd.read_csv(expertise_path)
    lang = expertise["lang"].values[0]

    with open(data_path, "r") as f:
        data = json.load(f)

    dataset = create_icl_dataset(
        data=data,
        dataset_name=dataset_name,
        n_examples=n_examples,
        lang=lang,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    readable_model = PytorchTransformersModel(
        model_name=model_name,
        seq_len=128,
        cache_dir=None,
        device=device,
    )

    layer_names = (
        list(expertise.sort_values("layer").layer.unique())
        if per_layer
        else [
            None,
        ]
    )
    
    generation_results = []
    set_seed(0, gpu=device != "cpu")

    for force_layer in layer_names:
        pbar = tqdm(
            total=len(dataset),
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

        for inputs in dataset:
            sentence = generate_sentence(
                model=readable_model.module,
                tokenizer=tokenizer,
                prompt=inputs["prompt"],
                length=generation_length,
                temperature=temperature,
            )
            generation_results.append(
                [forcing_value, num_units, top_n, inputs["prompt"], inputs["label"], sentence, force_layer]
            )
            pbar.update()

        if num_units > 0:
            readable_model.restore_units()

        pbar.close()
        
    if results_path is None:
        results_path: Path = (
            expertise_path.parent
            / f"icl_sentences_{lang}.csv"
        )
    else:
        results_path: Path = results_path
        
    generated_df = pd.DataFrame(
        columns=[
            "forcing_value",
            "num_units",
            "top_n",
            "prompt",
            "label",
            "sentence",
            "forced_layer",
        ],
        data=generation_results,
    )
    generated_df["lang"] = [lang] * len(generated_df)
    generated_df.to_csv(results_path)
    

@hydra.main(
    config_path="../config", config_name="generate_icl_config", version_base=None
)
def main(cfg):
    model_short_name = cfg.model_name.rstrip("/").split("/")[-1]
    expertise_path = Path(cfg.output_dir) / model_short_name / cfg.data_type / cfg.lang / f"expertise/{cfg.expert_file}.csv"
    results_path = Path(cfg.output_dir) / model_short_name / cfg.data_type / cfg.lang / f"expertise/icl_sentence_{cfg.forcing_value}_{cfg.num_units}_{cfg.expert_file}.csv"
    data_path = Path(cfg.data_dir) / cfg.dataset_name / f"{cfg.lang}.json"
    
    generate(
        model_name=cfg.model_name,
        dataset_name=cfg.dataset_name,
        n_examples=cfg.n_examples,
        forcing_value=cfg.forcing_value,
        top_n=cfg.top_n,
        num_units=cfg.num_units,
        generation_length=cfg.generation_length,
        temperature=cfg.temperature,
        metric=cfg.metric,
        data_path=data_path,
        expertise_path=expertise_path,
        results_path=results_path,
        per_layer=cfg.per_layer,
        only_last_token=cfg.only_last_token,
    )


if __name__ == "__main__":
    main()
