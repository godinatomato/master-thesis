#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

# Original file from:
#
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
from typing import List, Union

import hydra
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from neuron.generation import force_units_hooks, generate_sentence, set_seed, get_translation_prompt
from neuron.models import PytorchTransformersModel


def generate(
    model_name: str,
    prompt: str,
    forcing_value: str,
    top_n: int,
    num_units: int,
    generation_length: int,
    temperature: float,
    metric: str,
    seed: List[int],
    expertise_path: Path,
    results_path: Union[Path, None] = None,
    prompt_format_id: int = 1,
    per_layer: bool = False,
    only_last_token: bool = False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    expertise = pd.read_csv(expertise_path)
    lang = expertise["lang"].values[0]

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

    sweep_seed = range(seed[0], seed[1]) if len(seed) == 2 else seed
    for force_layer in layer_names:
        pbar = tqdm(
            total=len(sweep_seed),
            desc=(
                "Generating"
                f" [force={forcing_value} units={num_units}/{len(expertise)} ({100 * num_units / len(expertise):0.3f}%)"
                f" top_n={top_n} layers={force_layer}]"
            ),
        )

        mean_metric = 0
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
            mean_metric = float(df_force[metric].mean())
        else:
            model = readable_model
            mean_metric = 0.0

        for seed in sweep_seed:
            set_seed(seed, gpu=device != "cpu")
                
            if "translation_text_" in prompt:
                prompt = get_translation_prompt(
                    input_path=prompt,
                    prompt_format_id=prompt_format_id,
                    seed=seed
                )
            else:
                prompt = prompt

            sentence = generate_sentence(
                model=readable_model.module,
                tokenizer=tokenizer,
                prompt=prompt,
                length=generation_length,
                temperature=temperature,
            )
            generation_results.append(
                [
                    forcing_value,
                    num_units,
                    top_n,
                    seed,
                    sentence,
                    mean_metric,
                    force_layer,
                ]
            )
            pbar.update()

        if num_units > 0:
            readable_model.restore_units()

        pbar.close()

    if results_path is None:
        results_path: Path = (
            expertise_path.parent
            / f"forced_sentences_{lang}_{prompt.replace('_', '')}.csv"
        )
    else:
        results_path: Path = results_path

    generated_df = pd.DataFrame(
        columns=[
            "forcing_value",
            "num_units",
            "top_n",
            "seed",
            "sentence",
            "mean_metric",
            "forced_layer",
        ],
        data=generation_results,
    )
    generated_df["context"] = [prompt] * len(generated_df)
    generated_df["lang"] = [lang] * len(generated_df)
    generated_df.to_csv(results_path)

@hydra.main(config_path="../config", config_name="generate_sequences_config", version_base=None)
def main(cfg):
    model_short_name = cfg.model_name.rstrip("/").split("/")[-1]
    expertise_path = Path(cfg.output_dir) / model_short_name / cfg.data_type / cfg.lang / f"expertise/{cfg.expert_file}.csv"
    results_path = Path(cfg.output_dir) / model_short_name / cfg.data_type / cfg.lang / f"expertise/generated_sentence_{cfg.forcing_value}_{cfg.num_units}_{cfg.expert_file}.csv"
    
    generate(
        model_name=cfg.model_name,
        prompt=cfg.prompt,
        forcing_value=cfg.forcing_value,
        top_n=cfg.top_n,
        num_units=cfg.num_units,
        generation_length=cfg.generation_length,
        temperature=cfg.temperature,
        metric=cfg.metric,
        seed=cfg.seed,
        expertise_path=expertise_path,
        results_path=results_path,
        prompt_format_id=cfg.prompt_format_id,
        per_layer=cfg.per_layer,
        only_last_token=cfg.only_last_token,
    )


if __name__ == "__main__":
    main()
