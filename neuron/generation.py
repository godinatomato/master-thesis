#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pickle
from typing import Sequence, Dict, Tuple, Union, List

import numpy as np
import os
import pandas as pd
import random
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from neuron.models import PytorchTransformersModel


MAX_LENGTH: int = 10000  # Hardcoded max length to avoid infinite loop
EOT_TOKEN = "<|endoftext|>"


def set_seed(seed, gpu: bool):
    if seed:
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        if gpu:
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True  # type: ignore
            torch.backends.cudnn.benchmark = False  # type: ignore
        np.random.seed(seed)
        random.seed(seed)


def decode_sentence(
    token_ids: Sequence[torch.Tensor], tokenizer: PreTrainedTokenizer
) -> str:
    sentence = tokenizer.decode(
        token_ids, clean_up_tokenization_spaces=True, skip_special_tokens=True
    )
    return sentence


def sample_sequence(
    model: torch.nn.Module,
    length: int,
    inputs: Dict,
    temperature: float = 0.0,
) -> torch.Tensor:
    inputs = {k: v.to(torch.device("cuda", 0)) for k, v in inputs.items()}

    past = None
    last_token = None
    inputs["use_cache"] = True
    generated = inputs["input_ids"].cpu()

    with torch.no_grad():
        for _ in range(length):
            inputs["past_key_values"] = past
            if last_token is not None:
                inputs["input_ids"] = last_token.unsqueeze(0)
                inputs["attention_mask"] = torch.cat(
                    (
                        inputs["attention_mask"],
                        torch.tensor([1]).unsqueeze(-1).to(torch.device("cuda", 0)),
                    ),
                    dim=1,
                )

            outputs = model(**inputs)
            past = outputs.past_key_values
            if temperature == 0.0:
                next_token_logits = outputs.logits[0, -1, :]
                last_token = torch.argmax(next_token_logits.float(), dim=-1).unsqueeze(
                    -1
                )
            else:
                next_token_logits = outputs.logits[0, -1, :] / temperature
                last_token = torch.multinomial(
                    F.softmax(next_token_logits.float(), dim=-1), num_samples=1
                )
            generated = torch.cat((generated, last_token.unsqueeze(0).cpu()), dim=1)

        del inputs

    return generated


def generate_sentence(
    model: PreTrainedModel,
    tokenizer,
    prompt: str,
    length: int = 128,
    temperature: float = 0.0,
) -> Tuple[str, float]:
    raw_prompt_text = prompt
    inputs = tokenizer(raw_prompt_text, return_tensors="pt")

    out = sample_sequence(
        model=model,
        length=length,
        inputs=inputs,
        temperature=temperature,
    )

    out_list = out[0, :].tolist()
    generated_sentence = decode_sentence(out_list, tokenizer)

    return generated_sentence


def force_units_hooks(
    model: PytorchTransformersModel,
    expertise: pd.DataFrame,
    value: str,
    metric: str,
    num_units: int = 1,
    top_n: int = 1,
    use_layers: Union[str, List[str]] = None,
    only_last_token: bool = False,
) -> Tuple[PytorchTransformersModel, pd.DataFrame]:
    assert value is not None

    if use_layers is None:
        use_layers = []
    elif isinstance(use_layers, str):
        use_layers = [
            use_layers,
        ]

    if len(use_layers) > 0:
        selected_rows = expertise["layer"].str.contains("|".join(use_layers))
        df = expertise[selected_rows].copy()
    else:
        df = expertise.copy()

    if top_n <= 0:
        rs = np.random.RandomState(None)
        df = df.sample(n=num_units, replace=False, random_state=rs)
    else:
        df = df.sort_values(by=metric, ascending=False).iloc[
            range((top_n - 1) * num_units, top_n * num_units)
        ]

    for layer_name, layer_df in df.groupby("layer", sort=True):
        units_force = torch.tensor(layer_df["unit"].values, dtype=torch.int64)
        if value == "zero":
            vals_force = torch.zeros_like(units_force, dtype=torch.float32)
        else:
            vals_force = torch.tensor(layer_df[value].values, dtype=torch.float32)

        model.set_units_in_layer(
            layer_name=layer_name,
            units=units_force,
            values=vals_force,
            only_last_token=only_last_token,
        )

    return model, df


def get_translation_prompt(input_path: str, prompt_format_id: int, seed: int) -> str:
    with open(input_path, "rb") as f:
        source_text = pickle.load(f)[seed - 1]["en"]
    
    if prompt_format_id == 0:
        prompt = f"Translate a sentence from English to a target language.\nEnglish: {source_text}\nTarget Language:"
    elif prompt_format_id == 1:
        prompt = f"Translate English to a target language.\nEnglish: {source_text}\nTarget Language:"
    elif prompt_format_id == 2:
        prompt = f"Translate an English sentence into a target language.\nEnglish: {source_text}\nTarget Language:"
    elif prompt_format_id == 3:
        prompt = f"Translate an English sentence into German.\nEnglish: {source_text}\nGerman:"
    elif prompt_format_id == 4:
        prompt = f"Translate an English sentence into Japanese.\nEnglish: {source_text}\nJapanese:"
    elif prompt_format_id == 5:
        prompt = f"Translate an English sentence into French.\nEnglish: {source_text}\nFrench:"
    elif prompt_format_id == 6:
        prompt = f"Translate an English sentence into Spanish.\nEnglish: {source_text}\nSpanish:"
    elif prompt_format_id == 7:
        prompt = f"Translate an English sentence into Chinese.\nEnglish: {source_text}\nChinese:"
    else:
        raise ValueError(
            "error! prompt_format_id_for_translation is not properly defined!"
        )
        
    return prompt
