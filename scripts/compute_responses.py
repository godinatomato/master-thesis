#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import logging
import pathlib

import hydra

from neuron.data import lang_list_to_df, LangDataset
from neuron.tokenizer import PytorchTransformersTokenizer
from neuron.responses import cache_responses
from neuron.models import collect_responses_info, PytorchTransformersModel

logging.basicConfig(level=logging.WARNING)


def compute_and_save_responses(
    model_name: str,
    model_cache_dir: pathlib.Path,
    data_path: pathlib.Path,
    lang_group: str,
    lang: str,
    tokenizer: PytorchTransformersTokenizer,
    batch_size: int,
    response_save_path: pathlib.Path,
    num_per_lang: int,
    seq_len: int,
    device: str,
) -> None:
    model_short_name = model_name.rstrip("/").split("/")[-1]
    local_data_file = data_path / lang_group / f"{lang}.json"
    if not local_data_file.exists():
        print(f"Skipping {local_data_file}, file not found.")
        return

    if (
        response_save_path / model_short_name / lang_group / lang / "responses"
    ).exists():
        print(response_save_path / model_short_name / lang_group / lang / "responses")
        print(f"Skipping, already computed responses {local_data_file}")
        return

    random_seed = 1234

    dataset = LangDataset(
        json_file=local_data_file,
        seq_len=seq_len,
        num_per_lang=num_per_lang,
        random_seed=random_seed,
        tokenizer=tokenizer,
    )

    save_path = (
        response_save_path / model_short_name / dataset.lang_group / dataset.lang
    )
    if (save_path / "responses").exists():
        print(f"Skipping {dataset.lang_group}/{dataset.lang}")
        return

    save_path.mkdir(parents=True, exist_ok=True)

    tm_model = PytorchTransformersModel(
        model_name, seq_len=dataset.seq_len, cache_dir=model_cache_dir, device=device
    )

    responses_info_interm = collect_responses_info(
        model_name=model_name, model=tm_model
    )

    cache_responses(
        model=tm_model,
        dataset=dataset,
        batch_size=batch_size,
        response_infos=responses_info_interm,
        save_path=save_path / "responses",
    )

    return


@hydra.main(
    config_path="../config", config_name="compute_responses_config", version_base=None
)
def main(cfg):
    model_cache = cfg.model_cache if cfg.model_cache else None
    tokenizer_cache = cfg.tokenizer_cache
    if cfg.model_cache is not None and tokenizer_cache is None:
        tokenizer_cache = cfg.model_cache

    data_path = cfg.data_path
    responses_path = cfg.responses_path
    responses_path.mkdir(exist_ok=True, parents=True)

    if not cfg.langs:
        assert (data_path / "lang_list.csv").exists()
        langs_requested = data_path / "lang_list.csv"
    else:
        langs_requested = cfg.langs.split(",")

    lang_df = lang_list_to_df(langs_requested)

    tokenizer = PytorchTransformersTokenizer(cfg.model_name_or_path, tokenizer_cache)

    for _, row in lang_df.iterrows():
        lang, lang_group = row["lang"], row["group"]

        if lang in ["positive", "negative"] and lang_group == "keyword":
            continue

        compute_and_save_responses(
            model_name=cfg.model_name_or_path,
            model_cache_dir=model_cache,
            data_path=data_path,
            lang_group=lang_group,
            lang=lang,
            seq_len=cfg.seq_len,
            num_per_lang=cfg.num_per_lang,
            batch_size=cfg.inf_batch_size,
            response_save_path=responses_path,
            tokenizer=tokenizer,
            device=cfg.device,
        )


if __name__ == "__main__":
    main()
