#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path

import hydra
import torch

from neuron.data import LangDataset
from neuron.tokenizer import PytorchTransformersTokenizer
from neuron.responses import cache_responses
from neuron.models import collect_responses_info, PytorchTransformersModel


def compute_and_save_responses(
    model_name: str,
    data_path: Path,
    data_type: str,
    lang: str,
    batch_size: int,
    response_save_path: Path,
    num_per_lang: int,
    seq_len: int,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_short_name = model_name.rstrip("/").split("/")[-1]
    local_data_file = data_path / f"{lang}.json"
    if not local_data_file.exists():
        print(f"Skipping {local_data_file}, file not found.")
        return

    save_path = response_save_path / model_short_name / data_type / lang
    if (
        response_save_path / model_short_name / data_type / lang / "responses"
    ).exists():
        print(f"Skipping, already computed responses {local_data_file}")
        return
    save_path.mkdir(parents=True, exist_ok=True)

    seed = 0

    tokenizer = PytorchTransformersTokenizer(model_name=model_name, cache_dir=None)
    dataset = LangDataset(
        json_file=local_data_file,
        seq_len=seq_len,
        num_per_lang=num_per_lang,
        random_seed=seed,
        tokenizer=tokenizer,
    )
    tm_model = PytorchTransformersModel(
        model_name, seq_len=dataset.seq_len, cache_dir=None, device=device
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


@hydra.main(
    config_path="../config", config_name="compute_responses_config", version_base=None
)
def main(cfg):
    data_path = Path(cfg.data_path) / cfg.data_type
    responses_path = Path(cfg.responses_path)
    responses_path.mkdir(exist_ok=True, parents=True)

    compute_and_save_responses(
        model_name=cfg.model_name,
        data_path=data_path,
        data_type=cfg.data_type,
        lang=cfg.lang,
        batch_size=cfg.inf_batch_size,
        response_save_path=responses_path,
        num_per_lang=cfg.num_per_lang,
        seq_len=cfg.seq_len,
    )


if __name__ == "__main__":
    main()
