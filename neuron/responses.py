#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pathlib
import pickle
import typing as t

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from neuron.models import (
    TorchModel,
    ResponseInfo,
    processors_per_model,
    MODEL_INPUT_FIELDS,
    LABELS_FIELD,
)


def save_batch(
    batch: t.Dict[str, np.ndarray], batch_index: int, save_path: pathlib.Path
) -> None:
    with (save_path / f"{batch_index:05d}.pkl").open("wb") as fp:
        pickle.dump(batch, fp)


def cache_responses(
    model: TorchModel,
    dataset: Dataset,
    response_infos: t.List[ResponseInfo],
    batch_size: int,
    save_path: pathlib.Path,
) -> None:
    def _concatenate_data(x):
        new_batch = dict()
        for key in x[0].keys():
            if isinstance(x[0][key], str):
                new_batch[key] = np.array([x[idx][key] for idx in range(len(x))])
            else:
                new_batch[key] = torch.tensor([x[idx][key] for idx in range(len(x))])
        return new_batch

    save_path.mkdir(parents=True, exist_ok=True)
    process_fn_list = processors_per_model(model)

    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=_concatenate_data
    )
    for i, batch in tqdm(enumerate(data_loader), desc="Caching inference"):
        input_batch = {k: v for k, v in batch.items() if k in MODEL_INPUT_FIELDS}
        num_effective_token = torch.sum(input_batch["attention_mask"][0])
        num_effective_token = num_effective_token.detach().cpu().item()
        input_batch["attention_mask"] = input_batch["attention_mask"][
            :, :num_effective_token
        ]
        input_batch["input_ids"] = input_batch["input_ids"][:, :num_effective_token]
        response_batch = model.run_inference(
            inputs=input_batch, outputs={ri.name for ri in response_infos}
        )
        for process_fn in process_fn_list:
            response_batch = process_fn(response_batch)
        response_batch[LABELS_FIELD] = batch[LABELS_FIELD].detach().cpu().numpy()
        save_batch(batch=response_batch, batch_index=i, save_path=save_path)


def read_responses_from_cached(
    cached_dir: pathlib.Path, lang: str
) -> t.Tuple[t.Dict[str, np.ndarray], t.Optional[np.ndarray], t.Set[str]]:
    data: t.Dict[str, np.ndarray] = {}
    labels: t.List[float] = []
    response_names: t.Set[str] = set()
    labels_name = LABELS_FIELD
    all_files = sorted(list(cached_dir.glob("*.pkl")))
    if not all_files:
        raise RuntimeError("No responses found")

    data_as_lists: t.Dict[str, t.List[np.ndarray]] = {}
    for file_name in tqdm(all_files, total=len(all_files), desc=f"Loading {lang}"):
        with file_name.open("rb") as fp:
            response_batch = pickle.load(fp)
            if not response_names:
                response_names = set(response_batch.keys()) - set(labels_name)
            for l_name in response_names:
                if l_name not in data_as_lists:
                    data_as_lists[l_name] = []
                data_as_lists[l_name].append(response_batch[l_name].tolist())
            if LABELS_FIELD in response_batch:
                labels.extend(response_batch[LABELS_FIELD])
    for l_name in data_as_lists.keys():
        if l_name in ["labels"]:
            continue
        data[l_name] = np.concatenate(data_as_lists[l_name], axis=0).transpose()
        assert len(data[l_name].shape) == 2, "Wrong dimensionality of responses"

    return data, np.array(labels) if labels else None, response_names
