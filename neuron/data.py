#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import json
import pathlib
from typing import Any, List, Iterable, Dict, Union, Tuple, Optional

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tokenizer import PytorchTransformersTokenizer


class DatasetForSeqModels(Dataset):
    def __init__(
        self,
        path: pathlib.Path,
        seq_len: int,
        tokenizer: PytorchTransformersTokenizer,
        num_per_lang: int = None,
        random_seed: int = None,
    ) -> None:
        super().__init__()
        self._data: Dict[str, List] = {}
        self._model_input_fields: List[str] = []
        self._seq_len = seq_len
        self._num_per_lang = num_per_lang
        self._tokenizer = tokenizer

        unprocessed_data, self._labels = self._load_data(
            path=path,
            seq_len=seq_len,
            random_seed=random_seed,
            num_per_lang=num_per_lang,
        )

        preprocessed_named_data = self._tokenizer.preprocess_dataset(
            sentence_list=unprocessed_data,
            min_num_tokens=self.seq_len,
        )

        self._model_input_fields = list(preprocessed_named_data.keys())

        self._data["data"] = unprocessed_data
        self._data["labels"] = self._labels
        self._data.update(preprocessed_named_data)

        self._remove_too_long_data()

        self._verify_data_integrity()

    def __str__(self) -> str:
        msg = f"Dataset fields:\n"
        msg += f'\tData {len(self._data["data"])}'
        msg += f'\tTokens {np.array(self._data["input_ids"]).shape}\n'
        msg += f"\tLang Labels\n"
        v = self.data["labels"]
        msg += f"\t\t {np.sum(v)}/{len(v) - np.sum(v)} pos/neg examples.\n"
        return msg

    def _load_data(
        self,
        path: pathlib.Path,
        seq_len: int = 20,
        num_per_lang: int = None,
        random_seed: int = None,
    ) -> Tuple[List[str], List[int]]:
        """TO BE IMPLEMENTED IN CHILD CLASSES"""
        pass

    def _verify_data_integrity(self) -> None:
        for k, v in self.data.items():
            assert isinstance(v, list)
            msg = f"Dataset field {k}: List of {type(v[0])}"
            assert (
                isinstance(v[0], list) or isinstance(v[0], str) or isinstance(v[0], int)
            ), type(v[0])
            msg += f" of {type(v[0][0])}" if isinstance(v[0], list) else ""
        assert isinstance(self._data["input_ids"][0], list)
        assert isinstance(self._data["input_ids"][0][0], int)

    def _remove_too_long_data(self) -> None:
        remove_idx = []
        for idx, tokens in enumerate(self._data["input_ids"]):
            extra_tokens = 0
            if len(tokens) > self.seq_len + extra_tokens:
                print(
                    f"Removing data ({len(tokens) - extra_tokens} > {self.seq_len} tokens)"
                )
                remove_idx.append(idx)
        remove_idx = sorted(remove_idx, reverse=True)

        for key in self._data.keys():
            for i in remove_idx:
                del self._data[key][i]

    def get_input_fields(self) -> List[Union[str, Iterable[str]]]:
        return list(self._model_input_fields)

    @property
    def data(self) -> Dict[str, List]:
        return self._data

    @property
    def seq_len(self) -> int:
        return self._seq_len

    @property
    def num_per_lang(self) -> Optional[int]:
        return self._num_per_lang

    def __len__(self):
        return len(self._data[list(self._data.keys())[0]])

    def __getitem__(self, idx):
        batch_data = {k: self.data[k][idx] for k in self.data.keys()}
        return batch_data


class LangDataset(DatasetForSeqModels):
    def __init__(
        self,
        json_file: pathlib.Path,
        tokenizer: PytorchTransformersTokenizer,
        seq_len: int = 100,
        num_per_lang: int = None,
        random_seed: int = None,
    ) -> None:
        print(f"Creating dataset from {json_file}")
        with json_file.open("r", encoding="utf-8") as fp:
            json_data = json.load(fp)
        self._lang = json_data["lang"]
        self._lang_group = json_data["group"]
        super().__init__(
            path=json_file,
            seq_len=seq_len,
            num_per_lang=num_per_lang,
            tokenizer=tokenizer,
            random_seed=random_seed,
        )

    def _load_data(
        self,
        path: pathlib.Path,
        num_per_lang: int = None,
        random_seed: int = None,
    ) -> Tuple[List[str], List[int]]:
        random_state = np.random.RandomState(random_seed)

        label_map = {"positive": 1, "negative": 0}

        with path.open("r") as fp:
            json_data = json.load(fp)

        json_sentences = json_data["sentences"]

        unique_labels = sorted(list(json_sentences.keys()))

        sentences: List[str] = []
        labels: List[int] = []
        for label in unique_labels:
            if num_per_lang is not None and num_per_lang < len(json_sentences[label]):
                idx = random_state.choice(
                    len(json_sentences[label]), num_per_lang, replace=False
                )
            else:
                idx = np.arange(len(json_sentences[label]))
            sentences += [json_sentences[label][i] for i in idx]
            labels += [label_map[label]] * len(idx)
        return sentences, labels

    @property
    def lang(self) -> str:
        return self._lang

    @property
    def lang_group(self) -> str:
        return self._lang_group


class ICLPromptDataset(Dataset):
    def __init__(self, processed_data: List[Dict[str, Any]]):
        self.data = processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]

    """
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]

        return {
            "prompt": sample["prompt"],
            "true_label_id": sample["true_label_id"],
            "true_label_string": sample["true_label_string"],
        }
    """
